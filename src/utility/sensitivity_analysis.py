import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path("C:/Users/marck/Downloads/nahr_ibrahim_watershed")
SEQ_DIR = ROOT / "data" / "sequences"
SPLIT_DIR = ROOT / "data" / "splits"
MODEL_DIR = ROOT / "models" / "trained"
FIG_DIR = ROOT / "results" / "figures"
MET_DIR = ROOT / "results" / "metrics"

FIG_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)

# 18 features — must match training order exactly
FEATURE_COLS = [
    "precip_mm_day",
    "precip_3day",
    "precip_7day",
    "temp_mean_c",
    "temp_max_c",
    "temp_min_c",
    "temp_range_c",
    "swe_mm",
    "swe_delta",
    "snow_cover_pct",
    "month_sin",
    "month_cos",
    "soil_moisture_mm",
    "sm_7day_mean",
    "sm_anomaly",
    "pet_mm_day",
    "spi_3month",
    "spei_3month",
]

# ── Custom layers for loading Transformer ──────────────────────────────────────
from tensorflow.keras import layers


class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        pos = np.arange(max_len)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]
        angles = pos / np.power(10000, (2 * (dims // 2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self.pos_encoding = tf.cast(angles[np.newaxis], dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, : tf.shape(x)[1], :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"max_len": self.max_len, "d_model": self.d_model})
        return cfg


class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, n_heads, ffn_dim, dropout, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attention = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads, dropout=dropout
        )
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(ffn_dim, activation="relu"),
                layers.Dropout(dropout),
                layers.Dense(d_model),
            ]
        )
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn = self.drop1(self.attention(x, x, training=training), training=training)
        x = self.ln1(x + attn)
        ffn = self.drop2(self.ffn(x, training=training), training=training)
        return self.ln2(x + ffn)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "ffn_dim": self.ffn_dim,
                "dropout": self.dropout,
            }
        )
        return cfg


def nse_metric(y_true, y_pred):
    num = tf.reduce_sum(tf.square(y_true - y_pred))
    den = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1.0 - num / (den + tf.keras.backend.epsilon())


custom_obj = {
    "nse_metric": nse_metric,
    "PositionalEncoding": PositionalEncoding,
    "TransformerEncoderBlock": TransformerEncoderBlock,
}

# ── Load test data ─────────────────────────────────────────────────────────────
print("Loading test sequences ...")
X_test = np.load(SEQ_DIR / "X_test.npy")
y_test = np.load(SEQ_DIR / "y_test.npy")

scaler = pd.read_csv(SPLIT_DIR / "scaler_params.csv", index_col=0)
q_min = scaler.loc["discharge_m3s", "min"]
q_max = scaler.loc["discharge_m3s", "max"]


def inverse_q(q):
    return np.clip(q * (q_max - q_min) + q_min, 0, None)


y_obs = inverse_q(y_test)
print(f"  Test samples : {len(y_obs)}")
print(f"  Input shape  : {X_test.shape}\n")


# ── Metrics ────────────────────────────────────────────────────────────────────
def nse(obs, pred):
    return 1 - np.sum((obs - pred) ** 2) / np.sum((obs - np.mean(obs)) ** 2)


def kge(obs, pred):
    r = pearsonr(obs, pred)[0]
    alpha = np.std(pred) / np.std(obs)
    beta = np.mean(pred) / np.mean(obs)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


# ── Load models ────────────────────────────────────────────────────────────────
model_files = {
    "LSTM": "lstm_final.keras",
    "CNN-LSTM": "cnn_lstm_final.keras",
    "Transformer": "transformer_final.keras",
    "PI-LSTM": "pi_lstm_final.keras",
    "PI-Transformer": "pi_transformer_final.keras",
    "TCN": "tcn_final.keras",
    "TCAN": "tcan_final.keras",
}

models = {}
print("Loading models ...")
for name, fname in model_files.items():
    path = MODEL_DIR / fname
    if not path.exists():
        print(f"  skip  {name} — not found")
        continue
    try:
        m = tf.keras.models.load_model(
            str(path), custom_objects=custom_obj, compile=False
        )
        models[name] = m
        print(f"  ok    {name}")
    except Exception as e:
        print(f"  fail  {name} — {e}")

print(f"\n  {len(models)} models loaded\n")

# ── Permutation importance ─────────────────────────────────────────────────────
# For each feature:
#   1. Make a copy of X_test
#   2. Shuffle that feature across all samples (breaks input-output relationship)
#   3. Run the model on the shuffled data
#   4. Compute NSE drop = baseline NSE - shuffled NSE
#   5. Large drop = feature is important
#
# We repeat the shuffle 10 times and average to reduce randomness.
# This is computationally expensive but gives stable estimates.

N_REPEATS = 10  # number of shuffle repetitions per feature
rng = np.random.RandomState(42)

all_results = {}

for model_name, model in models.items():
    print(f"{'='*55}")
    print(f"  Permutation importance — {model_name}")
    print(f"{'='*55}")

    # Baseline performance on unshuffled test set
    y_pred_base = inverse_q(model.predict(X_test, batch_size=512, verbose=0).flatten())
    baseline_nse = nse(y_obs, y_pred_base)
    baseline_kge = kge(y_obs, y_pred_base)
    print(f"  Baseline  NSE={baseline_nse:.4f}  KGE={baseline_kge:.4f}\n")

    feature_results = []

    for feat_idx, feat_name in enumerate(FEATURE_COLS):
        drops_nse = []
        drops_kge = []

        for _ in range(N_REPEATS):
            X_perm = X_test.copy()

            # Shuffle this feature across all samples
            # We shuffle along the sample axis (axis=0) for each timestep
            # This destroys the temporal relationship between this feature and Q
            perm_idx = rng.permutation(len(X_perm))
            X_perm[:, :, feat_idx] = X_perm[perm_idx, :, feat_idx]

            y_perm = inverse_q(
                model.predict(X_perm, batch_size=512, verbose=0).flatten()
            )
            perm_nse = nse(y_obs, y_perm)
            perm_kge = kge(y_obs, y_perm)

            drops_nse.append(baseline_nse - perm_nse)
            drops_kge.append(baseline_kge - perm_kge)

        mean_drop_nse = np.mean(drops_nse)
        mean_drop_kge = np.mean(drops_kge)
        std_drop_nse = np.std(drops_nse)

        feature_results.append(
            {
                "feature": feat_name,
                "delta_nse": round(mean_drop_nse, 4),
                "delta_kge": round(mean_drop_kge, 4),
                "std_nse": round(std_drop_nse, 4),
                "baseline_nse": round(baseline_nse, 4),
            }
        )

        print(
            f"  [{feat_idx+1:>2}/18] {feat_name:<22} "
            f"ΔNSE={mean_drop_nse:+.4f}  ΔKGE={mean_drop_kge:+.4f}"
        )

    df_feat = pd.DataFrame(feature_results).sort_values("delta_nse", ascending=False)
    all_results[model_name] = df_feat

    # Save per-model results
    out = MET_DIR / f"permutation_importance_{model_name.lower().replace('-','_')}.csv"
    df_feat.to_csv(out, index=False)
    print(f"\n  Saved → {out.name}")
    print(f"\n  Top 5 most important features:")
    for _, row in df_feat.head(5).iterrows():
        bar = "█" * max(1, int(row["delta_nse"] * 100))
        print(f"    {row['feature']:<22} ΔNSE={row['delta_nse']:+.4f}  {bar}")
    print()

# ── Cross-model summary ────────────────────────────────────────────────────────
# Average importance across all models for each feature
print(f"{'='*55}")
print(f"  Cross-model feature importance summary")
print(f"{'='*55}")

summary = pd.DataFrame({"feature": FEATURE_COLS})
for model_name, df_feat in all_results.items():
    merged = summary.merge(
        df_feat[["feature", "delta_nse"]].rename(columns={"delta_nse": model_name}),
        on="feature",
        how="left",
    )
    summary[model_name] = merged[model_name]

summary["mean_delta_nse"] = summary[[m for m in models.keys()]].mean(axis=1)
summary["std_delta_nse"] = summary[[m for m in models.keys()]].std(axis=1)
summary = summary.sort_values("mean_delta_nse", ascending=False).reset_index(drop=True)

summary.to_csv(MET_DIR / "permutation_importance_summary.csv", index=False)

print(f"\n  {'Rank':<6} {'Feature':<22} {'Mean ΔNSE':>10} {'Std':>8}")
print(f"  {'-'*50}")
for i, row in summary.iterrows():
    print(
        f"  {i+1:<6} {row['feature']:<22} "
        f"{row['mean_delta_nse']:>+10.4f} {row['std_delta_nse']:>8.4f}"
    )

# ── Figures ────────────────────────────────────────────────────────────────────
print("\nGenerating figures ...")

# Figure 1 — cross-model summary heatmap
fig1, ax = plt.subplots(figsize=(14, 8))
fig1.patch.set_facecolor("#080f1a")
ax.set_facecolor("#0d1825")

model_names = list(models.keys())
feat_names = summary["feature"].tolist()
matrix = np.zeros((len(feat_names), len(model_names)))

for j, mname in enumerate(model_names):
    for i, feat in enumerate(feat_names):
        row = all_results[mname][all_results[mname]["feature"] == feat]
        if not row.empty:
            matrix[i, j] = row["delta_nse"].values[0]

im = ax.imshow(
    matrix, cmap="RdYlGn", aspect="auto", vmin=-0.05, vmax=matrix.max() * 1.1
)

ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, color="#e8f4f8", fontsize=9, rotation=30, ha="right")
ax.set_yticks(range(len(feat_names)))
ax.set_yticklabels(feat_names, color="#e8f4f8", fontsize=8, fontfamily="monospace")

for i in range(len(feat_names)):
    for j in range(len(model_names)):
        val = matrix[i, j]
        ax.text(
            j,
            i,
            f"{val:+.3f}",
            ha="center",
            va="center",
            fontsize=6.5,
            color="#000000" if abs(val) > 0.02 else "#666666",
        )

cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
cbar.set_label("NSE drop when feature is permuted", color="#8aafc4", fontsize=9)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8aafc4")

ax.set_title(
    "Permutation Feature Importance — All Models\n"
    "Larger value = feature more important",
    color="#e8f4f8",
    fontsize=12,
    pad=15,
)
ax.tick_params(colors="#4a6a82")
ax.spines[:].set_color("#1e3448")

plt.tight_layout()
plt.savefig(
    FIG_DIR / "permutation_importance_heatmap.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="#080f1a",
)
plt.show()
print("  Saved → permutation_importance_heatmap.png")

# Figure 2 — cross-model average bar chart
fig2, ax2 = plt.subplots(figsize=(12, 7))
fig2.patch.set_facecolor("#080f1a")
ax2.set_facecolor("#0d1825")

colors_feat = [
    (
        "#e76f51"
        if "soil" in f or "sm_" in f or "pet" in f or "spi" in f or "spei" in f
        else "#3b9eff"
    )
    for f in summary["feature"]
]

bars = ax2.barh(
    summary["feature"][::-1],
    summary["mean_delta_nse"][::-1],
    xerr=summary["std_delta_nse"][::-1],
    color=colors_feat[::-1],
    alpha=0.85,
    error_kw={"ecolor": "#8aafc4", "capsize": 3, "linewidth": 1},
)

ax2.axvline(0, color="#8aafc4", linewidth=1)
ax2.set_xlabel(
    "Mean NSE drop when feature permuted (averaged across 7 models)", color="#8aafc4"
)
ax2.set_title(
    "Cross-model Feature Importance\nOrange = new features | Blue = original features",
    color="#e8f4f8",
    fontsize=12,
)
ax2.tick_params(colors="#4a6a82")
ax2.spines[:].set_color("#1e3448")
ax2.grid(axis="x", alpha=0.08)
ax2.set_facecolor("#0d1825")

plt.tight_layout()
plt.savefig(
    FIG_DIR / "permutation_importance_summary.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="#080f1a",
)
plt.show()
print("  Saved → permutation_importance_summary.png")

print(f"\n{'='*55}")
print(f"  Done. Files saved:")
print(f"    results/metrics/permutation_importance_summary.csv")
for m in models:
    print(
        f"    results/metrics/permutation_importance_{m.lower().replace('-','_')}.csv"
    )
print(f"    results/figures/permutation_importance_heatmap.png")
print(f"    results/figures/permutation_importance_summary.png")
print(f"{'='*55}")
