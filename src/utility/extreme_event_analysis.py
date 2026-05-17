import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

# ── Custom layers ──────────────────────────────────────────────────────────────
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


def nse_loss(y_true, y_pred):
    num = tf.reduce_sum(tf.square(y_true - y_pred))
    den = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1.0 - num / (den + tf.keras.backend.epsilon())


custom_obj = {
    "nse_metric": nse_loss,
    "PositionalEncoding": PositionalEncoding,
    "TransformerEncoderBlock": TransformerEncoderBlock,
}

# ── Load test data ─────────────────────────────────────────────────────────────
print("Loading test data ...")
X_test = np.load(SEQ_DIR / "X_test.npy")
y_test = np.load(SEQ_DIR / "y_test.npy")
dates_test = pd.to_datetime(np.load(SEQ_DIR / "dates_test.npy", allow_pickle=True))

scaler = pd.read_csv(SPLIT_DIR / "scaler_params.csv", index_col=0)
q_min = scaler.loc["discharge_m3s", "min"]
q_max = scaler.loc["discharge_m3s", "max"]


def inverse_q(q):
    return np.clip(q * (q_max - q_min) + q_min, 0, None)


y_obs = inverse_q(y_test)

# ── Flow regime thresholds ─────────────────────────────────────────────────────
# Defined on the full observed test series
# High flow  : top 10%    — karstic spring pulses and flood events
# Normal flow: middle 80% — typical daily conditions
# Low flow   : bottom 10% — summer baseflow and drought conditions
p10 = np.percentile(y_obs, 10)
p90 = np.percentile(y_obs, 90)

mask_high = y_obs >= p90
mask_low = y_obs <= p10
mask_normal = ~mask_high & ~mask_low

print(f"  Total test days   : {len(y_obs)}")
print(f"  Q10 threshold     : {p10:.3f} m³/s  ({mask_low.sum()} days)")
print(f"  Q90 threshold     : {p90:.3f} m³/s  ({mask_high.sum()} days)")
print(f"  Normal flow days  : {mask_normal.sum()}")


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(obs, pred):
    if len(obs) < 5:
        return {k: np.nan for k in ["NSE", "KGE", "RMSE", "MAE", "PBIAS"]}
    nse = 1 - np.sum((obs - pred) ** 2) / np.sum((obs - np.mean(obs)) ** 2)
    r = pearsonr(obs, pred)[0] if len(obs) > 2 else np.nan
    alpha = np.std(pred) / np.std(obs) if np.std(obs) > 0 else np.nan
    beta = np.mean(pred) / np.mean(obs) if np.mean(obs) > 0 else np.nan
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    rmse = np.sqrt(np.mean((obs - pred) ** 2))
    mae = np.mean(np.abs(obs - pred))
    pbias = 100 * np.sum(pred - obs) / np.sum(obs)
    return {
        "NSE": round(nse, 4),
        "KGE": round(kge, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "PBIAS_%": round(pbias, 2),
    }


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
print("\nLoading models ...")
for name, fname in model_files.items():
    path = MODEL_DIR / fname
    if not path.exists():
        print(f"  skip  {name}")
        continue
    try:
        m = tf.keras.models.load_model(
            str(path), custom_objects=custom_obj, compile=False
        )
        models[name] = m
        print(f"  ok    {name}")
    except Exception as e:
        print(f"  fail  {name} — {e}")

# ── Run extreme event analysis ────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  Extreme event analysis — test period 2021–2025")
print(f"{'='*65}\n")

all_rows = []
all_preds = {}

for model_name, model in models.items():
    y_pred = inverse_q(model.predict(X_test, batch_size=512, verbose=0).flatten())
    all_preds[model_name] = y_pred

    # Metrics per flow regime
    for regime, mask, label in [
        ("overall", np.ones(len(y_obs), dtype=bool), "Overall"),
        ("high", mask_high, "High flow (Q > Q90)"),
        ("normal", mask_normal, "Normal flow"),
        ("low", mask_low, "Low flow (Q < Q10)"),
    ]:
        m = compute_metrics(y_obs[mask], y_pred[mask])
        all_rows.append(
            {
                "model": model_name,
                "regime": regime,
                "label": label,
                "n_days": int(mask.sum()),
                **m,
            }
        )

    # Print summary for this model
    print(f"  {model_name}")
    print(f"  {'Regime':<22} {'NSE':>7} {'KGE':>7} {'RMSE':>7} {'PBIAS':>8}")
    print(f"  {'-'*52}")
    for regime, label in [
        ("overall", "Overall"),
        ("high", "High flow"),
        ("normal", "Normal"),
        ("low", "Low flow"),
    ]:
        row = next(
            r for r in all_rows if r["model"] == model_name and r["regime"] == regime
        )
        print(
            f"  {label:<22} {row['NSE']:>7} {row['KGE']:>7} "
            f"{row['RMSE']:>7} {row['PBIAS_%']:>7}%"
        )
    print()

# ── Save results ───────────────────────────────────────────────────────────────
df_results = pd.DataFrame(all_rows)
df_results.to_csv(MET_DIR / "extreme_event_analysis.csv", index=False)
print(f"  Saved → results/metrics/extreme_event_analysis.csv")

# ── Cross-model comparison table ───────────────────────────────────────────────
print(f"\n  Cross-model high flow (Q > Q90) performance:")
print(f"  {'Model':<18} {'NSE':>7} {'KGE':>7} {'RMSE':>7} {'PBIAS':>8}")
print(f"  {'-'*48}")
high_rows = df_results[df_results.regime == "high"].sort_values("NSE", ascending=False)
for _, row in high_rows.iterrows():
    print(
        f"  {row['model']:<18} {row['NSE']:>7} {row['KGE']:>7} "
        f"{row['RMSE']:>7} {row['PBIAS_%']:>7}%"
    )

print(f"\n  Cross-model low flow (Q < Q10) performance:")
print(f"  {'Model':<18} {'NSE':>7} {'KGE':>7} {'RMSE':>7} {'PBIAS':>8}")
print(f"  {'-'*48}")
low_rows = df_results[df_results.regime == "low"].sort_values("NSE", ascending=False)
for _, row in low_rows.iterrows():
    print(
        f"  {row['model']:<18} {row['NSE']:>7} {row['KGE']:>7} "
        f"{row['RMSE']:>7} {row['PBIAS_%']:>7}%"
    )

# ── Figures ────────────────────────────────────────────────────────────────────
print("\nGenerating figures ...")

COLORS = {
    "LSTM": "#3b9eff",
    "CNN-LSTM": "#00b4a0",
    "Transformer": "#00d4ff",
    "TCN": "#a855f7",
    "TCAN": "#22d3ee",
    "PI-LSTM": "#f4a261",
    "PI-Transformer": "#e76f51",
}

model_names = list(models.keys())

# Figure 1 — NSE by regime for all models
fig1, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=False)
fig1.patch.set_facecolor("#080f1a")

regime_info = [
    ("high", "High Flow (Q > Q90)", "#e76f51"),
    ("normal", "Normal Flow", "#3b9eff"),
    ("low", "Low Flow (Q < Q10)", "#00b4a0"),
]

for ax, (regime, title, color) in zip(axes, regime_info):
    ax.set_facecolor("#0d1825")
    subset = df_results[df_results.regime == regime].set_index("model")
    nse_vals = [
        subset.loc[m, "NSE"] if m in subset.index else np.nan for m in model_names
    ]
    bar_colors = [COLORS.get(m, "#8aafc4") for m in model_names]
    bars = ax.bar(model_names, nse_vals, color=bar_colors, alpha=0.85, width=0.6)

    for bar, val in zip(bars, nse_vals):
        if not np.isnan(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                color="#e8f4f8",
                fontsize=8,
                fontfamily="monospace",
            )

    ax.axhline(0, color="#8aafc4", linewidth=0.8, linestyle="--")
    ax.set_title(title, color="#e8f4f8", fontsize=11)
    ax.set_ylabel("NSE", color="#8aafc4")
    ax.set_ylim(
        min(0, min(v for v in nse_vals if not np.isnan(v))) - 0.05,
        max(v for v in nse_vals if not np.isnan(v)) + 0.12,
    )
    ax.tick_params(colors="#4a6a82", labelrotation=30)
    ax.spines[:].set_color("#1e3448")
    ax.grid(axis="y", alpha=0.08)

fig1.suptitle(
    "NSE by Flow Regime — Test Period 2021–2025\n"
    "High flow = karstic spring pulses | Low flow = summer baseflow",
    color="#e8f4f8",
    fontsize=13,
    y=1.02,
    fontfamily="monospace",
)
plt.tight_layout()
plt.savefig(
    FIG_DIR / "extreme_nse_by_regime.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="#080f1a",
)
plt.show()
print("  Saved → extreme_nse_by_regime.png")

# Figure 2 — Time series with flow regimes highlighted
fig2, axes2 = plt.subplots(len(models), 1, figsize=(18, 3.5 * len(models)), sharex=True)
fig2.patch.set_facecolor("#080f1a")

if len(models) == 1:
    axes2 = [axes2]

for ax, (model_name, y_pred) in zip(axes2, all_preds.items()):
    ax.set_facecolor("#0d1825")

    # Shade high and low flow regions
    ax.fill_between(
        dates_test, 0, p90, where=np.ones(len(dates_test), dtype=bool), alpha=0.0
    )
    ax.axhspan(
        p90, y_obs.max() * 1.1, alpha=0.06, color="#e76f51", label="High flow zone"
    )
    ax.axhspan(0, p10, alpha=0.06, color="#00b4a0", label="Low flow zone")

    ax.plot(
        dates_test, y_obs, color="#ffffff", linewidth=0.9, alpha=0.7, label="Observed"
    )
    ax.plot(
        dates_test,
        y_pred,
        color=COLORS.get(model_name, "#8aafc4"),
        linewidth=1.0,
        alpha=0.85,
        label=model_name,
    )

    # Get high/low NSE for annotation
    row_h = df_results[
        (df_results.model == model_name) & (df_results.regime == "high")
    ].iloc[0]
    row_l = df_results[
        (df_results.model == model_name) & (df_results.regime == "low")
    ].iloc[0]
    row_o = df_results[
        (df_results.model == model_name) & (df_results.regime == "overall")
    ].iloc[0]

    ax.set_title(
        f"{model_name}  |  Overall NSE={row_o['NSE']}  "
        f"High-flow NSE={row_h['NSE']}  Low-flow NSE={row_l['NSE']}",
        color="#e8f4f8",
        fontsize=10,
    )
    ax.set_ylabel("Q (m³/s)", color="#8aafc4")
    ax.tick_params(colors="#4a6a82")
    ax.spines[:].set_color("#1e3448")
    ax.legend(
        facecolor="#0d1825",
        edgecolor="#1e3448",
        labelcolor="#8aafc4",
        fontsize=7,
        ncol=4,
    )
    ax.grid(alpha=0.06)
    ax.axhline(p90, color="#e76f51", linewidth=0.7, linestyle=":", alpha=0.6)
    ax.axhline(p10, color="#00b4a0", linewidth=0.7, linestyle=":", alpha=0.6)

fig2.suptitle(
    "Observed vs Predicted Discharge — Flow Regime Analysis\n"
    "Red zone = high flow  |  Green zone = low flow",
    color="#e8f4f8",
    fontsize=13,
    y=1.01,
    fontfamily="monospace",
)
plt.tight_layout()
plt.savefig(
    FIG_DIR / "extreme_timeseries_regimes.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="#080f1a",
)
plt.show()
print("  Saved → extreme_timeseries_regimes.png")

# Figure 3 — Scatter plots high flow only for each model
fig3, axes3 = plt.subplots(2, 4, figsize=(20, 10))
fig3.patch.set_facecolor("#080f1a")
axes3 = axes3.flatten()

for i, (model_name, y_pred) in enumerate(all_preds.items()):
    ax = axes3[i]
    ax.set_facecolor("#0d1825")

    obs_high = y_obs[mask_high]
    pred_high = y_pred[mask_high]

    ax.scatter(
        obs_high, pred_high, alpha=0.5, s=15, color=COLORS.get(model_name, "#8aafc4")
    )
    lim = max(obs_high.max(), pred_high.max()) * 1.05
    ax.plot(
        [0, lim], [0, lim], color="#e76f51", linewidth=1.5, linestyle="--", label="1:1"
    )

    row = df_results[
        (df_results.model == model_name) & (df_results.regime == "high")
    ].iloc[0]
    ax.set_title(
        f"{model_name}\nNSE={row['NSE']}  PBIAS={row['PBIAS_%']}%",
        color="#e8f4f8",
        fontsize=10,
    )
    ax.set_xlabel("Observed Q (m³/s)", color="#8aafc4")
    ax.set_ylabel("Predicted Q (m³/s)", color="#8aafc4")
    ax.tick_params(colors="#4a6a82")
    ax.spines[:].set_color("#1e3448")
    ax.grid(alpha=0.08)

# Hide unused subplot if 7 models
if len(models) < len(axes3):
    for j in range(len(models), len(axes3)):
        axes3[j].set_visible(False)

fig3.suptitle(
    "High Flow Scatter — Observed vs Predicted (Q > Q90)\n"
    "Karstic spring pulse events only",
    color="#e8f4f8",
    fontsize=13,
    y=1.02,
    fontfamily="monospace",
)
plt.tight_layout()
plt.savefig(
    FIG_DIR / "extreme_highflow_scatter.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="#080f1a",
)
plt.show()
print("  Saved → extreme_highflow_scatter.png")

print(f"\n{'='*65}")
print(f"  Done. Files saved:")
print(f"    results/metrics/extreme_event_analysis.csv")
print(f"    results/figures/extreme_nse_by_regime.png")
print(f"    results/figures/extreme_timeseries_regimes.png")
print(f"    results/figures/extreme_highflow_scatter.png")
print(f"{'='*65}")
