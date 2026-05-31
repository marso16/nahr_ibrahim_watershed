import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(
    os.environ.get("WATERSHED_ROOT", "C:/Users/marck/Downloads/nahr_ibrahim_watershed")
)
sys.path.insert(0, str(ROOT / "src" / "models"))

# ─── Configuration ──────────────────────────────────────────────────────────
HORIZON = 1
LOOKBACK = 30
N_PERMUTATIONS = 10  # repeat shuffling N times for stable importance
RANDOM_SEED = 42

SEQ_DIR = ROOT / "data" / "sequences"
SPLITS_DIR = ROOT / "data" / "splits"
MODEL_DIR = ROOT / "models" / "trained"
OUT_DIR = ROOT / "results" / "sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature names — must match the order used in training
FEATURE_COLS = [
    "precip_mm_day",
    "precip_3day",
    "precip_7day",
    "precip_14day",
    "precip_30day",
    "precip_60day",
    "precip_90day",
    "precip_lag1",
    "precip_lag2",
    "precip_lag3",
    "precip_lag5",
    "api_15d",
    "api_30d",
    "api_60d",
    "temp_mean_c",
    "temp_max_c",
    "temp_min_c",
    "temp_range_c",
    "swe_mm",
    "swe_delta",
    "snow_cover_pct",
    "soil_moisture_mm",
    "sm_7day_mean",
    "sm_30day_mean",
    "sm_anomaly",
    "sm_deep_30day",
    "sm_deep_anomaly",
    "pet_mm_day",
    "spi_3month",
    "spei_3month",
    "month_sin",
    "month_cos",
]
assert len(FEATURE_COLS) == 32

# Group features for thematic analysis
FEATURE_GROUP = {
    **{
        n: "Precipitation"
        for n in [
            "precip_mm_day",
            "precip_3day",
            "precip_7day",
            "precip_14day",
            "precip_30day",
            "precip_60day",
            "precip_90day",
            "precip_lag1",
            "precip_lag2",
            "precip_lag3",
            "precip_lag5",
            "api_15d",
            "api_30d",
            "api_60d",
        ]
    },
    **{
        n: "Temperature"
        for n in ["temp_mean_c", "temp_max_c", "temp_min_c", "temp_range_c"]
    },
    **{n: "Snow" for n in ["swe_mm", "swe_delta", "snow_cover_pct"]},
    **{
        n: "Soil moisture"
        for n in [
            "soil_moisture_mm",
            "sm_7day_mean",
            "sm_30day_mean",
            "sm_anomaly",
            "sm_deep_30day",
            "sm_deep_anomaly",
        ]
    },
    "pet_mm_day": "PET",
    **{n: "Drought indices" for n in ["spi_3month", "spei_3month"]},
    **{n: "Seasonal" for n in ["month_sin", "month_cos"]},
}

# ─── Load model + sequences ─────────────────────────────────────────────────
print(f"Sensitivity analysis at horizon h={HORIZON}")
print(f"Lookback: {LOOKBACK} days, permutations per feature: {N_PERMUTATIONS}\n")

from lstm import WatershedLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load model
model_path = MODEL_DIR / f"lstm_final_strategy_a_h{HORIZON}.pt"
print(f"Loading: {model_path.name}")
model = WatershedLSTM(input_dim=len(FEATURE_COLS)).to(device)
state = torch.load(model_path, map_location=device, weights_only=False)
if isinstance(state, dict) and "model_state_dict" in state:
    model.load_state_dict(state["model_state_dict"])
else:
    model.load_state_dict(state)
model.eval()

# Load test sequences
suffix = f"_h{HORIZON}_lb{LOOKBACK}"
X_test = np.load(SEQ_DIR / f"X_test{suffix}.npy")
y_test = np.load(SEQ_DIR / f"y_test{suffix}.npy")
print(f"Test sequences: X shape {X_test.shape}, y shape {y_test.shape}")

# Load discharge scaler so we can invert predictions back to m³/s
scaler_df = pd.read_csv(SPLITS_DIR / "scaler_params.csv", index_col=0)
Q_MIN = float(scaler_df.loc["discharge_m3s", "min"])
Q_MAX = float(scaler_df.loc["discharge_m3s", "max"])
if "__meta__" in scaler_df.index:
    LOG_TRANSFORM = bool(float(scaler_df.loc["__meta__", "min"]))
    LOG_EPS = float(scaler_df.loc["__meta__", "max"])
else:
    LOG_TRANSFORM = False
    LOG_EPS = 0.0


def invert(y_norm):
    y_lin = y_norm * (Q_MAX - Q_MIN) + Q_MIN
    if LOG_TRANSFORM:
        return np.maximum(np.exp(y_lin) - LOG_EPS, 0.0)
    return y_lin


def nse(obs, pred):
    obs = np.asarray(obs)
    pred = np.asarray(pred)
    denom = np.sum((obs - obs.mean()) ** 2)
    return np.nan if denom < 1e-12 else 1.0 - np.sum((obs - pred) ** 2) / denom


def predict(X):
    """Run LSTM forward in batches."""
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            batch = torch.from_numpy(X[i : i + 256].astype(np.float32)).to(device)
            out = model(batch).cpu().numpy().flatten()
            preds.append(out)
    return np.concatenate(preds)


# ─── Reference NSE ──────────────────────────────────────────────────────────
print("\nComputing reference NSE...")
y_pred_norm = predict(X_test)
y_pred_real = invert(y_pred_norm)
y_obs_real = invert(y_test)
ref_nse = nse(y_obs_real, y_pred_real)
print(f"  Reference NSE: {ref_nse:.4f}")

# ─── Permutation importance ─────────────────────────────────────────────────
print(f"\nRunning permutation importance ({N_PERMUTATIONS} shuffles per feature)...")
rng = np.random.default_rng(RANDOM_SEED)
results = []

for fi, fname in enumerate(FEATURE_COLS):
    drops = []
    for k in range(N_PERMUTATIONS):
        Xp = X_test.copy()
        # Shuffle this feature column across all time steps of all samples
        # We shuffle the sample order of (time, feature) slabs to preserve the
        # within-sequence pattern but break the relationship to the target.
        perm = rng.permutation(len(Xp))
        Xp[:, :, fi] = X_test[perm, :, fi]
        y_perm = predict(Xp)
        y_perm_real = invert(y_perm)
        n_perm = nse(y_obs_real, y_perm_real)
        drops.append(ref_nse - n_perm)
    mean_drop = float(np.mean(drops))
    std_drop = float(np.std(drops))
    results.append(
        {
            "feature": fname,
            "group": FEATURE_GROUP.get(fname, "Other"),
            "importance_nse_drop": round(mean_drop, 4),
            "importance_std": round(std_drop, 4),
        }
    )
    print(
        f"  {fi+1:>2}/{len(FEATURE_COLS)}  {fname:<22} drop={mean_drop:+.4f}  ±{std_drop:.4f}"
    )

# ─── Save and plot ──────────────────────────────────────────────────────────
df = pd.DataFrame(results).sort_values("importance_nse_drop", ascending=False)
out_csv = OUT_DIR / f"feature_importance_h{HORIZON}.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved → {out_csv.relative_to(ROOT)}")

# Top features bar plot
fig, ax = plt.subplots(figsize=(10, 9))
df_sorted = df.sort_values("importance_nse_drop")  # ascending so largest is on top
group_colors = {
    "Precipitation": "#1f77b4",
    "Temperature": "#d62728",
    "Snow": "#9467bd",
    "Soil moisture": "#2ca02c",
    "PET": "#ff7f0e",
    "Drought indices": "#17becf",
    "Seasonal": "#7f7f7f",
}
colors = [group_colors.get(g, "#888") for g in df_sorted["group"]]
ax.barh(
    df_sorted["feature"],
    df_sorted["importance_nse_drop"],
    xerr=df_sorted["importance_std"],
    color=colors,
    error_kw={"linewidth": 0.8, "ecolor": "#333"},
)
ax.set_xlabel("NSE drop when feature is shuffled")
ax.set_title(
    f"Permutation feature importance — LSTM h={HORIZON}\n(higher = more important)"
)
ax.axvline(0, color="#444", lw=1)
ax.grid(alpha=0.3, axis="x")

# Add legend
from matplotlib.patches import Patch

legend_items = [
    Patch(facecolor=c, label=g)
    for g, c in group_colors.items()
    if g in df_sorted["group"].values
]
ax.legend(handles=legend_items, loc="lower right", fontsize=9)
plt.tight_layout()
out_fig = OUT_DIR / f"feature_importance_h{HORIZON}.png"
plt.savefig(out_fig, dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved → {out_fig.relative_to(ROOT)}")

# ─── Group-level summary ────────────────────────────────────────────────────
group_sum = (
    df.groupby("group")["importance_nse_drop"].sum().sort_values(ascending=False)
)
print(f"\nImportance summed by feature group:")
for g, v in group_sum.items():
    print(f"  {g:<20} {v:+.4f}")

print(f"\nDone.")
