import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(
    os.environ.get("WATERSHED_ROOT", "C:/Users/marck/Downloads/nahr_ibrahim_watershed")
)
MASTER = ROOT / "data" / "master"
SPLIT_DIR = ROOT / "data" / "splits"
FIG_DIR = ROOT / "results" / "figures"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Load master dataset ────────────────────────────────────────────────────────
df = pd.read_csv(MASTER / "nahr_ibrahim_master_model.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

print(
    f"Master dataset: {len(df)} rows  "
    f"({df.date.min().date()} → {df.date.max().date()})\n"
)

# ── Chronological split ────────────────────────────────────────────────────────
# TRAIN_END = "2010-12-31"
# VAL_END   = "2013-12-31"
TRAIN_END = "2017-12-31"
VAL_END = "2020-12-31"

train = df[df.date <= TRAIN_END].reset_index(drop=True)
val = df[(df.date > TRAIN_END) & (df.date <= VAL_END)].reset_index(drop=True)
test = df[df.date > VAL_END].reset_index(drop=True)

total = len(df)
print(f"{'Split':<12} {'Period':<24} {'Days':>6} {'%':>6}")
print("-" * 52)
for name, s in [("Train", train), ("Validation", val), ("Test", test)]:
    period = f"{s.date.min().date()} → {s.date.max().date()}"
    print(f"{name:<12} {period:<24} {len(s):>6} {len(s) / total * 100:>5.1f}%")
print("-" * 52)
print(f"{'Total':<12} {'':<24} {total:>6}  100.0%\n")

# ── Per-split statistics ───────────────────────────────────────────────────────
stats_cols = [
    "precip_mm_day",
    "temp_mean_c",
    "swe_mm",
    "snow_cover_pct",
    "soil_moisture_mm",
    "pet_mm_day",
    "discharge_m3s",
]
print(f"{'Variable':<25} {'Train':>10} {'Val':>10} {'Test':>10}")
print("-" * 58)
for col in stats_cols:
    if col in df.columns:
        print(
            f"{col:<25} {train[col].mean():>10.3f} "
            f"{val[col].mean():>10.3f} {test[col].mean():>10.3f}"
        )

# ── Add discharge lags (computed on RAW discharge, before any transform) ───────
# Important: lags are computed per-split to avoid leakage across boundaries.
# The first 3 rows of val/test will have NaN lags filled with split-local backfill.
# DISCHARGE_LAG_COLS = ["discharge_lag1", "discharge_lag2", "discharge_lag3"]

# for s in [train, val, test]:
#     s["discharge_lag1"] = s["discharge_m3s"].shift(1)
#     s["discharge_lag2"] = s["discharge_m3s"].shift(2)
#     s["discharge_lag3"] = s["discharge_m3s"].shift(3)
#     for col in DISCHARGE_LAG_COLS:
#         s[col] = s[col].fillna(s["discharge_m3s"].iloc[0])

# print(f"\nAdded discharge lag features: {DISCHARGE_LAG_COLS}")

# ── Feature list ───────────────────────────────────────────────────────────────
FEATURE_COLS = [
    # Precipitation
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
    # Temperature
    "temp_mean_c",
    "temp_max_c",
    "temp_min_c",
    "temp_range_c",
    # Snow
    "swe_mm",
    "swe_delta",
    "snow_cover_pct",
    # Soil moisture
    "soil_moisture_mm",
    "sm_7day_mean",
    "sm_30day_mean",
    "sm_anomaly",
    "sm_deep_30day",
    "sm_deep_anomaly",
    # Energy / PET
    "pet_mm_day",
    # Drought
    "spi_3month",
    "spei_3month",
    # Cyclical
    "month_sin",
    "month_cos",
]
# FEATURE_COLS = [
#     "precip_mm_day",
#     "precip_3day",
#     "precip_7day",
#     "precip_lag1",
#     "precip_lag2",
#     "precip_lag3",
#     "precip_lag5",
#     "api_15d",
#     "temp_mean_c",
#     "temp_max_c",
#     "temp_min_c",
#     "temp_range_c",
#     "swe_mm",
#     "swe_delta",
#     "snow_cover_pct",
#     "month_sin",
#     "month_cos",
#     "soil_moisture_mm",
#     "sm_7day_mean",
#     "sm_anomaly",
#     "pet_mm_day",
#     "spi_3month",
#     "spei_3month",
#     # "discharge_lag1",
#     # "discharge_lag2",
#     # "discharge_lag3",
# ]
TARGET = "discharge_m3s"

print(f"\nFeatures ({len(FEATURE_COLS)}):")
for i, col in enumerate(FEATURE_COLS):
    print(f"  [{i:>2}] {col}")

# ── Log-transform target ───────────────────────────────────────────────────────
LOG_TRANSFORM_Q = os.environ.get("LOG_TRANSFORM_Q", "1") == "1"
LOG_EPS = 1e-3

train_q_raw = train[TARGET].copy()
val_q_raw = val[TARGET].copy()
test_q_raw = test[TARGET].copy()

# Save raw discharge as a separate column BEFORE any transform.
# Used by baselines.py and for plotting in real m³/s.
for s in [train, val, test]:
    s["discharge_m3s_raw"] = s[TARGET].copy()

if LOG_TRANSFORM_Q:
    print(f"\nLog-transforming target: y' = log(Q + {LOG_EPS})")
    for s in [train, val, test]:
        s[TARGET] = np.log(s[TARGET].clip(lower=0) + LOG_EPS)
else:
    print("\nUsing raw discharge as target (no log-transform)")

# ── Min-max normalisation (fit on train only) ──────────────────────────────────
all_cols = FEATURE_COLS + [TARGET]
train_min = train[all_cols].min()
train_max = train[all_cols].max()

scaler = pd.DataFrame({"min": train_min, "max": train_max})

# ── Write __meta__ row so lstm.py can invert the log-transform correctly ───────
# lstm.py checks:  if "__meta__" in scaler_df.index:
#                      log_transform = bool(float(scaler_df.loc["__meta__", "min"]))
#                      log_eps       = float(scaler_df.loc["__meta__", "max"])
meta_row = pd.DataFrame(
    {"min": [float(LOG_TRANSFORM_Q)], "max": [float(LOG_EPS)]},
    index=["__meta__"],
)
scaler = pd.concat([scaler, meta_row])
scaler.to_csv(SPLIT_DIR / "scaler_params.csv")

print(f"Scaler saved → data/splits/scaler_params.csv")
print(f"  __meta__ row written: log_transform={LOG_TRANSFORM_Q}, log_eps={LOG_EPS}")


def normalise(df_in, cols, lo, hi):
    out = df_in.copy()
    for col in cols:
        r = hi[col] - lo[col]
        out[col] = 0.0 if r == 0 else (df_in[col] - lo[col]) / r
    return out


train_norm = normalise(train, all_cols, train_min, train_max)
val_norm = normalise(val, all_cols, train_min, train_max)
test_norm = normalise(test, all_cols, train_min, train_max)

val_oob = ((val_norm[FEATURE_COLS] < 0) | (val_norm[FEATURE_COLS] > 1)).sum().sum()
test_oob = ((test_norm[FEATURE_COLS] < 0) | (test_norm[FEATURE_COLS] > 1)).sum().sum()
print(f"Out-of-range values: Val={val_oob}  Test={test_oob}")
if val_oob or test_oob:
    print("  (expected — reflects warming trend and distribution shift)")

# ── Save splits ────────────────────────────────────────────────────────────────
keep = ["date"] + FEATURE_COLS + [TARGET]

train.to_csv(SPLIT_DIR / "train_raw.csv", index=False)
val.to_csv(SPLIT_DIR / "val_raw.csv", index=False)
test.to_csv(SPLIT_DIR / "test_raw.csv", index=False)
train_norm[keep].to_csv(SPLIT_DIR / "train_norm.csv", index=False)
val_norm[keep].to_csv(SPLIT_DIR / "val_norm.csv", index=False)
test_norm[keep].to_csv(SPLIT_DIR / "test_norm.csv", index=False)

print(f"\nSaved:")
for name, n in [("train", len(train)), ("val", len(val)), ("test", len(test))]:
    print(f"  data/splits/{name}_raw.csv  /  {name}_norm.csv  ({n} rows)")
print(f"  data/splits/scaler_params.csv")

# ── Visualisation ──────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

for s, q_raw, color, label in [
    (train, train_q_raw, "steelblue", "Train (2000–2017)"),
    (val, val_q_raw, "orange", "Validation (2018–2020)"),
    (test, test_q_raw, "tomato", "Test (2021–2025)"),
]:
    ax1.fill_between(s.date, q_raw, alpha=0.5, color=color, label=label)

ax1.axvline(pd.Timestamp(TRAIN_END), color="orange", linestyle="--", linewidth=1.5)
ax1.axvline(pd.Timestamp(VAL_END), color="tomato", linestyle="--", linewidth=1.5)
ax1.set_title("Discharge time series — chronological split", fontsize=13)
ax1.set_ylabel("Discharge (m³/s)")
ax1.legend(loc="upper right")
ax1.grid(alpha=0.3)

bp = ax2.boxplot(
    [train_q_raw.values, val_q_raw.values, test_q_raw.values],
    patch_artist=True,
    widths=0.5,
    tick_labels=["Train\n2000–2017", "Validation\n2018–2020", "Test\n2021–2025"],
)
for patch, color in zip(bp["boxes"], ["steelblue", "orange", "tomato"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax2.set_title("Discharge distribution per split", fontsize=13)
ax2.set_ylabel("Discharge (m³/s)")
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "train_val_test_split.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigure saved → results/figures/train_val_test_split.png")
