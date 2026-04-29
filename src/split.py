"""
=============================================================================
Nahr Ibrahim Watershed — Train / Validation / Test Split
=============================================================================
Strategy : Chronological split (never random for time series)
Split     : 70% Train | 15% Validation | 15% Test
Periods   : Train 2000–2017 | Val 2018–2020 | Test 2021–2025
=============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
ROOT       = Path("C:/Users/marck/Downloads/nahr_ibrahim_watershed")
MASTER_DIR = ROOT / "data" / "master"
SPLIT_DIR  = ROOT / "data" / "splits"
FIG_DIR    = ROOT / "results" / "figures"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# LOAD MASTER DATASET
# =============================================================================
df = pd.read_csv(MASTER_DIR / "nahr_ibrahim_master_model.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

print("=" * 60)
print("  Nahr Ibrahim — Train / Validation / Test Split")
print("=" * 60)
print(f"\n  Master dataset: {len(df)} rows")
print(f"  Period: {df.date.min().date()} → {df.date.max().date()}")

# =============================================================================
# DEFINE SPLIT BOUNDARIES
# =============================================================================
TRAIN_END = "2017-12-31"
VAL_END   = "2020-12-31"

train = df[df["date"] <= TRAIN_END].reset_index(drop=True)
val   = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)].reset_index(drop=True)
test  = df[df["date"] > VAL_END].reset_index(drop=True)

# =============================================================================
# SUMMARY
# =============================================================================
total = len(df)
print(f"\n  {'Split':<12} {'Start':<12} {'End':<12} {'Days':>6} {'Pct':>6}")
print(f"  {'-'*50}")
print(f"  {'Train':<12} {str(train.date.min().date()):<12} {str(train.date.max().date()):<12} {len(train):>6} {len(train)/total*100:>5.1f}%")
print(f"  {'Validation':<12} {str(val.date.min().date()):<12} {str(val.date.max().date()):<12} {len(val):>6} {len(val)/total*100:>5.1f}%")
print(f"  {'Test':<12} {str(test.date.min().date()):<12} {str(test.date.max().date()):<12} {len(test):>6} {len(test)/total*100:>5.1f}%")
print(f"  {'-'*50}")
print(f"  {'Total':<12} {'':>12} {'':>12} {total:>6} {'100.0%':>6}")

# =============================================================================
# STATISTICS PER SPLIT — check no data leakage or distribution shift
# =============================================================================
print(f"\n  Variable statistics per split:")
print(f"  {'Variable':<22} {'Train mean':>12} {'Val mean':>12} {'Test mean':>12}")
print(f"  {'-'*60}")

numeric_cols = ["precip_mm_day", "temp_mean_c", "swe_mm",
                "snow_cover_pct", "discharge_m3s"]

for col in numeric_cols:
    print(f"  {col:<22} {train[col].mean():>12.3f} {val[col].mean():>12.3f} {test[col].mean():>12.3f}")

# =============================================================================
# NORMALIZATION 
# =============================================================================
print(f"\n  Fitting normalizer on training set only ...")

# Features (inputs to be included in the model)
feature_cols = [
    "precip_mm_day", "precip_3day", "precip_7day",
    "temp_mean_c", "temp_max_c", "temp_min_c", "temp_range_c",
    "swe_mm", "swe_delta", "snow_cover_pct",
    "month_sin", "month_cos",
]

target_col = "discharge_m3s" # using now glofas, to be replaced by gauged data if available later

# Compute min/max from training set only
train_min = train[feature_cols + [target_col]].min()
train_max = train[feature_cols + [target_col]].max()
train_range = train_max - train_min

# Save scaler parameters for inverse transform later
scaler_df = pd.DataFrame({"min": train_min, "max": train_max})
scaler_df.to_csv(SPLIT_DIR / "scaler_params.csv")
print(f"  Scaler parameters saved → data/splits/scaler_params.csv")

def normalize(df_in, cols, min_vals, max_vals):
    df_out = df_in.copy()
    for col in cols:
        r = max_vals[col] - min_vals[col]
        if r == 0:
            df_out[col] = 0.0
        else:
            df_out[col] = (df_in[col] - min_vals[col]) / r
    return df_out

all_cols = feature_cols + [target_col]

train_norm = normalize(train, all_cols, train_min, train_max)
val_norm   = normalize(val,   all_cols, train_min, train_max)
test_norm  = normalize(test,  all_cols, train_min, train_max)

# Check for out-of-range values in val/test (distribution shift indicator)
val_oob  = ((val_norm[feature_cols] < 0) | (val_norm[feature_cols] > 1)).sum().sum()
test_oob = ((test_norm[feature_cols] < 0) | (test_norm[feature_cols] > 1)).sum().sum()
print(f"  Out-of-range values after normalization: Val={val_oob} | Test={test_oob}")
if val_oob > 0 or test_oob > 0:
    print(f"  NOTE: Some val/test values fall outside training range — expected under climate change")

# =============================================================================
# SAVE SPLITS (both raw and normalized)
# =============================================================================
train.to_csv(SPLIT_DIR / "train_raw.csv", index=False)
val.to_csv(SPLIT_DIR   / "val_raw.csv",   index=False)
test.to_csv(SPLIT_DIR  / "test_raw.csv",  index=False)

# Normalized splits (features + target only, no date/season)
keep_cols = ["date"] + feature_cols + [target_col]
train_norm[keep_cols].to_csv(SPLIT_DIR / "train_norm.csv", index=False)
val_norm[keep_cols].to_csv(SPLIT_DIR   / "val_norm.csv",   index=False)
test_norm[keep_cols].to_csv(SPLIT_DIR  / "test_norm.csv",  index=False)

print(f"\n  Files saved:")
print(f"    data/splits/train_raw.csv   ({len(train)} rows)")
print(f"    data/splits/val_raw.csv     ({len(val)} rows)")
print(f"    data/splits/test_raw.csv    ({len(test)} rows)")
print(f"    data/splits/train_norm.csv  ({len(train_norm)} rows)")
print(f"    data/splits/val_norm.csv    ({len(val_norm)} rows)")
print(f"    data/splits/test_norm.csv   ({len(test_norm)} rows)")
print(f"    data/splits/scaler_params.csv")

# =============================================================================
# VISUALIZATION — discharge time series with split regions highlighted
# =============================================================================
fig, axes = plt.subplots(2, 1, figsize=(16, 8))

# --- Top: full discharge time series with split shading ---
ax = axes[0]
ax.fill_between(train["date"], train["discharge_m3s"],
                alpha=0.4, color="steelblue", label="Train (2000–2017)")
ax.fill_between(val["date"], val["discharge_m3s"],
                alpha=0.6, color="orange", label="Validation (2018–2020)")
ax.fill_between(test["date"], test["discharge_m3s"],
                alpha=0.6, color="tomato", label="Test (2021–2025)")
ax.axvline(pd.Timestamp(TRAIN_END), color="orange", linestyle="--", linewidth=1.5)
ax.axvline(pd.Timestamp(VAL_END),   color="tomato",  linestyle="--", linewidth=1.5)
ax.set_title("Discharge Time Series — Train / Validation / Test Split", fontsize=13)
ax.set_ylabel("Discharge (m³/s)")
ax.legend(loc="upper right")
ax.grid(alpha=0.3)

# --- Bottom: discharge distribution per split (box plot) ---
ax2 = axes[1]
data_to_plot = [
    train["discharge_m3s"].values,
    val["discharge_m3s"].values,
    test["discharge_m3s"].values
]
bp = ax2.boxplot(data_to_plot, patch_artist=True, widths=0.5,
                  labels=["Train\n(2000–2017)", "Validation\n(2018–2020)", "Test\n(2021–2025)"])

colors = ["steelblue", "orange", "tomato"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax2.set_title("Discharge Distribution per Split", fontsize=13)
ax2.set_ylabel("Discharge (m³/s)")
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "train_val_test_split.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  Figure saved → results/figures/train_val_test_split.png")
print("\n  Split complete. Ready for sequence windowing and model training.")
print("=" * 60)