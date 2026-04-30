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
df = pd.read_csv(
    MASTER_DIR / "nahr_ibrahim_master_model.csv",
    parse_dates=["date"]
)
df = df.sort_values("date").reset_index(drop=True)

print("=" * 65)
print("  Nahr Ibrahim — Train / Validation / Test Split")
print("=" * 65)
print(f"\n  Master dataset : {len(df)} rows")
print(f"  Period         : {df.date.min().date()} → {df.date.max().date()}")
print(f"  Columns        : {df.columns.tolist()}")

# =============================================================================
# SPLIT BOUNDARIES
# =============================================================================
TRAIN_END = "2017-12-31"
VAL_END   = "2020-12-31"

train = df[df["date"] <= TRAIN_END].reset_index(drop=True)
val   = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)].reset_index(drop=True)
test  = df[df["date"] > VAL_END].reset_index(drop=True)

total = len(df)
print(f"\n  {'Split':<12} {'Start':<12} {'End':<12} {'Days':>6} {'Pct':>6}")
print(f"  {'-'*52}")
print(f"  {'Train':<12} {str(train.date.min().date()):<12} "
      f"{str(train.date.max().date()):<12} {len(train):>6} "
      f"{len(train)/total*100:>5.1f}%")
print(f"  {'Validation':<12} {str(val.date.min().date()):<12} "
      f"{str(val.date.max().date()):<12} {len(val):>6} "
      f"{len(val)/total*100:>5.1f}%")
print(f"  {'Test':<12} {str(test.date.min().date()):<12} "
      f"{str(test.date.max().date()):<12} {len(test):>6} "
      f"{len(test)/total*100:>5.1f}%")
print(f"  {'-'*52}")
print(f"  {'Total':<12} {'':>12} {'':>12} {total:>6} {'100.0%':>6}")

# =============================================================================
# STATISTICS PER SPLIT
# =============================================================================
print(f"\n  Variable statistics per split:")
print(f"  {'Variable':<25} {'Train mean':>12} {'Val mean':>12} {'Test mean':>12}")
print(f"  {'-'*64}")

check_cols = [
    "precip_mm_day", "temp_mean_c", "swe_mm",
    "snow_cover_pct", "soil_moisture_mm", "pet_mm_day",
    "discharge_m3s",
]
for col in check_cols:
    if col in df.columns:
        print(f"  {col:<25} {train[col].mean():>12.3f} "
              f"{val[col].mean():>12.3f} {test[col].mean():>12.3f}")

# =============================================================================
# FEATURE COLUMNS — 16 total
# =============================================================================
FEATURE_COLS = [
    "precip_mm_day", "precip_3day",    "precip_7day",
    "temp_mean_c",   "temp_max_c",     "temp_min_c",   "temp_range_c",
    "swe_mm",        "swe_delta",      "snow_cover_pct",
    "month_sin",     "month_cos",
    "soil_moisture_mm", "sm_7day_mean", "sm_anomaly",
    "pet_mm_day",
]
target_col = "discharge_m3s"

print(f"\n  Feature columns ({len(FEATURE_COLS)}):")
for i, col in enumerate(FEATURE_COLS):
    tag = " ← new" if col in [
        "soil_moisture_mm","sm_7day_mean","sm_anomaly","pet_mm_day"
    ] else ""
    print(f"    [{i:>2}] {col}{tag}")

# =============================================================================
# NORMALISATION — fit on training set only
# =============================================================================
print(f"\n  Fitting normaliser on training set only ...")

train_min   = train[FEATURE_COLS + [target_col]].min()
train_max   = train[FEATURE_COLS + [target_col]].max()
train_range = train_max - train_min

scaler_df = pd.DataFrame({"min": train_min, "max": train_max})
scaler_df.to_csv(SPLIT_DIR / "scaler_params.csv")
print(f"  Scaler parameters saved → data/splits/scaler_params.csv")


def normalize(df_in, cols, min_vals, max_vals):
    df_out = df_in.copy()
    for col in cols:
        r = max_vals[col] - min_vals[col]
        df_out[col] = 0.0 if r == 0 else (df_in[col] - min_vals[col]) / r
    return df_out


all_cols = FEATURE_COLS + [target_col]
train_norm = normalize(train, all_cols, train_min, train_max)
val_norm   = normalize(val,   all_cols, train_min, train_max)
test_norm  = normalize(test,  all_cols, train_min, train_max)

val_oob  = ((val_norm[FEATURE_COLS]  < 0) | (val_norm[FEATURE_COLS]  > 1)).sum().sum()
test_oob = ((test_norm[FEATURE_COLS] < 0) | (test_norm[FEATURE_COLS] > 1)).sum().sum()
print(f"  Out-of-range values: Val={val_oob} | Test={test_oob}")
if val_oob > 0 or test_oob > 0:
    print(f"  NOTE: Values outside training range are expected under climate shift")

# =============================================================================
# SAVE SPLITS
# =============================================================================
train.to_csv(SPLIT_DIR / "train_raw.csv",  index=False)
val.to_csv(  SPLIT_DIR / "val_raw.csv",    index=False)
test.to_csv( SPLIT_DIR / "test_raw.csv",   index=False)

keep_cols = ["date"] + FEATURE_COLS + [target_col]
train_norm[keep_cols].to_csv(SPLIT_DIR / "train_norm.csv", index=False)
val_norm[keep_cols].to_csv(  SPLIT_DIR / "val_norm.csv",   index=False)
test_norm[keep_cols].to_csv( SPLIT_DIR / "test_norm.csv",  index=False)

print(f"\n  Files saved:")
print(f"    data/splits/train_raw.csv   ({len(train)} rows)")
print(f"    data/splits/val_raw.csv     ({len(val)} rows)")
print(f"    data/splits/test_raw.csv    ({len(test)} rows)")
print(f"    data/splits/train_norm.csv  ({len(train_norm)} rows)")
print(f"    data/splits/val_norm.csv    ({len(val_norm)} rows)")
print(f"    data/splits/test_norm.csv   ({len(test_norm)} rows)")
print(f"    data/splits/scaler_params.csv")

# =============================================================================
# VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(2, 1, figsize=(16, 8))

ax = axes[0]
ax.fill_between(train["date"], train["discharge_m3s"],
                alpha=0.4, color="steelblue", label="Train (2000–2017)")
ax.fill_between(val["date"], val["discharge_m3s"],
                alpha=0.6, color="orange", label="Validation (2018–2020)")
ax.fill_between(test["date"], test["discharge_m3s"],
                alpha=0.6, color="tomato", label="Test (2021–2025)")
ax.axvline(pd.Timestamp(TRAIN_END), color="orange",
           linestyle="--", linewidth=1.5)
ax.axvline(pd.Timestamp(VAL_END),   color="tomato",
           linestyle="--", linewidth=1.5)
ax.set_title("Discharge Time Series — Train / Validation / Test Split",
             fontsize=13)
ax.set_ylabel("Discharge (m³/s)")
ax.legend(loc="upper right")
ax.grid(alpha=0.3)

ax2 = axes[1]
bp = ax2.boxplot(
    [train["discharge_m3s"].values,
     val["discharge_m3s"].values,
     test["discharge_m3s"].values],
    patch_artist=True, widths=0.5,
    tick_labels=["Train\n(2000-2017)", "Validation\n(2018-2020)",
            "Test\n(2021-2025)"]
)
for patch, color in zip(bp["boxes"], ["steelblue","orange","tomato"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax2.set_title("Discharge Distribution per Split", fontsize=13)
ax2.set_ylabel("Discharge (m³/s)")
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "train_val_test_split.png",
            dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  Figure saved → results/figures/train_val_test_split.png")
print("=" * 65)