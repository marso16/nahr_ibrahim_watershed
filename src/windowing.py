import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
ROOT      = Path("C:/Users/marck/Downloads/nahr_ibrahim_watershed")
SPLIT_DIR = ROOT / "data" / "splits"
SEQ_DIR   = ROOT / "data" / "sequences"
FIG_DIR   = ROOT / "results" / "figures"

SEQ_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK = 30   # days of history fed to the model
HORIZON  = 1    # days ahead to predict

FEATURE_COLS = [
    "precip_mm_day", "precip_3day", "precip_7day", "temp_mean_c",   
    "temp_max_c", "temp_min_c", "temp_range_c", "swe_mm",        
    "swe_delta", "snow_cover_pct", "month_sin", "month_cos",
    "soil_moisture_mm", "sm_7day_mean", "sm_anomaly", "pet_mm_day",
]
TARGET_COL = "discharge_m3s"

print("=" * 65)
print("  Nahr Ibrahim — Sequence Windowing")
print("=" * 65)
print(f"  Lookback  : {LOOKBACK} days")
print(f"  Horizon   : {HORIZON} day ahead")
print(f"  Features  : {len(FEATURE_COLS)} ")

# =============================================================================
# LOAD NORMALISED SPLITS
# =============================================================================
print("\n  Loading normalised splits ...")

train_df = pd.read_csv(SPLIT_DIR / "train_norm.csv", parse_dates=["date"])
val_df   = pd.read_csv(SPLIT_DIR / "val_norm.csv",   parse_dates=["date"])
test_df  = pd.read_csv(SPLIT_DIR / "test_norm.csv",  parse_dates=["date"])

print(f"  Train : {len(train_df)} rows | "
      f"Val : {len(val_df)} rows | "
      f"Test : {len(test_df)} rows")

# Verify all feature columns are present
missing = [c for c in FEATURE_COLS if c not in train_df.columns]
if missing:
    raise ValueError(
        f"Missing columns in normalised split: {missing}\n"
        f"Re-run split.py first."
    )
print(f"  ✓ All {len(FEATURE_COLS)} feature columns confirmed present")

# =============================================================================
# WINDOWING FUNCTION
# =============================================================================
def create_sequences(df, feature_cols, target_col, lookback, horizon):
    """
    Convert a flat daily DataFrame into 3D sequences.

    Returns
    -------
    X     : (n_samples, lookback, n_features)
    y     : (n_samples,)
    dates : prediction date for each sample
    """
    features = df[feature_cols].values
    target   = df[target_col].values
    dates    = df["date"].values

    X, y, pred_dates = [], [], []
    for i in range(lookback, len(df) - horizon + 1):
        X.append(features[i - lookback : i])
        y.append(target[i + horizon - 1])
        pred_dates.append(dates[i + horizon - 1])

    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            np.array(pred_dates))


# =============================================================================
# CREATE SEQUENCES
# =============================================================================
print("\n  Creating sequences ...")

X_train, y_train, dates_train = create_sequences(
    train_df, FEATURE_COLS, TARGET_COL, LOOKBACK, HORIZON)
X_val,   y_val,   dates_val   = create_sequences(
    val_df,   FEATURE_COLS, TARGET_COL, LOOKBACK, HORIZON)
X_test,  y_test,  dates_test  = create_sequences(
    test_df,  FEATURE_COLS, TARGET_COL, LOOKBACK, HORIZON)

# =============================================================================
# SHAPE REPORT
# =============================================================================
print(f"\n  {'Split':<10} {'X shape':<28} {'y shape':<16} {'Samples':>8}")
print(f"  {'-'*64}")
print(f"  {'Train':<10} {str(X_train.shape):<28} "
      f"{str(y_train.shape):<16} {len(y_train):>8,}")
print(f"  {'Val':<10} {str(X_val.shape):<28} "
      f"{str(y_val.shape):<16} {len(y_val):>8,}")
print(f"  {'Test':<10} {str(X_test.shape):<28} "
      f"{str(y_test.shape):<16} {len(y_test):>8,}")
print(f"  {'-'*64}")
print(f"  {'Total':<10} {'':28} {'':16} "
      f"{len(y_train)+len(y_val)+len(y_test):>8,}")

# =============================================================================
# INTEGRITY CHECKS
# =============================================================================
print("\n  Running integrity checks ...")

checks_passed = True

# Fix the 1 known NaN in swe_delta day 1
nan_mask = np.isnan(X_train)
if nan_mask.sum() > 0:
    print(f"  Fixing {nan_mask.sum()} NaN(s) in X_train → replaced with 0")
    X_train[nan_mask] = 0.0

for name, X, y in [("Train", X_train, y_train),
                    ("Val",   X_val,   y_val),
                    ("Test",  X_test,  y_test)]:
    nan_X = np.isnan(X).sum()
    nan_y = np.isnan(y).sum()
    if nan_X > 0 or nan_y > 0:
        print(f"  ✗ {name}: {nan_X} NaN in X, {nan_y} NaN in y")
        checks_passed = False
    else:
        print(f"  ✓ {name}: No NaN values")

for name, X in [("Train", X_train), ("Val", X_val), ("Test", X_test)]:
    oob = ((X < -0.1) | (X > 1.1)).sum()
    if oob > 0:
        print(f"  ⚠ {name}: {oob} values outside [0,1] "
              f"(expected under climate shift)")
    else:
        print(f"  ✓ {name}: All values within expected range")

print(f"\n  ✓ Feature count per timestep : {X_train.shape[2]}")
print(f"  ✓ Lookback window            : {LOOKBACK} days")
if checks_passed:
    print(f"  ✓ All integrity checks passed")

# =============================================================================
# SAVE ARRAYS
# =============================================================================
print("\n  Saving sequence arrays ...")

np.save(SEQ_DIR / "X_train.npy",     X_train)
np.save(SEQ_DIR / "y_train.npy",     y_train)
np.save(SEQ_DIR / "X_val.npy",       X_val)
np.save(SEQ_DIR / "y_val.npy",       y_val)
np.save(SEQ_DIR / "X_test.npy",      X_test)
np.save(SEQ_DIR / "y_test.npy",      y_test)
np.save(SEQ_DIR / "dates_train.npy", dates_train)
np.save(SEQ_DIR / "dates_val.npy",   dates_val)
np.save(SEQ_DIR / "dates_test.npy",  dates_test)

pd.DataFrame({
    "parameter": ["lookback", "horizon", "n_features",
                  "feature_cols", "target_col",
                  "n_train", "n_val", "n_test"],
    "value"    : [LOOKBACK, HORIZON, len(FEATURE_COLS),
                  str(FEATURE_COLS), TARGET_COL,
                  len(y_train), len(y_val), len(y_test)]
}).to_csv(SEQ_DIR / "sequence_metadata.csv", index=False)

print(f"  X_train.npy  : {X_train.nbytes / 1e6:.1f} MB")
print(f"  X_val.npy    : {X_val.nbytes   / 1e6:.1f} MB")
print(f"  X_test.npy   : {X_test.nbytes  / 1e6:.1f} MB")
print(f"  Saved → data/sequences/")

# =============================================================================
# VISUALIZATION
# =============================================================================
print("\n  Generating visualisations ...")

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#080f1a")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# ── Window diagram ───────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, :2])
ax0.set_facecolor("#0d1825")
n_demo = 45
t      = np.arange(n_demo)
rng    = np.random.RandomState(42)
demo_q = np.sin(t * 0.25) * 0.3 + 0.5 + rng.randn(n_demo) * 0.05

ax0.fill_between(t[:LOOKBACK], demo_q[:LOOKBACK],
                 alpha=0.3, color="#3b9eff",
                 label="Input window (30 days)")
ax0.fill_between(t[LOOKBACK:LOOKBACK+1], demo_q[LOOKBACK:LOOKBACK+1],
                 alpha=0.8, color="#e76f51",
                 label="Prediction target (day 31)")
ax0.plot(t, demo_q, color="#8aafc4", linewidth=1.2, alpha=0.6)
ax0.axvline(LOOKBACK, color="#e76f51", linestyle="--",
            linewidth=1.5, alpha=0.8)
ax0.scatter([LOOKBACK], [demo_q[LOOKBACK]],
            color="#e76f51", s=80, zorder=5)
ax0.annotate("← 30-day input window →", xy=(14, 0.92),
             fontsize=10, color="#3b9eff",
             fontfamily="monospace", ha="center")
ax0.annotate("Predict\nday t+1",
             xy=(LOOKBACK, demo_q[LOOKBACK] + 0.08),
             fontsize=9, color="#e76f51",
             fontfamily="monospace", ha="center")
ax0.set_title(
    f"Sliding Window — {LOOKBACK}-day Lookback · 1-day Ahead · "
    f"{len(FEATURE_COLS)} Features",
    color="#e8f4f8", fontsize=11, pad=10)
ax0.set_xlabel("Time (days)", color="#8aafc4")
ax0.set_ylabel("Normalised discharge", color="#8aafc4")
ax0.tick_params(colors="#4a6a82")
ax0.spines[:].set_color("#1e3448")
ax0.legend(facecolor="#0d1825", edgecolor="#1e3448",
           labelcolor="#8aafc4", fontsize=9)
ax0.set_facecolor("#0d1825")

# ── Sample counts ─────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 2])
ax1.set_facecolor("#0d1825")
splits_n = ["Train\n2000–2017", "Val\n2018–2020", "Test\n2021–2025"]
counts   = [len(y_train), len(y_val), len(y_test)]
colors_b = ["#3b9eff", "#f4a261", "#e76f51"]
bars = ax1.bar(splits_n, counts, color=colors_b, alpha=0.8, width=0.5)
for bar, count in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 30, f"{count:,}",
             ha="center", va="bottom",
             color="#e8f4f8", fontsize=9, fontfamily="monospace")
ax1.set_title("Sequence Counts per Split",
              color="#e8f4f8", fontsize=11, pad=10)
ax1.set_ylabel("Number of sequences", color="#8aafc4")
ax1.tick_params(colors="#4a6a82")
ax1.spines[:].set_color("#1e3448")
ax1.set_facecolor("#0d1825")

# ── Feature heatmap (one sample window) ──────────────────────
ax2 = fig.add_subplot(gs[1, :2])
ax2.set_facecolor("#0d1825")
sample_idx    = 500
sample_window = X_train[sample_idx]  # (30, 16)
im = ax2.imshow(sample_window.T, aspect="auto",
                cmap="RdYlBu_r", vmin=0, vmax=1,
                interpolation="nearest")
ax2.set_yticks(range(len(FEATURE_COLS)))
ax2.set_yticklabels(FEATURE_COLS, fontsize=7,
                    color="#8aafc4", fontfamily="monospace")
ax2.set_xlabel("Day within window (0 = oldest, 29 = most recent)",
               color="#8aafc4")
ax2.set_title(
    f"Feature Heatmap — Example Window (sample #{sample_idx})",
    color="#e8f4f8", fontsize=11, pad=10)
ax2.tick_params(colors="#4a6a82")
ax2.spines[:].set_color("#1e3448")
cbar = plt.colorbar(im, ax=ax2, fraction=0.015, pad=0.02)
cbar.set_label("Normalised value", color="#8aafc4", fontsize=8)
cbar.ax.yaxis.set_tick_params(color="#4a6a82")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#4a6a82")
ax2.set_facecolor("#0d1825")

# ── Target distribution ───────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 2])
ax3.set_facecolor("#0d1825")
ax3.hist(y_train, bins=60, color="#3b9eff", alpha=0.7,
         label="Train", density=True, orientation="horizontal")
ax3.hist(y_val,   bins=40, color="#f4a261", alpha=0.7,
         label="Val",   density=True, orientation="horizontal")
ax3.hist(y_test,  bins=40, color="#e76f51", alpha=0.7,
         label="Test",  density=True, orientation="horizontal")
ax3.set_title("Target (Q) Distribution",
              color="#e8f4f8", fontsize=11, pad=10)
ax3.set_ylabel("Normalised discharge", color="#8aafc4")
ax3.set_xlabel("Density", color="#8aafc4")
ax3.tick_params(colors="#4a6a82")
ax3.spines[:].set_color("#1e3448")
ax3.legend(facecolor="#0d1825", edgecolor="#1e3448",
           labelcolor="#8aafc4", fontsize=8)
ax3.set_facecolor("#0d1825")

fig.suptitle(
    f"Sequence Windowing — {LOOKBACK}-day Lookback · 1-day Ahead · "
    f"{len(FEATURE_COLS)} Features",
    color="#e8f4f8", fontsize=13, y=0.98, fontfamily="monospace")

plt.savefig(FIG_DIR / "sequence_windowing.png",
            dpi=150, bbox_inches="tight", facecolor="#080f1a")
plt.show()

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("  SEQUENCE WINDOWING SUMMARY")
print("=" * 65)
print(f"\n  Window configuration:")
print(f"    Lookback  : {LOOKBACK} days")
print(f"    Horizon   : {HORIZON} day ahead")
print(f"    Features  : {len(FEATURE_COLS)} ")
print(f"\n  Array shapes:")
print(f"    X_train   : {X_train.shape}  →  (samples, timesteps, features)")
print(f"    y_train   : {y_train.shape}")
print(f"    X_val     : {X_val.shape}")
print(f"    y_val     : {y_val.shape}")
print(f"    X_test    : {X_test.shape}")
print(f"    y_test    : {y_test.shape}")

total_mb = (X_train.nbytes + X_val.nbytes + X_test.nbytes +
            y_train.nbytes + y_val.nbytes + y_test.nbytes) / 1e6
print(f"\n  Memory footprint : {total_mb:.1f} MB total")
print(f"\n  Files saved → data/sequences/")
print("=" * 65)