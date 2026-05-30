import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(
    os.environ.get("WATERSHED_ROOT", "C:/Users/marck/Downloads/nahr_ibrahim_watershed")
)
SPLIT_DIR = ROOT / "data" / "splits"
SEQ_DIR = ROOT / "data" / "sequences"
FIG_DIR = ROOT / "results" / "figures"

SEQ_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--horizon", type=int, default=1)
parser.add_argument("--lookback", type=int, default=30)
args, _ = parser.parse_known_args()

LOOKBACK = args.lookback
HORIZON = args.horizon
print(f"Running with HORIZON={HORIZON}, LOOKBACK={LOOKBACK}")

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

print(
    f"Sequence windowing — {LOOKBACK}-day lookback, "
    f"{len(FEATURE_COLS)} features, {HORIZON}-day ahead\n"
)


# ── Load normalised splits ─────────────────────────────────────────────────────
train_df = pd.read_csv(SPLIT_DIR / "train_norm.csv", parse_dates=["date"])
val_df = pd.read_csv(SPLIT_DIR / "val_norm.csv", parse_dates=["date"])
test_df = pd.read_csv(SPLIT_DIR / "test_norm.csv", parse_dates=["date"])

print(f"Loaded:  train {len(train_df)}  val {len(val_df)}  test {len(test_df)} rows")

missing = [c for c in FEATURE_COLS if c not in train_df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}\nRe-run split.py first.")
print(f"All {len(FEATURE_COLS)} feature columns present\n")


# ── Sliding window ─────────────────────────────────────────────────────────────
def make_sequences(df, features, target, lookback, horizon):
    X, y, dates = [], [], []
    feat_arr = df[features].values
    target_arr = df[target].values
    date_arr = df["date"].values
    for i in range(lookback, len(df) - horizon + 1):
        X.append(feat_arr[i - lookback : i])
        y.append(target_arr[i + horizon - 1])
        dates.append(date_arr[i + horizon - 1])
    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(dates, dtype="datetime64[ns]"),
    )


X_train, y_train, dates_train = make_sequences(
    train_df, FEATURE_COLS, TARGET, LOOKBACK, HORIZON
)
X_val, y_val, dates_val = make_sequences(
    val_df, FEATURE_COLS, TARGET, LOOKBACK, HORIZON
)
X_test, y_test, dates_test = make_sequences(
    test_df, FEATURE_COLS, TARGET, LOOKBACK, HORIZON
)

print(f"{'Split':<8} {'X shape':<26} {'samples':>8}")
print("-" * 44)
for name, X, y in [
    ("Train", X_train, y_train),
    ("Val", X_val, y_val),
    ("Test", X_test, y_test),
]:
    print(f"{name:<8} {str(X.shape):<26} {len(y):>8,}")
print(f"{'Total':<8} {'':<26} {len(y_train) + len(y_val) + len(y_test):>8,}\n")


# ── Integrity checks ───────────────────────────────────────────────────────────
for name, X in [("Train", X_train), ("Val", X_val), ("Test", X_test)]:
    n = np.isnan(X).sum()
    assert n == 0, f"{name} has {n} NaN(s) — fix in preprocess.py, not here"

print("All integrity checks passed")

# ── Save arrays ───────────────────────────────────────────────────────────────
suffix = f"_h{HORIZON}_lb{LOOKBACK}"
for fname, arr in [
    ("X_train", X_train),
    ("y_train", y_train),
    ("dates_train", dates_train),
    ("X_val", X_val),
    ("y_val", y_val),
    ("dates_val", dates_val),
    ("X_test", X_test),
    ("y_test", y_test),
    ("dates_test", dates_test),
]:
    np.save(SEQ_DIR / f"{fname}{suffix}.npy", arr)

pd.DataFrame(
    {
        "parameter": [
            "lookback",
            "horizon",
            "n_features",
            "feature_cols",
            "target",
            "n_train",
            "n_val",
            "n_test",
        ],
        "value": [
            LOOKBACK,
            HORIZON,
            len(FEATURE_COLS),
            str(FEATURE_COLS),
            TARGET,
            len(y_train),
            len(y_val),
            len(y_test),
        ],
    }
).to_csv(SEQ_DIR / "sequence_metadata.csv", index=False)

import json

with open(SEQ_DIR / "features.json", "w") as f:
    json.dump(
        {
            "feature_cols": FEATURE_COLS,
            "target": TARGET,
            "lookback": LOOKBACK,
            "horizon": HORIZON,
        },
        f,
        indent=2,
    )

total_mb = sum(a.nbytes for a in [X_train, X_val, X_test, y_train, y_val, y_test]) / 1e6
print(f"\nArrays saved → data/sequences/  ({total_mb:.1f} MB total)")
print(f"  X_train {X_train.shape}  X_val {X_val.shape}  X_test {X_test.shape}")


# ── Figures ───────────────────────────────────────────────────────────────────
def style(ax):
    ax.set_facecolor("#ffffff")
    ax.tick_params(colors="#333333")
    ax.spines[:].set_color("#cccccc")

    ax.title.set_color("#111111")
    ax.xaxis.label.set_color("#444444")
    ax.yaxis.label.set_color("#444444")


fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#ffffff")

gs = gridspec.GridSpec(
    2,
    3,
    figure=fig,
    hspace=0.4,
    wspace=0.35,
)

# panel 1 — sliding window diagram
ax0 = fig.add_subplot(gs[0, :2])
style(ax0)

t = np.arange(LOOKBACK + 15)
rng = np.random.RandomState(42)

demo = np.sin(t * 0.25) * 0.3 + 0.5 + rng.randn(LOOKBACK + 15) * 0.05

ax0.fill_between(
    t[:LOOKBACK],
    demo[:LOOKBACK],
    alpha=0.3,
    color="#1f77b4",
    label=f"Input window ({LOOKBACK} days)",
)

ax0.fill_between(
    t[LOOKBACK : LOOKBACK + 1],
    demo[LOOKBACK : LOOKBACK + 1],
    alpha=0.8,
    color="#d62728",
    label="Target (day t+1)",
)

ax0.plot(
    t,
    demo,
    color="#4c72b0",
    linewidth=1.2,
    alpha=0.7,
)

ax0.axvline(
    LOOKBACK,
    color="#d62728",
    linestyle="--",
    linewidth=1.5,
    alpha=0.8,
)

ax0.scatter(
    [LOOKBACK],
    [demo[LOOKBACK]],
    color="#d62728",
    s=80,
    zorder=5,
)

ax0.annotate(
    f"← {LOOKBACK}-day input →",
    xy=(LOOKBACK / 2, 0.92),
    fontsize=10,
    color="#1f77b4",
    fontfamily="monospace",
    ha="center",
)

ax0.annotate(
    "Predict\nday t+1",
    xy=(LOOKBACK, demo[LOOKBACK] + 0.08),
    fontsize=9,
    color="#d62728",
    fontfamily="monospace",
    ha="center",
)

ax0.set_title(
    f"Sliding window — {LOOKBACK}-day lookback · "
    f"1-day ahead · {len(FEATURE_COLS)} features",
    fontsize=11,
    pad=10,
)

ax0.set_xlabel("Time (days)")
ax0.set_ylabel("Normalised discharge")

ax0.legend(
    facecolor="#ffffff",
    edgecolor="#cccccc",
    labelcolor="#333333",
    fontsize=9,
)

# panel 2 — sequence counts
ax1 = fig.add_subplot(gs[0, 2])
style(ax1)

labels = [
    "Train\n2000–2017",
    "Val\n2018–2020",
    "Test\n2021–2025",
]

counts = [
    len(y_train),
    len(y_val),
    len(y_test),
]

colors = [
    "#1f77b4",
    "#ff7f0e",
    "#d62728",
]

bars = ax1.bar(
    labels,
    counts,
    color=colors,
    alpha=0.85,
    width=0.5,
)

for bar, n in zip(bars, counts):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 30,
        f"{n:,}",
        ha="center",
        va="bottom",
        color="#111111",
        fontsize=9,
        fontfamily="monospace",
    )

ax1.set_title(
    "Sequence counts",
    fontsize=11,
    pad=10,
)

ax1.set_ylabel("Number of sequences")

# panel 3 — feature heatmap for one example window
ax2 = fig.add_subplot(gs[1, :2])
style(ax2)

im = ax2.imshow(
    X_train[500].T,
    aspect="auto",
    cmap="RdYlBu_r",
    vmin=0,
    vmax=1,
    interpolation="nearest",
)

ax2.set_yticks(range(len(FEATURE_COLS)))

ax2.set_yticklabels(
    FEATURE_COLS,
    fontsize=7,
    color="#444444",
    fontfamily="monospace",
)

ax2.set_xlabel(f"Day in window (0 = oldest, {LOOKBACK-1} = latest)")

ax2.set_title(
    "Feature heatmap — example window (sample #500)",
    fontsize=11,
    pad=10,
)

cbar = plt.colorbar(
    im,
    ax=ax2,
    fraction=0.015,
    pad=0.02,
)

cbar.set_label(
    "Normalised value",
    color="#444444",
    fontsize=8,
)

plt.setp(
    cbar.ax.yaxis.get_ticklabels(),
    color="#666666",
)

# panel 4 — target distribution
ax3 = fig.add_subplot(gs[1, 2])
style(ax3)

for y, color, label in [
    (y_train, "#1f77b4", "Train"),
    (y_val, "#ff7f0e", "Val"),
    (y_test, "#d62728", "Test"),
]:
    ax3.hist(
        y,
        bins=50,
        color=color,
        alpha=0.7,
        label=label,
        density=True,
        orientation="horizontal",
    )

ax3.set_title(
    "Target (Q) distribution",
    fontsize=11,
    pad=10,
)

ax3.set_ylabel("Normalised discharge")
ax3.set_xlabel("Density")

ax3.legend(
    facecolor="#ffffff",
    edgecolor="#cccccc",
    labelcolor="#333333",
    fontsize=8,
)

fig.suptitle(
    f"Sequence windowing — {LOOKBACK}-day lookback · "
    f"1-day ahead · {len(FEATURE_COLS)} features",
    color="#111111",
    fontsize=13,
    y=0.98,
    fontfamily="monospace",
)

plt.savefig(
    FIG_DIR / "sequence_windowing.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="#ffffff",
)

plt.close()

print("Figure saved → results/figures/sequence_windowing.png")
