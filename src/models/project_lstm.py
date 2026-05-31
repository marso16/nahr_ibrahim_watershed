import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime

# ─── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(
    os.environ.get("WATERSHED_ROOT", "C:/Users/marck/Downloads/nahr_ibrahim_watershed")
)
sys.path.insert(0, str(ROOT / "src" / "models"))

FUTURE_DIR = ROOT / "data" / "master" / "future"
SCALER_FILE = ROOT / "data" / "splits" / "scaler_params.csv"
MODEL_DIR = ROOT / "models" / "trained"
OUT_DIR = ROOT / "data" / "projections"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Configuration ──────────────────────────────────────────────────────────
GCMS = [
    "MPI-ESM1-2-HR",
    "EC-Earth3",
    "ACCESS-CM2",
    "NorESM2-MM",
    "MRI-ESM2-0",
    "CMCC-ESM2",
    "INM-CM5-0",
]
SCENARIOS = ["ssp245", "ssp585"]
HORIZONS = [1, 3, 7, 14, 30]
LOOKBACK = 30

# Feature columns — must match training order exactly!
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


# ─── Import the model architecture ──────────────────────────────────────────
print("Importing model architecture...")
try:
    from lstm import WatershedLSTM  # adjust if your class name differs

    print("  WatershedLSTM imported from lstm.py")
except ImportError as e:
    print(f"  FAILED: {e}")
    print(f"  Check that src/models/lstm.py exists and exports the model class.")
    print(f"  If your class is named differently, edit the import in this script.")
    sys.exit(1)


# ─── Load scaler parameters ─────────────────────────────────────────────────
print("\nLoading scaler parameters from training...")
scaler_df = pd.read_csv(SCALER_FILE, index_col=0)
print(f"  Scaler entries: {len(scaler_df)} rows")

# Extract per-feature min/max
feature_scaler = {}
for col in FEATURE_COLS:
    if col not in scaler_df.index:
        raise KeyError(f"Feature '{col}' missing from scaler_params.csv")
    feature_scaler[col] = {
        "min": float(scaler_df.loc[col, "min"]),
        "max": float(scaler_df.loc[col, "max"]),
    }

# Discharge scaler + log-transform metadata
Q_MIN = float(scaler_df.loc["discharge_m3s", "min"])
Q_MAX = float(scaler_df.loc["discharge_m3s", "max"])
if "__meta__" in scaler_df.index:
    LOG_TRANSFORM = bool(float(scaler_df.loc["__meta__", "min"]))
    LOG_EPS = float(scaler_df.loc["__meta__", "max"])
else:
    LOG_TRANSFORM = False
    LOG_EPS = 0.0
print(f"  Discharge range (in transform space): [{Q_MIN:.4f}, {Q_MAX:.4f}]")
print(f"  Log transform: {LOG_TRANSFORM}  log_eps={LOG_EPS}")


# ─── Normalize / inverse-transform helpers ──────────────────────────────────
def normalize_features(df: pd.DataFrame) -> np.ndarray:
    """Min-max normalize each feature column using training scaler."""
    X = np.zeros((len(df), len(FEATURE_COLS)), dtype=np.float32)
    for i, col in enumerate(FEATURE_COLS):
        v = df[col].values.astype(np.float32)
        v_min = feature_scaler[col]["min"]
        v_max = feature_scaler[col]["max"]
        rng = v_max - v_min
        if rng < 1e-12:
            X[:, i] = 0.0
        else:
            X[:, i] = (v - v_min) / rng
        # Clip to [0,1] — future values may exceed training range
        X[:, i] = np.clip(X[:, i], 0.0, 1.0)
    return X


def build_windows(X: np.ndarray, lookback: int):
    """Build sliding lookback windows (T - lookback + 1, lookback, F)."""
    n = len(X)
    if n < lookback:
        return None
    n_windows = n - lookback + 1
    return np.stack([X[i : i + lookback] for i in range(n_windows)], axis=0)


def inverse_discharge(y_pred_norm: np.ndarray) -> np.ndarray:
    """Inverse-transform predictions from normalized → m³/s."""
    y_lin = y_pred_norm * (Q_MAX - Q_MIN) + Q_MIN
    if LOG_TRANSFORM:
        y_real = np.maximum(np.exp(y_lin) - LOG_EPS, 0.0)
    else:
        y_real = y_lin
    return y_real


# ─── Main projection loop ───────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

t_start = datetime.now()
summary = []
failed = []

for horizon in HORIZONS:
    print(f"\n{'═' * 70}\n  HORIZON: {horizon} days\n{'═' * 70}")

    # ─── Load model for this horizon ────────────────────────────────────
    # Adjust filename pattern to match what your training script saves
    candidates = [
        MODEL_DIR / f"lstm_final_strategy_a_h{horizon}.pt",
        MODEL_DIR / f"lstm_final_strategy_a_h{horizon}_lb{LOOKBACK}.pt",
        MODEL_DIR / f"lstm_final_h{horizon}.pt",
    ]
    model_path = next((p for p in candidates if p.exists()), None)
    if model_path is None:
        print(f"  WARNING: No trained model found for h={horizon}. Tried:")
        for c in candidates:
            print(f"    {c}")
        print(f"  Skipping horizon {horizon}.")
        continue
    print(f"  Loading model: {model_path.name}")

    # Initialize model architecture — must match training
    model = WatershedLSTM(input_dim=len(FEATURE_COLS)).to(device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Loaded {n_params:,} parameters")

    # ─── Iterate over GCMs × scenarios ─────────────────────────────────
    for gcm in GCMS:
        for scen in SCENARIOS:
            in_file = FUTURE_DIR / f"{gcm}__{scen}.csv"
            if not in_file.exists():
                failed.append((gcm, scen, horizon, "missing input"))
                continue

            df = pd.read_csv(in_file, parse_dates=["date"])
            df = df.sort_values("date").reset_index(drop=True)

            # Check for NaN before normalizing
            n_nan = df[FEATURE_COLS].isna().sum().sum()
            if n_nan > 0:
                print(
                    f"    [warn] {gcm}/{scen}: {n_nan} NaN in features — filling with 0"
                )
                df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)

            X = normalize_features(df)
            windows = build_windows(X, LOOKBACK)
            if windows is None:
                failed.append((gcm, scen, horizon, "too few rows"))
                continue

            # Inference in batches to manage GPU memory
            batch_size = 512
            preds = []
            with torch.no_grad():
                for i in range(0, len(windows), batch_size):
                    batch = torch.from_numpy(windows[i : i + batch_size]).to(device)
                    out = model(batch)
                    preds.append(out.cpu().numpy().flatten())
            y_norm = np.concatenate(preds)
            y_real = inverse_discharge(y_norm)

            # Predictions align with the LAST day of each window + horizon offset.
            # Window 0 covers dates[0:lookback], LSTM predicts horizon days ahead.
            # Target date for window i is dates[i + lookback - 1 + horizon].
            n_pred = len(y_real)
            pred_dates = df["date"].values[
                LOOKBACK - 1 + horizon : LOOKBACK - 1 + horizon + n_pred
            ]
            # If we went past the end, trim
            n_keep = min(len(pred_dates), len(y_real))

            out_df = pd.DataFrame(
                {
                    "date": pred_dates[:n_keep],
                    "discharge_m3s_pred": y_real[:n_keep],
                }
            )

            out_path = OUT_DIR / f"{gcm}__{scen}__h{horizon}.csv"
            out_df.to_csv(out_path, index=False)

            summary.append(
                {
                    "gcm": gcm,
                    "scenario": scen,
                    "horizon": horizon,
                    "n_predictions": int(n_keep),
                    "first_date": str(pd.Timestamp(pred_dates[0]).date()),
                    "last_date": str(pd.Timestamp(pred_dates[n_keep - 1]).date()),
                    "mean_m3s": float(np.mean(y_real[:n_keep])),
                    "median_m3s": float(np.median(y_real[:n_keep])),
                    "max_m3s": float(np.max(y_real[:n_keep])),
                    "p95_m3s": float(np.percentile(y_real[:n_keep], 95)),
                }
            )

            print(
                f"    {gcm:<18} {scen:<8} → {n_keep} days, "
                f"mean={np.mean(y_real[:n_keep]):.2f} m³/s, "
                f"max={np.max(y_real[:n_keep]):.2f}"
            )

    # Free GPU memory between horizons
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

# ─── Summary ────────────────────────────────────────────────────────────────
elapsed = (datetime.now() - t_start).total_seconds() / 60
print(f"\n{'═' * 70}")
print(f"  Done in {elapsed:.1f} minutes")
print(f"  Successful projections: {len(summary)}")
print(f"  Failed:                 {len(failed)}")
print(f"{'═' * 70}\n")

if summary:
    sdf = pd.DataFrame(summary)
    sdf.to_csv(OUT_DIR / "_projection_summary.csv", index=False)
    print("Summary by horizon (mean discharge in m³/s):")
    pivot = sdf.pivot_table(
        index=["gcm", "scenario"],
        columns="horizon",
        values="mean_m3s",
        aggfunc="first",
    ).round(2)
    print(pivot.to_string())

if failed:
    print(f"\nFailed combinations:")
    for f in failed:
        print(f"  {f}")
