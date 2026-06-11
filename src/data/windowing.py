import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(
    os.environ.get("WATERSHED_ROOT", "C:/Users/marck/Downloads/nahr_ibrahim_watershed")
)
MASTER = ROOT / "data" / "master" / "nahr_ibrahim_master_model.csv"
OUT = ROOT / "data" / "sequences"
SPLITS = ROOT / "data" / "splits"
OUT.mkdir(parents=True, exist_ok=True)
SPLITS.mkdir(parents=True, exist_ok=True)

# ── Standard 32-feature set (matches Table 2 in article) ───────────────────
FEATURES_32 = [
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


def build_sequences(
    horizon: int, lookback: int, log_transform: bool = True, log_eps: float = 0.001
):
    print(
        f"\n[windowing.py] h={horizon}, lookback={lookback}, log_transform={log_transform}"
    )

    df = pd.read_csv(MASTER, parse_dates=["date"])

    # ── 32-feature guard ──────────────────────────────────────────────────
    extra = [c for c in df.columns if c not in FEATURES_32 + ["date", "discharge_m3s"]]
    if extra:
        print(f"  Dropping {len(extra)} extra columns: {extra}")
        df = df.drop(columns=extra)

    features = [c for c in df.columns if c in FEATURES_32]
    assert len(features) == 32, f"Expected 32 features, got {len(features)}: {features}"
    # ───────────────────────────────────────────────────────────────────────

    # Target
    y = df["discharge_m3s"].values.astype(np.float64)
    if log_transform:
        y = np.log(y + log_eps)

    # Sliding window
    X_list, y_list, dates = [], [], []
    n = len(df)
    for i in range(n - lookback - horizon + 1):
        seq = df[features].iloc[i : i + lookback].values.astype(np.float32)
        tgt = y[i + lookback + horizon - 1]
        dt = df["date"].iloc[i + lookback + horizon - 1]
        X_list.append(seq)
        y_list.append(tgt)
        dates.append(dt)

    X = np.stack(X_list)  # (samples, lookback, 32)
    y_out = np.array(y_list, dtype=np.float32)
    dates = pd.DatetimeIndex(dates)

    # Chronological split
    train_mask = dates.year <= 2017
    val_mask = (dates.year >= 2018) & (dates.year <= 2020)
    test_mask = dates.year >= 2021

    # Scaling — fit on train only
    X_train = X[train_mask]
    y_train = y_out[train_mask]

    x_min = X_train.min(axis=(0, 1))
    x_max = X_train.max(axis=(0, 1))
    y_min = float(y_train.min())
    y_max = float(y_train.max())

    x_range = x_max - x_min
    x_range[x_range == 0] = 1.0
    y_range = y_max - y_min
    if y_range == 0:
        y_range = 1.0

    X_scaled = (X - x_min) / x_range
    y_scaled = (y_out - y_min) / y_range

    # Save sequences
    suffix = f"_h{horizon}_lb{lookback}"
    np.save(OUT / f"X_train{suffix}.npy", X_scaled[train_mask])
    np.save(OUT / f"y_train{suffix}.npy", y_scaled[train_mask])
    np.save(OUT / f"X_val{suffix}.npy", X_scaled[val_mask])
    np.save(OUT / f"y_val{suffix}.npy", y_scaled[val_mask])
    np.save(OUT / f"X_test{suffix}.npy", X_scaled[test_mask])
    np.save(OUT / f"y_test{suffix}.npy", y_scaled[test_mask])

    # Save dates (needed by training scripts)
    np.save(OUT / f"dates_train{suffix}.npy", dates[train_mask].values)
    np.save(OUT / f"dates_val{suffix}.npy", dates[val_mask].values)
    np.save(OUT / f"dates_test{suffix}.npy", dates[test_mask].values)

    # Save scaler CSV with __target__ and __meta__ rows
    scaler_df = pd.DataFrame(
        {
            "min": np.concatenate([x_min, [y_min]]),
            "max": np.concatenate([x_max, [y_max]]),
        },
        index=features + ["__target__"],
    )
    scaler_df.loc["__meta__"] = [float(log_transform), log_eps]

    scaler_path = SPLITS / f"scaler_params{suffix}.csv"
    scaler_df.to_csv(scaler_path)

    print(f"  X shape: {X_scaled.shape}, y shape: {y_scaled.shape}")
    print(
        f"  Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}"
    )
    print(f"  Scaler saved: {scaler_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--lookback", type=int, required=True)
    p.add_argument("--log_transform", type=int, default=1, help="1=True, 0=False")
    p.add_argument("--log_eps", type=float, default=0.001)
    args = p.parse_args()
    build_sequences(args.horizon, args.lookback, bool(args.log_transform), args.log_eps)
