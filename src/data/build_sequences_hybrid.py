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

# ── Standard 32 meteorological features ───────────────────────────────────
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


# ── Standalone GR4J forward pass (numpy) ─────────────────────────────────
def gr4j_forward(precip, pet, x1=300.0, x2=0.5, x3=80.0, x4=2.5):
    n = len(precip)
    S = np.zeros(n)
    R = np.zeros(n)
    states = np.zeros((n, 10))

    for t in range(n):
        P = float(precip[t])
        E = float(pet[t])

        Pn = max(P - E, 0.0)
        En = max(E - P, 0.0)

        # Production store
        s_prev = S[t - 1] if t > 0 else 0.0
        ratio = min(s_prev / x1, 1.0) if x1 > 0 else 0.0
        Ps = Pn * (1.0 - ratio**2)
        Es = En * (2.0 - ratio) * ratio if ratio < 1.0 else En

        s_temp = s_prev + Ps - Es
        s_temp = max(s_temp, 0.0)

        # Percolation
        pratio = s_temp / x1 if x1 > 0 else 0.0
        Perc = s_temp * (1.0 - (1.0 + pratio**4) ** (-0.25))
        s_t = s_temp - Perc
        s_t = max(s_t, 0.0)

        # Routing input
        Pr = Pn - Ps + Perc

        # Groundwater exchange
        r_prev = R[t - 1] if t > 0 else 0.0
        exch = x2 * (r_prev / x3) ** 3.5 if x3 > 0 else 0.0

        # Routing store
        r_temp = r_prev + Pr + exch
        r_temp = max(r_temp, 0.0)

        # Outflow
        rratio = r_temp / x3 if x3 > 0 else 0.0
        Qr = r_temp * (1.0 - (1.0 + rratio**4) ** (-0.25))
        r_t = r_temp - Qr
        r_t = max(r_t, 0.0)

        Q = Qr  # mm/day

        # Log-transformed states (log1p for non-negative)
        states[t, 0] = np.log1p(s_t)
        states[t, 1] = np.log1p(r_t)
        states[t, 2] = np.log1p(Ps)
        states[t, 3] = np.log1p(Es)
        states[t, 4] = np.log1p(Perc)
        states[t, 5] = np.log1p(Pr)
        states[t, 6] = np.log1p(Qr + abs(exch))
        states[t, 7] = exch  # signed
        states[t, 8] = np.log1p(Qr)
        states[t, 9] = np.log1p(Q)

        S[t] = s_t
        R[t] = r_t

    return states


def build_hybrid_sequences(
    horizon: int,
    lookback: int,
    log_transform: bool = True,
    log_eps: float = 0.001,
    x1=300.0,
    x2=0.5,
    x3=80.0,
    x4=2.5,
):
    print(
        f"\n[build_sequences_hybrid.py] h={horizon}, lookback={lookback}, log={log_transform}"
    )

    df = pd.read_csv(MASTER, parse_dates=["date"])

    # ── 32-feature guard ──────────────────────────────────────────────────
    extra = [c for c in df.columns if c not in FEATURES_32 + ["date", "discharge_m3s"]]
    if extra:
        print(f"  Dropping {len(extra)} extra columns: {extra}")
        df = df.drop(columns=extra)

    features = [c for c in df.columns if c in FEATURES_32]
    assert len(features) == 32, f"Expected 32 features, got {len(features)}: {features}"
    # ─────────────────────────────────────────────────────────────────────

    # Compute GR4J states (10 variables) from physical P and PET
    p_arr = df["precip_mm_day"].fillna(0.0).values
    pet_arr = df["pet_mm_day"].fillna(0.0).values
    gr4j_states = gr4j_forward(p_arr, pet_arr, x1, x2, x3, x4)

    # Augmented feature matrix: 32 met + 10 GR4J = 42 columns
    met_vals = df[features].values.astype(np.float32)
    aug = np.concatenate([met_vals, gr4j_states], axis=1)  # (n_timesteps, 42)

    # Target
    y = df["discharge_m3s"].values.astype(np.float64)
    if log_transform:
        y = np.log(y + log_eps)

    # Sliding window
    X_list, y_list, dates = [], [], []
    n = len(df)
    for i in range(n - lookback - horizon + 1):
        seq = aug[i : i + lookback].astype(np.float32)
        tgt = y[i + lookback + horizon - 1]
        dt = df["date"].iloc[i + lookback + horizon - 1]
        X_list.append(seq)
        y_list.append(tgt)
        dates.append(dt)

    X = np.stack(X_list)  # (samples, lookback, 42)
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

    p_idx = features.index("precip_mm_day")
    pet_idx = features.index("pet_mm_day")
    X_scaled[:, :, p_idx] = X[:, :, p_idx]
    X_scaled[:, :, pet_idx] = X[:, :, pet_idx]

    y_scaled = (y_out - y_min) / y_range

    # Save sequences
    suffix = f"_h{horizon}_lb{lookback}_hybrid"
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

    # Feature names for scaler
    gr4j_names = [
        "gr4j_prod_fill",
        "gr4j_rout_fill",
        "gr4j_infil",
        "gr4j_ae",
        "gr4j_perc",
        "gr4j_rout_in",
        "gr4j_rout_total",
        "gr4j_exch",
        "gr4j_rout_out",
        "gr4j_qsim",
    ]
    all_features = features + gr4j_names

    # Save scaler CSV
    scaler_df = pd.DataFrame(
        {
            "min": np.concatenate([x_min, [y_min]]),
            "max": np.concatenate([x_max, [y_max]]),
        },
        index=all_features + ["__target__"],
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
    p.add_argument(
        "--x1", type=float, default=300.0, help="GR4J production store capacity"
    )
    p.add_argument("--x2", type=float, default=0.5, help="GR4J exchange coefficient")
    p.add_argument("--x3", type=float, default=80.0, help="GR4J routing store capacity")
    p.add_argument("--x4", type=float, default=2.5, help="GR4J time constant")
    args = p.parse_args()
    build_hybrid_sequences(
        args.horizon,
        args.lookback,
        bool(args.log_transform),
        args.log_eps,
        args.x1,
        args.x2,
        args.x3,
        args.x4,
    )
