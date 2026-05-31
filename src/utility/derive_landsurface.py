"""
Calibrate offline land-surface models (degree-day snow + bucket soil moisture)
against historical ERA5-Land data. Saves the calibrated parameters.

These calibrated models will later be applied to bias-corrected CMIP6 future
climate forcings to produce projected soil moisture and SWE consistent with
the LSTM's training distribution.

Inputs (already on disk):
  data/raw/era5_land/era5land_other_daily.csv  — historical T, SWE, soil
                                                  moisture, PET from ERA5-Land
  data/raw/chirps/chirps_nahr_ibrahim_2000_2025_daily.csv — historical precip

Outputs:
  models/trained/landsurface_params.json
  results/figures/landsurface_calibration.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(
    os.environ.get("WATERSHED_ROOT", "C:/Users/marck/Downloads/nahr_ibrahim_watershed")
)
RAW = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models" / "trained"
FIG_DIR = ROOT / "results" / "figures"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Calibrate over the LSTM training period only to avoid leakage into test
TRAIN_START = pd.Timestamp("2000-01-01")
TRAIN_END = pd.Timestamp("2017-12-31")


def nse(obs, sim):
    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom < 1e-12:
        return np.nan
    return 1 - np.sum((obs - sim) ** 2) / denom


def corr(obs, sim):
    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    if obs.std() < 1e-8 or sim.std() < 1e-8:
        return np.nan
    return np.corrcoef(obs, sim)[0, 1]


# ═══════════════════════════════════════════════════════════════════════════
# Snow model
# ═══════════════════════════════════════════════════════════════════════════
def run_snow_model(P_mm, T_C, T_snow: float, melt_factor: float, swe0: float = 0.0):
    """
    Degree-day snow model.

    Parameters
    ----------
    P_mm : array of daily precipitation (mm/day)
    T_C  : array of daily mean temperature (°C)
    T_snow : threshold separating rain from snow (°C)
    melt_factor : snowmelt rate (mm/°C/day)
    swe0 : initial SWE (mm)

    Returns
    -------
    swe : array of daily snow water equivalent (mm)
    rain : array of daily rain that reaches ground (mm)  [excludes snowfall]
    melt : array of daily snowmelt (mm)
    """
    n = len(P_mm)
    swe = np.zeros(n)
    rain = np.zeros(n)
    melt = np.zeros(n)
    state = swe0

    for i in range(n):
        if T_C[i] < T_snow:
            # All precip is snow
            state = state + P_mm[i]
            rain[i] = 0.0
            melt[i] = 0.0
        else:
            # All precip is rain
            rain[i] = P_mm[i]
            # Potential melt
            pot_melt = melt_factor * (T_C[i] - T_snow)
            melt[i] = min(state, pot_melt)
            state = state - melt[i]
        swe[i] = state
    return swe, rain, melt


def calibrate_snow_model(P_mm, T_C, SWE_obs):
    """Calibrate T_snow and melt_factor by maximizing NSE against ERA5 SWE."""

    def objective(params):
        T_snow, melt_factor = params
        swe_sim, _, _ = run_snow_model(P_mm, T_C, T_snow, melt_factor)
        val = nse(SWE_obs, swe_sim)
        return 1.0 if np.isnan(val) else 1.0 - val

    bounds = [
        (-10.0, 5.0),  # T_snow: widen lower
        (0.5, 15.0),  # melt_factor: widen both
    ]
    print("  Calibrating snow model (degree-day)...")
    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=42,
        maxiter=60,
        popsize=15,
        tol=1e-5,
        polish=True,
        disp=False,
    )
    return result.x, 1.0 - result.fun


# ═══════════════════════════════════════════════════════════════════════════
# Bucket soil-moisture model
# ═══════════════════════════════════════════════════════════════════════════
def run_bucket_model(
    input_water_mm,
    PET_mm,
    FC: float,
    WP: float,
    drainage_rate: float,
    ET_scale: float,
    sm0: float,
):
    """
    Single-bucket soil moisture model.

    Parameters
    ----------
    input_water_mm : daily rain + snowmelt reaching the soil (mm/day)
    PET_mm : daily potential evapotranspiration (mm/day)
    FC : field capacity (mm)
    WP : wilting point (mm)
    drainage_rate : fraction of (SM - FC) draining per day when above FC
    ET_scale : maximum AET / PET ratio when soil is fully saturated
    sm0 : initial soil moisture state (mm)

    Returns
    -------
    sm : daily soil moisture (mm)
    aet : daily actual ET (mm)
    """
    n = len(input_water_mm)
    sm = np.zeros(n)
    aet = np.zeros(n)
    state = sm0

    for i in range(n):
        # Inflow
        state = state + input_water_mm[i]

        # Drainage if above field capacity
        if state > FC:
            drain = drainage_rate * (state - FC)
            state = state - drain

        # Actual ET — limited by soil saturation between WP and FC
        if state <= WP:
            actual_et = 0.0
        elif state >= FC:
            actual_et = ET_scale * PET_mm[i]
        else:
            sat = (state - WP) / max(FC - WP, 1e-6)
            actual_et = ET_scale * PET_mm[i] * sat

        actual_et = min(actual_et, max(state - WP, 0.0))
        state = state - actual_et
        state = max(state, 0.0)

        sm[i] = state
        aet[i] = actual_et

    return sm, aet


def calibrate_bucket_model(input_water_mm, PET_mm, SM_obs):
    """
    Calibrate bucket parameters against an observed soil moisture series.
    Target: ERA5-Land 28-100 cm layer (sm_28_100cm_mm) which is what your
    LSTM uses as 'soil_moisture_mm'.
    """
    sm_mean_obs = float(np.mean(SM_obs))

    def objective(params):
        FC, WP, drain, ET_scl = params
        if WP >= FC:
            return 1.0  # invalid
        sm_sim, _ = run_bucket_model(
            input_water_mm,
            PET_mm,
            FC=FC,
            WP=WP,
            drainage_rate=drain,
            ET_scale=ET_scl,
            sm0=sm_mean_obs,  # start near climatology
        )
        # Combined objective: NSE + small penalty on mean bias
        val_nse = nse(SM_obs, sm_sim)
        if np.isnan(val_nse):
            return 1.0
        bias_pen = abs(sm_sim.mean() - SM_obs.mean()) / (sm_mean_obs + 1e-6)
        return (1.0 - val_nse) + 0.1 * bias_pen

    # Bounds chosen to span typical Mediterranean values for the
    # 28-100cm layer (your sm_28_100cm_mm ranges ~187-346 mm).
    bounds = [
        (150.0, 600.0),  # FC: widen
        (20.0, 250.0),  # WP: widen
        (0.001, 0.50),  # drainage_rate: widen upper much more
        (0.1, 1.5),  # ET_scale: widen both
    ]
    print("  Calibrating bucket soil-moisture model...")
    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=42,
        maxiter=80,
        popsize=20,
        tol=1e-5,
        polish=True,
        disp=False,
    )
    FC, WP, drain, ET_scl = result.x
    # Re-run to compute final NSE without the bias penalty
    sm_sim, _ = run_bucket_model(
        input_water_mm,
        PET_mm,
        FC=FC,
        WP=WP,
        drainage_rate=drain,
        ET_scale=ET_scl,
        sm0=sm_mean_obs,
    )
    final_nse = nse(SM_obs, sm_sim)
    return result.x, final_nse


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("Offline land-surface model calibration\n")

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading historical forcing data...")
    chirps = pd.read_csv(
        RAW / "chirps" / "chirps_nahr_ibrahim_2000_2025_daily.csv",
        parse_dates=["date"],
    )
    era5 = pd.read_csv(
        RAW / "era5_land" / "era5land_other_daily.csv",
        parse_dates=["date"],
    )
    print(f"  CHIRPS: {len(chirps)} days")
    print(f"  ERA5-Land: {len(era5)} days\n")

    # Merge and restrict to training period
    df = chirps.merge(era5, on="date", how="inner")
    df = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)].reset_index(
        drop=True
    )
    print(f"  Training window: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Days available: {len(df)}\n")

    # Arrays
    P_mm = df["precip_mm_day"].fillna(0.0).values
    T_C = df["temp_mean_c"].values
    PET_mm = df["pet_mm_day"].values

    # Targets from ERA5-Land
    SWE_obs = df["swe_mm"].values
    SM_obs = df["sm_28_100cm_mm"].values

    # ── Calibrate snow model ──────────────────────────────────────────────
    print("=" * 60)
    print("  Snow model (degree-day)")
    print("=" * 60)
    (T_snow, melt_factor), snow_nse_val = calibrate_snow_model(P_mm, T_C, SWE_obs)
    print(f"  T_snow:      {T_snow:+.2f} °C")
    print(f"  melt_factor: {melt_factor:.2f} mm/°C/day")
    print(f"  NSE vs ERA5-Land SWE: {snow_nse_val:.4f}")

    swe_sim, rain, melt = run_snow_model(P_mm, T_C, T_snow, melt_factor)
    snow_r = corr(SWE_obs, swe_sim)
    snow_mae = mean_absolute_error(SWE_obs, swe_sim)
    print(f"  Correlation: {snow_r:.4f}")
    print(f"  MAE:         {snow_mae:.3f} mm")
    print()

    # ── Calibrate bucket model ────────────────────────────────────────────
    print("=" * 60)
    print("  Bucket soil-moisture model")
    print("=" * 60)
    # Bucket input = rain (excluding snow) + snowmelt
    bucket_input = rain + melt

    (FC, WP, drain, ET_scl), bucket_nse_val = calibrate_bucket_model(
        bucket_input, PET_mm, SM_obs
    )
    print(f"  Field capacity (FC):  {FC:.1f} mm")
    print(f"  Wilting point (WP):   {WP:.1f} mm")
    print(f"  Drainage rate:        {drain:.4f} /day")
    print(f"  ET scale:             {ET_scl:.3f}")
    print(f"  NSE vs ERA5-Land soil moisture: {bucket_nse_val:.4f}")

    sm_sim, aet = run_bucket_model(
        bucket_input,
        PET_mm,
        FC=FC,
        WP=WP,
        drainage_rate=drain,
        ET_scale=ET_scl,
        sm0=float(SM_obs.mean()),
    )
    bucket_r = corr(SM_obs, sm_sim)
    bucket_mae = mean_absolute_error(SM_obs, sm_sim)
    print(f"  Correlation: {bucket_r:.4f}")
    print(f"  MAE:         {bucket_mae:.3f} mm")
    print()

    # ── Save parameters ───────────────────────────────────────────────────
    params = {
        "calibration_window": {
            "start": str(TRAIN_START.date()),
            "end": str(TRAIN_END.date()),
            "n_days": int(len(df)),
        },
        "snow_model": {
            "T_snow_C": float(T_snow),
            "melt_factor_mm_per_C_per_day": float(melt_factor),
            "nse": float(snow_nse_val),
            "correlation": float(snow_r),
            "mae_mm": float(snow_mae),
        },
        "bucket_model": {
            "field_capacity_mm": float(FC),
            "wilting_point_mm": float(WP),
            "drainage_rate_per_day": float(drain),
            "ET_scale": float(ET_scl),
            "nse": float(bucket_nse_val),
            "correlation": float(bucket_r),
            "mae_mm": float(bucket_mae),
            "calibrated_against": "ERA5-Land sm_28_100cm_mm",
        },
        "notes": [
            "Snow model: standard degree-day, calibrated against ERA5-Land swe_mm.",
            "Bucket: single-layer water balance against the 28-100cm soil moisture layer.",
            "Inflow to bucket is rain (excluding snowfall) plus snowmelt.",
            "Target = sm_28_100cm_mm because that's what the LSTM uses as 'soil_moisture_mm'.",
        ],
    }
    out_path = MODEL_DIR / "landsurface_params.json"
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Parameters saved → {out_path}")

    # ── Calibration plots ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle(
        f"Offline land-surface model calibration "
        f"({TRAIN_START.year}–{TRAIN_END.year} ERA5-Land)",
        fontsize=12,
    )

    # SWE time series
    ax = axes[0, 0]
    ax.plot(df["date"], SWE_obs, label="ERA5-Land", color="#1f77b4", lw=1.0)
    ax.plot(df["date"], swe_sim, label="Simulated", color="#ff7f0e", lw=1.0, alpha=0.85)
    ax.set_title(f"SWE — NSE={snow_nse_val:.3f}, r={snow_r:.3f}")
    ax.set_ylabel("SWE (mm)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # SWE scatter
    ax = axes[0, 1]
    ax.scatter(SWE_obs, swe_sim, s=2, alpha=0.4, color="#9467bd")
    lo, hi = 0, max(float(SWE_obs.max()), float(swe_sim.max())) * 1.05
    ax.plot([lo, hi], [lo, hi], "--", color="#888", lw=1)
    ax.set_xlabel("ERA5-Land SWE (mm)")
    ax.set_ylabel("Simulated SWE (mm)")
    ax.set_title("SWE observed vs simulated")
    ax.grid(alpha=0.3)

    # Soil moisture time series
    ax = axes[1, 0]
    ax.plot(df["date"], SM_obs, label="ERA5-Land (28-100cm)", color="#1f77b4", lw=1.0)
    ax.plot(df["date"], sm_sim, label="Bucket sim", color="#ff7f0e", lw=1.0, alpha=0.85)
    ax.set_title(f"Soil moisture — NSE={bucket_nse_val:.3f}, r={bucket_r:.3f}")
    ax.set_ylabel("Soil moisture (mm)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # Soil moisture scatter
    ax = axes[1, 1]
    ax.scatter(SM_obs, sm_sim, s=2, alpha=0.4, color="#9467bd")
    lo = min(float(SM_obs.min()), float(sm_sim.min())) * 0.95
    hi = max(float(SM_obs.max()), float(sm_sim.max())) * 1.05
    ax.plot([lo, hi], [lo, hi], "--", color="#888", lw=1)
    ax.set_xlabel("ERA5-Land soil moisture (mm)")
    ax.set_ylabel("Simulated soil moisture (mm)")
    ax.set_title("Soil moisture observed vs simulated")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = FIG_DIR / "landsurface_calibration.png"
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Plot saved      → {fig_path}\n")

    # Final verdict
    print("=" * 60)
    print("  Verdict")
    print("=" * 60)
    snow_ok = snow_nse_val > 0.5
    bucket_ok = bucket_nse_val > 0.5
    if snow_ok and bucket_ok:
        print(f"  Both models calibrated successfully (NSE > 0.5).")
        print(f"  Ready to apply to CMIP6 future projections.")
    else:
        if not snow_ok:
            print(f"  WARNING: Snow model NSE = {snow_nse_val:.3f} (< 0.5).")
            print(f"           SWE may be poorly captured. Discuss in limitations.")
        if not bucket_ok:
            print(f"  WARNING: Bucket NSE = {bucket_nse_val:.3f} (< 0.5).")
            print(f"           Single-layer bucket may not capture karst storage well.")
        print(
            f"  Models will still apply, but check the calibration plot before proceeding."
        )


if __name__ == "__main__":
    main()
