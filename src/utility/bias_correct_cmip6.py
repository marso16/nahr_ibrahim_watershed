"""
Bias-correct CMIP6 atmospheric variables against CHIRPS (precip) and
ERA5-Land (temperature) using monthly empirical quantile mapping.

Inputs:
  data/raw/cmip6/atmospheric/{gcm}/{scenario}/{year}.csv
  data/raw/chirps/chirps_nahr_ibrahim_2000_2025_daily.csv
  data/raw/era5_land/era5land_other_daily.csv

Outputs:
  data/raw/cmip6/atmospheric_bc/{gcm}/{scenario}/merged_daily.csv
    columns: date, precip_mm_day, temp_mean_c, temp_max_c, temp_min_c

The historical overlap window used for fitting the correction is 1995-2014.
Inside this window, ERA5/CHIRPS data is treated as observational truth.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(
    os.environ.get("WATERSHED_ROOT", "C:/Users/marck/Downloads/nahr_ibrahim_watershed")
)
CMIP_DIR = ROOT / "data" / "raw" / "cmip6" / "atmospheric"
BC_OUT_DIR = ROOT / "data" / "raw" / "cmip6" / "atmospheric_bc"
CHIRPS_FILE = (
    ROOT / "data" / "raw" / "chirps" / "chirps_nahr_ibrahim_2000_2025_daily.csv"
)
ERA5_FILE = ROOT / "data" / "raw" / "era5_land" / "era5land_other_daily.csv"
LOG_DIR = ROOT / "logs"
BC_OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ────────────────────────────────────────────────────────────
GCMS = [
    "MPI-ESM1-2-HR",
    "EC-Earth3",
    "ACCESS-CM2",
    "NorESM2-MM",
    "MRI-ESM2-0",
    "CMCC-ESM2",
    "INM-CM5-0",
]
SCENARIOS = ["historical", "ssp245", "ssp585"]

# Calibration window: overlap between observations and CMIP6 historical
CAL_START = pd.Timestamp("2000-01-01")  # earliest observation date
CAL_END = pd.Timestamp("2014-12-31")  # latest CMIP6 historical date

# Wet-day threshold (mm/day). Days below this are treated as zero for
# the wet-day frequency adjustment.
WET_DAY_MM = 0.1


# ── Helpers ──────────────────────────────────────────────────────────────────
def empirical_cdf_value(value, sorted_values):
    """Return the empirical CDF of `value` given a sorted array of historical samples."""
    n = len(sorted_values)
    if n == 0:
        return 0.5
    # Position of value in sorted array
    idx = np.searchsorted(sorted_values, value, side="right")
    return idx / n


def inverse_empirical_cdf(prob, sorted_values):
    """Return the value at quantile `prob` from a sorted historical sample."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    idx = int(np.clip(prob * n, 0, n - 1))
    return float(sorted_values[idx])


def qmap_temperature(obs_train, mod_train, mod_target):
    """
    Additive quantile mapping for temperature-like variables.
    For each value in mod_target:
      q = empirical CDF of mod_train at that value
      obs_val = inverse empirical CDF of obs_train at q
      correction = obs_val - mod_val (at that quantile)
      corrected = mod_value + correction
    Returns array of corrected values, same length as mod_target.
    """
    obs_sorted = np.sort(obs_train)
    mod_sorted = np.sort(mod_train)
    corrected = np.zeros_like(mod_target, dtype=np.float64)
    for i, v in enumerate(mod_target):
        q = empirical_cdf_value(v, mod_sorted)
        mod_val_at_q = inverse_empirical_cdf(q, mod_sorted)
        obs_val_at_q = inverse_empirical_cdf(q, obs_sorted)
        # Additive correction
        corrected[i] = v + (obs_val_at_q - mod_val_at_q)
    return corrected


def qmap_precipitation(obs_train, mod_train, mod_target, wet_thr=WET_DAY_MM):
    """
    Multiplicative quantile mapping for precipitation, with wet-day adjustment.

    Steps:
      1. Determine observed wet-day frequency in training data.
      2. Threshold mod_train so it has the same wet-day frequency
         (set the smallest mod_train values to zero until frequencies match).
      3. Apply quantile mapping to wet days only, multiplicative scaling.
      4. Days below the wet threshold (in scaled mod_target) are set to 0.

    Returns array of corrected values.
    """
    obs = np.asarray(obs_train, dtype=np.float64)
    mod = np.asarray(mod_train, dtype=np.float64)
    target = np.asarray(mod_target, dtype=np.float64)

    # 1. Observed wet-day fraction
    obs_wet_frac = (obs >= wet_thr).mean() if len(obs) > 0 else 0.0

    # 2. Find a threshold on mod that produces the same wet-day fraction
    if len(mod) > 0:
        # The value of mod_train at the (1 - obs_wet_frac) quantile is the threshold
        mod_thr = np.quantile(mod, max(0.0, 1.0 - obs_wet_frac))
    else:
        mod_thr = wet_thr

    obs_wet_sorted = np.sort(obs[obs >= wet_thr])
    mod_wet_sorted = np.sort(mod[mod > mod_thr])

    if len(obs_wet_sorted) < 5 or len(mod_wet_sorted) < 5:
        # Not enough data for meaningful QM — return target unchanged
        return np.maximum(target, 0.0)

    corrected = np.zeros_like(target)
    for i, v in enumerate(target):
        if v <= mod_thr:
            corrected[i] = 0.0  # dry day in corrected output
            continue
        # Quantile of v in mod_wet distribution
        q = empirical_cdf_value(v, mod_wet_sorted)
        # Corresponding observed wet-day quantile
        obs_val_at_q = inverse_empirical_cdf(q, obs_wet_sorted)
        corrected[i] = obs_val_at_q

    return np.maximum(corrected, 0.0)


def load_cmip6_series(gcm, scenario):
    """Concatenate all yearly CSVs for one GCM × scenario into a single dataframe."""
    folder = CMIP_DIR / gcm / scenario
    files = sorted(folder.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f, parse_dates=["date"]) for f in files]
    df = (
        pd.concat(dfs, ignore_index=True)
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return df


def metrics(obs, sim):
    """Quick metric summary for QC reporting."""
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    if len(obs) == 0:
        return {}
    r = np.corrcoef(obs, sim)[0, 1] if obs.std() > 0 and sim.std() > 0 else np.nan
    mean_bias = sim.mean() - obs.mean()
    rmse = float(np.sqrt(np.mean((sim - obs) ** 2)))
    return {
        "obs_mean": float(obs.mean()),
        "sim_mean": float(sim.mean()),
        "mean_bias": float(mean_bias),
        "rmse": rmse,
        "corr": float(r) if not np.isnan(r) else None,
    }


# ── Load observations ────────────────────────────────────────────────────────
print("Loading observations...")
chirps = pd.read_csv(CHIRPS_FILE, parse_dates=["date"])
era5 = pd.read_csv(ERA5_FILE, parse_dates=["date"])
obs = chirps[["date", "precip_mm_day"]].merge(
    era5[["date", "temp_mean_c", "temp_max_c", "temp_min_c"]], on="date", how="inner"
)
obs = obs[(obs["date"] >= CAL_START) & (obs["date"] <= CAL_END)].reset_index(drop=True)
obs["month"] = obs["date"].dt.month
print(
    f"  Observations: {len(obs)} days ({obs['date'].min().date()} → {obs['date'].max().date()})\n"
)

# ── Bias-correct each GCM × scenario ─────────────────────────────────────────
qc_summary = []
all_t0 = datetime.now()

for gcm in GCMS:
    print(f"\n{'=' * 70}\n  {gcm}\n{'=' * 70}")

    # Load CMIP6 historical for this GCM (needed for fitting QM)
    cmip_hist = load_cmip6_series(gcm, "historical")
    if cmip_hist.empty:
        print(f"  No historical CMIP6 data for {gcm}, skipping.")
        continue
    cmip_hist["month"] = cmip_hist["date"].dt.month
    cmip_hist_cal = cmip_hist[
        (cmip_hist["date"] >= CAL_START) & (cmip_hist["date"] <= CAL_END)
    ].reset_index(drop=True)
    print(f"  CMIP6 historical calibration window: {len(cmip_hist_cal)} days")

    for scenario in SCENARIOS:
        print(f"\n  Scenario: {scenario}")
        cmip_target = load_cmip6_series(gcm, scenario)
        if cmip_target.empty:
            print(f"    No data, skipping.")
            continue
        cmip_target["month"] = cmip_target["date"].dt.month

        # Storage for corrected columns
        corrected = pd.DataFrame({"date": cmip_target["date"].values})
        var_metrics = {}

        for var in ["precip_mm_day", "temp_mean_c", "temp_max_c", "temp_min_c"]:
            corrected[var] = np.nan

            for month in range(1, 13):
                obs_m = obs.loc[obs["month"] == month, var].dropna().values
                mod_train_m = (
                    cmip_hist_cal.loc[cmip_hist_cal["month"] == month, var]
                    .dropna()
                    .values
                )
                mask_target = cmip_target["month"] == month
                mod_target_m = cmip_target.loc[mask_target, var].dropna().values

                if len(obs_m) < 30 or len(mod_train_m) < 30:
                    # Insufficient data — fall back to passthrough
                    corrected.loc[mask_target, var] = mod_target_m
                    continue

                if var == "precip_mm_day":
                    corrected_m = qmap_precipitation(obs_m, mod_train_m, mod_target_m)
                else:
                    corrected_m = qmap_temperature(obs_m, mod_train_m, mod_target_m)

                corrected.loc[mask_target, var] = corrected_m

            # QC: for historical scenario, compare corrected vs observed in overlap
            if scenario == "historical":
                merged = obs.merge(
                    corrected.rename(columns={var: f"{var}_bc"})[["date", f"{var}_bc"]],
                    on="date",
                    how="inner",
                )
                if len(merged) > 0:
                    var_metrics[var] = metrics(
                        merged[var].values, merged[f"{var}_bc"].values
                    )

        # Save the corrected series
        out_dir = BC_OUT_DIR / gcm / scenario
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "merged_daily.csv"
        corrected.to_csv(out_file, index=False)

        print(f"    Saved → {out_file.relative_to(ROOT)} ({len(corrected)} days)")
        if var_metrics:
            for v, m in var_metrics.items():
                print(
                    f"    QC ({v}): obs={m['obs_mean']:.2f}  bc={m['sim_mean']:.2f}  "
                    f"bias={m['mean_bias']:+.3f}  rmse={m['rmse']:.3f}  r={m['corr']:.3f}"
                )
                qc_summary.append(
                    {
                        "gcm": gcm,
                        "scenario": scenario,
                        "variable": v,
                        **m,
                    }
                )

# ── Write QC report ──────────────────────────────────────────────────────────
qc_df = pd.DataFrame(qc_summary)
qc_path = LOG_DIR / "bias_correction_qc.csv"
qc_df.to_csv(qc_path, index=False)

elapsed = (datetime.now() - all_t0).total_seconds() / 60
print(f"\n{'=' * 70}")
print(f"  Done. {elapsed:.1f} min total.")
print(f"  QC summary → {qc_path}")
print(f"  Bias-corrected data → {BC_OUT_DIR.relative_to(ROOT)}")
print(f"{'=' * 70}")

if not qc_df.empty:
    print(
        f"\nBias correction QC (historical period only — should show near-zero bias):"
    )
    pivot = qc_df.pivot_table(
        index="gcm", columns="variable", values="mean_bias", aggfunc="first"
    ).round(3)
    print(pivot.to_string())
