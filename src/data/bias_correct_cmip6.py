import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

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
CAL_START = pd.Timestamp("2000-01-01")
CAL_END = pd.Timestamp("2014-12-31")
WET_DAY_MM = 0.1


def empirical_cdf_value(value, sorted_values):
    n = len(sorted_values)
    if n == 0:
        return 0.5
    idx = np.searchsorted(sorted_values, value, side="right")
    return idx / n


def inverse_empirical_cdf(prob, sorted_values):
    n = len(sorted_values)
    if n == 0:
        return 0.0
    idx = int(np.clip(prob * n, 0, n - 1))
    return float(sorted_values[idx])


def qmap_temperature(obs_train, mod_train, mod_target):
    obs_sorted = np.sort(obs_train)
    mod_sorted = np.sort(mod_train)
    corrected = np.zeros_like(mod_target, dtype=np.float64)
    for i, v in enumerate(mod_target):
        q = empirical_cdf_value(v, mod_sorted)
        mod_val_at_q = inverse_empirical_cdf(q, mod_sorted)
        obs_val_at_q = inverse_empirical_cdf(q, obs_sorted)
        corrected[i] = v + (obs_val_at_q - mod_val_at_q)
    return corrected


def qmap_precipitation(obs_train, mod_train, mod_target, wet_thr=WET_DAY_MM):
    obs = np.asarray(obs_train, dtype=np.float64)
    mod = np.asarray(mod_train, dtype=np.float64)
    target = np.asarray(mod_target, dtype=np.float64)
    obs_wet_frac = (obs >= wet_thr).mean() if len(obs) > 0 else 0.0

    if len(mod) > 0:
        mod_thr = np.quantile(mod, max(0.0, 1.0 - obs_wet_frac))
    else:
        mod_thr = wet_thr

    obs_wet_sorted = np.sort(obs[obs >= wet_thr])
    mod_wet_sorted = np.sort(mod[mod > mod_thr])

    if len(obs_wet_sorted) < 5 or len(mod_wet_sorted) < 5:
        return np.maximum(target, 0.0)

    corrected = np.zeros_like(target)
    for i, v in enumerate(target):
        if v <= mod_thr:
            corrected[i] = 0.0
            continue
        q = empirical_cdf_value(v, mod_wet_sorted)
        obs_val_at_q = inverse_empirical_cdf(q, obs_wet_sorted)
        corrected[i] = obs_val_at_q

    return np.maximum(corrected, 0.0)


def load_cmip6_series(gcm, scenario):
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

qc_summary = []
all_t0 = datetime.now()

for gcm in GCMS:
    print(f"\n{'=' * 70}\n  {gcm}\n{'=' * 70}")

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
                    corrected.loc[mask_target, var] = mod_target_m
                    continue

                if var == "precip_mm_day":
                    corrected_m = qmap_precipitation(obs_m, mod_train_m, mod_target_m)
                else:
                    corrected_m = qmap_temperature(obs_m, mod_train_m, mod_target_m)

                corrected.loc[mask_target, var] = corrected_m

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
