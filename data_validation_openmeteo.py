"""
=============================================================================
Nahr Ibrahim Watershed — Data Validation
Giovanni (IMERG + MERRA-2) vs Open-Meteo (ERA5)
=============================================================================
Validates satellite/reanalysis inputs against an independent ERA5-based
reference dataset for precipitation and temperature.
=============================================================================
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

ROOT    = Path("C:/Users/marck/Downloads/nahr_ibrahim_watershed")
FIG_DIR = ROOT / "results" / "figures"
MET_DIR = ROOT / "results" / "metrics"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)

LAT = 34.093
LON = 35.878

print("=" * 65)
print("  Nahr Ibrahim — Data Validation")
print("  Giovanni vs Open-Meteo (ERA5)")
print("=" * 65)

# =============================================================================
# FETCH OPEN-METEO ERA5 DATA
# =============================================================================
print("\n[1/4] Fetching Open-Meteo ERA5 reference data ...")

def get_open_meteo(lat, lon, start, end):
    url    = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude"  : lat,
        "longitude" : lon,
        "start_date": start,
        "end_date"  : end,
        "daily"     : [
            "precipitation_sum",
            "temperature_2m_mean",
            "temperature_2m_min",
            "temperature_2m_max",
        ],
        "timezone"  : "Asia/Beirut",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()["daily"]
    df   = pd.DataFrame({
        "date": pd.to_datetime(data["time"]),
        "prcp": data["precipitation_sum"],
        "tavg": data["temperature_2m_mean"],
        "tmin": data["temperature_2m_min"],
        "tmax": data["temperature_2m_max"],
    })
    df.set_index("date", inplace=True)
    return df


ms_data = get_open_meteo(LAT, LON, "2000-01-01", "2025-12-31")

print(f"  Records : {len(ms_data):,}")
print(f"  Period  : {ms_data.index.min().date()} → {ms_data.index.max().date()}")
print(f"  Missing :\n{ms_data.isnull().sum().to_string()}")
print(f"\n  Sample:")
print(ms_data.head(5).to_string())

# =============================================================================
# LOAD GIOVANNI MASTER DATA
# =============================================================================

print("\n[2/4] Loading Giovanni master dataset ...")

master_path = ROOT / "data" / "master" / "nahr_ibrahim_master_model.csv"
if not master_path.exists():
    master_path = ROOT / "data" / "master" / "nahr_ibrahim_master_full.csv"

master = pd.read_csv(master_path, parse_dates=["date"])
master.set_index("date", inplace=True)

print(f"  Records : {len(master):,}")
print(f"  Columns : {master.columns.tolist()}")

# =============================================================================
# MERGE AND VALIDATE
# =============================================================================

print("\n[3/4] Computing validation metrics ...")

# Merge on date
merged = master.join(ms_data, how="inner").dropna(
    subset=["precip_mm_day", "temp_mean_c",
            "temp_min_c", "temp_max_c",
            "prcp", "tavg", "tmin", "tmax"]
)

print(f"  Overlapping days: {len(merged):,}")

def validate_pair(obs, pred, var_name, units):
    mask = ~(np.isnan(obs) | np.isnan(pred))
    obs  = obs[mask]
    pred = pred[mask]

    r, p_val = stats.pearsonr(obs, pred)
    bias     = np.mean(pred - obs)
    rmse     = np.sqrt(np.mean((pred - obs)**2))
    mae      = np.mean(np.abs(pred - obs))

    pbias = 100 * np.sum(pred - obs) / np.sum(obs) if np.sum(obs) > 0 else np.nan

    print(f"\n  {var_name} ({units})")
    print(f"    Pearson r   : {r:.4f}  (p = {p_val:.2e})")
    print(f"    Bias        : {bias:+.3f} {units}")
    print(f"    RMSE        : {rmse:.3f} {units}")
    print(f"    MAE         : {mae:.3f} {units}")
    if not np.isnan(pbias):
        print(f"    PBIAS       : {pbias:+.1f}%")

    if r >= 0.90:   quality = "Excellent"
    elif r >= 0.80: quality = "Good"
    elif r >= 0.70: quality = "Acceptable"
    elif r >= 0.60: quality = "Moderate"
    else:           quality = "Poor"
    print(f"    Quality     : {quality}")

    return {
        "variable" : var_name,
        "units"    : units,
        "r"        : round(r, 4),
        "p_value"  : round(p_val, 6),
        "bias"     : round(bias, 4),
        "rmse"     : round(rmse, 4),
        "mae"      : round(mae, 4),
        "pbias_%"  : round(pbias, 2) if not np.isnan(pbias) else None,
        "quality"  : quality,
        "n_days"   : int(mask.sum()),
    }, obs, pred


results = []

r1, obs_prcp, pred_prcp = validate_pair(
    merged["prcp"].values,
    merged["precip_mm_day"].values,
    "Precipitation", "mm/day"
)
results.append(r1)

r2, obs_tavg, pred_tavg = validate_pair(
    merged["tavg"].values,
    merged["temp_mean_c"].values,
    "Mean Temperature", "°C"
)
results.append(r2)

r3, obs_tmin, pred_tmin = validate_pair(
    merged["tmin"].values,
    merged["temp_min_c"].values,
    "Min Temperature", "°C"
)
results.append(r3)

r4, obs_tmax, pred_tmax = validate_pair(
    merged["tmax"].values,
    merged["temp_max_c"].values,
    "Max Temperature", "°C"
)
results.append(r4)

# Save metrics
val_df = pd.DataFrame(results)
val_df.to_csv(MET_DIR / "data_validation_metrics.csv", index=False)
print(f"\n  Metrics saved → results/metrics/data_validation_metrics.csv")

# =============================================================================
# VISUALIZATION
# =============================================================================
print("\n[4/4] Generating validation figures ...")

fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor("#080f1a")
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.35)

pairs = [
    ("Precipitation (mm/day)", obs_prcp, pred_prcp, r1["r"], "#3b9eff"),
    ("Mean Temperature (°C)",  obs_tavg, pred_tavg, r2["r"], "#f4a261"),
    ("Min Temperature (°C)",   obs_tmin, pred_tmin, r3["r"], "#00b4a0"),
    ("Max Temperature (°C)",   obs_tmax, pred_tmax, r4["r"], "#a855f7"),
]

for row, (label, obs, pred, r_val, color) in enumerate(pairs):

    # ── Scatter plot ──────────────────────────────────────
    ax_sc = fig.add_subplot(gs[row, 0])
    ax_sc.set_facecolor("#0d1825")
    ax_sc.scatter(obs, pred, alpha=0.15, s=4, color=color, edgecolors="none")
    lim = max(np.nanmax(obs), np.nanmax(pred)) * 1.05
    lo  = min(np.nanmin(obs), np.nanmin(pred)) - 0.5
    ax_sc.plot([lo, lim], [lo, lim], color="#e76f51", linewidth=1.5, linestyle="--", label="1:1 line")
    ax_sc.set_xlabel("Open-Meteo ERA5", color="#8aafc4", fontsize=9)
    ax_sc.set_ylabel("Giovanni", color="#8aafc4", fontsize=9)
    ax_sc.set_title(f"{label}\nr = {r_val:.3f}", color="#e8f4f8", fontsize=10)
    ax_sc.tick_params(colors="#4a6a82", labelsize=8)
    ax_sc.spines[:].set_color("#1e3448")
    ax_sc.legend(facecolor="#0d1825", edgecolor="#1e3448", labelcolor="#8aafc4", fontsize=7)
    ax_sc.set_facecolor("#0d1825")

    # ── Time series (last 3 years) ────────────────────────
    ax_ts = fig.add_subplot(gs[row, 1:])
    ax_ts.set_facecolor("#0d1825")
    dates = merged.index[-1096:] 
    obs_r  = obs[-1096:]
    pred_r = pred[-1096:]

    ax_ts.plot(dates, obs_r,  color="#8aafc4",  linewidth=0.8, alpha=0.9, label="Open-Meteo ERA5")
    ax_ts.plot(dates, pred_r, color=color,      linewidth=0.8, alpha=0.9, linestyle="--", label="Giovanni")
    ax_ts.set_title(f"{label} — 2023–2025 Comparison", color="#e8f4f8", fontsize=10)
    ax_ts.set_ylabel(label.split("(")[1].replace(")", ""), color="#8aafc4", fontsize=9)
    ax_ts.tick_params(colors="#4a6a82", labelsize=8)
    ax_ts.spines[:].set_color("#1e3448")
    ax_ts.legend(facecolor="#0d1825", edgecolor="#1e3448", labelcolor="#8aafc4", fontsize=7)
    ax_ts.set_facecolor("#0d1825")

fig.suptitle(
    "Data Validation — Giovanni vs Open-Meteo ERA5",
    color="#e8f4f8", fontsize=14, y=1.01, fontfamily="monospace"
)

plt.savefig(FIG_DIR / "data_validation_giovanni_vs_openmeteo.png", dpi=150, bbox_inches="tight", facecolor="#080f1a")
plt.show()

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("  VALIDATION SUMMARY")
print("=" * 65)
print(f"\n  {'Variable':<25} {'r':>8} {'Bias':>10} {'RMSE':>10} {'Quality':>12}")
print(f"  {'-'*68}")
for row in results:
    units = row["units"]
    print(f"  {row['variable']:<25} {row['r']:>8.4f} "
          f"{row['bias']:>+9.3f}{units[0]:>1} "
          f"{row['rmse']:>9.3f}{units[0]:>1} "
          f"{row['quality']:>12}")

print(f"\n  Interpretation:")
print(f"  Both Giovanni and Open-Meteo use ERA5 as a base,")
print(f"  so strong agreement is expected and confirms data")
print(f"  integrity. Precipitation differences reflect IMERG")
print(f"  satellite correction vs ERA5 model simulation.")
print(f"\n  Files saved:")
print(f"    results/metrics/data_validation_metrics.csv")
print(f"    results/figures/data_validation_giovanni_vs_openmeteo.png")
print("=" * 65)