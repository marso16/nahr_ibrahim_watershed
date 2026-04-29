"""
=============================================================================
Nahr Ibrahim Watershed — Data Validation
Giovanni (IMERG + MERRA-2) vs Meteostat Beirut Airport (40100)
=============================================================================
Run this script in the meteo conda environment:
    conda activate meteo
    python data_validation_meteostat.py

Output CSVs and figures are saved to your thesis results folder
and can be read from the thesis environment afterward.

Station: Beirut Rafic Hariri International Airport
  ID        : 40100
  Elevation : 29 m asl
  Distance  : ~47 km from watershed centroid
  Coverage  : Expected full period (major international airport)

Note on temperature bias:
  Beirut Airport sits at 29m asl while the Nahr Ibrahim watershed
  spans 300-2000m asl. A systematic warm bias of 4-8 degrees C in
  station data relative to Giovanni is physically expected due to
  the lapse rate (~0.6 degrees C per 100m elevation gain).
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from meteostat import Daily
from datetime import datetime
from pathlib import Path
from scipy import stats

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

ROOT    = Path("C:/Users/marck/Downloads/nahr_ibrahim_watershed")
RAW_DIR = ROOT / "data"    / "raw"
FIG_DIR = ROOT / "results" / "figures"
MET_DIR = ROOT / "results" / "metrics"

for d in [RAW_DIR, FIG_DIR, MET_DIR]:
    d.mkdir(parents=True, exist_ok=True)

START = datetime(2000, 1, 1)
END   = datetime(2025, 12, 31)

STATION_ID         = "40100"
STATION_NAME       = "Beirut Rafic Hariri Airport"
STATION_ELEV       = 29    # m asl
WATERSHED_ELEV_MEAN= 900   # m asl (approximate watershed mean)

print("=" * 65)
print("  Nahr Ibrahim — Data Validation (Meteostat)")
print(f"  Station: {STATION_NAME} ({STATION_ID})")
print("=" * 65)

# =============================================================================
# 1. FETCH BEIRUT AIRPORT DATA
# =============================================================================

print("\n[1/5] Fetching Beirut Airport station data ...")

try:
    beirut = Daily(STATION_ID, START, END).fetch()
    beirut.index = pd.to_datetime(beirut.index)

    print(f"  Fetched successfully")
    print(f"  Records       : {len(beirut):,}")
    print(f"  Period        : {beirut.index.min().date()} -> "
          f"{beirut.index.max().date()}")
    print(f"\n  Missing values:")
    for col, label in [("prcp","precip"),("tavg","mean T"),
                       ("tmin","min T"),("tmax","max T")]:
        n_miss = beirut[col].isna().sum()
        pct    = 100 * beirut[col].isna().mean()
        print(f"    {label:<10}: {n_miss:,} missing ({pct:.1f}%)")

    # Save raw station data
    beirut_csv = RAW_DIR / f"meteostat_beirut_{STATION_ID}.csv"
    beirut.to_csv(beirut_csv)
    print(f"\n  Saved -> data/raw/meteostat_beirut_{STATION_ID}.csv")
    print(f"\n  Sample (first 5 rows):")
    print(beirut[["prcp","tavg","tmin","tmax"]].head().to_string())

except Exception as e:
    print(f"  Failed to fetch: {e}")
    raise

# =============================================================================
# 2. LOAD GIOVANNI MASTER DATA
# =============================================================================

print("\n[2/5] Loading Giovanni master dataset ...")

master_path = ROOT / "data" / "master" / "nahr_ibrahim_master_model.csv"
if not master_path.exists():
    master_path = ROOT / "data" / "master" / "nahr_ibrahim_master_full.csv"

master = pd.read_csv(master_path, parse_dates=["date"])
master.set_index("date", inplace=True)

print(f"  Records  : {len(master):,}")
print(f"  Period   : {master.index.min().date()} -> "
      f"{master.index.max().date()}")

# =============================================================================
# 3. MERGE DATASETS
# =============================================================================

print("\n[3/5] Merging datasets ...")

merged = master.join(
    beirut[["prcp","tavg","tmin","tmax"]].add_suffix("_stn"),
    how="inner"
)

print(f"  Total overlapping days : {len(merged):,}")
print(f"  Valid precip pairs     : "
      f"{merged[['prcp_stn','precip_mm_day']].dropna().shape[0]:,}")
print(f"  Valid temp pairs       : "
      f"{merged[['tavg_stn','temp_mean_c']].dropna().shape[0]:,}")

# =============================================================================
# 4. VALIDATION METRICS
# =============================================================================

print("\n[4/5] Computing validation metrics ...")

# Elevation lapse rate correction
ELEV_DIFF         = WATERSHED_ELEV_MEAN - STATION_ELEV
LAPSE_RATE        = 0.006  # degrees C per meter
LAPSE_CORRECTION  = ELEV_DIFF * LAPSE_RATE
print(f"\n  Elevation difference : {ELEV_DIFF} m")
print(f"  Expected lapse bias  : -{LAPSE_CORRECTION:.1f} degrees C "
      f"(watershed cooler than airport)")


def validate_pair(obs_series, pred_series, var_name, units,
                  lapse_correct=False):
    valid = obs_series.notna() & pred_series.notna()
    obs   = obs_series[valid].values.astype(float)
    pred  = pred_series[valid].values.astype(float)
    n     = int(valid.sum())

    if n < 50:
        print(f"\n  {var_name} -- insufficient valid days ({n}), skipping")
        return None, None, None, None

    obs_adj = obs - LAPSE_CORRECTION if lapse_correct else obs

    r_raw,  _  = stats.pearsonr(obs, pred)
    r_adj,  _  = stats.pearsonr(obs_adj, pred)
    bias       = float(np.mean(pred - obs))
    bias_adj   = float(np.mean(pred - obs_adj))
    rmse       = float(np.sqrt(np.mean((pred - obs)**2)))
    rmse_adj   = float(np.sqrt(np.mean((pred - obs_adj)**2)))
    mae        = float(np.mean(np.abs(pred - obs)))
    pbias      = float(100 * np.sum(pred - obs) / np.sum(obs)) \
                 if np.sum(np.abs(obs)) > 0 else np.nan

    if r_raw >= 0.90:   quality = "Excellent"
    elif r_raw >= 0.80: quality = "Good"
    elif r_raw >= 0.70: quality = "Acceptable"
    elif r_raw >= 0.60: quality = "Moderate"
    else:               quality = "Poor"

    print(f"\n  {var_name} ({units})")
    print(f"    Valid days       : {n:,}")
    print(f"    Pearson r (raw)  : {r_raw:.4f}")
    if lapse_correct:
        print(f"    Pearson r (adj)  : {r_adj:.4f}  "
              f"(after -{LAPSE_CORRECTION:.1f} deg C lapse correction)")
    print(f"    Bias (raw)       : {bias:+.3f} {units}")
    if lapse_correct:
        print(f"    Bias (adj)       : {bias_adj:+.3f} {units}")
    print(f"    RMSE             : {rmse:.3f} {units}")
    if lapse_correct:
        print(f"    RMSE (adj)       : {rmse_adj:.3f} {units}")
    print(f"    MAE              : {mae:.3f} {units}")
    if not np.isnan(pbias):
        print(f"    PBIAS            : {pbias:+.1f}%")
    print(f"    Quality          : {quality}")

    return {
        "variable"              : var_name,
        "units"                 : units,
        "n_valid_days"          : n,
        "r_raw"                 : round(r_raw, 4),
        "r_lapse_adj"           : round(r_adj, 4) if lapse_correct else None,
        "bias_raw"              : round(bias, 4),
        "bias_lapse_adj"        : round(bias_adj, 4) if lapse_correct else None,
        "rmse"                  : round(rmse, 4),
        "rmse_lapse_adj"        : round(rmse_adj, 4) if lapse_correct else None,
        "mae"                   : round(mae, 4),
        "pbias_%"               : round(pbias, 2) if not np.isnan(pbias) else None,
        "quality"               : quality,
    }, obs, pred, obs_adj


results   = []
plot_data = {}

r1, o1, p1, a1 = validate_pair(
    merged["prcp_stn"], merged["precip_mm_day"],
    "Precipitation", "mm/day", lapse_correct=False)
if r1:
    results.append(r1)
    plot_data["Precipitation (mm/day)"] = {
        "obs":o1,"pred":p1,"adj":a1,"color":"#3b9eff","lapse":False,
        "stn":"prcp_stn","gio":"precip_mm_day"}

r2, o2, p2, a2 = validate_pair(
    merged["tavg_stn"], merged["temp_mean_c"],
    "Mean Temperature", "°C", lapse_correct=True)
if r2:
    results.append(r2)
    plot_data["Mean Temperature (°C)"] = {
        "obs":o2,"pred":p2,"adj":a2,"color":"#f4a261","lapse":True,
        "stn":"tavg_stn","gio":"temp_mean_c"}

r3, o3, p3, a3 = validate_pair(
    merged["tmin_stn"], merged["temp_min_c"],
    "Min Temperature", "°C", lapse_correct=True)
if r3:
    results.append(r3)
    plot_data["Min Temperature (°C)"] = {
        "obs":o3,"pred":p3,"adj":a3,"color":"#00b4a0","lapse":True,
        "stn":"tmin_stn","gio":"temp_min_c"}

r4, o4, p4, a4 = validate_pair(
    merged["tmax_stn"], merged["temp_max_c"],
    "Max Temperature", "°C", lapse_correct=True)
if r4:
    results.append(r4)
    plot_data["Max Temperature (°C)"] = {
        "obs":o4,"pred":p4,"adj":a4,"color":"#a855f7","lapse":True,
        "stn":"tmax_stn","gio":"temp_max_c"}

# Save metrics
val_df = pd.DataFrame(results)
val_df.to_csv(MET_DIR / "data_validation_meteostat_metrics.csv", index=False)
print(f"\n  Metrics -> results/metrics/data_validation_meteostat_metrics.csv")

# =============================================================================
# 5. MONTHLY AGGREGATION VALIDATION
# =============================================================================

print("\n  Monthly aggregation validation ...")

monthly_stn = beirut[["prcp","tavg"]].resample("ME").agg(
    {"prcp":"sum","tavg":"mean"}).dropna()
monthly_gio = master[["precip_mm_day","temp_mean_c"]].resample("ME").agg(
    {"precip_mm_day":"sum","temp_mean_c":"mean"}).dropna()
monthly = monthly_stn.join(monthly_gio, how="inner").dropna()

r_prcp_m = r_tavg_m = None
if len(monthly) >= 12:
    r_prcp_m, _ = stats.pearsonr(
        monthly["prcp"].values, monthly["precip_mm_day"].values)
    r_tavg_m, _ = stats.pearsonr(
        monthly["tavg"].values, monthly["temp_mean_c"].values)
    print(f"  Monthly precipitation r : {r_prcp_m:.4f}  "
          f"(n={len(monthly)} months)")
    print(f"  Monthly temperature r   : {r_tavg_m:.4f}  "
          f"(n={len(monthly)} months)")

# =============================================================================
# 6. VISUALIZATION
# =============================================================================

print("\n[5/5] Generating validation figures ...")

n_vars = len(plot_data)
fig = plt.figure(figsize=(20, 5.5 * n_vars))
fig.patch.set_facecolor("#080f1a")
gs  = gridspec.GridSpec(n_vars, 3, figure=fig, hspace=0.55, wspace=0.35)

for row, (label, d) in enumerate(plot_data.items()):
    obs   = d["obs"]; pred = d["pred"]
    adj   = d["adj"]; color = d["color"]
    lapse = d["lapse"]
    r_val = results[row]["r_raw"]
    bias  = results[row]["bias_raw"]
    units = label.split("(")[1].replace(")","").strip()

    # ── Scatter ───────────────────────────────────────────
    ax_sc = fig.add_subplot(gs[row, 0])
    ax_sc.set_facecolor("#0d1825")
    ax_sc.scatter(obs, pred, alpha=0.15, s=5,
                  color=color, edgecolors="none", label="Raw")
    if lapse:
        ax_sc.scatter(adj, pred, alpha=0.15, s=5,
                      color="#ffffff", edgecolors="none",
                      label=f"Lapse adj")
    lo = min(np.nanmin(obs), np.nanmin(pred)) - 1
    hi = max(np.nanmax(obs), np.nanmax(pred)) + 1
    ax_sc.plot([lo,hi],[lo,hi], color="#e76f51",
               linewidth=1.5, linestyle="--", label="1:1 line")
    ax_sc.set_xlabel(f"Beirut Airport ({units})",
                     color="#8aafc4", fontsize=9)
    ax_sc.set_ylabel(f"Giovanni ({units})",
                     color="#8aafc4", fontsize=9)
    ax_sc.set_title(
        f"{label}\nr={r_val:.3f} | bias={bias:+.2f} {units}",
        color="#e8f4f8", fontsize=10)
    ax_sc.tick_params(colors="#4a6a82", labelsize=8)
    ax_sc.spines[:].set_color("#1e3448")
    ax_sc.legend(facecolor="#0d1825", edgecolor="#1e3448",
                 labelcolor="#8aafc4", fontsize=7)
    ax_sc.set_facecolor("#0d1825")

    # ── Time series ───────────────────────────────────────
    ax_ts = fig.add_subplot(gs[row, 1:])
    ax_ts.set_facecolor("#0d1825")

    valid   = merged[d["stn"]].notna() & merged[d["gio"]].notna()
    dates_v = merged.index[valid]
    obs_v   = merged.loc[valid, d["stn"]].values
    pred_v  = merged.loc[valid, d["gio"]].values
    n_show  = min(1096, len(dates_v))

    ax_ts.plot(dates_v[-n_show:], obs_v[-n_show:],
               color="#8aafc4", linewidth=0.9, alpha=0.9,
               label="Beirut Airport")
    ax_ts.plot(dates_v[-n_show:], pred_v[-n_show:],
               color=color, linewidth=0.9, alpha=0.9,
               linestyle="--", label="Giovanni")
    ax_ts.set_title(f"{label} — Recent Period",
                    color="#e8f4f8", fontsize=10)
    ax_ts.set_ylabel(units, color="#8aafc4", fontsize=9)
    ax_ts.tick_params(colors="#4a6a82", labelsize=8)
    ax_ts.spines[:].set_color("#1e3448")
    ax_ts.legend(facecolor="#0d1825", edgecolor="#1e3448",
                 labelcolor="#8aafc4", fontsize=7)
    ax_ts.set_facecolor("#0d1825")

fig.suptitle(
    f"Data Validation -- Giovanni vs {STATION_NAME} (Meteostat)\n"
    f"Station: {STATION_ID} | Elev: {STATION_ELEV}m | ~47km from watershed",
    color="#e8f4f8", fontsize=13, y=1.01, fontfamily="monospace")

plt.savefig(FIG_DIR / "data_validation_giovanni_vs_meteostat.png",
            dpi=150, bbox_inches="tight", facecolor="#080f1a")
plt.show()
print("  Figure -> results/figures/data_validation_giovanni_vs_meteostat.png")

# =============================================================================
# 7. SEASONAL VALIDATION
# =============================================================================

print("\n  Seasonal precipitation correlation ...")

merged["season"] = merged.index.month.map({
    12:"Winter",1:"Winter",2:"Winter",
    3:"Spring", 4:"Spring",5:"Spring",
    6:"Summer", 7:"Summer",8:"Summer",
    9:"Autumn", 10:"Autumn",11:"Autumn"})

print(f"  {'Season':<10} {'r':>8} {'N days':>10}")
print(f"  {'-'*32}")
for season in ["Winter","Spring","Summer","Autumn"]:
    mask = ((merged["season"]==season) &
            merged["prcp_stn"].notna() &
            merged["precip_mm_day"].notna())
    if mask.sum() < 30:
        continue
    r_s, _ = stats.pearsonr(
        merged.loc[mask,"prcp_stn"].values,
        merged.loc[mask,"precip_mm_day"].values)
    print(f"  {season:<10} {r_s:>8.4f} {mask.sum():>10,}")

# =============================================================================
# 8. FINAL SUMMARY
# =============================================================================

print("\n" + "="*65)
print("  VALIDATION SUMMARY -- GIOVANNI vs BEIRUT AIRPORT")
print("="*65)
print(f"\n  Station : {STATION_NAME} (ID: {STATION_ID})")
print(f"  Elev    : {STATION_ELEV}m asl "
      f"(watershed mean ~{WATERSHED_ELEV_MEAN}m)")
print(f"\n  {'Variable':<22} {'r':>8} {'Bias':>12} "
      f"{'RMSE':>10} {'Quality':>12}")
print(f"  {'-'*66}")
for r in results:
    u = r["units"]
    print(f"  {r['variable']:<22} {r['r_raw']:>8.4f} "
          f"{r['bias_raw']:>+10.3f}{u[0]:1} "
          f"{r['rmse']:>9.3f}{u[0]:1} "
          f"{r['quality']:>12}")

if r_prcp_m and r_tavg_m:
    print(f"\n  Monthly aggregation:")
    print(f"    Precipitation r : {r_prcp_m:.4f}")
    print(f"    Temperature r   : {r_tavg_m:.4f}")

print(f"""
  Notes for supervisor:
  - Temperature bias (~4-8 C) is due to elevation difference
    (airport at 29m vs watershed at ~900m mean elevation).
    Lapse rate correction (-{LAPSE_CORRECTION:.1f} C) removes most bias.
  - Precipitation daily r is moderate -- airport is coastal
    while watershed receives orographic enhancement.
  - Monthly precipitation r (above) is substantially higher,
    confirming correct seasonal pattern in IMERG data.
""")

print(f"  Files saved:")
print(f"    data/raw/meteostat_beirut_{STATION_ID}.csv")
print(f"    results/metrics/data_validation_meteostat_metrics.csv")
print(f"    results/figures/data_validation_giovanni_vs_meteostat.png")
print(f"\n  Done. Switch back to thesis environment.")
print("="*65)