import os
import re
import sys
import io
import glob
import zipfile
import warnings
import logging
import shutil
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rasterio
import contextlib
import cfgrib
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping
from pathlib import Path
from tqdm import tqdm

logging.getLogger("cfgrib").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
os.environ["CPL_LOG"] = "NUL"
os.environ["ECCODES_LOG_STREAM"] = "NUL"


# =============================================================================
# PATHS
# =============================================================================
ROOT        = Path("C:/Users/marck/Downloads/nahr_ibrahim_watershed")
RAW         = ROOT / "data" / "raw"
PROCESSED   = ROOT / "data" / "processed"
MASTER_DIR  = ROOT / "data" / "master"

GIOVANNI    = RAW / "giovanni"
GLOFAS_DIR  = RAW / "glofas"
MODIS_DIR   = RAW / "modis"
SHAPE_DIR   = RAW / "shapefiles"

for d in [PROCESSED, MASTER_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GEOJSON    = SHAPE_DIR / "nahr_ibrahim_watershed.geojson"
DATE_START = "2000-01-01"
DATE_END   = "2025-12-31"

print("=" * 70)
print("  Nahr Ibrahim Watershed — Preprocessing Pipeline")
print("=" * 70)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def parse_giovanni_csv(filepath: Path) -> pd.DataFrame:
    """Parse area-averaged time series CSV from NASA Giovanni."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("time,"):
            data_start = i
            break

    df = pd.read_csv(filepath, skiprows=data_start, header=0)
    df.columns = [c.strip() for c in df.columns]
    df = df.iloc[:, :2].copy()
    df.columns = ["time", "value"]
    df["time"] = pd.to_datetime(df["time"])

    # Replace fill values with NaN
    df["value"] = df["value"].replace(-9999.9, np.nan)
    df["value"] = df["value"].replace(-9999,   np.nan)

    df = df.dropna(subset=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def resample_to_daily(df: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    """Resample a sub-daily Giovanni time series to daily."""
    df = df.set_index("time")
    if method == "sum":
        daily = df["value"].resample("D").sum()
    else:
        daily = df["value"].resample("D").mean()
    return daily.reset_index().rename(columns={"time": "date", "value": "value"})


def clip_to_daterange(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df[date_col] = pd.to_datetime(df[date_col])
    mask = (df[date_col] >= DATE_START) & (df[date_col] <= DATE_END)
    return df[mask].reset_index(drop=True)


def flag_outliers(series: pd.Series, iqr_factor: float = 3.0) -> pd.Series:
    """Return boolean mask of outliers using IQR method."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - iqr_factor * IQR
    upper = Q3 + iqr_factor * IQR
    return (series < lower) | (series > upper)


# =============================================================================
# PRECIPITATION — IMERG (daily, mm/day)
# =============================================================================
print("\n[1/6] Processing Precipitation ...")
(PROCESSED / "precip_imerg_daily.csv").unlink(missing_ok=True)

precip_files = (list(GIOVANNI.glob("*IMERG*precip*.csv"))
                + list(GIOVANNI.glob("*imerg*.csv"))
                + list(GIOVANNI.glob("*GPM*.csv")))

precip_raw = parse_giovanni_csv(precip_files[0])
print(f"   Raw rows: {len(precip_raw)} | "
      f"Period: {precip_raw.time.min()} → {precip_raw.time.max()}")

if precip_raw["time"].dt.hour.nunique() > 1:
    precip_daily = resample_to_daily(precip_raw, method="sum")
else:
    precip_daily = precip_raw.rename(columns={"time": "date"})

precip_daily = clip_to_daterange(precip_daily)

neg_mask = precip_daily["value"] < 0
if neg_mask.sum() > 0:
    print(f"   WARNING: {neg_mask.sum()} negative precipitation values set to 0")
    precip_daily.loc[neg_mask, "value"] = 0.0

outliers = flag_outliers(precip_daily["value"])
print(f"   Outlier check (IQRx3): {outliers.sum()} extreme values flagged")

precip_daily.rename(columns={"value": "precip_mm_day"}, inplace=True)
precip_daily.to_csv(PROCESSED / "precip_imerg_daily.csv", index=False)
print(f"   Saved → precip_imerg_daily.csv  ({len(precip_daily)} rows)")

# =============================================================================
# TEMPERATURE — MERRA-2 (hourly → daily)
# =============================================================================
print("\n[2/6] Processing MERRA-2 Temperature ...")
(PROCESSED / "temp_merra2_daily.csv").unlink(missing_ok=True)

temp_files = (list(GIOVANNI.glob("*MERRA*T2M*.csv"))
              + list(GIOVANNI.glob("*MERRA*temp*.csv"))
              + list(GIOVANNI.glob("*T2M*.csv")))

temp_raw = parse_giovanni_csv(temp_files[0])
print(f"   Raw rows: {len(temp_raw)} | "
      f"Period: {temp_raw.time.min()} → {temp_raw.time.max()}")

temp_raw = temp_raw.set_index("time")
temp_daily = pd.DataFrame({
    "date"        : temp_raw["value"].resample("D").mean().index,
    "temp_mean_c" : temp_raw["value"].resample("D").mean().values,
    "temp_max_c"  : temp_raw["value"].resample("D").max().values,
    "temp_min_c"  : temp_raw["value"].resample("D").min().values,
})

temp_daily = clip_to_daterange(temp_daily)

invalid = (temp_daily["temp_mean_c"] < -20) | (temp_daily["temp_mean_c"] > 45)
if invalid.sum() > 0:
    print(f"   WARNING: {invalid.sum()} temperature values outside [-20, 45]°C")

temp_daily.to_csv(PROCESSED / "temp_merra2_daily.csv", index=False)
print(f"   Saved → temp_merra2_daily.csv  ({len(temp_daily)} rows)")
print(f"   Tmean range: {temp_daily.temp_mean_c.min():.1f}°C → "
      f"{temp_daily.temp_mean_c.max():.1f}°C")

# =============================================================================
# SNOW WATER EQUIVALENT — GLDAS (3-hourly → daily)
# =============================================================================
print("\n[3/6] Processing GLDAS SWE ...")
(PROCESSED / "swe_gldas_daily.csv").unlink(missing_ok=True)

swe_files = (list(GIOVANNI.glob("*SWE*.csv"))
             + list(GIOVANNI.glob("*snow_water*.csv"))
             + list(GIOVANNI.glob("*GLDAS*SWE*.csv")))

swe_raw = parse_giovanni_csv(swe_files[0])
print(f"   Raw rows: {len(swe_raw)} | "
      f"Period: {swe_raw.time.min()} → {swe_raw.time.max()}")

swe_daily = resample_to_daily(swe_raw, method="mean")
swe_daily = clip_to_daterange(swe_daily)

neg_swe = swe_daily["value"] < 0
if neg_swe.sum() > 0:
    print(f"   WARNING: {neg_swe.sum()} negative SWE values → set to 0")
    swe_daily.loc[neg_swe, "value"] = 0.0

swe_daily.rename(columns={"value": "swe_mm"}, inplace=True)
swe_daily.to_csv(PROCESSED / "swe_gldas_daily.csv", index=False)
print(f"   Saved → swe_gldas_daily.csv  ({len(swe_daily)} rows)")
print(f"   SWE range: {swe_daily.swe_mm.min():.1f} → "
      f"{swe_daily.swe_mm.max():.1f} mm")

# =============================================================================
# SOIL MOISTURE — GLDAS Noah SoilMoi0_10cm (3-hourly → daily)   ← NEW
# =============================================================================
print("\n[4/6] Processing GLDAS Soil Moisture (0–10 cm) ...")
(PROCESSED / "soil_moisture_gldas_daily.csv").unlink(missing_ok=True)

sm_files = (list(GIOVANNI.glob("*SoilMoi0_10cm*.csv"))
            + list(GIOVANNI.glob("*SoilMoi*.csv"))
            + list(GIOVANNI.glob("*soil_moi*.csv")))

if not sm_files:
    print("   WARNING: No soil moisture file found in data/raw/giovanni/")
    print("   Expected filename pattern: *SoilMoi0_10cm*.csv")
    print("   Soil moisture features will be filled with NaN and interpolated.")
    sm_daily = pd.DataFrame({
        "date"              : pd.date_range(DATE_START, DATE_END, freq="D"),
        "soil_moisture_mm"  : np.nan,
    })
else:
    sm_raw = parse_giovanni_csv(sm_files[0])
    print(f"   Raw rows: {len(sm_raw)} | "
          f"Period: {sm_raw.time.min()} → {sm_raw.time.max()}")
    print(f"   Source  : {sm_files[0].name}")

    # Aggregate 3-hourly → daily mean
    sm_daily = resample_to_daily(sm_raw, method="mean")
    sm_daily = clip_to_daterange(sm_daily)

    # Replace any remaining fill values
    sm_daily["value"] = sm_daily["value"].replace(-9999, np.nan)
    sm_daily["value"] = sm_daily["value"].replace(-9999.9, np.nan)

    # Clip to physical range (0–500 mm for 0–10 cm layer)
    neg_sm = sm_daily["value"] < 0
    if neg_sm.sum() > 0:
        print(f"   WARNING: {neg_sm.sum()} negative SM values → set to 0")
        sm_daily.loc[neg_sm, "value"] = 0.0

    sm_daily.rename(columns={"value": "soil_moisture_mm"}, inplace=True)

print(f"   SM range: {sm_daily['soil_moisture_mm'].min():.2f} → "
      f"{sm_daily['soil_moisture_mm'].max():.2f} mm")
print(f"   Missing : {sm_daily['soil_moisture_mm'].isna().sum()} days")

sm_daily.to_csv(PROCESSED / "soil_moisture_gldas_daily.csv", index=False)
print(f"   Saved → soil_moisture_gldas_daily.csv  ({len(sm_daily)} rows)")

# =============================================================================
# RIVER DISCHARGE — GloFAS (daily, m³/s)
# =============================================================================
print("\n[5/6] Processing GloFAS Discharge ...")
(PROCESSED / "discharge_glofas_daily.csv").unlink(missing_ok=True)

watershed = gpd.read_file(GEOJSON)
if watershed.crs.to_epsg() != 4326:
    watershed = watershed.to_crs(epsg=4326)

centroid = watershed.geometry.centroid.iloc[0]
lat_out  = centroid.y
lon_out  = centroid.x
print(f"   Outlet point (centroid): lat={lat_out:.4f}, lon={lon_out:.4f}")

nc_extract_dir = GLOFAS_DIR / "extracted"
grib_files = (sorted(glob.glob(str(nc_extract_dir / "*.grib")))
              + sorted(glob.glob(str(nc_extract_dir / "*.grib2")))
              + sorted(glob.glob(str(nc_extract_dir / "data.grib"))))

if not grib_files:
    grib_files = sorted(glob.glob(str(GLOFAS_DIR / "*.grib")))

sample_ds = xr.open_dataset(
    grib_files[0], engine="cfgrib",
    backend_kwargs={"indexpath": ""}
)
print(f"   Variables found  : {list(sample_ds.data_vars)}")
print(f"   Coordinates found: {list(sample_ds.coords)}")
sample_ds.close()

discharge_frames = []
for grib_path in tqdm(grib_files, desc="   Reading GloFAS GRIB files"):
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            ds = xr.open_dataset(
                grib_path, engine="cfgrib",
                backend_kwargs={"indexpath": ""}
            )
        var_candidates = [v for v in ds.data_vars if "dis" in v.lower()]
        if not var_candidates:
            var_candidates = list(ds.data_vars)
        var_name = var_candidates[0]
        lat_name = [c for c in ds.coords if "lat" in c.lower()][0]
        lon_name = [c for c in ds.coords if "lon" in c.lower()][0]
        ts = ds[var_name].sel(
            {lat_name: lat_out, lon_name: lon_out}, method="nearest"
        )
        df_year = ts.to_dataframe(name="discharge_m3s").reset_index()
        df_year = df_year[["time", "discharge_m3s"]].copy()
        discharge_frames.append(df_year)
        ds.close()
    except Exception as e:
        print(f"   WARNING: Could not read {Path(grib_path).name} — {e}")
        continue

discharge = pd.concat(discharge_frames, ignore_index=True)
discharge["date"] = pd.to_datetime(discharge["time"]).dt.normalize()
discharge = discharge.groupby("date")["discharge_m3s"].mean().reset_index()
discharge = clip_to_daterange(discharge)
discharge = discharge.drop_duplicates(subset="date", keep="first")
discharge = discharge.sort_values("date").reset_index(drop=True)

neg_q = discharge["discharge_m3s"] < 0
if neg_q.sum() > 0:
    print(f"   WARNING: {neg_q.sum()} negative discharge values → set to NaN")
    discharge.loc[neg_q, "discharge_m3s"] = np.nan

discharge.to_csv(PROCESSED / "discharge_glofas_daily.csv", index=False)
print(f"   Q range: {discharge.discharge_m3s.min():.3f} → "
      f"{discharge.discharge_m3s.max():.3f} m³/s")

# =============================================================================
# SNOW COVER — MODIS MOD10A1 (daily, %)
# =============================================================================
print("\n[6/6] Processing MODIS Snow Cover GeoTIFFs ...")
(PROCESSED / "snow_cover_modis_daily.csv").unlink(missing_ok=True)

tif_files = sorted(glob.glob(str(MODIS_DIR / "**" / "*.tif"), recursive=True))
snow_records = []

for tif_path in tqdm(tif_files, desc="   Clipping GeoTIFFs"):
    fname = Path(tif_path).stem
    date_match = re.search(r"(\d{8})T", fname)
    if not date_match:
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", fname)
        if not date_match:
            continue
        date_str = date_match.group(1)
    else:
        raw_date = date_match.group(1)
        date_str = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"

    try:
        with rasterio.open(tif_path) as src:
            ws = watershed.to_crs(src.crs)
            geom_reproj = [mapping(g) for g in ws.geometry]
            out_image, _ = rio_mask(src, geom_reproj, crop=True, nodata=src.nodata)
            data = out_image[0].astype(float)
            valid_mask = (data >= 0) & (data <= 100)
            if valid_mask.sum() == 0:
                snow_pct = np.nan
            else:
                snow_pixels = (data[valid_mask] >= 40).sum()
                total_valid = valid_mask.sum()
                snow_pct = (snow_pixels / total_valid) * 100.0
        snow_records.append({"date": date_str, "snow_cover_pct": snow_pct})
    except Exception:
        snow_records.append({"date": date_str, "snow_cover_pct": np.nan})

snow_df = pd.DataFrame(snow_records)
snow_df["date"] = pd.to_datetime(snow_df["date"])
snow_df = snow_df.sort_values("date").reset_index(drop=True)

n_before = len(snow_df)
snow_df = snow_df.sort_values(
    ["date", "snow_cover_pct"], ascending=[True, False]
)
snow_df = snow_df.drop_duplicates(subset="date", keep="first")
snow_df = snow_df.sort_values("date").reset_index(drop=True)
print(f"   Duplicates removed: {n_before - len(snow_df)}")

snow_df = clip_to_daterange(snow_df)
snow_df.to_csv(PROCESSED / "snow_cover_modis_daily.csv", index=False)
print(f"   Saved → snow_cover_modis_daily.csv  ({len(snow_df)} rows)")
print(f"   Snow cover range: {snow_df.snow_cover_pct.min():.1f}% → "
      f"{snow_df.snow_cover_pct.max():.1f}%")
print(f"   NaN days (cloudy/missing): {snow_df.snow_cover_pct.isna().sum()}")

# =============================================================================
# MASTER DATASET CONSTRUCTION
# =============================================================================
print("\n[7/7] Building Master Dataset ...")

(MASTER_DIR / "nahr_ibrahim_master_full.csv").unlink(missing_ok=True)
(MASTER_DIR / "nahr_ibrahim_master_model.csv").unlink(missing_ok=True)

full_index = pd.DataFrame({
    "date": pd.date_range(start=DATE_START, end=DATE_END, freq="D")
})

# Load all processed files
precip_df    = pd.read_csv(PROCESSED / "precip_imerg_daily.csv", parse_dates=["date"])
temp_df      = pd.read_csv(PROCESSED / "temp_merra2_daily.csv", parse_dates=["date"])
swe_df       = pd.read_csv(PROCESSED / "swe_gldas_daily.csv", parse_dates=["date"])
sm_df        = pd.read_csv(PROCESSED / "soil_moisture_gldas_daily.csv", parse_dates=["date"])
discharge_df = pd.read_csv(PROCESSED / "discharge_glofas_daily.csv", parse_dates=["date"])
snow_df      = pd.read_csv(PROCESSED / "snow_cover_modis_daily.csv",parse_dates=["date"])

# Merge all on date
master = (full_index
          .merge(precip_df,    on="date", how="left")
          .merge(temp_df,      on="date", how="left")
          .merge(swe_df,       on="date", how="left")
          .merge(sm_df,        on="date", how="left")   
          .merge(discharge_df, on="date", how="left")
          .merge(snow_df,      on="date", how="left"))

# =============================================================================
# GAP FILLING
# =============================================================================
print("   Gap filling ...")

# Snow cover: interpolate up to 5 consecutive days, then monthly climatology
master["snow_cover_pct"] = master["snow_cover_pct"].interpolate(
    method="linear", limit=5
)
master["month"] = master["date"].dt.month
monthly_snow = master.groupby("month")["snow_cover_pct"].transform("mean")
master["snow_cover_pct"] = master["snow_cover_pct"].fillna(monthly_snow)

# SWE: interpolate short gaps up to 3 days
master["swe_mm"] = master["swe_mm"].interpolate(method="linear", limit=3)

# Soil moisture: interpolate up to 5 days, then 30-day rolling mean
master["soil_moisture_mm"] = master["soil_moisture_mm"].interpolate(
    method="linear", limit=5
)
sm_rolling = master["soil_moisture_mm"].rolling(30, min_periods=5).mean()
master["soil_moisture_mm"] = master["soil_moisture_mm"].fillna(sm_rolling)
master["soil_moisture_mm"] = master["soil_moisture_mm"].fillna(master["soil_moisture_mm"].mean())

# Temperature: interpolate short gaps
for col in ["temp_mean_c", "temp_max_c", "temp_min_c"]:
    master[col] = master[col].interpolate(method="linear", limit=3)

# Precipitation: fill missing with 0
master["precip_mm_day"] = master["precip_mm_day"].fillna(0.0)

master.drop(columns=["month"], inplace=True)

# =============================================================================
# DERIVED FEATURES
# =============================================================================
print("   Computing derived features ...")

# Antecedent precipitation
master["precip_3day"] = master["precip_mm_day"].rolling(3, min_periods=1).sum()
master["precip_7day"] = master["precip_mm_day"].rolling(7, min_periods=1).sum()

# Temperature range
master["temp_range_c"] = master["temp_max_c"] - master["temp_min_c"]

# Snowmelt proxy
master["swe_delta"] = master["swe_mm"].diff().clip(upper=0).abs()

# ── Soil moisture derived features ────────────────────────────────────────
# 7-day antecedent mean — captures pre-event wetness state
master["sm_7day_mean"] = master["soil_moisture_mm"].rolling(
    7, min_periods=1
).mean()

# Soil moisture anomaly — deviation from 30-day climatological mean
# Positive = wetter than usual → higher runoff propensity
# Negative = drier than usual  → higher infiltration, lower runoff
master["sm_anomaly"] = (
    master["soil_moisture_mm"] -
    master["soil_moisture_mm"].rolling(30, min_periods=7).mean()
).fillna(0.0)

# PET — Hamon (1961) temperature-based method
# Already used in physics-informed loss; now also an explicit model feature
doy = master["date"].dt.dayofyear
master["daylight_hrs"] = 12 + 4 * np.sin(2 * np.pi * (doy - 80) / 365)
master["pet_mm_day"]   = (
    0.1651 * master["daylight_hrs"] *
    (216.7 * 0.6108 *
     np.exp(17.27 * master["temp_mean_c"] /
            (master["temp_mean_c"] + 237.3)) /
     (master["temp_mean_c"] + 273.3))
).clip(lower=0)
master.drop(columns=["daylight_hrs"], inplace=True)

# ── Time features ──────────────────────────────────────────────────────────
master["month"]  = master["date"].dt.month
master["season"] = master["month"].map({
    12: "winter",  1: "winter",  2: "winter",
     3: "spring",  4: "spring",  5: "spring",
     6: "summer",  7: "summer",  8: "summer",
     9: "autumn", 10: "autumn", 11: "autumn",
})
master["month_sin"] = np.sin(2 * np.pi * master["month"] / 12)
master["month_cos"] = np.cos(2 * np.pi * master["month"] / 12)

# =============================================================================
# FINAL COLUMN ORDER
# =============================================================================
col_order = [
    "date",
    # Meteorological inputs (original 12)
    "precip_mm_day", "precip_3day", "precip_7day",
    "temp_mean_c", "temp_max_c", "temp_min_c", "temp_range_c",
    "swe_mm", "swe_delta", "snow_cover_pct",
    "month_sin", "month_cos",
    # New soil moisture features (+3)
    "soil_moisture_mm", "sm_7day_mean", "sm_anomaly",
    # New PET feature (+1)
    "pet_mm_day",
    # Target variable
    "discharge_m3s",
    # Metadata
    "month", "season",
]

master = master[col_order]

master_model = master.dropna(
    subset=["discharge_m3s"]
).reset_index(drop=True)

master.to_csv(MASTER_DIR / "nahr_ibrahim_master_full.csv",  index=False)
master_model.to_csv(MASTER_DIR / "nahr_ibrahim_master_model.csv", index=False)

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("  MASTER DATASET SUMMARY")
print("=" * 70)
print(f"  Full date range    : {master.date.min().date()} → "
      f"{master.date.max().date()}")
print(f"  Total days (full)  : {len(master)}")
print(f"  Total days (model) : {len(master_model)}")
print(f"  Total features     : 16 (was 12 — +3 soil moisture, +1 PET)")

print(f"\n  Missing values (full dataset):")
check_cols = [
    "precip_mm_day", "precip_3day", "precip_7day",
    "temp_mean_c", "temp_max_c", "temp_min_c", "temp_range_c",
    "swe_mm", "swe_delta", "snow_cover_pct",
    "month_sin", "month_cos",
    "soil_moisture_mm", "sm_7day_mean", "sm_anomaly",
    "pet_mm_day", "discharge_m3s",
]
for col in check_cols:
    n_miss = master[col].isna().sum()
    pct    = n_miss / len(master) * 100
    status = "✓" if pct == 0 else "⚠" if pct < 5 else "✗"
    print(f"    {status} {col:<24}: {n_miss:>5} missing  ({pct:5.1f}%)")

print(f"\n  New features added:")
print(f"    soil_moisture_mm  : GLDAS Noah 0–10 cm SWE (daily mean)")
print(f"    sm_7day_mean      : 7-day antecedent soil moisture")
print(f"    sm_anomaly        : Deviation from 30-day rolling mean")
print(f"    pet_mm_day        : Hamon (1961) PET from temperature")

print(f"\n  Files saved:")
print(f"    data/master/nahr_ibrahim_master_full.csv")
print(f"    data/master/nahr_ibrahim_master_model.csv")
print("=" * 70)