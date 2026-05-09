import re
import io
import glob
import warnings
import logging
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

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path("C:/Users/marck/Downloads/nahr_ibrahim_watershed")
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MASTER = ROOT / "data" / "master"

ERA5_FILE = RAW / "era5" / "era5_precip_2000_2025_daily.csv"
GIOVANNI = RAW / "giovanni"
GLOFAS = RAW / "glofas"
MODIS = RAW / "modis"
GEOJSON = RAW / "shapefiles" / "nahr_ibrahim_watershed.geojson"

PROCESSED.mkdir(parents=True, exist_ok=True)
MASTER.mkdir(parents=True, exist_ok=True)

START = "2000-01-01"
END = "2025-12-31"

print("Nahr Ibrahim — preprocessing pipeline")
print(f"Period: {START} to {END}\n")


# ── Helper: clip a dataframe to the study period ───────────────────────────────
def to_study_period(df, col="date"):
    df[col] = pd.to_datetime(df[col])
    return df[(df[col] >= START) & (df[col] <= END)].reset_index(drop=True)


# ── Helper: parse NASA Giovanni area-averaged CSVs ─────────────────────────────
# Giovanni exports have a variable number of header lines before the data starts.
# We scan until we find a line beginning with "time," and read from there.
def read_giovanni(path):
    with open(path) as f:
        lines = f.readlines()
    start = next(
        i for i, l in enumerate(lines) if l.strip().lower().startswith("time,")
    )
    df = pd.read_csv(path, skiprows=start)
    df.columns = [c.strip() for c in df.columns]
    df = df.iloc[:, :2].copy()
    df.columns = ["time", "value"]
    df["time"] = pd.to_datetime(df["time"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["value"] = df["value"].replace([-9999, -9999.9], np.nan)
    return df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)


# ── Helper: resample sub-daily Giovanni series to daily ───────────────────────
def daily(df, method="mean"):
    df = df.set_index("time")
    resampled = (
        df["value"].resample("D").sum()
        if method == "sum"
        else df["value"].resample("D").mean()
    )
    return resampled.reset_index().rename(columns={"time": "date", "value": "value"})


# ── Helper: flag extreme outliers using IQR ────────────────────────────────────
def iqr_outliers(s, k=3.0):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


# ==============================================================================
# 1. Precipitation — ERA5-Land total_precipitation (daily sum, mm/day)
#    Downloaded via GEE: ECMWF/ERA5_LAND/DAILY_AGGR, spatially averaged
#    over the spring recharge bounding box (34.02–34.16N, 35.84–35.96E)
# ==============================================================================
print("[1/6] Precipitation (ERA5-Land) ...")
(PROCESSED / "precip_era5_daily.csv").unlink(missing_ok=True)

precip = pd.read_csv(ERA5_FILE, parse_dates=["date"])
precip = to_study_period(precip)

# Sanity checks
neg = precip["precip_mm_day"] < 0
if neg.any():
    print(f"  {neg.sum()} negative values clipped to 0")
    precip.loc[neg, "precip_mm_day"] = 0.0

n_outliers = iqr_outliers(precip["precip_mm_day"]).sum()
print(f"  IQR outlier check: {n_outliers} flagged (retained — extreme rain events)")

precip.to_csv(PROCESSED / "precip_era5_daily.csv", index=False)
print(
    f"  {len(precip)} days | mean {precip.precip_mm_day.mean():.2f} mm/day "
    f"| max {precip.precip_mm_day.max():.1f} mm/day\n"
)


# ==============================================================================
# 2. Temperature — MERRA-2 T2MMEAN/T2MMAX/T2MMIN (hourly → daily)
#    Area-averaged over watershed bounding box via NASA Giovanni.
#    Single Giovanni file provides all three statistics as hourly values;
#    we resample to daily mean/max/min here.
# ==============================================================================
print("[2/6] Temperature (MERRA-2) ...")
(PROCESSED / "temp_merra2_daily.csv").unlink(missing_ok=True)

temp_file = (list(GIOVANNI.glob("*MERRA*T2M*.csv")) + list(GIOVANNI.glob("*T2M*.csv")))[
    0
]
t = read_giovanni(temp_file).set_index("time")
print(f"  Raw rows: {len(t)} | {t.index.min()} → {t.index.max()}")

temp = pd.DataFrame(
    {
        "date": t["value"].resample("D").mean().index,
        "temp_mean_c": t["value"].resample("D").mean().values,
        "temp_max_c": t["value"].resample("D").max().values,
        "temp_min_c": t["value"].resample("D").min().values,
    }
)
temp = to_study_period(temp)

out_of_range = ((temp.temp_mean_c < -20) | (temp.temp_mean_c > 45)).sum()
if out_of_range:
    print(f"  WARNING: {out_of_range} Tmean values outside physical range")

temp.to_csv(PROCESSED / "temp_merra2_daily.csv", index=False)
print(
    f"  {len(temp)} days | Tmean {temp.temp_mean_c.min():.1f}°C → "
    f"{temp.temp_mean_c.max():.1f}°C\n"
)


# ==============================================================================
# 3. Snow water equivalent — GLDAS Noah v2.1 SWE_inst (3-hourly → daily)
#    Area-averaged via NASA Giovanni. Negative values are physical artefacts
#    from the land surface model and are set to zero.
# ==============================================================================
print("[3/6] SWE (GLDAS Noah) ...")
(PROCESSED / "swe_gldas_daily.csv").unlink(missing_ok=True)

swe_file = (list(GIOVANNI.glob("*SWE*.csv")) + list(GIOVANNI.glob("*snow_water*.csv")))[
    0
]
swe_raw = read_giovanni(swe_file)
swe = daily(swe_raw, method="mean")
swe = to_study_period(swe)

neg_swe = swe["value"] < 0
if neg_swe.any():
    print(f"  {neg_swe.sum()} negative SWE values → set to 0")
    swe.loc[neg_swe, "value"] = 0.0

swe.rename(columns={"value": "swe_mm"}, inplace=True)
swe.to_csv(PROCESSED / "swe_gldas_daily.csv", index=False)
print(f"  {len(swe)} days | range {swe.swe_mm.min():.1f}–{swe.swe_mm.max():.1f} mm\n")


# ==============================================================================
# 4. Soil moisture — GLDAS Noah v2.1 SoilMoi0_10cm (3-hourly → daily)
#    0–10 cm layer, area-averaged via NASA Giovanni.
#    If the file is missing (not yet downloaded), we fill with NaN and rely
#    on gap-filling later — the pipeline will still run.
# ==============================================================================
print("[4/6] Soil moisture (GLDAS Noah 0–10 cm) ...")
(PROCESSED / "soil_moisture_gldas_daily.csv").unlink(missing_ok=True)

sm_files = list(GIOVANNI.glob("*SoilMoi0_10cm*.csv")) + list(
    GIOVANNI.glob("*SoilMoi*.csv")
)

if not sm_files:
    print("  WARNING: no soil moisture file found — filling with NaN")
    sm = pd.DataFrame(
        {
            "date": pd.date_range(START, END, freq="D"),
            "soil_moisture_mm": np.nan,
        }
    )
else:
    sm_raw = read_giovanni(sm_files[0])
    sm = daily(sm_raw, method="mean")
    sm = to_study_period(sm)
    sm["value"] = sm["value"].replace([-9999, -9999.9], np.nan)
    sm.loc[sm["value"] < 0, "value"] = 0.0
    sm.rename(columns={"value": "soil_moisture_mm"}, inplace=True)
    print(f"  Source: {sm_files[0].name}")

print(
    f"  Range: {sm.soil_moisture_mm.min():.2f}–{sm.soil_moisture_mm.max():.2f} mm "
    f"| missing: {sm.soil_moisture_mm.isna().sum()} days"
)
sm.to_csv(PROCESSED / "soil_moisture_gldas_daily.csv", index=False)
print(f"  {len(sm)} days saved\n")


# ==============================================================================
# 5. River discharge — GloFAS ERA5 v4.0 (daily, m³/s)
#    Downloaded from Copernicus CDS as monthly GRIB files and extracted at
#    the watershed centroid using nearest-neighbour interpolation.
#    Monthly ZIPs were unzipped into data/raw/glofas/unzipped/.
#    The 2013 and 2014 full-year files are named glofas_2013.grib /
#    glofas_2014.grib to avoid data.grib naming collisions.
# ==============================================================================
print("[5/6] River discharge (GloFAS ERA5 v4.0) ...")
(PROCESSED / "discharge_glofas_daily.csv").unlink(missing_ok=True)

ws = gpd.read_file(GEOJSON)
ws = ws.to_crs(epsg=4326) if ws.crs.to_epsg() != 4326 else ws
centroid = ws.geometry.centroid.iloc[0]
lat_out, lon_out = centroid.y, centroid.x
print(f"  Extraction point (centroid): {lat_out:.4f}°N, {lon_out:.4f}°E")

unzipped = GLOFAS / "unzipped"
gribs = sorted(glob.glob(str(unzipped / "**" / "*.grib"), recursive=True)) + sorted(
    glob.glob(str(unzipped / "**" / "*.grib2"), recursive=True)
)
print(f"  GRIB files found: {len(gribs)}")

frames = []
for g in tqdm(gribs, desc="  Reading GRIB"):
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            ds = xr.open_dataset(g, engine="cfgrib", backend_kwargs={"indexpath": ""})
        var = next(
            (v for v in ds.data_vars if "dis" in v.lower()), list(ds.data_vars)[0]
        )
        latn = next(c for c in ds.coords if "lat" in c.lower())
        lonn = next(c for c in ds.coords if "lon" in c.lower())
        ts = ds[var].sel({latn: lat_out, lonn: lon_out}, method="nearest")
        df = ts.to_dataframe(name="discharge_m3s").reset_index()
        frames.append(df[["time", "discharge_m3s"]])
        ds.close()
    except Exception as e:
        print(f"  skip {Path(g).name}: {e}")

q = pd.concat(frames, ignore_index=True)
q["date"] = pd.to_datetime(q["time"]).dt.normalize()
q = q.groupby("date")["discharge_m3s"].mean().reset_index()
q = to_study_period(q)
q = q.drop_duplicates("date").sort_values("date").reset_index(drop=True)
q.loc[q.discharge_m3s < 0, "discharge_m3s"] = np.nan

q.to_csv(PROCESSED / "discharge_glofas_daily.csv", index=False)
print(
    f"  {len(q)} days | Q range {q.discharge_m3s.min():.3f}–"
    f"{q.discharge_m3s.max():.3f} m³/s\n"
)


# ==============================================================================
# 6. Snow cover — MODIS MOD10A1.061 (daily, %)
#    Downloaded via NASA AppEEARS as GeoTIFFs, chunked by 5-year periods.
#    Each tile is clipped to the watershed polygon and the fraction of pixels
#    with NDSI >= 40 is used as the snow cover estimate (standard threshold).
#    Duplicate dates (from overlapping chunks) keep the highest value —
#    this selects the least-cloudy observation for that day.
# ==============================================================================
print("[6/6] Snow cover (MODIS MOD10A1.061) ...")
(PROCESSED / "snow_cover_modis_daily.csv").unlink(missing_ok=True)

tifs = sorted(glob.glob(str(MODIS / "**" / "*.tif"), recursive=True))
records = []

for tif in tqdm(tifs, desc="  Clipping"):
    stem = Path(tif).stem
    m = re.search(r"(\d{8})T", stem) or re.search(r"(\d{4}-\d{2}-\d{2})", stem)
    if not m:
        continue
    raw = m.group(1)
    date = raw if "-" in raw else f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"

    try:
        with rasterio.open(tif) as src:
            geom = [mapping(g) for g in ws.to_crs(src.crs).geometry]
            arr, _ = rio_mask(src, geom, crop=True, nodata=src.nodata)
            data = arr[0].astype(float)
            valid = (data >= 0) & (data <= 100)
            pct = (
                (data[valid] >= 40).sum() / valid.sum() * 100 if valid.any() else np.nan
            )
    except Exception:
        pct = np.nan

    records.append({"date": date, "snow_cover_pct": pct})

snow = pd.DataFrame(records)
snow["date"] = pd.to_datetime(snow["date"])
snow = (
    snow.sort_values(["date", "snow_cover_pct"], ascending=[True, False])
    .drop_duplicates("date", keep="first")
    .sort_values("date")
    .reset_index(drop=True)
)
snow = to_study_period(snow)

snow.to_csv(PROCESSED / "snow_cover_modis_daily.csv", index=False)
print(
    f"  {len(snow)} unique days | range {snow.snow_cover_pct.min():.1f}–"
    f"{snow.snow_cover_pct.max():.1f}% | "
    f"{snow.snow_cover_pct.isna().sum()} cloudy/missing\n"
)


# ==============================================================================
# 7. Build master dataset
#    All variables are merged onto a complete daily index for 2000–2025.
#    Gap-filling strategy:
#      - Snow cover  : linear interpolation ≤5 days, then monthly climatology
#      - SWE         : linear interpolation ≤3 days
#      - Soil moisture: linear interpolation ≤5 days, then 30-day rolling mean
#      - Temperature : linear interpolation ≤3 days
#      - Precipitation: missing → 0 (ERA5-Land has very few gaps)
# ==============================================================================
print("[7/7] Building master dataset ...")
(MASTER / "nahr_ibrahim_master_full.csv").unlink(missing_ok=True)
(MASTER / "nahr_ibrahim_master_model.csv").unlink(missing_ok=True)

idx = pd.DataFrame({"date": pd.date_range(START, END, freq="D")})

p = pd.read_csv(PROCESSED / "precip_era5_daily.csv", parse_dates=["date"])
t = pd.read_csv(PROCESSED / "temp_merra2_daily.csv", parse_dates=["date"])
s = pd.read_csv(PROCESSED / "swe_gldas_daily.csv", parse_dates=["date"])
sm = pd.read_csv(PROCESSED / "soil_moisture_gldas_daily.csv", parse_dates=["date"])
q = pd.read_csv(PROCESSED / "discharge_glofas_daily.csv", parse_dates=["date"])
sn = pd.read_csv(PROCESSED / "snow_cover_modis_daily.csv", parse_dates=["date"])

df = (
    idx.merge(p, on="date", how="left")
    .merge(t, on="date", how="left")
    .merge(s, on="date", how="left")
    .merge(sm, on="date", how="left")
    .merge(q, on="date", how="left")
    .merge(sn, on="date", how="left")
)

# ── Gap filling ────────────────────────────────────────────────────────────────
df["month"] = df["date"].dt.month

df["snow_cover_pct"] = df["snow_cover_pct"].interpolate(method="linear", limit=5)
monthly_snow = df.groupby("month")["snow_cover_pct"].transform("mean")
df["snow_cover_pct"] = df["snow_cover_pct"].fillna(monthly_snow)

df["swe_mm"] = df["swe_mm"].interpolate(method="linear", limit=3)

df["soil_moisture_mm"] = df["soil_moisture_mm"].interpolate(method="linear", limit=5)
df["soil_moisture_mm"] = df["soil_moisture_mm"].fillna(
    df["soil_moisture_mm"].rolling(30, min_periods=5).mean()
)
df["soil_moisture_mm"] = df["soil_moisture_mm"].fillna(df["soil_moisture_mm"].mean())

for col in ["temp_mean_c", "temp_max_c", "temp_min_c"]:
    df[col] = df[col].interpolate(method="linear", limit=3)

df["precip_mm_day"] = df["precip_mm_day"].fillna(0.0)
df.drop(columns=["month"], inplace=True)

# ── Derived features ───────────────────────────────────────────────────────────
# Antecedent precipitation indices
df["precip_3day"] = df["precip_mm_day"].rolling(3, min_periods=1).sum()
df["precip_7day"] = df["precip_mm_day"].rolling(7, min_periods=1).sum()

# Diurnal temperature range
df["temp_range_c"] = df["temp_max_c"] - df["temp_min_c"]

# Snowmelt proxy — daily SWE decrease (positive when snow melts)
# fillna(0) handles the first row where diff() is undefined
df["swe_delta"] = df["swe_mm"].diff().clip(upper=0).abs().fillna(0.0)

# Soil moisture indices
df["sm_7day_mean"] = df["soil_moisture_mm"].rolling(7, min_periods=1).mean()
df["sm_anomaly"] = (
    df["soil_moisture_mm"] - df["soil_moisture_mm"].rolling(30, min_periods=7).mean()
).fillna(0.0)

# Potential evapotranspiration — Hamon (1961), calibrated for ~34°N
doy = df["date"].dt.dayofyear
daylight = 12 + 4 * np.sin(2 * np.pi * (doy - 80) / 365)
sat_vp = (
    216.7
    * 0.6108
    * np.exp(17.27 * df["temp_mean_c"] / (df["temp_mean_c"] + 237.3))
    / (df["temp_mean_c"] + 273.3)
)
df["pet_mm_day"] = (0.1651 * daylight * sat_vp).clip(lower=0)

# ── Drought indices — SPI and SPEI ────────────────────────────────────────────
# SPI-3  : Standardised Precipitation Index over 3-month rolling window
#          Positive = wetter than normal, Negative = drier than normal
#          Values < -1 indicate moderate drought, < -2 severe drought
# SPEI-3 : Same but accounts for atmospheric water demand via PET
#          More sensitive to warming-driven drought than SPI alone
from scipy.stats import norm, gamma as gamma_dist


def compute_spi(precip_series, scale=3):
    rolling = precip_series.rolling(scale, min_periods=scale).sum()
    result = np.full(len(rolling), np.nan)
    valid = rolling.notna() & (rolling > 0)
    if valid.sum() < 10:
        return pd.Series(result, index=precip_series.index)
    try:
        params = gamma_dist.fit(rolling[valid].values, floc=0)
        prob = gamma_dist.cdf(rolling[valid].values, *params)
        prob = np.clip(prob, 0.001, 0.999)
        result[valid.values] = norm.ppf(prob)
    except Exception:
        pass
    return pd.Series(result, index=precip_series.index)


def compute_spei(precip_series, pet_series, scale=3):
    # Water balance deficit/surplus — core of SPEI
    wb = precip_series - pet_series
    rolling = wb.rolling(scale, min_periods=scale).sum()
    result = np.full(len(rolling), np.nan)
    valid = rolling.notna()
    if valid.sum() < 10:
        return pd.Series(result, index=precip_series.index)
    try:
        # Shift to positive range for gamma fitting
        shifted = rolling[valid] - rolling[valid].min() + 0.001
        params = gamma_dist.fit(shifted.values, floc=0)
        prob = gamma_dist.cdf(shifted.values, *params)
        prob = np.clip(prob, 0.001, 0.999)
        result[valid.values] = norm.ppf(prob)
    except Exception:
        pass
    return pd.Series(result, index=precip_series.index)


df["spi_3month"] = compute_spi(df["precip_mm_day"], scale=3)
df["spei_3month"] = compute_spei(df["precip_mm_day"], df["pet_mm_day"], scale=3)

# Fill the first 2 rows (insufficient rolling window) with 0
df["spi_3month"] = df["spi_3month"].fillna(0.0)
df["spei_3month"] = df["spei_3month"].fillna(0.0)

# ── Flood exceedance index ─────────────────────────────────────────────────────
# Binary flag: 1 when discharge exceeds the 90th percentile of the training
# period (2000–2017). Captures high-flow / flood conditions explicitly.
# We use the full series percentile as a proxy since training split
# is not yet defined here — will be recalculated in split.py if needed.
q90 = df["discharge_m3s"].quantile(0.90)
df["flood_index"] = (df["discharge_m3s"] > q90).astype(float)
df["flood_index"] = df["flood_index"].fillna(0.0)

print(f"  Drought/flood indices computed:")
print(
    f"    SPI-3  : mean={df['spi_3month'].mean():.3f}  "
    f"min={df['spi_3month'].min():.2f}  max={df['spi_3month'].max():.2f}"
)
print(
    f"    SPEI-3 : mean={df['spei_3month'].mean():.3f}  "
    f"min={df['spei_3month'].min():.2f}  max={df['spei_3month'].max():.2f}"
)
print(
    f"    Flood days (Q > Q90={q90:.3f} m³/s): "
    f"{df['flood_index'].sum():.0f} days "
    f"({df['flood_index'].mean() * 100:.1f}%)"
)

# Cyclical month encoding — avoids discontinuity between December and January
df["month"] = df["date"].dt.month
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["season"] = df["month"].map(
    {
        12: "winter",
        1: "winter",
        2: "winter",
        3: "spring",
        4: "spring",
        5: "spring",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "autumn",
        10: "autumn",
        11: "autumn",
    }
)

# ── Final column order ─────────────────────────────────────────────────────────
cols = [
    "date",
    "precip_mm_day",
    "precip_3day",
    "precip_7day",
    "temp_mean_c",
    "temp_max_c",
    "temp_min_c",
    "temp_range_c",
    "swe_mm",
    "swe_delta",
    "snow_cover_pct",
    "month_sin",
    "month_cos",
    "soil_moisture_mm",
    "sm_7day_mean",
    "sm_anomaly",
    "pet_mm_day",
    # drought and flood indices  ← new
    "spi_3month",
    "spei_3month",
    "flood_index",
    "discharge_m3s",
    "month",
    "season",
]
df = df[cols]

df_model = df.dropna(subset=["discharge_m3s"]).reset_index(drop=True)

df.to_csv(MASTER / "nahr_ibrahim_master_full.csv", index=False)
df_model.to_csv(MASTER / "nahr_ibrahim_master_model.csv", index=False)

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"  Period  : {df.date.min().date()} → {df.date.max().date()}")
print(f"  Full    : {len(df):,} days  |  Model: {len(df_model):,} days")
print(f"  Features: 19")
print(f"\n  Missing values:")
check = [
    "precip_mm_day",
    "precip_3day",
    "precip_7day",
    "temp_mean_c",
    "temp_max_c",
    "temp_min_c",
    "temp_range_c",
    "swe_mm",
    "swe_delta",
    "snow_cover_pct",
    "month_sin",
    "month_cos",
    "soil_moisture_mm",
    "sm_7day_mean",
    "sm_anomaly",
    "pet_mm_day",
    "spi_3month",
    "spei_3month",
    "flood_index",
    "discharge_m3s",
]
for col in check:
    n = df[col].isna().sum()
    pct = n / len(df) * 100
    tag = "ok" if n == 0 else "warn"
    print(f"  [{tag}] {col:<24}: {n:>4} ({pct:.1f}%)")
print(f"\n  Saved → data/master/nahr_ibrahim_master_full.csv")
print(f"  Saved → data/master/nahr_ibrahim_master_model.csv")
print(f"{'=' * 60}")
