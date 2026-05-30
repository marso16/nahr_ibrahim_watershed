"""
Strategy A preprocessing: full ERA5-Land replacement.

Inputs (already on disk):
  - data/raw/chirps/chirps_nahr_ibrahim_2000_2025_daily.csv
  - data/raw/era5_land/era5land_other_daily.csv       (used for T, SM, SWE, PET)
  - data/processed/discharge_glofas_daily.csv         (existing)
  - data/processed/snow_cover_modis_daily.csv         (existing, starts 2006)

Outputs:
  - data/master/nahr_ibrahim_master_full.csv
  - data/master/nahr_ibrahim_master_model.csv
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm, gamma as gamma_dist

ROOT = Path(
    os.environ.get("WATERSHED_ROOT", "C:/Users/marck/Downloads/nahr_ibrahim_watershed")
)
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MASTER = ROOT / "data" / "master"
MASTER.mkdir(parents=True, exist_ok=True)

START = "2000-01-01"
END = "2025-12-31"
TRAIN_END_FOR_FIT = pd.Timestamp("2017-12-31")  # KEEP IN SYNC with split.py

print("Strategy A preprocessing — ERA5-Land replacement\n")

# ── Load all sources ─────────────────────────────────────────────────────────
idx = pd.DataFrame({"date": pd.date_range(START, END, freq="D")})

# Precipitation from CHIRPS (kept)
chirps = pd.read_csv(
    RAW / "chirps" / "chirps_nahr_ibrahim_2000_2025_daily.csv",
    parse_dates=["date"],
)

# ERA5-Land variables (new). Select only the columns we want — defensive
# against any extra columns (e.g. the broken snow_cover_pct we removed).
era5_all = pd.read_csv(
    RAW / "era5_land" / "era5land_other_daily.csv",
    parse_dates=["date"],
)
era5_cols = [
    "date",
    "temp_mean_c",
    "temp_max_c",
    "temp_min_c",
    "sm_0_7cm_mm",
    "sm_7_28cm_mm",
    "sm_28_100cm_mm",
    "sm_100_289cm_mm",
    "swe_mm",
    "pet_mm_day",
]
missing_era5 = [c for c in era5_cols if c not in era5_all.columns]
if missing_era5:
    raise ValueError(
        f"ERA5 CSV is missing expected columns: {missing_era5}\n"
        f"Available columns: {era5_all.columns.tolist()}"
    )
era5 = era5_all[era5_cols].copy()

# Discharge target (existing GloFAS)
q = pd.read_csv(PROCESSED / "discharge_glofas_daily.csv", parse_dates=["date"])

# Snow cover from MODIS (kept). Starts 2006 — earlier dates will be NaN
# and filled with train-only monthly means below.
sn = pd.read_csv(
    PROCESSED / "snow_cover_modis_daily.csv",
    parse_dates=["date"],
)

print(
    f"  CHIRPS:   {len(chirps):>6} rows  ({chirps.date.min().date()} → {chirps.date.max().date()})"
)
print(
    f"  ERA5:     {len(era5):>6} rows  ({era5.date.min().date()} → {era5.date.max().date()})"
)
print(f"  GloFAS:   {len(q):>6} rows  ({q.date.min().date()} → {q.date.max().date()})")
print(
    f"  MODIS:    {len(sn):>6} rows  ({sn.date.min().date()} → {sn.date.max().date()})\n"
)

# ── Merge into single dataframe ──────────────────────────────────────────────
df = (
    idx.merge(chirps[["date", "precip_mm_day"]], on="date", how="left")
    .merge(era5, on="date", how="left")
    .merge(q[["date", "discharge_m3s"]], on="date", how="left")
    .merge(sn[["date", "snow_cover_pct"]], on="date", how="left")
)

print(f"[debug] Merged df has {len(df.columns)} columns, {len(df)} rows")
print(f"[debug] snow_cover_pct non-null: {df['snow_cover_pct'].notna().sum()}")
print(f"[debug] temp_mean_c non-null:    {df['temp_mean_c'].notna().sum()}\n")

# ── Gap filling (train-only stats, no leakage) ───────────────────────────────
df["month"] = df["date"].dt.month
train_only = df[df["date"] <= TRAIN_END_FOR_FIT]

# Precipitation — fill with 0 (CHIRPS rarely has gaps)
df["precip_mm_day"] = df["precip_mm_day"].fillna(0.0)

# Temperature — interpolate short gaps, then train-only monthly mean
for col in ["temp_mean_c", "temp_max_c", "temp_min_c"]:
    df[col] = df[col].interpolate(method="linear", limit=3)
    monthly_train = train_only.groupby("month")[col].mean()
    df[col] = df[col].fillna(df["month"].map(monthly_train))
    df[col] = df[col].fillna(train_only[col].mean())

# Soil moisture (4 layers) — interpolate, then train-only mean
for col in ["sm_0_7cm_mm", "sm_7_28cm_mm", "sm_28_100cm_mm", "sm_100_289cm_mm"]:
    df[col] = df[col].interpolate(method="linear", limit=5)
    df[col] = df[col].fillna(train_only[col].mean())

# SWE — interpolate short gaps, fill rest with 0 (no snow)
df["swe_mm"] = df["swe_mm"].interpolate(method="linear", limit=3).fillna(0.0)
df["swe_mm"] = df["swe_mm"].clip(lower=0)

# PET — interpolate, fall back to train mean
df["pet_mm_day"] = df["pet_mm_day"].interpolate(method="linear", limit=3)
df["pet_mm_day"] = df["pet_mm_day"].fillna(train_only["pet_mm_day"].mean())
df["pet_mm_day"] = df["pet_mm_day"].clip(lower=0)

# Snow cover (MODIS) — interpolate, then train-only monthly mean.
# Note: MODIS data starts 2006, so 2000-2005 will be filled entirely from
# the climatological monthly means computed on training data.
df["snow_cover_pct"] = df["snow_cover_pct"].interpolate(method="linear", limit=5)
sn_monthly_train = train_only.groupby("month")["snow_cover_pct"].mean()
df["snow_cover_pct"] = df["snow_cover_pct"].fillna(df["month"].map(sn_monthly_train))
df["snow_cover_pct"] = df["snow_cover_pct"].fillna(train_only["snow_cover_pct"].mean())

df.drop(columns=["month"], inplace=True)

# ── Derived features ─────────────────────────────────────────────────────────
# Short-window precipitation
df["precip_3day"] = df["precip_mm_day"].rolling(3, min_periods=1).sum()
df["precip_7day"] = df["precip_mm_day"].rolling(7, min_periods=1).sum()

# NEW: Long-window precipitation (Step 4 — for karst storage memory)
df["precip_14day"] = df["precip_mm_day"].rolling(14, min_periods=1).sum()
df["precip_30day"] = df["precip_mm_day"].rolling(30, min_periods=1).sum()
df["precip_60day"] = df["precip_mm_day"].rolling(60, min_periods=1).sum()
df["precip_90day"] = df["precip_mm_day"].rolling(90, min_periods=1).sum()

# Precipitation lags
df["precip_lag1"] = df["precip_mm_day"].shift(1).fillna(0.0)
df["precip_lag2"] = df["precip_mm_day"].shift(2).fillna(0.0)
df["precip_lag3"] = df["precip_mm_day"].shift(3).fillna(0.0)
df["precip_lag5"] = df["precip_mm_day"].shift(5).fillna(0.0)


# Antecedent Precipitation Index at multiple half-lives (Kohler & Linsley 1951)
# k=0.92 → ~8-day half-life  (flashy response, original feature)
# k=0.98 → ~35-day half-life (medium storage)
# k=0.99 → ~70-day half-life (karst memory)
def compute_api(precip_arr, k):
    out = np.zeros(len(precip_arr))
    for i in range(1, len(precip_arr)):
        out[i] = k * out[i - 1] + precip_arr[i]
    return out


p_arr = df["precip_mm_day"].fillna(0.0).values
df["api_15d"] = compute_api(p_arr, 0.92)
df["api_30d"] = compute_api(p_arr, 0.98)  # NEW
df["api_60d"] = compute_api(p_arr, 0.99)  # NEW

# Temperature derived
df["temp_range_c"] = df["temp_max_c"] - df["temp_min_c"]

# Snow derived (melt proxy: absolute value of negative SWE change)
df["swe_delta"] = df["swe_mm"].diff().clip(upper=0).abs().fillna(0.0)

# Soil moisture: use sm_28_100cm as the main variable (mid-depth, balances
# responsiveness with storage memory). Replaces former GLDAS 0-10 cm.
df["soil_moisture_mm"] = df["sm_28_100cm_mm"]
df["sm_7day_mean"] = df["soil_moisture_mm"].rolling(7, min_periods=1).mean()
df["sm_30day_mean"] = df["soil_moisture_mm"].rolling(30, min_periods=1).mean()  # NEW
df["sm_anomaly"] = (
    df["soil_moisture_mm"] - df["soil_moisture_mm"].rolling(30, min_periods=7).mean()
).fillna(0.0)

# NEW: deep soil moisture rolling means (karst storage proxy)
df["sm_deep_30day"] = df["sm_100_289cm_mm"].rolling(30, min_periods=1).mean()
df["sm_deep_anomaly"] = (
    df["sm_100_289cm_mm"] - df["sm_100_289cm_mm"].rolling(90, min_periods=30).mean()
).fillna(0.0)


# ── Drought indices — proper 90-day, train-fit, mixed distribution ───────────
def compute_spi_proper(precip_series, train_mask, scale_days=90):
    rolling = precip_series.rolling(scale_days, min_periods=scale_days).sum()
    train_rolling = rolling[train_mask]
    train_positive = train_rolling[train_rolling > 0].dropna().values
    if len(train_positive) < 30:
        return pd.Series(np.full(len(rolling), np.nan), index=precip_series.index)
    n_train_total = train_rolling.dropna().shape[0]
    p_zero = (n_train_total - len(train_positive)) / max(n_train_total, 1)
    try:
        params = gamma_dist.fit(train_positive, floc=0)
    except Exception:
        return pd.Series(np.full(len(rolling), np.nan), index=precip_series.index)
    result = np.full(len(rolling), np.nan)
    not_nan = rolling.notna()
    vals = rolling[not_nan].values
    cdf = np.where(
        vals > 0,
        p_zero + (1 - p_zero) * gamma_dist.cdf(vals, *params),
        p_zero / 2.0,
    )
    cdf = np.clip(cdf, 0.001, 0.999)
    result[not_nan.values] = norm.ppf(cdf)
    return pd.Series(result, index=precip_series.index)


def compute_spei_proper(precip_series, pet_series, train_mask, scale_days=90):
    wb = (precip_series - pet_series).rolling(scale_days, min_periods=scale_days).sum()
    train_vals = wb[train_mask].dropna().values
    if len(train_vals) < 30:
        return pd.Series(np.full(len(wb), np.nan), index=precip_series.index)
    mu, sigma = train_vals.mean(), train_vals.std()
    if sigma < 1e-8:
        return pd.Series(np.full(len(wb), np.nan), index=precip_series.index)
    result = np.full(len(wb), np.nan)
    not_nan = wb.notna()
    result[not_nan.values] = (wb[not_nan].values - mu) / sigma
    return pd.Series(result, index=precip_series.index)


train_mask = df["date"] <= TRAIN_END_FOR_FIT
print(
    f"SPI/SPEI fitted on training period only "
    f"({train_mask.sum()} days <= {TRAIN_END_FOR_FIT.date()})"
)

df["spi_3month"] = compute_spi_proper(df["precip_mm_day"], train_mask, scale_days=90)
df["spei_3month"] = compute_spei_proper(
    df["precip_mm_day"], df["pet_mm_day"], train_mask, scale_days=90
)
df["spi_3month"] = df["spi_3month"].bfill().fillna(0.0)
df["spei_3month"] = df["spei_3month"].bfill().fillna(0.0)

# ── Cyclical month encoding ──────────────────────────────────────────────────
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

# ── Final column order ───────────────────────────────────────────────────────
cols = [
    "date",
    # Precipitation
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
    # Temperature
    "temp_mean_c",
    "temp_max_c",
    "temp_min_c",
    "temp_range_c",
    # Snow
    "swe_mm",
    "swe_delta",
    "snow_cover_pct",
    # Soil moisture
    "soil_moisture_mm",
    "sm_7day_mean",
    "sm_30day_mean",
    "sm_anomaly",
    "sm_deep_30day",
    "sm_deep_anomaly",
    # Energy / PET
    "pet_mm_day",
    # Drought
    "spi_3month",
    "spei_3month",
    # Cyclical
    "month_sin",
    "month_cos",
    # Target + metadata (not features)
    "discharge_m3s",
    "month",
    "season",
]

feature_cols_check = [
    c for c in cols if c not in ("date", "discharge_m3s", "month", "season")
]

# Check for NaN in features
nan_report = df[feature_cols_check].isna().sum()
bad = nan_report[nan_report > 0]
if not bad.empty:
    raise AssertionError(f"NaN remaining in features:\n{bad}")

df = df[cols]
df_model = df.dropna(subset=["discharge_m3s"]).reset_index(drop=True)

df.to_csv(MASTER / "nahr_ibrahim_master_full.csv", index=False)
df_model.to_csv(MASTER / "nahr_ibrahim_master_model.csv", index=False)

print(f"\n{'=' * 60}")
print(f"  Period   : {df.date.min().date()} → {df.date.max().date()}")
print(f"  Full     : {len(df):,} days  |  Model: {len(df_model):,} days")
print(f"  Features : {len(feature_cols_check)} (excluding date/target/month/season)")
print(f"\n  Saved → data/master/nahr_ibrahim_master_full.csv")
print(f"  Saved → data/master/nahr_ibrahim_master_model.csv")
print(f"{'=' * 60}")

# ── Sanity checks ────────────────────────────────────────────────────────────
print("\nSPI/SPEI sanity check — July 2010 (should be negative for dry period):")
print(
    df[df["date"].between("2010-07-01", "2010-07-15")][
        ["date", "precip_mm_day", "spi_3month", "spei_3month"]
    ].to_string(index=False)
)

print("\nDeep soil moisture seasonal range (sm_100_289cm_mm) — June 2010 vs Feb 2010:")
sm_jun = df[df["date"].between("2010-06-01", "2010-06-30")]["sm_deep_30day"].mean()
sm_feb = df[df["date"].between("2010-02-01", "2010-02-28")]["sm_deep_30day"].mean()
print(f"  Feb 2010 mean: {sm_feb:.1f} mm  (wet season)")
print(f"  Jun 2010 mean: {sm_jun:.1f} mm  (dry season approaching)")
