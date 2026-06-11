import os
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm, gamma as gamma_dist

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(
    os.environ.get("WATERSHED_ROOT", "C:/Users/marck/Downloads/nahr_ibrahim_watershed")
)
BC_DIR = ROOT / "data" / "raw" / "cmip6" / "atmospheric_bc"
MASTER_FILE = ROOT / "data" / "master" / "nahr_ibrahim_master_model.csv"
LS_PARAMS = ROOT / "models" / "trained" / "landsurface_params.json"
OUT_DIR = ROOT / "data" / "master" / "future"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
SCENARIOS = ["ssp245", "ssp585"]

# Training period — for fitting climatologies and SPI/SPEI
TRAIN_START = pd.Timestamp("2000-01-01")
TRAIN_END = pd.Timestamp("2017-12-31")

# Latitude for Hamon PET (Nahr Ibrahim watershed center)
WATERSHED_LAT_DEG = 34.10


# ── Hamon PET ────────────────────────────────────────────────────────────────
def daylight_hours(doy: int, lat_deg: float) -> float:
    """Daylight hours from day-of-year and latitude (radians math)."""
    lat = math.radians(lat_deg)
    # Solar declination
    decl = 0.409 * math.sin(2 * math.pi * doy / 365.0 - 1.39)
    # Sunset hour angle, with safety clip
    cos_ws = -math.tan(lat) * math.tan(decl)
    cos_ws = max(min(cos_ws, 1.0), -1.0)
    ws = math.acos(cos_ws)
    return (24.0 / math.pi) * ws


def hamon_pet(t_mean_c, doy, lat_deg=WATERSHED_LAT_DEG):
    """
    Hamon (1961) reference evapotranspiration in mm/day.

    PET = k × (N/12) × ρ_sat(T)
        where
            N        = daylight hours (depends on latitude and DOY)
            ρ_sat(T) = saturation vapour density (g/m³)
                     ≈ 216.7 × es(T) / (T + 273.3)
            es(T)    = 6.108 × exp(17.27 T / (T + 237.3))      [hPa]
            k        = empirical Hamon coefficient, ~0.1651 in mm/day units

    This formulation produces values around 0-7 mm/day for temperate /
    Mediterranean climates, which matches ERA5-Land PET in this watershed.
    """
    t = np.asarray(t_mean_c, dtype=np.float64)
    doy_arr = np.asarray(doy, dtype=np.int64)
    Ld = np.array([daylight_hours(int(d), lat_deg) for d in doy_arr])

    # Saturation vapor pressure (hPa) — Tetens formula
    es = 6.108 * np.exp(17.27 * t / (t + 237.3))
    # Saturation vapor density (g/m³) from ideal gas approximation
    rho_sat = 216.7 * es / (t + 273.3)
    # Hamon PET (mm/day)
    pet = 0.1651 * (Ld / 12.0) * rho_sat
    return np.maximum(pet, 0.0)


# ── Bucket soil-moisture model (reused from derive_landsurface.py) ──────────
def run_bucket_model(input_water_mm, PET_mm, FC, WP, drainage_rate, ET_scale, sm0):
    """Single-bucket soil moisture model — identical to derive_landsurface.py."""
    n = len(input_water_mm)
    sm = np.zeros(n)
    state = sm0
    for i in range(n):
        state = state + input_water_mm[i]
        if state > FC:
            state = state - drainage_rate * (state - FC)
        if state <= WP:
            aet = 0.0
        elif state >= FC:
            aet = ET_scale * PET_mm[i]
        else:
            sat = (state - WP) / max(FC - WP, 1e-6)
            aet = ET_scale * PET_mm[i] * sat
        aet = min(aet, max(state - WP, 0.0))
        state = max(state - aet, 0.0)
        sm[i] = state
    return sm


# ── Snow model (reused from derive_landsurface.py) ──────────────────────────
def run_snow_model(P_mm, T_C, T_snow, melt_factor, swe0=0.0):
    n = len(P_mm)
    swe = np.zeros(n)
    rain = np.zeros(n)
    melt = np.zeros(n)
    state = swe0
    for i in range(n):
        if T_C[i] < T_snow:
            state = state + P_mm[i]
            rain[i] = 0.0
            melt[i] = 0.0
        else:
            rain[i] = P_mm[i]
            pot_melt = melt_factor * (T_C[i] - T_snow)
            melt[i] = min(state, pot_melt)
            state = state - melt[i]
        swe[i] = state
    return swe, rain, melt


def compute_api(precip_arr, k):
    """Exponential API — identical to preprocess2.py."""
    out = np.zeros(len(precip_arr))
    for i in range(1, len(precip_arr)):
        out[i] = k * out[i - 1] + precip_arr[i]
    return out


# ── SPI fitter (mirrors compute_spi_proper but stores params for reuse) ─────
def fit_spi_params(precip_series, scale_days=90):
    """Fit gamma parameters on a precipitation series for SPI computation."""
    rolling = precip_series.rolling(scale_days, min_periods=scale_days).sum()
    positive = rolling[rolling > 0].dropna().values
    n_total = rolling.dropna().shape[0]
    if len(positive) < 30:
        return None
    p_zero = (n_total - len(positive)) / max(n_total, 1)
    try:
        params = gamma_dist.fit(positive, floc=0)
    except Exception:
        return None
    return {
        "gamma_params": params,
        "p_zero": float(p_zero),
        "scale_days": int(scale_days),
    }


def apply_spi(precip_series, fit, scale_days=90):
    """Apply fitted SPI parameters to a precipitation series."""
    rolling = precip_series.rolling(scale_days, min_periods=scale_days).sum()
    if fit is None:
        return pd.Series(np.full(len(rolling), np.nan), index=precip_series.index)
    params = fit["gamma_params"]
    p_zero = fit["p_zero"]
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


def fit_spei_params(precip_series, pet_series, scale_days=90):
    """Fit SPEI parameters (mean, std of water balance) on training data."""
    wb = (precip_series - pet_series).rolling(scale_days, min_periods=scale_days).sum()
    vals = wb.dropna().values
    if len(vals) < 30:
        return None
    return {
        "mu": float(vals.mean()),
        "sigma": float(vals.std()),
        "scale_days": int(scale_days),
    }


def apply_spei(precip_series, pet_series, fit, scale_days=90):
    if fit is None:
        return pd.Series(np.full(len(precip_series), np.nan), index=precip_series.index)
    wb = (precip_series - pet_series).rolling(scale_days, min_periods=scale_days).sum()
    sigma = fit["sigma"]
    if sigma < 1e-8:
        return pd.Series(np.full(len(wb), np.nan), index=precip_series.index)
    result = np.full(len(wb), np.nan)
    not_nan = wb.notna()
    result[not_nan.values] = (wb[not_nan].values - fit["mu"]) / sigma
    return pd.Series(result, index=precip_series.index)


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Load training data and fit everything we need
# ═══════════════════════════════════════════════════════════════════════════
print("Step 1: Loading training data and fitting historical climatologies...")
master = (
    pd.read_csv(MASTER_FILE, parse_dates=["date"])
    .sort_values("date")
    .reset_index(drop=True)
)
train_mask = (master["date"] >= TRAIN_START) & (master["date"] <= TRAIN_END)
train = master.loc[train_mask].reset_index(drop=True)
print(f"  Master rows: {len(master)}, training rows: {len(train)}")

# 1a. PET bias correction: monthly factor to scale Hamon PET → ERA5 PET
print("\n  1a. PET correction factor (monthly):")
train_doy = train["date"].dt.dayofyear.values
train_t = train["temp_mean_c"].values
hamon_train = hamon_pet(train_t, train_doy)
era5_train = train["pet_mm_day"].values

train_month = train["date"].dt.month.values
pet_factor = {}
for m in range(1, 13):
    mask = train_month == m
    mean_era5 = era5_train[mask].mean()
    mean_hamon = hamon_train[mask].mean()
    factor = mean_era5 / mean_hamon if mean_hamon > 0 else 1.0
    pet_factor[m] = float(factor)
print(
    f"     Monthly factors (ERA5/Hamon): {[round(pet_factor[m], 2) for m in range(1, 13)]}"
)
print(
    f"     Overall ERA5 mean: {era5_train.mean():.2f}  Hamon mean: {hamon_train.mean():.2f}"
)

# 1b. SPI / SPEI fits using training-period precipitation and PET
print("\n  1b. Fitting SPI/SPEI parameters on training period...")
spi_fit = fit_spi_params(train["precip_mm_day"], scale_days=90)
spei_fit = fit_spei_params(train["precip_mm_day"], train["pet_mm_day"], scale_days=90)
if spi_fit is None or spei_fit is None:
    raise RuntimeError("Failed to fit SPI/SPEI — check training data.")
print(
    f"     SPI gamma params: shape={spi_fit['gamma_params'][0]:.3f}, "
    f"scale={spi_fit['gamma_params'][2]:.3f}, p_zero={spi_fit['p_zero']:.3f}"
)
print(f"     SPEI mu={spei_fit['mu']:.2f}, sigma={spei_fit['sigma']:.2f}")

# 1c. Day-of-year climatologies for snow features and deep soil moisture
print("\n  1c. Building DOY climatologies for snow + deep soil moisture...")
train["doy"] = train["date"].dt.dayofyear
doy_clim = train.groupby("doy")[
    ["swe_mm", "swe_delta", "snow_cover_pct", "sm_deep_30day", "sm_deep_anomaly"]
].mean()
# Fill any gaps (leap-year DOY 366) by reindexing then ffill
doy_clim = doy_clim.reindex(range(1, 367)).interpolate(
    method="linear", limit_direction="both"
)
print(f"     Climatology built for {len(doy_clim)} DOYs.")

# 1d. Load calibrated land-surface params
print("\n  1d. Loading calibrated land-surface parameters...")
with open(LS_PARAMS) as f:
    ls = json.load(f)
T_snow = float(ls["snow_model"]["T_snow_C"])
melt_f = float(ls["snow_model"]["melt_factor_mm_per_C_per_day"])
FC = float(ls["bucket_model"]["field_capacity_mm"])
WP = float(ls["bucket_model"]["wilting_point_mm"])
drain = float(ls["bucket_model"]["drainage_rate_per_day"])
ET_scl = float(ls["bucket_model"]["ET_scale"])
sm_mean_train = float(train["soil_moisture_mm"].mean())
print(f"     Snow: T_snow={T_snow:.2f}°C  melt={melt_f:.2f} mm/°C/day")
print(f"     Bucket: FC={FC:.1f}  WP={WP:.1f}  drain={drain:.4f}  ET_scl={ET_scl:.3f}")
print(f"     Bucket init: SM₀ = {sm_mean_train:.1f} mm (train mean)")

# Final column order — match preprocess2.py exactly
FEATURE_COLS = [
    "date",
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


def build_features_for_scenario(gcm: str, scenario: str) -> pd.DataFrame:
    """Build the full feature set for one GCM × scenario."""
    bc_file = BC_DIR / gcm / scenario / "merged_daily.csv"
    if not bc_file.exists():
        raise FileNotFoundError(bc_file)

    bc = (
        pd.read_csv(bc_file, parse_dates=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Concatenate last 90 days of historical for SPI warm-up
    hist_file = BC_DIR / gcm / "historical" / "merged_daily.csv"
    hist = (
        pd.read_csv(hist_file, parse_dates=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    warmup = hist[hist["date"] >= bc["date"].min() - pd.Timedelta(days=90)]
    warmup = warmup[warmup["date"] < bc["date"].min()]
    full = (
        pd.concat([warmup, bc], ignore_index=True)
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )

    # ── Temperature derived ──────────────────────────────────────────────
    full["temp_range_c"] = full["temp_max_c"] - full["temp_min_c"]

    # ── PET (Hamon + monthly correction) ─────────────────────────────────
    doy_arr = full["date"].dt.dayofyear.values
    pet_hamon = hamon_pet(full["temp_mean_c"].values, doy_arr)
    month_arr = full["date"].dt.month.values
    factor_arr = np.array([pet_factor[int(m)] for m in month_arr])
    full["pet_mm_day"] = pet_hamon * factor_arr

    # ── Snow + bucket ────────────────────────────────────────────────────
    P = full["precip_mm_day"].values
    T = full["temp_mean_c"].values
    PET = full["pet_mm_day"].values

    swe_sim, rain, melt = run_snow_model(P, T, T_snow, melt_f, swe0=0.0)
    bucket_input = rain + melt
    sm_sim = run_bucket_model(
        bucket_input,
        PET,
        FC=FC,
        WP=WP,
        drainage_rate=drain,
        ET_scale=ET_scl,
        sm0=sm_mean_train,
    )

    # We use the simulated soil moisture, but snow features come from
    # historical day-of-year climatology (per the methodology decision).
    full["soil_moisture_mm"] = sm_sim

    # Map DOY → climatology for snow features and deep soil moisture
    full["swe_mm"] = full["date"].dt.dayofyear.map(doy_clim["swe_mm"]).values
    full["swe_delta"] = full["date"].dt.dayofyear.map(doy_clim["swe_delta"]).values
    full["snow_cover_pct"] = (
        full["date"].dt.dayofyear.map(doy_clim["snow_cover_pct"]).values
    )
    full["sm_deep_30day"] = (
        full["date"].dt.dayofyear.map(doy_clim["sm_deep_30day"]).values
    )
    full["sm_deep_anomaly"] = (
        full["date"].dt.dayofyear.map(doy_clim["sm_deep_anomaly"]).values
    )

    # ── Precipitation features (mirror preprocess2.py) ──────────────────
    full["precip_3day"] = full["precip_mm_day"].rolling(3, min_periods=1).sum()
    full["precip_7day"] = full["precip_mm_day"].rolling(7, min_periods=1).sum()
    full["precip_14day"] = full["precip_mm_day"].rolling(14, min_periods=1).sum()
    full["precip_30day"] = full["precip_mm_day"].rolling(30, min_periods=1).sum()
    full["precip_60day"] = full["precip_mm_day"].rolling(60, min_periods=1).sum()
    full["precip_90day"] = full["precip_mm_day"].rolling(90, min_periods=1).sum()

    full["precip_lag1"] = full["precip_mm_day"].shift(1).fillna(0.0)
    full["precip_lag2"] = full["precip_mm_day"].shift(2).fillna(0.0)
    full["precip_lag3"] = full["precip_mm_day"].shift(3).fillna(0.0)
    full["precip_lag5"] = full["precip_mm_day"].shift(5).fillna(0.0)

    p_arr = full["precip_mm_day"].fillna(0.0).values
    full["api_15d"] = compute_api(p_arr, 0.92)
    full["api_30d"] = compute_api(p_arr, 0.98)
    full["api_60d"] = compute_api(p_arr, 0.99)

    # ── Soil moisture derived ────────────────────────────────────────────
    full["sm_7day_mean"] = full["soil_moisture_mm"].rolling(7, min_periods=1).mean()
    full["sm_30day_mean"] = full["soil_moisture_mm"].rolling(30, min_periods=1).mean()
    full["sm_anomaly"] = (
        full["soil_moisture_mm"]
        - full["soil_moisture_mm"].rolling(30, min_periods=7).mean()
    ).fillna(0.0)

    # ── SPI / SPEI using fitted params ──────────────────────────────────
    full["spi_3month"] = apply_spi(full["precip_mm_day"], spi_fit, scale_days=90)
    full["spei_3month"] = apply_spei(
        full["precip_mm_day"], full["pet_mm_day"], spei_fit, scale_days=90
    )
    full["spi_3month"] = full["spi_3month"].bfill().fillna(0.0)
    full["spei_3month"] = full["spei_3month"].bfill().fillna(0.0)

    # ── Cyclical month ──────────────────────────────────────────────────
    full["month_sin"] = np.sin(2 * np.pi * full["date"].dt.month / 12.0)
    full["month_cos"] = np.cos(2 * np.pi * full["date"].dt.month / 12.0)

    # ── Drop the warmup period; only return the requested scenario window ─
    out = full[full["date"] >= bc["date"].min()].reset_index(drop=True)
    return out[FEATURE_COLS]


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Build features for each GCM × scenario
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}\nStep 2: Building future feature CSVs\n{'=' * 70}\n")

summary = []
for gcm in GCMS:
    for scen in SCENARIOS:
        print(f"  {gcm} / {scen}")
        try:
            feat = build_features_for_scenario(gcm, scen)
        except Exception as e:
            print(f"    FAILED: {e}")
            continue
        out_path = OUT_DIR / f"{gcm}__{scen}.csv"
        feat.to_csv(out_path, index=False)
        print(f"    Saved → {out_path.relative_to(ROOT)}")
        print(
            f"    Rows: {len(feat)}  Date range: {feat['date'].min().date()} → {feat['date'].max().date()}"
        )

        # Sanity stats
        summary.append(
            {
                "gcm": gcm,
                "scenario": scen,
                "n_days": len(feat),
                "precip_mean": float(feat["precip_mm_day"].mean()),
                "tmean_mean": float(feat["temp_mean_c"].mean()),
                "pet_mean": float(feat["pet_mm_day"].mean()),
                "sm_mean": float(feat["soil_moisture_mm"].mean()),
                "spi_mean": float(feat["spi_3month"].mean()),
                "spei_mean": float(feat["spei_3month"].mean()),
            }
        )

# Save summary
summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUT_DIR / "_summary.csv", index=False)
print(f"\n{'=' * 70}")
print(f"  Done. Summary:")
print(summary_df.round(3).to_string(index=False))
print(f"\n  Output dir: {OUT_DIR}")
