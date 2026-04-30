import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats
from tensorflow.keras import layers

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

# =============================================================================
# CUSTOM LAYERS (REQUIRED FOR TRANSFORMER LOADING)
# =============================================================================
class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

        positions = np.arange(max_len)[:, np.newaxis]
        dims      = np.arange(d_model)[np.newaxis, :]
        angles    = positions / np.power(10000, (2 * (dims // 2)) / d_model)

        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])

        self.pos_encoding = tf.cast(
            angles[np.newaxis, :, :], dtype=tf.float32
        )

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_len": self.max_len,
            "d_model": self.d_model,
        })
        return config


class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, n_heads, ffn_dim, dropout, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout

        self.attention = layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads,
            dropout=dropout,
        )

        self.ffn = tf.keras.Sequential([
            layers.Dense(ffn_dim, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_output = self.attention(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout,
        })
        return config

# =============================================================================
# CONFIGURATION
# =============================================================================
ROOT      = Path("C:/Users/marck/Downloads/nahr_ibrahim_watershed")
CMIP6_DIR = ROOT / "data"    / "raw"     / "cmip6"
SPLIT_DIR = ROOT / "data"    / "splits"
MASTER_DIR= ROOT / "data"    / "master"
MODEL_DIR = ROOT / "models"  / "trained"
PRED_DIR  = ROOT / "results" / "predictions"
FIG_DIR   = ROOT / "results" / "figures"
MET_DIR   = ROOT / "results" / "metrics"
SCEN_DIR  = ROOT / "results" / "scenarios"

for d in [SCEN_DIR, FIG_DIR, MET_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["ssp245", "ssp585"]
LOOKBACK  = 30

FEATURE_COLS = [
    "precip_mm_day", "precip_3day",    "precip_7day",
    "temp_mean_c",   "temp_max_c",     "temp_min_c",   "temp_range_c",
    "swe_mm",        "swe_delta",      "snow_cover_pct",
    "month_sin",     "month_cos",
    "soil_moisture_mm", "sm_7day_mean", "sm_anomaly",
    "pet_mm_day",
]

N_FEATURES = len(FEATURE_COLS)  # 16
IDX_PRECIP = 0
IDX_SWE    = 7

print("=" * 65)
print("  Nahr Ibrahim — Climate Scenario Pipeline")
print(f"  Models: 5 | Features: {N_FEATURES} | Scenarios: SSP245, SSP585")
print("=" * 65)

# =============================================================================
# 1. LOAD SCALER
# =============================================================================

scaler = pd.read_csv(SPLIT_DIR / "scaler_params.csv", index_col=0)
q_min  = scaler.loc["discharge_m3s", "min"]
q_max  = scaler.loc["discharge_m3s", "max"]
print(f"\n  Scaler loaded | Q range: [{q_min:.3f}, {q_max:.3f}] m³/s")


def inverse_transform_q(q_norm):
    return np.clip(q_norm * (q_max - q_min) + q_min, 0, None)

# =============================================================================
# 2. HAMON PET — used in feature engineering for CMIP6 inputs
# =============================================================================

def hamon_pet(temp_c: np.ndarray, doy: np.ndarray) -> np.ndarray:
    """
    Hamon (1961) potential evapotranspiration.
    Calibrated for Lebanon latitude (~34°N).
    """
    Ld      = 12 + 4 * np.sin(2 * np.pi * (doy - 80) / 365)
    rho_sat = (216.7 * 0.6108 *
               np.exp(17.27 * temp_c / (temp_c + 237.3)) /
               (temp_c + 273.3))
    return np.clip(0.1651 * Ld * rho_sat, 0, None)

# =============================================================================
# 3. LOAD ALL 5 MODELS
# =============================================================================

print("\n  Loading trained models ...")


def nse_metric(y_true, y_pred):
    num = tf.reduce_sum(tf.square(y_true - y_pred))
    den = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1.0 - num / (den + tf.keras.backend.epsilon())


# custom_objects = {"nse_metric": nse_metric}
custom_objects = {
    "nse_metric": nse_metric,
    "PositionalEncoding": PositionalEncoding,
    "TransformerEncoderBlock": TransformerEncoderBlock,
}

model_files = {
    "LSTM"          : MODEL_DIR / "lstm_final.keras",
    "CNN-LSTM"      : MODEL_DIR / "cnn_lstm_final.keras",
    "Transformer"   : MODEL_DIR / "transformer_final.keras",
    "PI-LSTM"       : MODEL_DIR / "pi_lstm_final.keras",
    "PI-Transformer": MODEL_DIR / "pi_transformer_final.keras",
}

models = {}
for name, path in model_files.items():
    if not path.exists():
        print(f"  ✗ {name:<18} — file not found: {path.name}")
        continue
    try:
        m = tf.keras.models.load_model(
            str(path),
            custom_objects=custom_objects,
            compile=False
        )
        models[name] = m
        print(f"  ✓ {name:<18} ({m.count_params():,} params)")
    except Exception as e:
        print(f"  ✗ {name:<18} — {type(e).__name__}: {e}")

print(f"\n  Loaded {len(models)} model(s): {list(models.keys())}")

if not models:
    raise RuntimeError("No models loaded. Check model files exist.")

# =============================================================================
# 4. LOAD CMIP6 DATA
# =============================================================================

def load_cmip6_scenario(scenario: str) -> pd.DataFrame:
    print(f"\n  Loading CMIP6 {scenario.upper()} ...")
    nc_dir = CMIP6_DIR / scenario / "pr"
    years  = sorted([int(f.stem) for f in nc_dir.glob("*.nc")])
    print(f"  Years: {years[0]}–{years[-1]} ({len(years)} files)")

    records = []
    for year in years:
        try:
            pr_ds  = xr.open_dataset(CMIP6_DIR / scenario / "pr"     / f"{year}.nc")
            tas_ds = xr.open_dataset(CMIP6_DIR / scenario / "tas"    / f"{year}.nc")
            tn_ds  = xr.open_dataset(CMIP6_DIR / scenario / "tasmin" / f"{year}.nc")
            tx_path = CMIP6_DIR / scenario / "tasmax" / f"{year}.nc"
            tx_ds   = xr.open_dataset(tx_path) if tx_path.exists() else None

            times = pd.to_datetime(pr_ds["time"].values)
            pr_v  = pr_ds["pr"].mean(dim=["lat","lon"]).values * 86400
            tas_v = tas_ds["tas"].mean(dim=["lat","lon"]).values - 273.15
            tn_v  = tn_ds["tasmin"].mean(dim=["lat","lon"]).values - 273.15
            tx_v  = (tx_ds["tasmax"].mean(dim=["lat","lon"]).values - 273.15
                     if tx_ds is not None else tas_v + 5.0)

            for i, t in enumerate(times):
                records.append({
                    "date"       : t,
                    "pr_raw"     : max(0.0, float(pr_v[i])),
                    "tas_raw"    : float(tas_v[i]),
                    "tasmin_raw" : float(tn_v[i]),
                    "tasmax_raw" : float(tx_v[i]),
                })
            pr_ds.close(); tas_ds.close(); tn_ds.close()
            if tx_ds is not None: tx_ds.close()

        except Exception as e:
            print(f"    WARNING: {year} — {e}")
            continue

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  Loaded {len(df):,} records: "
          f"{df.date.min().date()} → {df.date.max().date()}")
    return df

# =============================================================================
# 5. DEGREE-DAY SNOW MODEL
# =============================================================================

def compute_snow(df: pd.DataFrame,
                 melt_rate: float = 3.0,
                 snow_temp: float = 2.0) -> pd.DataFrame:
    """Degree-day snow accumulation and melt model."""
    df  = df.copy()
    swe = np.zeros(len(df))
    for i in range(1, len(df)):
        t      = df["tas_raw"].iloc[i]
        p      = df["pr_raw"].iloc[i]
        accum  = p if t < snow_temp else 0.0
        melt   = max(0.0, melt_rate * t) if t > 0 else 0.0
        swe[i] = max(0.0, swe[i-1] + accum - melt)
    df["swe_mm"]         = swe
    df["snow_cover_pct"] = np.clip(swe / 50.0 * 100.0, 0.0, 100.0)
    return df

# =============================================================================
# 6. FEATURE ENGINEERING (all 16 features from CMIP6 forcing)
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all 16 model input features from CMIP6 raw fields.
    Mirrors preprocess.py feature engineering exactly.
    """
    df = df.copy()

    # ── Meteorological ──────────────────────────────────────────
    df["precip_mm_day"] = df["pr_raw"].clip(lower=0)
    df["temp_mean_c"]   = df["tas_raw"]
    df["temp_max_c"]    = df["tasmax_raw"]
    df["temp_min_c"]    = df["tasmin_raw"]
    df["temp_range_c"]  = df["temp_max_c"] - df["temp_min_c"]

    # ── Antecedent precipitation ─────────────────────────────────
    df["precip_3day"] = df["precip_mm_day"].rolling(3, min_periods=1).sum()
    df["precip_7day"] = df["precip_mm_day"].rolling(7, min_periods=1).sum()

    # ── Cryosphere ───────────────────────────────────────────────
    df = compute_snow(df)
    df["swe_delta"] = df["swe_mm"].diff().clip(upper=0).abs().fillna(0)

    # ── Cyclical time features ───────────────────────────────────
    df["month"]     = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ── Soil moisture proxy (degree-day water balance) ───────────
    # CMIP6 does not provide soil moisture — proxy from precip and temp
    # Calibrated to GLDAS historical range for Lebanon (5–80 mm)
    sm = np.zeros(len(df))
    sm[0] = 25.0   # typical GLDAS 0-10cm mean for Lebanon
    for i in range(1, len(df)):
        p      = df["precip_mm_day"].iloc[i]
        t      = df["temp_mean_c"].iloc[i]
        et_est = max(0.0, 0.15 * t)          # simplified ET loss (mm/day)
        sm[i]  = np.clip(sm[i-1] + p * 0.3 - et_est, 5.0, 80.0)

    df["soil_moisture_mm"] = sm
    df["sm_7day_mean"]     = df["soil_moisture_mm"].rolling(
        7, min_periods=1).mean()
    df["sm_anomaly"]       = (
        df["soil_moisture_mm"] -
        df["soil_moisture_mm"].rolling(30, min_periods=7).mean()
    ).fillna(0.0)

    # ── PET — Hamon (1961) ───────────────────────────────────────
    doy = df["date"].dt.dayofyear.values
    df["pet_mm_day"] = hamon_pet(df["temp_mean_c"].values, doy)

    return df

# =============================================================================
# 7. NORMALISE FEATURES
# =============================================================================

def normalize_features(df: pd.DataFrame) -> np.ndarray:
    """
    Apply training-set scaler to CMIP6 features.
    Uses same scaler_params.csv fitted on 2000-2017 training data.
    """
    out = np.zeros((len(df), N_FEATURES), dtype=np.float32)
    for j, col in enumerate(FEATURE_COLS):
        if col in scaler.index:
            f_min = scaler.loc[col, "min"]
            f_max = scaler.loc[col, "max"]
            r = f_max - f_min
            if r > 0:
                out[:, j] = ((df[col].values - f_min) / r).clip(0, None)
            # else leave as 0 for zero-range features
        else:
            # Column not in scaler (e.g. computed exactly in [0,1])
            out[:, j] = df[col].values.astype(np.float32)
    return out

# =============================================================================
# 8. CREATE SEQUENCES
# =============================================================================

def create_sequences(features: np.ndarray,
                     lookback: int = 30) -> np.ndarray:
    return np.array(
        [features[i - lookback:i] for i in range(lookback, len(features))],
        dtype=np.float32
    )

# =============================================================================
# 9. PREDICT SCENARIO
# =============================================================================

def predict_scenario(scenario: str) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print(f"  Scenario: {scenario.upper()}")
    print(f"{'='*55}")

    df_raw    = load_cmip6_scenario(scenario)
    df_feat   = engineer_features(df_raw)
    feat_norm = normalize_features(df_feat)
    X         = create_sequences(feat_norm, LOOKBACK)
    dates     = df_feat["date"].iloc[LOOKBACK:].reset_index(drop=True)

    print(f"  Sequences : {X.shape}  (samples, {LOOKBACK}, {N_FEATURES})")

    # Verify feature count matches model expectations
    for name, model in models.items():
        expected = model.input_shape[-1]
        if expected != N_FEATURES:
            print(f"  WARNING: {name} expects {expected} features, "
                  f"but sequences have {N_FEATURES}")

    results = {"date": pd.to_datetime(dates.values)}

    for name, model in models.items():
        print(f"  Predicting {name:<16} ...", end=" ", flush=True)
        try:
            y_norm = model.predict(X, batch_size=512, verbose=0).flatten()
            y_m3s  = inverse_transform_q(y_norm)
            results[f"Q_{name}"] = y_m3s
            print(f"mean={y_m3s.mean():.3f} | "
                  f"max={y_m3s.max():.3f} | "
                  f"min={y_m3s.min():.3f} m³/s")
        except Exception as e:
            print(f"FAILED — {e}")
            results[f"Q_{name}"] = np.full(len(dates), np.nan)

    df_out = pd.DataFrame(results)

    # ── Ensembles ─────────────────────────────────────────────────
    pure_cols = [f"Q_{n}" for n in ["LSTM", "CNN-LSTM", "Transformer"]
                 if f"Q_{n}" in df_out
                 and not df_out[f"Q_{n}"].isna().all()]
    pi_cols   = [f"Q_{n}" for n in ["PI-LSTM", "PI-Transformer"]
                 if f"Q_{n}" in df_out
                 and not df_out[f"Q_{n}"].isna().all()]
    all_good  = pure_cols + pi_cols

    if pure_cols:
        df_out["Q_ensemble_pure"] = df_out[pure_cols].mean(axis=1)
        print(f"  Ensemble (pure AI) : {len(pure_cols)} models")
    if pi_cols:
        df_out["Q_ensemble_pi"]   = df_out[pi_cols].mean(axis=1)
        print(f"  Ensemble (PI)      : {len(pi_cols)} models")
    if all_good:
        df_out["Q_ensemble_all"]  = df_out[all_good].mean(axis=1)
        df_out["Q_ensemble_std"]  = df_out[all_good].std(axis=1)
        print(f"  Ensemble (all)     : {len(all_good)} models")

    # ── Attach climate forcing columns for reference ───────────────
    df_out["precip_mm_day"] = df_feat["precip_mm_day"].iloc[LOOKBACK:].values
    df_out["temp_mean_c"]   = df_feat["temp_mean_c"].iloc[LOOKBACK:].values
    df_out["swe_mm"]        = df_feat["swe_mm"].iloc[LOOKBACK:].values
    df_out["pet_mm_day"]    = df_feat["pet_mm_day"].iloc[LOOKBACK:].values

    out_path = SCEN_DIR / f"discharge_{scenario}_daily.csv"
    df_out.to_csv(out_path, index=False)
    print(f"  Saved → results/scenarios/discharge_{scenario}_daily.csv")
    return df_out

# =============================================================================
# 10. TREND ANALYSIS (Linear regression + Mann-Kendall)
# =============================================================================

def compute_trends(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["date"].dt.year
    q_cols     = [c for c in df.columns if c.startswith("Q_")]
    annual     = df.groupby("year")[q_cols].mean()
    rows       = []

    for col in q_cols:
        y     = annual[col].dropna().values
        years = annual[col].dropna().index.values
        if len(y) < 10:
            continue

        # Linear regression
        slope, _, r, p_lr, _ = stats.linregress(years, y)

        # Mann-Kendall test
        n     = len(y)
        s     = sum(np.sign(y[j] - y[i])
                    for i in range(n - 1) for j in range(i + 1, n))
        var_s = n * (n - 1) * (2 * n + 5) / 18
        z_mk  = (s - np.sign(s)) / np.sqrt(var_s) if s != 0 else 0.0
        p_mk  = 2 * (1 - stats.norm.cdf(abs(z_mk)))

        rows.append({
            "scenario"          : scenario,
            "model"             : col.replace("Q_", ""),
            "slope_m3s_decade"  : round(slope * 10, 4),
            "r_squared"         : round(r ** 2, 4),
            "mk_z"              : round(z_mk, 4),
            "mk_p"              : round(p_mk, 4),
            "significant"       : p_mk < 0.05,
            "direction"         : "declining" if slope < 0 else "increasing",
            "mean_2015_2040"    : round(
                annual.loc[annual.index <= 2040, col].mean(), 4),
            "mean_2075_2100"    : round(
                annual.loc[annual.index >= 2075, col].mean(), 4),
            "pct_change"        : round(
                (annual.loc[annual.index >= 2075, col].mean() -
                 annual.loc[annual.index <= 2040, col].mean()) /
                annual.loc[annual.index <= 2040, col].mean() * 100, 2)
                if annual.loc[annual.index <= 2040, col].mean() != 0
                else np.nan,
        })

    return pd.DataFrame(rows)

# =============================================================================
# 11. RUN BOTH SCENARIOS
# =============================================================================

# Clear cached CSVs to force rerun
for scenario in SCENARIOS:
    cache = SCEN_DIR / f"discharge_{scenario}_daily.csv"
    if cache.exists():
        cache.unlink()
        print(f"  Cleared cache: {cache.name}")

scenario_results = {}
for scenario in SCENARIOS:
    scenario_results[scenario] = predict_scenario(scenario)

# Compute trends
all_trends = []
for scenario, df in scenario_results.items():
    all_trends.append(compute_trends(df, scenario))

trends_df = pd.concat(all_trends, ignore_index=True)
trends_df.to_csv(MET_DIR / "climate_scenario_trends_full.csv", index=False)

# Print trend table
key_models = ["LSTM", "CNN-LSTM", "Transformer",
              "PI-LSTM", "PI-Transformer",
              "ensemble_pure", "ensemble_pi", "ensemble_all"]

print(f"\n  {'Model':<22} {'SSP245 /decade':>16} {'SSP585 /decade':>16} "
      f"{'Sig245':>8} {'Sig585':>8} {'Chg245%':>9} {'Chg585%':>9}")
print(f"  {'-'*90}")

for m in key_models:
    r245 = trends_df[(trends_df.scenario == "ssp245") & (trends_df.model == m)]
    r585 = trends_df[(trends_df.scenario == "ssp585") & (trends_df.model == m)]
    if len(r245) == 0:
        continue
    s245   = r245["slope_m3s_decade"].values[0]
    s585   = r585["slope_m3s_decade"].values[0] if len(r585) > 0 else float("nan")
    g245   = "✓" if r245["significant"].values[0] else "✗"
    g585   = "✓" if len(r585) > 0 and r585["significant"].values[0] else "✗"
    c245   = r245["pct_change"].values[0] if len(r245) > 0 else float("nan")
    c585   = r585["pct_change"].values[0] if len(r585) > 0 else float("nan")
    print(f"  {m:<22} {s245:>16.4f} {s585:>16.4f} "
          f"{g245:>8} {g585:>8} {c245:>8.1f}% {c585:>8.1f}%")

# =============================================================================
# 12. VISUALISATION — 6 figures
# =============================================================================

print("\n  Generating thesis figures ...")

COLORS = {
    "LSTM"          : "#3b9eff",
    "CNN-LSTM"      : "#00b4a0",
    "Transformer"   : "#00d4ff",
    "PI-LSTM"       : "#f4a261",
    "PI-Transformer": "#e76f51",
    "ensemble_pure" : "#8aafc4",
    "ensemble_pi"   : "#f4a261",
    "ensemble_all"  : "#ffffff",
}

# ── Figure 1: Annual discharge projections ────────────────────
fig1, axes = plt.subplots(2, 1, figsize=(18, 14))
fig1.patch.set_facecolor("#080f1a")

for ax, (scenario, label) in zip(axes, [
    ("ssp245", "SSP2-4.5 — Moderate Emissions"),
    ("ssp585", "SSP5-8.5 — High Emissions"),
]):
    ax.set_facecolor("#0d1825")
    df = scenario_results[scenario].copy()
    df["year"] = df["date"].dt.year
    annual = df.groupby("year").mean(numeric_only=True)

    for name in ["LSTM", "CNN-LSTM", "Transformer", "PI-LSTM", "PI-Transformer"]:
        col = f"Q_{name}"
        if col not in annual or annual[col].isna().all():
            continue
        sm = annual[col].rolling(5, center=True, min_periods=1).mean()
        ls = "--" if "PI" in name else "-"
        ax.plot(annual.index, sm,
                color=COLORS.get(name, "#8aafc4"),
                linewidth=1.5, alpha=0.75, linestyle=ls, label=name)

    if "Q_ensemble_all" in annual:
        sm_e = annual["Q_ensemble_all"].rolling(5, center=True, min_periods=1).mean()
        sm_s = annual["Q_ensemble_std"].rolling(5, center=True, min_periods=1).mean()
        ax.plot(annual.index, sm_e, color="#ffffff",
                linewidth=2.5, label="All-model ensemble", zorder=5)
        ax.fill_between(annual.index, sm_e - sm_s, sm_e + sm_s,
                        alpha=0.08, color="#ffffff")
        hmean = annual.loc[annual.index <= 2025, "Q_ensemble_all"].mean()
        ax.axhline(hmean, color="#8aafc4", linewidth=1,
                   linestyle=":", alpha=0.5,
                   label=f"2015–2025 mean ({hmean:.3f} m³/s)")

    ax.axvline(2025, color="#4a6a82", linewidth=1.2,
               linestyle=":", alpha=0.8, label="Present (2025)")
    ax.set_title(label, color="#e8f4f8", fontsize=12)
    ax.set_xlabel("Year", color="#8aafc4")
    ax.set_ylabel("Annual Mean Discharge (m³/s)", color="#8aafc4")
    ax.tick_params(colors="#4a6a82")
    ax.spines[:].set_color("#1e3448")
    ax.legend(facecolor="#0d1825", edgecolor="#1e3448",
              labelcolor="#8aafc4", fontsize=8, ncol=4)
    ax.grid(alpha=0.08)
    ax.set_facecolor("#0d1825")

fig1.suptitle(
    "Nahr Ibrahim — Projected Discharge 2015–2100\n"
    "Pure AI vs Physics-Informed | MPI-ESM1-2-HR · NEX-GDDP-CMIP6",
    color="#e8f4f8", fontsize=13, y=1.01, fontfamily="monospace")
plt.tight_layout()
plt.savefig(FIG_DIR / "scenario_annual_discharge.png",
            dpi=150, bbox_inches="tight", facecolor="#080f1a")
plt.show()
print("  ✓ Figure 1: scenario_annual_discharge.png")

# ── Figure 2: Pure AI vs PI comparison ───────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))
fig2.patch.set_facecolor("#080f1a")

for ax, scenario in zip(axes2, SCENARIOS):
    ax.set_facecolor("#0d1825")
    df = scenario_results[scenario].copy()
    df["year"] = df["date"].dt.year
    annual = df.groupby("year").mean(numeric_only=True)

    for col_name, color, label, ls in [
        ("Q_Transformer",    "#00d4ff", "Transformer (pure)", "-"),
        ("Q_PI-Transformer", "#e76f51", "PI-Transformer",     "--"),
        ("Q_LSTM",           "#3b9eff", "LSTM (pure)",        "-"),
        ("Q_PI-LSTM",        "#f4a261", "PI-LSTM",            "--"),
    ]:
        if col_name not in annual or annual[col_name].isna().all():
            continue
        sm = annual[col_name].rolling(10, center=True, min_periods=3).mean()
        ax.plot(annual.index, sm, color=color, linewidth=2,
                linestyle=ls, label=label, alpha=0.9)

    ax.axvline(2025, color="#4a6a82", linewidth=1, linestyle=":")
    ax.set_title(f"Pure AI vs Physics-Informed — {scenario.upper()}",
                 color="#e8f4f8", fontsize=11)
    ax.set_xlabel("Year", color="#8aafc4")
    ax.set_ylabel("Discharge (m³/s)", color="#8aafc4")
    ax.tick_params(colors="#4a6a82")
    ax.spines[:].set_color("#1e3448")
    ax.legend(facecolor="#0d1825", edgecolor="#1e3448",
              labelcolor="#8aafc4", fontsize=9)
    ax.grid(alpha=0.08)
    ax.set_facecolor("#0d1825")

fig2.suptitle("Pure AI vs Physics-Informed — Discharge Projections",
              color="#e8f4f8", fontsize=13, y=1.02, fontfamily="monospace")
plt.tight_layout()
plt.savefig(FIG_DIR / "scenario_pure_vs_pi.png",
            dpi=150, bbox_inches="tight", facecolor="#080f1a")
plt.show()
print("  ✓ Figure 2: scenario_pure_vs_pi.png")

# ── Figure 3: Seasonal shift ──────────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 7))
fig3.patch.set_facecolor("#080f1a")
month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

for ax, scenario in zip(axes3, SCENARIOS):
    ax.set_facecolor("#0d1825")
    df = scenario_results[scenario].copy()
    df["month"] = df["date"].dt.month
    df["year"]  = df["date"].dt.year

    col = next((c for c in
                ["Q_ensemble_all", "Q_Transformer", "Q_LSTM"]
                if c in df and not df[c].isna().all()), None)
    if col is None:
        continue

    for label, mask, color in [
        ("2015–2040", (df.year >= 2015) & (df.year <= 2040), "#3b9eff"),
        ("2041–2070", (df.year >= 2041) & (df.year <= 2070), "#f4a261"),
        ("2071–2100", (df.year >= 2071) & (df.year <= 2100), "#e76f51"),
    ]:
        if mask.sum() == 0:
            continue
        monthly = df[mask].groupby("month")[col].mean()
        ax.plot(monthly.index, monthly.values, color=color,
                linewidth=2.5, marker="o", markersize=6, label=label)

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels, color="#8aafc4", fontsize=9)
    ax.set_title(f"Seasonal Shift — {scenario.upper()}",
                 color="#e8f4f8", fontsize=11)
    ax.set_ylabel("Monthly Mean Discharge (m³/s)", color="#8aafc4")
    ax.tick_params(colors="#4a6a82")
    ax.spines[:].set_color("#1e3448")
    ax.legend(facecolor="#0d1825", edgecolor="#1e3448", labelcolor="#8aafc4")
    ax.grid(alpha=0.08)
    ax.set_facecolor("#0d1825")

fig3.suptitle("Nahr Ibrahim — Seasonal Discharge Shift by Period",
              color="#e8f4f8", fontsize=13, y=1.02, fontfamily="monospace")
plt.tight_layout()
plt.savefig(FIG_DIR / "scenario_seasonal_shift.png",
            dpi=150, bbox_inches="tight", facecolor="#080f1a")
plt.show()
print("  ✓ Figure 3: scenario_seasonal_shift.png")

# ── Figure 4: Trend summary bar chart ────────────────────────
plot_m = [m for m in key_models if trends_df["model"].isin([m]).any()]
fig4, ax4 = plt.subplots(figsize=(16, 6))
fig4.patch.set_facecolor("#080f1a")
ax4.set_facecolor("#0d1825")
x = np.arange(len(plot_m))
w = 0.35

for i, (scenario, color) in enumerate(
    {"ssp245": "#3b9eff", "ssp585": "#e76f51"}.items()
):
    slopes = []
    for m in plot_m:
        row = trends_df[(trends_df.scenario == scenario) &
                        (trends_df.model == m)]
        slopes.append(row["slope_m3s_decade"].values[0] if len(row) > 0 else 0)
    bars = ax4.bar(x + i * w, slopes, w,
                   label=scenario.upper(), color=color, alpha=0.8)
    for bar, val in zip(bars, slopes):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.0003 * np.sign(val + 1e-9),
                 f"{val:.4f}", ha="center", va="bottom",
                 color="#e8f4f8", fontsize=7, fontfamily="monospace")

ax4.axhline(0, color="#8aafc4", linewidth=1)
ax4.set_xticks(x + w / 2)
ax4.set_xticklabels(plot_m, color="#8aafc4", fontsize=8, rotation=15)
ax4.set_ylabel("Discharge Trend (m³/s per decade)", color="#8aafc4")
ax4.set_title("Projected Discharge Trends 2015–2100 — All Models",
              color="#e8f4f8", fontsize=12)
ax4.tick_params(colors="#4a6a82")
ax4.spines[:].set_color("#1e3448")
ax4.legend(facecolor="#0d1825", edgecolor="#1e3448", labelcolor="#8aafc4")
ax4.grid(axis="y", alpha=0.08)
plt.tight_layout()
plt.savefig(FIG_DIR / "scenario_trend_summary.png",
            dpi=150, bbox_inches="tight", facecolor="#080f1a")
plt.show()
print("  ✓ Figure 4: scenario_trend_summary.png")

# ── Figure 5: Climate forcing ─────────────────────────────────
fig5, axes5 = plt.subplots(2, 2, figsize=(18, 10))
fig5.patch.set_facecolor("#080f1a")

for col_idx, scenario in enumerate(SCENARIOS):
    df = scenario_results[scenario].copy()
    df["year"] = df["date"].dt.year
    annual = df.groupby("year").mean(numeric_only=True)

    for row_idx, (var, ylabel, color) in enumerate([
        ("precip_mm_day", "Precipitation (mm/day)", "#3b9eff"),
        ("temp_mean_c",   "Temperature (°C)",       "#f4a261"),
    ]):
        ax = axes5[row_idx, col_idx]
        ax.set_facecolor("#0d1825")
        sm = annual[var].rolling(10, center=True, min_periods=3).mean()
        if row_idx == 0:
            ax.bar(annual.index, annual[var],
                   color=color, alpha=0.25, width=0.8)
        else:
            ax.plot(annual.index, annual[var],
                    color=color, alpha=0.25, linewidth=0.8)
        ax.plot(annual.index, sm, color=color,
                linewidth=2, label="10-yr smooth")
        ax.set_title(f"{ylabel} — {scenario.upper()}",
                     color="#e8f4f8", fontsize=10)
        ax.set_ylabel(ylabel, color="#8aafc4")
        ax.tick_params(colors="#4a6a82")
        ax.spines[:].set_color("#1e3448")
        ax.legend(facecolor="#0d1825", edgecolor="#1e3448",
                  labelcolor="#8aafc4", fontsize=8)
        ax.grid(alpha=0.08)
        ax.set_facecolor("#0d1825")

fig5.suptitle("CMIP6 Climate Forcing — MPI-ESM1-2-HR",
              color="#e8f4f8", fontsize=13, y=1.01, fontfamily="monospace")
plt.tight_layout()
plt.savefig(FIG_DIR / "scenario_climate_forcing.png",
            dpi=150, bbox_inches="tight", facecolor="#080f1a")
plt.show()
print("  ✓ Figure 5: scenario_climate_forcing.png")

# ── Figure 6: PI vs Pure ensemble divergence ──────────────────
fig6, ax6 = plt.subplots(figsize=(14, 7))
fig6.patch.set_facecolor("#080f1a")
ax6.set_facecolor("#0d1825")

for scenario, ls in [("ssp245", "-"), ("ssp585", "--")]:
    df = scenario_results[scenario].copy()
    df["year"] = df["date"].dt.year
    annual = df.groupby("year").mean(numeric_only=True)

    for col_name, color, label in [
        ("Q_ensemble_pure", "#00d4ff", f"Pure AI ensemble ({scenario})"),
        ("Q_ensemble_pi",   "#e76f51", f"PI ensemble ({scenario})"),
    ]:
        if col_name not in annual or annual[col_name].isna().all():
            continue
        sm = annual[col_name].rolling(10, center=True, min_periods=3).mean()
        ax6.plot(annual.index, sm, color=color, linewidth=2,
                 linestyle=ls, label=label, alpha=0.85)

ax6.axvline(2025, color="#4a6a82", linewidth=1.2, linestyle=":")
ax6.set_xlabel("Year", color="#8aafc4")
ax6.set_ylabel("Annual Mean Discharge (m³/s)", color="#8aafc4")
ax6.set_title("Pure AI vs Physics-Informed Ensemble — Long-term Divergence",
              color="#e8f4f8", fontsize=12)
ax6.tick_params(colors="#4a6a82")
ax6.spines[:].set_color("#1e3448")
ax6.legend(facecolor="#0d1825", edgecolor="#1e3448",
           labelcolor="#8aafc4", fontsize=9, ncol=2)
ax6.grid(alpha=0.08)
plt.tight_layout()
plt.savefig(FIG_DIR / "scenario_pi_vs_pure_divergence.png",
            dpi=150, bbox_inches="tight", facecolor="#080f1a")
plt.show()
print("  ✓ Figure 6: scenario_pi_vs_pure_divergence.png")

# =============================================================================
# 13. FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 65)
print("  CLIMATE SCENARIO SUMMARY — NAHR IBRAHIM")
print(f"  Models: {len(models)} | Features: {N_FEATURES}")
print("=" * 65)

for scenario in SCENARIOS:
    df = scenario_results[scenario].copy()
    df["year"] = df["date"].dt.year
    col = next((c for c in
                ["Q_ensemble_all", "Q_Transformer", "Q_LSTM"]
                if c in df and not df[c].isna().all()), None)
    if col is None:
        continue
    early   = df[df.year <= 2040][col].mean()
    late    = df[df.year >= 2075][col].mean()
    chg_pct = (late - early) / early * 100

    print(f"\n  {scenario.upper()}:")
    print(f"    Mean Q 2015–2040 : {early:.4f} m³/s")
    print(f"    Mean Q 2075–2100 : {late:.4f} m³/s")
    print(f"    Change           : {chg_pct:+.1f}%")
    print(f"    Direction        : "
          f"{'DECLINING ↓' if chg_pct < 0 else 'INCREASING ↑'}")

print(f"\n  Pure AI vs PI ensemble trend (SSP5-8.5):")
for ens in ["ensemble_pure", "ensemble_pi"]:
    row = trends_df[(trends_df.scenario == "ssp585") &
                    (trends_df.model == ens)]
    if len(row) == 0:
        continue
    print(f"    {ens:<22}: "
          f"{row['slope_m3s_decade'].values[0]:.4f} m³/s/decade")

print(f"\n  Files saved:")
for f in ["discharge_ssp245_daily.csv", "discharge_ssp585_daily.csv"]:
    print(f"    results/scenarios/{f}")
print(f"    results/metrics/climate_scenario_trends_full.csv")
for f in ["scenario_annual_discharge.png", "scenario_pure_vs_pi.png",
          "scenario_seasonal_shift.png",  "scenario_trend_summary.png",
          "scenario_climate_forcing.png", "scenario_pi_vs_pure_divergence.png"]:
    print(f"    results/figures/{f}")
print("=" * 65)