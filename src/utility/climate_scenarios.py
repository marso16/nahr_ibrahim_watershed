import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats
from tensorflow.keras import layers

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")


# ── Custom layers needed to reload the Transformer ────────────────────────────
# TensorFlow requires these classes to be registered before calling load_model.
# Keeping them here makes the script self-contained — no external imports needed.


class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        pos = np.arange(max_len)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]
        angles = pos / np.power(10000, (2 * (dims // 2)) / d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self.pos_encoding = tf.cast(angles[np.newaxis], dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, : tf.shape(x)[1], :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"max_len": self.max_len, "d_model": self.d_model})
        return cfg


class TransformerEncoderBlock(layers.Layer):
    def __init__(self, d_model, n_heads, ffn_dim, dropout, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attention = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads, dropout=dropout
        )
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(ffn_dim, activation="relu"),
                layers.Dropout(dropout),
                layers.Dense(d_model),
            ]
        )
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn = self.drop1(self.attention(x, x, training=training), training=training)
        x = self.ln1(x + attn)
        ffn = self.drop2(self.ffn(x, training=training), training=training)
        return self.ln2(x + ffn)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "ffn_dim": self.ffn_dim,
                "dropout": self.dropout,
            }
        )
        return cfg


# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path("C:/Users/marck/Downloads/nahr_ibrahim_watershed")
CMIP6_DIR = ROOT / "data" / "raw" / "cmip6"
SPLIT_DIR = ROOT / "data" / "splits"
MODEL_DIR = ROOT / "models" / "trained"
FIG_DIR = ROOT / "results" / "figures"
MET_DIR = ROOT / "results" / "metrics"
SCEN_DIR = ROOT / "results" / "scenarios"

for d in [SCEN_DIR, FIG_DIR, MET_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["ssp245", "ssp585"]
LOOKBACK = 30

FEATURE_COLS = [
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
]
N = len(FEATURE_COLS)

print(f"Climate scenario pipeline — {N} features, {len(SCENARIOS)} scenarios\n")


# ── Scaler — fitted on 2000–2017 training data ─────────────────────────────────
scaler = pd.read_csv(SPLIT_DIR / "scaler_params.csv", index_col=0)
q_min = scaler.loc["discharge_m3s", "min"]
q_max = scaler.loc["discharge_m3s", "max"]
print(f"Scaler loaded  Q ∈ [{q_min:.3f}, {q_max:.3f}] m³/s")


def inverse_q(q_norm):
    return np.clip(q_norm * (q_max - q_min) + q_min, 0, None)


# ── Hamon (1961) PET ───────────────────────────────────────────────────────────
# Temperature-based estimate calibrated for ~34°N (Lebanon).
# Used both here and in the physics-informed loss during training.
def hamon_pet(temp_c, doy):
    daylight = 12 + 4 * np.sin(2 * np.pi * (doy - 80) / 365)
    sat_vp = (
        216.7 * 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3)) / (temp_c + 273.3)
    )
    return np.clip(0.1651 * daylight * sat_vp, 0, None)


# ── Load trained models ─────────────────────────────────────────────────────────
def nse_metric(y_true, y_pred):
    num = tf.reduce_sum(tf.square(y_true - y_pred))
    den = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1.0 - num / (den + tf.keras.backend.epsilon())


custom_obj = {
    "nse_metric": nse_metric,
    "PositionalEncoding": PositionalEncoding,
    "TransformerEncoderBlock": TransformerEncoderBlock,
}

model_paths = {
    "LSTM": MODEL_DIR / "lstm_final.keras",
    "CNN-LSTM": MODEL_DIR / "cnn_lstm_final.keras",
    "Transformer": MODEL_DIR / "transformer_final.keras",
    "PI-LSTM": MODEL_DIR / "pi_lstm_final.keras",
    "PI-Transformer": MODEL_DIR / "pi_transformer_final.keras",
}

print("Loading models:")
models = {}
for name, path in model_paths.items():
    if not path.exists():
        print(f"  skip  {name} — not found")
        continue
    try:
        m = tf.keras.models.load_model(
            str(path), custom_objects=custom_obj, compile=False
        )
        models[name] = m
        print(f"  ok    {name} ({m.count_params():,} params)")
    except Exception as e:
        print(f"  fail  {name} — {e}")

if not models:
    raise RuntimeError("No models loaded.")
print(f"\n{len(models)} models ready: {list(models.keys())}\n")


# ── CMIP6 data loading ──────────────────────────────────────────────────────────
# Downloaded via GEE as yearly CSVs per variable.
# Units on disk: pr in kg/m²/s, temperatures in K.

UNIT_CONV = {
    "pr": lambda x: x * 86400,  # kg/m²/s → mm/day
    "tas": lambda x: x - 273.15,  # K → °C
    "tasmin": lambda x: x - 273.15,
    "tasmax": lambda x: x - 273.15,
}


def load_cmip6(scenario):
    print(f"Loading CMIP6 {scenario.upper()} ...")
    years = sorted(int(f.stem) for f in (CMIP6_DIR / scenario / "pr").glob("*.csv"))
    print(f"  {years[0]}–{years[-1]}  ({len(years)} years)")

    records = []
    for year in years:
        try:
            cols = {}
            for var in ["pr", "tas", "tasmin", "tasmax"]:
                p = CMIP6_DIR / scenario / var / f"{year}.csv"
                df = pd.read_csv(p, parse_dates=["date"])
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                df["value"] = UNIT_CONV[var](df["value"])
                cols[var] = df.set_index("date")["value"]
            merged = pd.DataFrame(cols).dropna()
            for date, row in merged.iterrows():
                records.append(
                    {
                        "date": date,
                        "pr_raw": max(0.0, float(row["pr"])),
                        "tas_raw": float(row["tas"]),
                        "tasmin_raw": float(row["tasmin"]),
                        "tasmax_raw": float(row["tasmax"]),
                    }
                )
        except Exception as e:
            print(f"  skip {year} — {e}")

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  {len(df):,} records  {df.date.min().date()} → {df.date.max().date()}")
    return df


# ── Degree-day snow model ───────────────────────────────────────────────────────
# Simple but effective for estimating SWE in CMIP6 projections where GLDAS
# snow data is not available. Parameters calibrated for Mount Lebanon.
def compute_snow(df, melt_rate=3.0, snow_temp=2.0):
    df = df.copy()
    swe = np.zeros(len(df))
    for i in range(1, len(df)):
        t = df["tas_raw"].iloc[i]
        p = df["pr_raw"].iloc[i]
        accum = p if t < snow_temp else 0.0
        melt = max(0.0, melt_rate * t) if t > 0 else 0.0
        swe[i] = max(0.0, swe[i - 1] + accum - melt)
    df["swe_mm"] = swe
    df["snow_cover_pct"] = np.clip(swe / 50.0 * 100.0, 0.0, 100.0)
    return df


# ── Feature engineering ─────────────────────────────────────────────────────────
# Mirrors preprocess.py exactly so the normalised inputs match what the models
# saw during training. The soil moisture proxy replaces GLDAS — calibrated to
# the GLDAS 0–10 cm range for Lebanon (5–80 mm).
def engineer_features(df):
    df = df.copy()

    df["precip_mm_day"] = df["pr_raw"].clip(lower=0)
    df["temp_mean_c"] = df["tas_raw"]
    df["temp_max_c"] = df["tasmax_raw"]
    df["temp_min_c"] = df["tasmin_raw"]
    df["temp_range_c"] = df["temp_max_c"] - df["temp_min_c"]

    df["precip_3day"] = df["precip_mm_day"].rolling(3, min_periods=1).sum()
    df["precip_7day"] = df["precip_mm_day"].rolling(7, min_periods=1).sum()

    df = compute_snow(df)
    df["swe_delta"] = df["swe_mm"].diff().clip(upper=0).abs().fillna(0)

    df["month"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Soil moisture proxy — water balance model, no GLDAS in CMIP6
    sm = np.zeros(len(df))
    sm[0] = 25.0  # start near the GLDAS Lebanon mean
    for i in range(1, len(df)):
        p = df["precip_mm_day"].iloc[i]
        t = df["temp_mean_c"].iloc[i]
        et = max(0.0, 0.15 * t)
        sm[i] = np.clip(sm[i - 1] + p * 0.3 - et, 5.0, 80.0)

    df["soil_moisture_mm"] = sm
    df["sm_7day_mean"] = df["soil_moisture_mm"].rolling(7, min_periods=1).mean()
    df["sm_anomaly"] = (
        df["soil_moisture_mm"]
        - df["soil_moisture_mm"].rolling(30, min_periods=7).mean()
    ).fillna(0.0)

    doy = df["date"].dt.dayofyear.values
    df["pet_mm_day"] = hamon_pet(df["temp_mean_c"].values, doy)

    # ── Drought indices proxy for CMIP6 ───────────────────────────────────────
    # SPI-3 and SPEI-3 computed from the derived precip and PET above.
    # These mirror the indices computed in preprocess.py for historical data.
    from scipy.stats import norm, gamma as gamma_dist

    def _spi(series, scale=3):
        rolling = series.rolling(scale, min_periods=scale).sum()
        result = np.zeros(len(rolling))
        valid = rolling.notna() & (rolling > 0)
        if valid.sum() > 10:
            try:
                params = gamma_dist.fit(rolling[valid].values, floc=0)
                prob = np.clip(
                    gamma_dist.cdf(rolling[valid].values, *params), 0.001, 0.999
                )
                result[valid.values] = norm.ppf(prob)
            except Exception:
                pass
        return pd.Series(result, index=series.index)

    df["spi_3month"] = _spi(df["precip_mm_day"]).fillna(0.0)
    df["spei_3month"] = _spi(df["precip_mm_day"] - df["pet_mm_day"]).fillna(0.0)

    # Flood index — top 10% precipitation days as proxy for high-flow conditions
    # Real discharge not available in CMIP6 so we use precipitation exceedance
    p90 = df["precip_mm_day"].quantile(0.90)
    df["flood_index"] = (df["precip_mm_day"] > p90).astype(float)

    return df


# ── Normalise using training scaler ────────────────────────────────────────────
def normalise(df):
    out = np.zeros((len(df), N), dtype=np.float32)
    for j, col in enumerate(FEATURE_COLS):
        if col in scaler.index:
            f_min = scaler.loc[col, "min"]
            f_max = scaler.loc[col, "max"]
            r = f_max - f_min
            if r > 0:
                out[:, j] = ((df[col].values - f_min) / r).clip(0, None)
        else:
            out[:, j] = df[col].values.astype(np.float32)
    return out


# ── Sliding window sequences ────────────────────────────────────────────────────
def make_sequences(features, lookback=30):
    return np.array(
        [features[i - lookback : i] for i in range(lookback, len(features))],
        dtype=np.float32,
    )


# ── Run one scenario ────────────────────────────────────────────────────────────
def predict_scenario(scenario):
    print(f"\n--- {scenario.upper()} ---")

    raw = load_cmip6(scenario)
    feat = engineer_features(raw)
    X = make_sequences(normalise(feat), LOOKBACK)
    dates = feat["date"].iloc[LOOKBACK:].reset_index(drop=True)
    print(f"  Sequences: {X.shape}")

    # quick sanity check
    for name, m in models.items():
        exp = m.input_shape[-1]
        if exp != N:
            print(f"  WARNING {name} expects {exp} features, got {N}")

    out = {"date": pd.to_datetime(dates.values)}
    for name, m in models.items():
        print(f"  {name:<18} ...", end=" ", flush=True)
        try:
            q = inverse_q(m.predict(X, batch_size=512, verbose=0).flatten())
            out[f"Q_{name}"] = q
            print(f"mean={q.mean():.3f}  max={q.max():.3f}  min={q.min():.3f} m³/s")
        except Exception as e:
            print(f"FAILED — {e}")
            out[f"Q_{name}"] = np.full(len(dates), np.nan)

    df = pd.DataFrame(out)

    # ensemble columns
    pure = [
        f"Q_{n}"
        for n in ["LSTM", "CNN-LSTM", "Transformer"]
        if f"Q_{n}" in df and not df[f"Q_{n}"].isna().all()
    ]
    pi = [
        f"Q_{n}"
        for n in ["PI-LSTM", "PI-Transformer"]
        if f"Q_{n}" in df and not df[f"Q_{n}"].isna().all()
    ]
    all_ = pure + pi

    if pure:
        df["Q_ensemble_pure"] = df[pure].mean(axis=1)
    if pi:
        df["Q_ensemble_pi"] = df[pi].mean(axis=1)
    if all_:
        df["Q_ensemble_all"] = df[all_].mean(axis=1)
        df["Q_ensemble_std"] = df[all_].std(axis=1)

    # keep climate forcing for reference
    for col in ["precip_mm_day", "temp_mean_c", "swe_mm", "pet_mm_day"]:
        df[col] = feat[col].iloc[LOOKBACK:].values

    df.to_csv(SCEN_DIR / f"discharge_{scenario}_daily.csv", index=False)
    print(f"  Saved → results/scenarios/discharge_{scenario}_daily.csv")
    return df


# ── Trend analysis ──────────────────────────────────────────────────────────────
# Linear regression + Mann-Kendall non-parametric test.
# Both are needed: LR gives slope magnitude, MK gives significance
# without assuming normality (appropriate for skewed discharge series).
def compute_trends(df, scenario):
    df = df.copy()
    df["year"] = df["date"].dt.year
    q_cols = [c for c in df.columns if c.startswith("Q_")]
    annual = df.groupby("year")[q_cols].mean()
    rows = []

    for col in q_cols:
        y = annual[col].dropna().values
        years = annual[col].dropna().index.values
        if len(y) < 10:
            continue

        slope, _, r, _, _ = stats.linregress(years, y)

        n = len(y)
        s = sum(np.sign(y[j] - y[i]) for i in range(n - 1) for j in range(i + 1, n))
        var_s = n * (n - 1) * (2 * n + 5) / 18
        z_mk = (s - np.sign(s)) / np.sqrt(var_s) if s != 0 else 0.0
        p_mk = 2 * (1 - stats.norm.cdf(abs(z_mk)))

        early = annual.loc[annual.index <= 2040, col].mean()
        late = annual.loc[annual.index >= 2075, col].mean()

        rows.append(
            {
                "scenario": scenario,
                "model": col.replace("Q_", ""),
                "slope_m3s_decade": round(slope * 10, 4),
                "r_squared": round(r**2, 4),
                "mk_z": round(z_mk, 4),
                "mk_p": round(p_mk, 4),
                "significant": p_mk < 0.05,
                "direction": "declining" if slope < 0 else "increasing",
                "mean_2015_2040": round(early, 4),
                "mean_2075_2100": round(late, 4),
                "pct_change": (
                    round((late - early) / early * 100, 2) if early else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


# ── Run ─────────────────────────────────────────────────────────────────────────
for scenario in SCENARIOS:
    p = SCEN_DIR / f"discharge_{scenario}_daily.csv"
    if p.exists():
        p.unlink()

results = {s: predict_scenario(s) for s in SCENARIOS}

trends_df = pd.concat(
    [compute_trends(df, s) for s, df in results.items()], ignore_index=True
)
trends_df.to_csv(MET_DIR / "climate_scenario_trends_full.csv", index=False)

key_models = [
    "LSTM",
    "CNN-LSTM",
    "Transformer",
    "PI-LSTM",
    "PI-Transformer",
    "ensemble_pure",
    "ensemble_pi",
    "ensemble_all",
]

print(
    f"\n{'Model':<22} {'SSP245/dec':>12} {'SSP585/dec':>12} "
    f"{'Sig245':>7} {'Sig585':>7} {'Chg245':>8} {'Chg585':>8}"
)
print("-" * 82)
for m in key_models:
    r245 = trends_df[(trends_df.scenario == "ssp245") & (trends_df.model == m)]
    r585 = trends_df[(trends_df.scenario == "ssp585") & (trends_df.model == m)]
    if r245.empty:
        continue
    s245 = r245["slope_m3s_decade"].values[0]
    s585 = r585["slope_m3s_decade"].values[0] if not r585.empty else float("nan")
    g245 = "✓" if r245["significant"].values[0] else "✗"
    g585 = "✓" if not r585.empty and r585["significant"].values[0] else "✗"
    c245 = r245["pct_change"].values[0]
    c585 = r585["pct_change"].values[0] if not r585.empty else float("nan")
    print(
        f"{m:<22} {s245:>12.4f} {s585:>12.4f} {g245:>7} {g585:>7} "
        f"{c245:>7.1f}% {c585:>7.1f}%"
    )


# ── Figures ─────────────────────────────────────────────────────────────────────
print("\nGenerating figures ...")

COLORS = {
    "LSTM": "#3b9eff",
    "CNN-LSTM": "#00b4a0",
    "Transformer": "#00d4ff",
    "PI-LSTM": "#f4a261",
    "PI-Transformer": "#e76f51",
    "ensemble_pure": "#8aafc4",
    "ensemble_pi": "#f4a261",
    "ensemble_all": "#ffffff",
}


def style_ax(ax):
    ax.set_facecolor("#0d1825")
    ax.tick_params(colors="#4a6a82")
    ax.spines[:].set_color("#1e3448")
    ax.grid(alpha=0.08)


def legend(ax):
    ax.legend(
        facecolor="#0d1825", edgecolor="#1e3448", labelcolor="#8aafc4", fontsize=8
    )


# Figure 1 — annual discharge projections
fig1, axes = plt.subplots(2, 1, figsize=(18, 14))
fig1.patch.set_facecolor("#080f1a")

for ax, (scen, label) in zip(
    axes,
    [
        ("ssp245", "SSP2-4.5 — Moderate Emissions"),
        ("ssp585", "SSP5-8.5 — High Emissions"),
    ],
):
    style_ax(ax)
    df = results[scen].copy()
    df["year"] = df["date"].dt.year
    ann = df.groupby("year").mean(numeric_only=True)

    for name in ["LSTM", "CNN-LSTM", "Transformer", "PI-LSTM", "PI-Transformer"]:
        col = f"Q_{name}"
        if col not in ann or ann[col].isna().all():
            continue
        sm = ann[col].rolling(5, center=True, min_periods=1).mean()
        ax.plot(
            ann.index,
            sm,
            color=COLORS[name],
            linewidth=1.5,
            alpha=0.75,
            linestyle="--" if "PI" in name else "-",
            label=name,
        )

    if "Q_ensemble_all" in ann:
        sm_e = ann["Q_ensemble_all"].rolling(5, center=True, min_periods=1).mean()
        sm_s = ann["Q_ensemble_std"].rolling(5, center=True, min_periods=1).mean()
        ax.plot(
            ann.index,
            sm_e,
            color="#ffffff",
            linewidth=2.5,
            label="All-model ensemble",
            zorder=5,
        )
        ax.fill_between(
            ann.index, sm_e - sm_s, sm_e + sm_s, alpha=0.08, color="#ffffff"
        )
        base = ann.loc[ann.index <= 2025, "Q_ensemble_all"].mean()
        ax.axhline(
            base,
            color="#8aafc4",
            linewidth=1,
            linestyle=":",
            alpha=0.5,
            label=f"2015–2025 mean ({base:.3f} m³/s)",
        )

    ax.axvline(
        2025,
        color="#4a6a82",
        linewidth=1.2,
        linestyle=":",
        alpha=0.8,
        label="Present (2025)",
    )
    ax.set_title(label, color="#e8f4f8", fontsize=12)
    ax.set_xlabel("Year", color="#8aafc4")
    ax.set_ylabel("Annual Mean Discharge (m³/s)", color="#8aafc4")
    legend(ax)

fig1.suptitle(
    "Nahr Ibrahim — Projected Discharge 2015–2100\n"
    "Pure AI vs Physics-Informed | MPI-ESM1-2-HR · NEX-GDDP-CMIP6",
    color="#e8f4f8",
    fontsize=13,
    y=1.01,
    fontfamily="monospace",
)
plt.tight_layout()
plt.savefig(
    FIG_DIR / "scenario_annual_discharge.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="#080f1a",
)
plt.show()
print("  fig 1 done")

# Figure 2 — pure AI vs PI
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))
fig2.patch.set_facecolor("#080f1a")

for ax, scen in zip(axes2, SCENARIOS):
    style_ax(ax)
    df = results[scen].copy()
    df["year"] = df["date"].dt.year
    ann = df.groupby("year").mean(numeric_only=True)
    for col, color, lbl, ls in [
        ("Q_Transformer", "#00d4ff", "Transformer", "-"),
        ("Q_PI-Transformer", "#e76f51", "PI-Transformer", "--"),
        ("Q_LSTM", "#3b9eff", "LSTM", "-"),
        ("Q_PI-LSTM", "#f4a261", "PI-LSTM", "--"),
    ]:
        if col not in ann or ann[col].isna().all():
            continue
        sm = ann[col].rolling(10, center=True, min_periods=3).mean()
        ax.plot(
            ann.index, sm, color=color, linewidth=2, linestyle=ls, label=lbl, alpha=0.9
        )
    ax.axvline(2025, color="#4a6a82", linewidth=1, linestyle=":")
    ax.set_title(f"Pure AI vs PI — {scen.upper()}", color="#e8f4f8", fontsize=11)
    ax.set_xlabel("Year", color="#8aafc4")
    ax.set_ylabel("Discharge (m³/s)", color="#8aafc4")
    legend(ax)

fig2.suptitle(
    "Pure AI vs Physics-Informed — Discharge Projections",
    color="#e8f4f8",
    fontsize=13,
    y=1.02,
    fontfamily="monospace",
)
plt.tight_layout()
plt.savefig(
    FIG_DIR / "scenario_pure_vs_pi.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="#080f1a",
)
plt.show()
print("  fig 2 done")

# Figure 3 — seasonal shift
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 7))
fig3.patch.set_facecolor("#080f1a")
mlabels = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

for ax, scen in zip(axes3, SCENARIOS):
    style_ax(ax)
    df = results[scen].copy()
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    col = next(
        (
            c
            for c in ["Q_ensemble_all", "Q_Transformer", "Q_LSTM"]
            if c in df and not df[c].isna().all()
        ),
        None,
    )
    if not col:
        continue
    for lbl, mask, color in [
        ("2015–2040", (df.year >= 2015) & (df.year <= 2040), "#3b9eff"),
        ("2041–2070", (df.year >= 2041) & (df.year <= 2070), "#f4a261"),
        ("2071–2100", (df.year >= 2071) & (df.year <= 2100), "#e76f51"),
    ]:
        if not mask.any():
            continue
        mon = df[mask].groupby("month")[col].mean()
        ax.plot(
            mon.index,
            mon.values,
            color=color,
            linewidth=2.5,
            marker="o",
            markersize=6,
            label=lbl,
        )
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(mlabels, color="#8aafc4", fontsize=9)
    ax.set_title(f"Seasonal shift — {scen.upper()}", color="#e8f4f8", fontsize=11)
    ax.set_ylabel("Monthly Mean Discharge (m³/s)", color="#8aafc4")
    legend(ax)

fig3.suptitle(
    "Nahr Ibrahim — Seasonal Discharge Shift by Period",
    color="#e8f4f8",
    fontsize=13,
    y=1.02,
    fontfamily="monospace",
)
plt.tight_layout()
plt.savefig(
    FIG_DIR / "scenario_seasonal_shift.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="#080f1a",
)
plt.show()
print("  fig 3 done")

# Figure 4 — trend bar chart
fig4, ax4 = plt.subplots(figsize=(16, 6))
fig4.patch.set_facecolor("#080f1a")
style_ax(ax4)
plot_m = [m for m in key_models if (trends_df["model"] == m).any()]
x, w = np.arange(len(plot_m)), 0.35

for i, (scen, color) in enumerate({"ssp245": "#3b9eff", "ssp585": "#e76f51"}.items()):
    slopes = []
    for m in plot_m:
        row = trends_df[(trends_df.scenario == scen) & (trends_df.model == m)]
        slopes.append(row["slope_m3s_decade"].values[0] if not row.empty else 0)
    bars = ax4.bar(x + i * w, slopes, w, label=scen.upper(), color=color, alpha=0.8)
    for bar, val in zip(bars, slopes):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0003 * np.sign(val + 1e-9),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            color="#e8f4f8",
            fontsize=7,
            fontfamily="monospace",
        )

ax4.axhline(0, color="#8aafc4", linewidth=1)
ax4.set_xticks(x + w / 2)
ax4.set_xticklabels(plot_m, color="#8aafc4", fontsize=8, rotation=15)
ax4.set_ylabel("Discharge trend (m³/s per decade)", color="#8aafc4")
ax4.set_title("Projected discharge trends 2015–2100", color="#e8f4f8", fontsize=12)
ax4.legend(facecolor="#0d1825", edgecolor="#1e3448", labelcolor="#8aafc4")
plt.tight_layout()
plt.savefig(
    FIG_DIR / "scenario_trend_summary.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="#080f1a",
)
plt.show()
print("  fig 4 done")

# Figure 5 — climate forcing
fig5, axes5 = plt.subplots(2, 2, figsize=(18, 10))
fig5.patch.set_facecolor("#080f1a")

for ci, scen in enumerate(SCENARIOS):
    df = results[scen].copy()
    df["year"] = df["date"].dt.year
    ann = df.groupby("year").mean(numeric_only=True)
    for ri, (var, ylabel, color) in enumerate(
        [
            ("precip_mm_day", "Precipitation (mm/day)", "#3b9eff"),
            ("temp_mean_c", "Temperature (°C)", "#f4a261"),
        ]
    ):
        ax = axes5[ri, ci]
        style_ax(ax)
        sm = ann[var].rolling(10, center=True, min_periods=3).mean()
        if ri == 0:
            ax.bar(ann.index, ann[var], color=color, alpha=0.25, width=0.8)
        else:
            ax.plot(ann.index, ann[var], color=color, alpha=0.25, linewidth=0.8)
        ax.plot(ann.index, sm, color=color, linewidth=2, label="10-yr smooth")
        ax.set_title(f"{ylabel} — {scen.upper()}", color="#e8f4f8", fontsize=10)
        ax.set_ylabel(ylabel, color="#8aafc4")
        ax.legend(
            facecolor="#0d1825", edgecolor="#1e3448", labelcolor="#8aafc4", fontsize=8
        )

fig5.suptitle(
    "CMIP6 Climate Forcing — MPI-ESM1-2-HR",
    color="#e8f4f8",
    fontsize=13,
    y=1.01,
    fontfamily="monospace",
)
plt.tight_layout()
plt.savefig(
    FIG_DIR / "scenario_climate_forcing.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="#080f1a",
)
plt.show()
print("  fig 5 done")

# Figure 6 — PI vs pure divergence
fig6, ax6 = plt.subplots(figsize=(14, 7))
fig6.patch.set_facecolor("#080f1a")
style_ax(ax6)

for scen, ls in [("ssp245", "-"), ("ssp585", "--")]:
    df = results[scen].copy()
    df["year"] = df["date"].dt.year
    ann = df.groupby("year").mean(numeric_only=True)
    for col, color, lbl in [
        ("Q_ensemble_pure", "#00d4ff", f"Pure AI ({scen})"),
        ("Q_ensemble_pi", "#e76f51", f"PI ({scen})"),
    ]:
        if col not in ann or ann[col].isna().all():
            continue
        sm = ann[col].rolling(10, center=True, min_periods=3).mean()
        ax6.plot(
            ann.index, sm, color=color, linewidth=2, linestyle=ls, label=lbl, alpha=0.85
        )

ax6.axvline(2025, color="#4a6a82", linewidth=1.2, linestyle=":")
ax6.set_xlabel("Year", color="#8aafc4")
ax6.set_ylabel("Annual Mean Discharge (m³/s)", color="#8aafc4")
ax6.set_title(
    "Pure AI vs PI ensemble — long-term divergence", color="#e8f4f8", fontsize=12
)
ax6.legend(
    facecolor="#0d1825", edgecolor="#1e3448", labelcolor="#8aafc4", fontsize=9, ncol=2
)
plt.tight_layout()
plt.savefig(
    FIG_DIR / "scenario_pi_vs_pure_divergence.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="#080f1a",
)
plt.show()
print("  fig 6 done")


# ── Summary ─────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"Climate scenario summary — {len(models)} models, {N} features")
print(f"{'=' * 60}")

for scen in SCENARIOS:
    df = results[scen].copy()
    df["year"] = df["date"].dt.year
    col = next(
        (
            c
            for c in ["Q_ensemble_all", "Q_Transformer", "Q_LSTM"]
            if c in df and not df[c].isna().all()
        ),
        None,
    )
    if not col:
        continue
    early = df[df.year <= 2040][col].mean()
    late = df[df.year >= 2075][col].mean()
    chg = (late - early) / early * 100
    print(f"\n  {scen.upper()}:")
    print(f"    2015–2040 mean : {early:.4f} m³/s")
    print(f"    2075–2100 mean : {late:.4f} m³/s")
    print(
        f"    Change         : {chg:+.1f}%  "
        f"({'declining' if chg < 0 else 'increasing'})"
    )

print(f"\n  PI vs pure ensemble (SSP5-8.5 slope):")
for ens in ["ensemble_pure", "ensemble_pi"]:
    row = trends_df[(trends_df.scenario == "ssp585") & (trends_df.model == ens)]
    if row.empty:
        continue
    print(f"    {ens:<20}: {row['slope_m3s_decade'].values[0]:.4f} m³/s/decade")

print(f"\n  Saved:")
for s in SCENARIOS:
    print(f"    results/scenarios/discharge_{s}_daily.csv")
print(f"    results/metrics/climate_scenario_trends_full.csv")
for f in [
    "scenario_annual_discharge",
    "scenario_pure_vs_pi",
    "scenario_seasonal_shift",
    "scenario_trend_summary",
    "scenario_climate_forcing",
    "scenario_pi_vs_pure_divergence",
]:
    print(f"    results/figures/{f}.png")
print(f"{'=' * 60}")
