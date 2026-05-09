import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path("C:/Users/marck/Downloads/nahr_ibrahim_watershed")
RESULTS = ROOT / "results"
MET_DIR = RESULTS / "metrics"
SCEN_DIR = RESULTS / "scenarios"
SPLIT_DIR = ROOT / "data" / "splits"
MASTER_DIR = ROOT / "data" / "master"
PRED_DIR = RESULTS / "predictions"

output_lines = []


def section(title):
    output_lines.append("\n" + "=" * 70)
    output_lines.append(f"  {title}")
    output_lines.append("=" * 70)


def sub(title):
    output_lines.append(f"\n  -- {title} --")


def line(text=""):
    output_lines.append(f"  {text}")


# =============================================================================
output_lines.append("=" * 70)
output_lines.append("  NAHR IBRAHIM WATERSHED — AI HYDROLOGY THESIS")
output_lines.append("  Complete Results Export")
output_lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
output_lines.append("=" * 70)

# =============================================================================
# 1. MASTER DATASET SUMMARY
# =============================================================================
section("1. MASTER DATASET SUMMARY")

master_path = MASTER_DIR / "nahr_ibrahim_master_full.csv"
if master_path.exists():
    df = pd.read_csv(master_path, parse_dates=["date"])
    line(f"Date range     : {df.date.min().date()} → {df.date.max().date()}")
    line(f"Total days     : {len(df)}")
    line(f"Total features : 16")
    line()
    line("Missing values per feature:")
    check_cols = [
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
        "discharge_m3s",
    ]
    for col in check_cols:
        if col in df.columns:
            n = df[col].isna().sum()
            pct = n / len(df) * 100
            status = "OK" if pct == 0 else "WARN" if pct < 5 else "FAIL"
            line(f"  [{status}] {col:<24}: {n:>5} missing ({pct:5.1f}%)")

    line()
    line("Feature statistics:")
    for col in check_cols:
        if col in df.columns and df[col].notna().sum() > 0:
            line(
                f"  {col:<24}: min={df[col].min():.3f}  max={df[col].max():.3f}  mean={df[col].mean():.3f}  std={df[col].std():.3f}"
            )
else:
    line("Master dataset not found.")

# =============================================================================
# 2. DATA SPLITS
# =============================================================================
section("2. DATA SPLITS")

for split in ["train", "val", "test"]:
    p = SPLIT_DIR / f"{split}_raw.csv"
    if p.exists():
        s = pd.read_csv(p, parse_dates=["date"])
        line(
            f"{split.upper():<12}: {s.date.min().date()} → {s.date.max().date()}  |  {len(s)} days  |  Q mean={s['discharge_m3s'].mean():.3f} m³/s"
        )
    else:
        line(f"{split.upper():<12}: not found")

scaler_path = SPLIT_DIR / "scaler_params.csv"
if scaler_path.exists():
    scaler = pd.read_csv(scaler_path, index_col=0)
    line()
    line("Scaler params (min-max, fitted on training set):")
    for idx in scaler.index:
        line(
            f"  {idx:<24}: min={scaler.loc[idx, 'min']:.4f}  max={scaler.loc[idx, 'max']:.4f}"
        )

# =============================================================================
# 3. MODEL BENCHMARK RESULTS
# =============================================================================
section("3. MODEL BENCHMARK RESULTS — TEST PERIOD 2021-2025")

metrics_files = list(MET_DIR.glob("*metrics*.csv")) if MET_DIR.exists() else []
if metrics_files:
    for mf in sorted(metrics_files):
        if "scenario" in mf.name or "trend" in mf.name or "stress" in mf.name:
            continue
        sub(mf.stem)
        try:
            mdf = pd.read_csv(mf)
            line(mdf.to_string(index=False))
        except Exception as e:
            line(f"Could not read: {e}")
else:
    line("No metrics CSV files found in results/metrics/")
    line("Adding hardcoded benchmark results from training session:")
    line()
    line(
        f"  {'Model':<18} {'Type':<10} {'NSE':>8} {'KGE':>8} {'RMSE':>8} {'Peak Bias':>12} {'PBIAS':>8}"
    )
    line(f"  {'-' * 70}")
    results = [
        ("Transformer", "Pure AI", 0.680, 0.671, 0.254, -37.0, -4.1),
        ("PI-Transformer", "Hybrid", 0.586, 0.730, 0.271, -29.7, +9.4),
        ("CNN-LSTM", "Pure AI", 0.559, 0.656, 0.298, -34.1, +13.0),
        ("LSTM", "Pure AI", 0.515, 0.423, 0.312, -51.7, +0.1),
        ("PI-LSTM", "Hybrid", 0.512, 0.486, 0.312, -47.2, +2.6),
    ]
    for r in results:
        line(
            f"  {r[0]:<18} {r[1]:<10} {r[2]:>8.3f} {r[3]:>8.3f} {r[4]:>8.3f} {r[5]:>11.1f}% {r[6]:>7.1f}%"
        )

    line()
    line("Feature impact (12 vs 16 features):")
    line(
        f"  {'Model':<18} {'NSE-12':>8} {'NSE-16':>8} {'ΔNSE':>8} {'KGE-12':>8} {'KGE-16':>8} {'ΔKGE':>8}"
    )
    line(f"  {'-' * 66}")
    impacts = [
        ("Transformer", 0.603, 0.680, 0.077, 0.632, 0.671, 0.039),
        ("PI-Transformer", 0.528, 0.586, 0.058, 0.574, 0.730, 0.156),
        ("LSTM", 0.518, 0.515, -0.003, 0.503, 0.423, -0.080),
    ]
    for r in impacts:
        line(
            f"  {r[0]:<18} {r[1]:>8.3f} {r[2]:>8.3f} {r[3]:>+8.3f} {r[4]:>8.3f} {r[5]:>8.3f} {r[6]:>+8.3f}"
        )

# =============================================================================
# 4. CLIMATE SCENARIO TRENDS
# =============================================================================
section("4. CLIMATE SCENARIO PROJECTIONS — MPI-ESM1-2-HR")

trends_path = MET_DIR / "climate_scenario_trends_full.csv"
if trends_path.exists():
    trends = pd.read_csv(trends_path)
    line(trends.to_string(index=False))
else:
    line("climate_scenario_trends_full.csv not found.")
    line("Adding results from previous run:")
    line()
    line(
        f"  {'Model':<22} {'Scenario':<10} {'Slope/decade':>14} {'Mean 2015-40':>14} {'Mean 2075-00':>14} {'%Change':>9} {'Sig':>5}"
    )
    line(f"  {'-' * 90}")
    climate = [
        ("Transformer", "ssp245", -0.0028, 0.851, 0.826, -3.3, True),
        ("Transformer", "ssp585", -0.0072, 0.851, 0.793, -6.8, True),
        ("PI-Transformer", "ssp245", -0.0034, 0.849, 0.819, -3.5, True),
        ("PI-Transformer", "ssp585", -0.0096, 0.849, 0.778, -8.5, True),
        ("ensemble_all", "ssp245", -0.0031, 0.851, 0.823, -3.3, True),
        ("ensemble_all", "ssp585", -0.0080, 0.851, 0.785, -7.8, True),
        ("ensemble_pi", "ssp585", -0.0096, 0.849, 0.776, -8.5, True),
        ("ensemble_pure", "ssp585", -0.0065, 0.852, 0.796, -6.6, True),
    ]
    for r in climate:
        sig = "Yes" if r[6] else "No"
        line(
            f"  {r[0]:<22} {r[1]:<10} {r[2]:>14.4f} {r[3]:>14.4f} {r[4]:>14.4f} {r[5]:>8.1f}% {sig:>5}"
        )

    line()
    line("Key finding: PI ensemble projects 48% steeper SSP5-8.5 decline")
    line("than pure AI ensemble (−0.0096 vs −0.0065 m³/s/decade)")

# =============================================================================
# 5. SCENARIO DAILY FILES SUMMARY
# =============================================================================
section("5. CLIMATE SCENARIO DAILY OUTPUT SUMMARY")

for scen in ["ssp245", "ssp585"]:
    p = SCEN_DIR / f"discharge_{scen}_daily.csv"
    if p.exists():
        df = pd.read_csv(p, parse_dates=["date"])
        q_cols = [c for c in df.columns if c.startswith("Q_")]
        sub(f"{scen.upper()} — {len(df)} days")
        line(f"  Period : {df.date.min().date()} → {df.date.max().date()}")
        for col in q_cols:
            if df[col].notna().any():
                early = df[df.date.dt.year <= 2040][col].mean()
                late = df[df.date.dt.year >= 2075][col].mean()
                chg = (late - early) / early * 100 if early > 0 else 0
                line(
                    f"  {col.replace('Q_', ''):<20}: mean={df[col].mean():.4f}  early={early:.4f}  late={late:.4f}  change={chg:+.1f}%"
                )
    else:
        line(f"{scen}: discharge_{scen}_daily.csv not found")

# =============================================================================
# 6. WATERSHED CHARACTERISTICS
# =============================================================================
section("6. WATERSHED CHARACTERISTICS")

line("Name          : Nahr Ibrahim watershed")
line("Location      : Mount Lebanon, Byblos District, Lebanon")
line("Area          : 329.8 km² (computed via QGIS UTM Zone 37N)")
line("Elevation     : 2 – 2,684 m asl (mean 1,571 m, std 547 m)")
line("Slope         : 0.12° – 62.7° (mean 14.9°, std 10.9°)")
line("CRS           : WGS84 UTM Zone 37N (EPSG:32637)")
line("Afqa spring   : 34.06789°N, 35.89354°E  (~1,200 m asl)")
line("Roueiss spring: 34.10876°N, 35.90846°E  (~1,265 m asl)")
line("River outlet  : 34.06249°N, 35.64235°E  (Mediterranean coast)")
line("Bounding box  : 34.02°–34.16°N, 35.84°–35.96°E")
line("Climate       : Mediterranean · wet winters · dry summers")
line("Precipitation : ~900–1,400 mm/yr · 25–35% as snow")
line("Data source   : HydroSHEDS (mghydro) · precision: low")

# =============================================================================
# 7. DATA SOURCES
# =============================================================================
section("7. DATA SOURCES")

sources = [
    ("Precipitation", "CHIRPS v2.0", "Google Earth Engine", "0.05° daily", "2000–2025"),
    ("Temperature", "MERRA-2 T2M", "NASA Giovanni", "0.5° hourly", "2000–2025"),
    ("SWE", "GLDAS Noah v2.1", "NASA Giovanni", "0.25° 3-hourly", "2000–2025"),
    (
        "Soil moisture",
        "GLDAS Noah SoilMoi",
        "NASA Giovanni",
        "0.25° 3-hourly",
        "2000–2025",
    ),
    ("Snow cover", "MODIS MOD10A1.061", "NASA AppEEARS", "500m daily", "2000–2025"),
    ("Discharge", "GloFAS ERA5 v4.0", "Copernicus CDS", "0.05° daily", "2000–2025"),
    (
        "Climate proj.",
        "NEX-GDDP-CMIP6",
        "Google Earth Engine",
        "0.25° daily",
        "2015–2100",
    ),
]
line(f"  {'Variable':<16} {'Product':<24} {'Source':<22} {'Resolution':<18} {'Period'}")
line(f"  {'-' * 90}")
for s in sources:
    line(f"  {s[0]:<16} {s[1]:<24} {s[2]:<22} {s[3]:<18} {s[4]}")

# =============================================================================
# 8. MODEL ARCHITECTURES
# =============================================================================
section("8. MODEL ARCHITECTURES")

models = [
    ("LSTM", "Pure AI", "2-layer stacked LSTM", "125K", "MSE", "Train 2000-2017"),
    ("CNN-LSTM", "Pure AI", "Conv1D + 2-layer LSTM", "143K", "MSE", "Train 2000-2017"),
    (
        "Transformer",
        "Pure AI",
        "3 encoder blocks d=64 4-head attn",
        "108K",
        "MSE",
        "Train 2000-2017",
    ),
    (
        "PI-LSTM",
        "Hybrid",
        "LSTM + water balance penalty",
        "125K",
        "MSE+0.05·WB²",
        "Train 2000-2017",
    ),
    (
        "PI-Transformer",
        "Hybrid",
        "Transformer + water balance penalty",
        "108K",
        "MSE+0.05·WB²",
        "Train 2000-2017",
    ),
]
line(
    f"  {'Model':<18} {'Type':<10} {'Architecture':<38} {'Params':<8} {'Loss':<16} {'Period'}"
)
line(f"  {'-' * 100}")
for m in models:
    line(f"  {m[0]:<18} {m[1]:<10} {m[2]:<38} {m[3]:<8} {m[4]:<16} {m[5]}")

line()
line("Training config:")
line("  Optimizer     : Adam  lr=0.001  gradient clip norm=1.0")
line("  Early stopping: patience=20  monitor=val_MSE")
line("  Lookback      : 30 days")
line("  Input shape   : (samples, 30, 16)")
line("  Batch size    : 32")
line("  Physics loss  : L = MSE + 0.05 * E[(P - ET - Q - dS)^2]")

# =============================================================================
# 9. VALIDATION RESULTS
# =============================================================================
section("9. INPUT DATA VALIDATION")

line("MERRA-2 temperature vs Meteostat WMO 40100 (Beirut Airport):")
line("  Daily  Pearson r : 0.975")
line("  Monthly Pearson r: 0.994")
line("  Mean bias        : -2.4 °C  (explained by 871 m elevation difference)")
line()
line("CHIRPS precipitation vs Beirut Airport:")
line("  Daily  Pearson r : 0.513")
line("  Monthly Pearson r: 0.653")
line("  PBIAS            : -6.2%")
line("  Note: orographic effect expected — watershed at higher elevation")

# =============================================================================
# WRITE OUTPUT FILE
# =============================================================================
out_path = RESULTS / "results.txt"
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print(f"\nExported → {out_path}")
print(f"Lines    : {len(output_lines)}")
