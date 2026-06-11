import os
import sys
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(
    os.environ.get("WATERSHED_ROOT", "C:/Users/marck/Downloads/nahr_ibrahim_watershed")
)

# ── Auto-create output file ───────────────────────────────────────────────
OUT_PATH = ROOT / "results" / "master_report.txt"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
sys.stdout = open(OUT_PATH, "w", encoding="utf-8")


def header(title, level="="):
    print(f"\n{level * 70}")
    print(f"  {title}")
    print(f"{level * 70}\n")


def subheader(title):
    print(f"\n  {title}")
    print(f"  {'-' * (len(title) + 2)}")


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATASET
# ═══════════════════════════════════════════════════════════════════════════
header("1. DATASET & PREPROCESSING")

try:
    master = pd.read_csv(
        ROOT / "data/master/nahr_ibrahim_master_model.csv", parse_dates=["date"]
    )
    feat_cols = [
        c
        for c in master.columns
        if c not in ["date", "discharge_m3s", "discharge_m3s_raw"]
    ]
    print(f"Master: {len(master)} rows, {len(feat_cols)} features")
    print(f"  Range: {master['date'].min().date()} → {master['date'].max().date()}")
    target_col = (
        "discharge_m3s_raw"
        if "discharge_m3s_raw" in master.columns
        else "discharge_m3s"
    )
    print(
        f"  Target mean: {master[target_col].mean():.3f} m³/s, P95: {master[target_col].quantile(0.95):.3f}, P05: {master[target_col].quantile(0.05):.3f}"
    )
except Exception as e:
    print(f"[SKIP] {e}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. LAND-SURFACE
# ═══════════════════════════════════════════════════════════════════════════
header("2. LAND-SURFACE CALIBRATION")

try:
    with open(ROOT / "models/trained/landsurface_params.json") as f:
        ls = json.load(f)
    print(
        f"Snow: T_snow={ls['snow_model']['T_snow_C']:+.2f}°C, melt={ls['snow_model']['melt_factor_mm_per_C_per_day']:.2f}"
    )
    print(
        f"Bucket: FC={ls['bucket_model']['field_capacity_mm']:.1f}, WP={ls['bucket_model']['wilting_point_mm']:.1f}, drain={ls['bucket_model']['drainage_rate_per_day']:.4f}"
    )
except Exception as e:
    print(f"[SKIP] {e}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. METRICS (parse model/horizon/seed from filename)
# ═══════════════════════════════════════════════════════════════════════════
header("3. PERFORMANCE METRICS")

metrics_dir = ROOT / "results/metrics"
all_metrics = []


def parse_metrics_filename(fname):
    stem = Path(fname).stem

    m_ens = re.search(r"^(tcn|lstm)_ensemble_metrics_h(\d+)$", stem)
    if m_ens:
        model = m_ens.group(1) + "_ensemble"
        h = int(m_ens.group(2))
        seed = 0  # ensemble has no single seed
        return model, h, seed

    # Standard pattern
    m_std = re.search(r"_h(\d+)_s(\d+)", stem)
    if not m_std:
        return None, None, None
    h = int(m_std.group(1))
    seed = int(m_std.group(2))

    # Model is everything before "_metrics_"
    parts = stem.split("_metrics_")
    model = parts[0] if len(parts) > 0 else "unknown"
    return model, h, seed


if metrics_dir.exists():
    files = sorted(metrics_dir.glob("*_metrics_*.csv"))
    print(f"Found {len(files)} metrics files")

    for mf in files:
        try:
            df = pd.read_csv(mf)
            model, h, seed = parse_metrics_filename(mf.name)
            if model is None:
                print(f"[SKIP] Cannot parse filename: {mf.name}")
                continue
            df["model"] = model
            df["horizon"] = h
            df["seed"] = seed
            df["source_file"] = mf.name
            all_metrics.append(df)
        except Exception as e:
            print(f"[SKIP] {mf.name}: {e}")

    if all_metrics:
        combined = pd.concat(all_metrics, ignore_index=True)
        print(f"\nCombined: {len(combined)} rows from {len(all_metrics)} files")
        print(f"Columns: {list(combined.columns)}")

        # Map to standard names (your CSVs already use NSE, KGE, MAE, RMSE, etc.)
        # Just verify they exist
        required = ["NSE", "KGE", "MAE", "RMSE"]
        missing = [r for r in required if r not in combined.columns]
        if missing:
            print(f"[WARNING] Missing metric columns: {missing}")

        # Best seed per model × horizon
        print(f"\nBest seed per model × horizon:")
        print(
            f"  {'Model':<<12} {'H':>3} {'Seed':>6} {'NSE':>7} {'KGE':>7} {'MAE':>7} {'RMSE':>7} {'PBIAS':>7} {'Peak%':>7}"
        )
        print(f"  {'-'*65}")

        for (model, h), group in combined.groupby(["model", "horizon"]):
            best = group.loc[group["NSE"].idxmax()]
            s = int(best["seed"])
            pbias = best.get("PBIAS_%", np.nan)
            peak = best.get("Peak_Bias_%", np.nan)
            print(
                f"  {str(model):<<12} {int(h):>3} {s:>6} {best['NSE']:>7.4f} {best['KGE']:>7.4f} "
                f"{best['MAE']:>7.4f} {best['RMSE']:>7.4f} {pbias:>7.2f} {peak:>7.2f}"
            )

        # Multi-seed summary
        print(f"\nMulti-seed summary (mean ± std):")
        print(
            f"  {'Model':<<12} {'H':>3} {'NSE':>15} {'KGE':>15} {'MAE':>7} {'RMSE':>7}"
        )
        print(f"  {'-'*60}")
        for (model, h), group in combined.groupby(["model", "horizon"]):
            n = len(group)
            if n > 1:
                print(
                    f"  {str(model):<<12} {int(h):>3} "
                    f"{group['NSE'].mean():.3f}±{group['NSE'].std():.3f} "
                    f"{group['KGE'].mean():.3f}±{group['KGE'].std():.3f} "
                    f"{group['MAE'].mean():.4f} {group['RMSE'].mean():.4f} (n={n})"
                )
            else:
                print(
                    f"  {str(model):<<12} {int(h):>3} "
                    f"{group['NSE'].mean():.3f} (n=1) "
                    f"{group['KGE'].mean():.3f} (n=1) "
                    f"{group['MAE'].mean():.4f} {group['RMSE'].mean():.4f}"
                )

        # Overall ranking
        print(f"\nOverall ranking (best seed only) by horizon:")
        best_per_model = combined.loc[
            combined.groupby(["model", "horizon"])["NSE"].idxmax()
        ]
        best_per_model = best_per_model.sort_values(
            ["horizon", "NSE"], ascending=[True, False]
        )
        for _, row in best_per_model.iterrows():
            print(
                f"    h={int(row['horizon']):>2} | {str(row['model']):<<12} | NSE={row['NSE']:.4f} | KGE={row['KGE']:.4f} | seed={int(row['seed'])}"
            )
else:
    print("[SKIP] No metrics directory")

# ═══════════════════════════════════════════════════════════════════════════
# 4. SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════
header("4. PERMUTATION FEATURE IMPORTANCE")

sens_dir = ROOT / "results/sensitivity"
if sens_dir.exists():
    for sf in sorted(sens_dir.glob("sensitivity_*.csv")):
        try:
            df = pd.read_csv(sf)
            if df.empty or "feature" not in df.columns:
                continue
            print(f"\n  {sf.name}")
            if "baseline_nse" in df.columns:
                print(f"    Baseline NSE: {df['baseline_nse'].iloc[0]:.4f}")
            for _, row in df.head(5).iterrows():
                drop = row.get("mean_nse_drop", row.get("nse_drop", 0))
                std = row.get("std_nse_drop", 0)
                print(f"    {str(row['feature']):<<20} drop={drop:>8.5f} ± {std:.5f}")
        except Exception as e:
            print(f"  {sf.name}: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. PROJECTIONS
# ═══════════════════════════════════════════════════════════════════════════
header("5. CLIMATE PROJECTIONS")

proj_dir = ROOT / "data/projections"
if proj_dir.exists():
    files = list(proj_dir.glob("*__h*.csv"))
    print(f"Projection files: {len(files)}")
    for pf in sorted(files)[:15]:
        try:
            df = pd.read_csv(pf, parse_dates=["date"])
            qcol = (
                "discharge_m3s_pred"
                if "discharge_m3s_pred" in df.columns
                else df.columns[1]
            )
            print(f"  {pf.name}: {len(df)} rows, mean={df[qcol].mean():.3f} m³/s")
        except Exception as e:
            print(f"  {pf.name}: {e}")
    if len(files) > 15:
        print(f"  ... and {len(files) - 15} more")
else:
    print("[SKIP] No projections directory")

# ═══════════════════════════════════════════════════════════════════════════
# 6. ARTICLE-READY BLOCK
# ═══════════════════════════════════════════════════════════════════════════
header("6. ARTICLE-READY COPY-PASTE BLOCK", "=")

print("\n[DATASET]")
try:
    print(f"Nahr Ibrahim watershed, 329 km², 2000–2025.")
    print(f"Master CSV: {len(master)} rows, {len(feat_cols)} features.")
    print(f"Train/Val/Test chronological split: 2000–2017 / 2018–2020 / 2021–2025.")
    print(
        f"Discharge target: GloFAS-ERA5 v4, mean {master[target_col].mean():.3f} m³/s, "
        f"P95 {master[target_col].quantile(0.95):.3f}, P05 {master[target_col].quantile(0.05):.3f}."
    )
except:
    pass

print("\n[LAND SURFACE]")
try:
    print(
        f"Snow model (degree-day): T_snow = {ls['snow_model']['T_snow_C']:+.2f}°C, "
        f"melt = {ls['snow_model']['melt_factor_mm_per_C_per_day']:.2f} mm/°C/day."
    )
    print(
        f"Bucket model: FC = {ls['bucket_model']['field_capacity_mm']:.1f} mm, "
        f"WP = {ls['bucket_model']['wilting_point_mm']:.1f} mm, "
        f"drain = {ls['bucket_model']['drainage_rate_per_day']:.4f} /day, "
        f"ET_scale = {ls['bucket_model']['ET_scale']:.3f}."
    )
except:
    pass

try:
    if "combined" in locals() and not combined.empty:
        print("\n[PERFORMANCE METRICS — BEST SEED]")
        for h in [1, 3, 14]:
            h_data = combined[combined["horizon"] == h]
            if not h_data.empty:
                print(f"\nh = {h}:")
                for model in sorted(h_data["model"].unique()):
                    m_data = h_data[h_data["model"] == model]
                    best = m_data.loc[m_data["NSE"].idxmax()]
                    print(
                        f"  {str(model):<<12} NSE={best['NSE']:.4f}, KGE={best['KGE']:.4f}, "
                        f"MAE={best['MAE']:.4f}, RMSE={best['RMSE']:.4f}, "
                        f"PeakBias={best.get('Peak_Bias_%', 'N/A')}%, seed={int(best['seed'])}"
                    )

        print("\n[PERFORMANCE METRICS — MULTI-SEED SUMMARY]")
        for (model, h), group in combined.groupby(["model", "horizon"]):
            if len(group) > 1:
                print(
                    f"  {str(model):<<12} h={int(h)}  NSE={group['NSE'].mean():.3f}±{group['NSE'].std():.3f} "
                    f"KGE={group['KGE'].mean():.3f}±{group['KGE'].std():.3f} (n={len(group)})"
                )
            else:
                print(
                    f"  {str(model):<<12} h={int(h)}  NSE={group['NSE'].mean():.3f} KGE={group['KGE'].mean():.3f} (n=1)"
                )
except Exception as e:
    print(f"[SKIP article block: {e}]")

print("\n" + "=" * 70)
print("  END OF REPORT")
print("=" * 70)
sys.stdout = sys.__stdout__
print(f"Saved to: {OUT_PATH}")
