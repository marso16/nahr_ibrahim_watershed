"""
collect_metrics.py — Scan results/metrics/ and summarize all trained models.

Detects model type from filename prefix, collects NSE, KGE, MAE, RMSE,
PBIAS, Peak_Bias, Log_NSE, and prints a sorted summary table.

Usage:
    python collect_metrics.py
    python collect_metrics.py --sort nse
    python collect_metrics.py --horizon 1
    python collect_metrics.py --csv results/all_metrics.csv
"""

import os
import argparse
import pandas as pd
from pathlib import Path


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=str,
        default=os.environ.get(
            "WATERSHED_ROOT", "C:/Users/marck/Downloads/nahr_ibrahim_watershed"
        ),
    )
    p.add_argument(
        "--sort",
        type=str,
        default="nse",
        choices=["nse", "kge", "mae", "rmse", "model", "horizon"],
        help="Column to sort by",
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Filter to a specific horizon (e.g. 1, 3, 7)",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to save the full table as CSV",
    )
    return p.parse_args()


def detect_model(filename: str) -> str:
    """Infer model name from metrics CSV filename."""
    fn = filename.lower()
    if "bilstm" in fn:
        return "BiLSTM"
    if "gr4j" in fn:
        return "GR4J"
    if "hybrid" in fn:
        return "Hybrid"
    if "ulstm" in fn:
        return "UniLSTM"
    if "lstm" in fn:
        return "LSTM"
    if "tcn" in fn:
        return "TCN"
    return "Unknown"


def detect_horizon(filename: str) -> str:
    import re

    m = re.search(r"[_\-]h(\d+)", filename.lower())
    if m:
        return int(m.group(1))
    return "?"


def detect_seed(filename: str) -> str:
    import re

    m = re.search(r"[_\-]s(\d+)", filename.lower())
    if m:
        return int(m.group(1))
    return "?"


def main():
    args = get_args()
    ROOT = Path(args.root)
    MET_DIR = ROOT / "results" / "metrics"

    if not MET_DIR.exists():
        print(f"ERROR: metrics directory not found: {MET_DIR}")
        print("Run your models first to generate metrics CSVs.")
        return

    # Collect all CSV files
    files = sorted(MET_DIR.glob("*.csv"))
    if not files:
        print(f"No metrics CSVs found in {MET_DIR}")
        return

    print(f"Found {len(files)} metrics file(s) in {MET_DIR}\n")

    rows = []
    skipped = []

    for fpath in files:
        try:
            df = pd.read_csv(fpath)

            # Some files may have multiple rows (train/val/test splits)
            # Keep only Test split if available, otherwise first row
            if "split" in df.columns:
                test_rows = df[df["split"].str.lower() == "test"]
                row = test_rows.iloc[0] if len(test_rows) > 0 else df.iloc[0]
            else:
                row = df.iloc[0]

            entry = {
                "File": fpath.name,
                "Model": detect_model(fpath.name),
                "Horizon": detect_horizon(fpath.name),
                "Seed": detect_seed(fpath.name),
                "NSE": round(float(row.get("NSE", float("nan"))), 4),
                "KGE": round(float(row.get("KGE", float("nan"))), 4),
                "logNSE": round(float(row.get("Log_NSE", float("nan"))), 4),
                "MAE": round(float(row.get("MAE", float("nan"))), 4),
                "RMSE": round(float(row.get("RMSE", float("nan"))), 4),
                "PBIAS%": round(float(row.get("PBIAS_%", float("nan"))), 2),
                "PeakBias%": round(float(row.get("Peak_Bias_%", float("nan"))), 2),
            }
            rows.append(entry)

        except Exception as e:
            skipped.append((fpath.name, str(e)))

    if not rows:
        print("No valid metrics found.")
        return

    result = pd.DataFrame(rows)

    # Filter by horizon if requested
    if args.horizon is not None:
        result = result[result["Horizon"] == args.horizon]
        if result.empty:
            print(f"No results found for horizon={args.horizon}")
            return

    # Sort
    sort_map = {
        "nse": ("NSE", False),
        "kge": ("KGE", False),
        "mae": ("MAE", True),
        "rmse": ("RMSE", True),
        "model": ("Model", True),
        "horizon": ("Horizon", True),
    }
    sort_col, ascending = sort_map[args.sort]
    result = result.sort_values(sort_col, ascending=ascending).reset_index(drop=True)

    # ── Print summary ────────────────────────────────────────────────────────
    print(f"{'='*110}")
    print(
        f"{'Model':<14} {'H':>2} {'Seed':>6}  {'NSE':>7} {'KGE':>7} "
        f"{'logNSE':>7} {'MAE':>7} {'RMSE':>7} {'PBIAS%':>8} {'Peak%':>7}  File"
    )
    print(f"{'='*110}")

    for _, r in result.iterrows():
        horizon = f"{r['Horizon']}" if r["Horizon"] != "?" else "?"
        seed = f"{r['Seed']}" if r["Seed"] != "?" else "?"

        # Flag bad results
        nse_str = f"{r['NSE']:>7.4f}"
        flag = ""
        if r["NSE"] < 0:
            flag = " ← NEGATIVE"
        elif r["NSE"] < 0.5:
            flag = " ← POOR"

        print(
            f"{r['Model']:<14} {horizon:>2} {seed:>6}  "
            f"{nse_str} {r['KGE']:>7.4f} "
            f"{r['logNSE']:>7.4f} {r['MAE']:>7.4f} {r['RMSE']:>7.4f} "
            f"{r['PBIAS%']:>8.2f} {r['PeakBias%']:>7.2f}  "
            f"{r['File']}{flag}"
        )

    print(f"{'='*110}")
    print(f"Total: {len(result)} result(s)")

    # ── Per-model summary (mean ± std across seeds) ───────────────────────────
    multi_seed = result.groupby(["Model", "Horizon"]).filter(lambda x: len(x) > 1)
    if not multi_seed.empty:
        print(f"\n{'─'*70}")
        print("Multi-seed summary (mean ± std):")
        print(f"{'─'*70}")
        summary = multi_seed.groupby(["Model", "Horizon"])[
            ["NSE", "KGE", "MAE", "RMSE"]
        ].agg(["mean", "std"])
        for (model, horizon), row in summary.iterrows():
            nse_mean = row[("NSE", "mean")]
            nse_std = row[("NSE", "std")]
            kge_mean = row[("KGE", "mean")]
            kge_std = row[("KGE", "std")]
            mae_mean = row[("MAE", "mean")]
            rmse_mean = row[("RMSE", "mean")]
            print(
                f"  {model:<12} h={horizon}  "
                f"NSE={nse_mean:.3f}±{nse_std:.3f}  "
                f"KGE={kge_mean:.3f}±{kge_std:.3f}  "
                f"MAE={mae_mean:.3f}  RMSE={rmse_mean:.3f}"
            )
        print(f"{'─'*70}")

    # ── Best per model (across seeds) ─────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("Best result per model (by NSE):")
    print(f"{'─'*70}")
    best = result.loc[result.groupby(["Model", "Horizon"])["NSE"].idxmax()]
    for _, r in best.sort_values("NSE", ascending=False).iterrows():
        print(
            f"  {r['Model']:<12} h={r['Horizon']}  "
            f"NSE={r['NSE']:.4f}  KGE={r['KGE']:.4f}  "
            f"seed={r['Seed']}  ({r['File']})"
        )
    print(f"{'─'*70}")

    # ── Skipped files ─────────────────────────────────────────────────────────
    if skipped:
        print(f"\nSkipped {len(skipped)} file(s) due to errors:")
        for fname, err in skipped:
            print(f"  {fname}: {err}")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out, index=False)
        print(f"\nSaved full table → {out}")


if __name__ == "__main__":
    main()
