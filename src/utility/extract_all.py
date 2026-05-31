"""
Consolidate all LSTM experiment metrics into a single comparison table.

Scans results/metrics/ for files matching lstm_metrics_*.csv and combines them.
Output: results/metrics/all_experiments.csv (and prints to stdout).
"""

import os
import re
import pandas as pd
from pathlib import Path

ROOT = Path(
    os.environ.get("WATERSHED_ROOT", "C:/Users/marck/Downloads/nahr_ibrahim_watershed")
)
MET_DIR = ROOT / "results" / "metrics"

# ── Find all LSTM metric files ───────────────────────────────────────────────
files = sorted(MET_DIR.glob("*_metrics_*.csv"))

if not files:
    raise SystemExit(f"No lstm_metrics_*.csv files found in {MET_DIR}")

print(f"Found {len(files)} experiment file(s) in {MET_DIR}\n")

# ── Load each and attach run_tag ─────────────────────────────────────────────
rows = []
for f in files:
    # Strip prefix and suffix: lstm_metrics_<tag>.csv → <tag>
    tag = re.sub(r"^(lstm|mlp|gru)_metrics_", "", f.stem)
    df = pd.read_csv(f)
    if df.empty:
        print(f"  [skip] {f.name} is empty")
        continue
    # If file has more than one row, take the last (typically the final test eval)
    row = df.iloc[-1].to_dict()
    row["run_tag"] = tag

    # Try to extract horizon from tag (e.g., "strategy_a_h7" → 7, "h1" → 1)
    h_match = re.search(r"h(\d+)", tag)
    row["horizon"] = int(h_match.group(1)) if h_match else None

    # Mark experiment family (everything before the _h{N} part)
    family = re.sub(r"_?h\d+$", "", tag) or "default"
    row["family"] = family

    rows.append(row)

if not rows:
    raise SystemExit("No valid metric rows found.")

df_all = pd.DataFrame(rows)

# ── Reorder columns: identifiers first, then key metrics, then the rest ──────
preferred_order = [
    "run_tag",
    "family",
    "horizon",
    "NSE",
    "KGE",
    "Log_NSE",
    "MAE",
    "RMSE",
    "PBIAS_%",
    "Peak_Bias_%",
    "Peak_MAE",
    "Peak_RMSE",
    "KGE_r",
    "KGE_alpha",
    "KGE_beta",
    "R2",
]
ordered = [c for c in preferred_order if c in df_all.columns]
ordered += [c for c in df_all.columns if c not in ordered]
df_all = df_all[ordered]

# ── Sort: by family, then by horizon ─────────────────────────────────────────
df_all = df_all.sort_values(
    by=["family", "horizon"],
    na_position="last",
).reset_index(drop=True)

# ── Save ─────────────────────────────────────────────────────────────────────
out_path = MET_DIR / "all_experiments.csv"
df_all.to_csv(out_path, index=False)

# ── Pretty-print headline columns ────────────────────────────────────────────
headline = ["run_tag", "horizon", "NSE", "KGE", "Log_NSE", "MAE", "Peak_Bias_%"]
headline = [c for c in headline if c in df_all.columns]

print(df_all[headline].to_string(index=False))
print(f"\nSaved → {out_path}")

# ── If baselines exist, append them for context ──────────────────────────────
baselines_path = MET_DIR / "baselines_multi_horizon.csv"
if baselines_path.exists():
    print(f"\n{'=' * 60}")
    print("Baselines (persistence + climatology) at same horizons:")
    print(f"{'=' * 60}")
    b = pd.read_csv(baselines_path)
    print(b.to_string(index=False))

    # ── Build side-by-side comparison if both exist ──────────────────────────
    print(f"\n{'=' * 60}")
    print("Comparison: LSTM vs baselines (NSE only)")
    print(f"{'=' * 60}")

    # Pivot baselines to horizon-rows, model-columns
    b_wide = b.pivot_table(
        index="horizon_days", columns="model", values="NSE", aggfunc="last"
    )

    # Pull NSE per horizon per family for LSTM
    lstm_wide = df_all.dropna(subset=["horizon"]).pivot_table(
        index="horizon", columns="family", values="NSE", aggfunc="last"
    )

    comparison = b_wide.join(lstm_wide, how="outer")
    comparison.index.name = "horizon"
    print(comparison.round(4).to_string())
