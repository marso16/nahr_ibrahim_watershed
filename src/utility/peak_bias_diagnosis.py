import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(
    os.environ.get("WATERSHED_ROOT", "C:/Users/marck/Downloads/nahr_ibrahim_watershed")
)
PRED_DIR = ROOT / "results" / "predictions"
OUT_DIR = ROOT / "results" / "peak_bias"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 3, 7, 14, 30]


def find_test_predictions_file(horizon: int) -> Path | None:
    """Locate the test-set predictions CSV for a horizon."""
    candidates = [
        PRED_DIR / f"lstm_predictions_test_strategy_a_h{horizon}.csv",
        PRED_DIR / f"lstm_predictions_test_lstm_strategy_a_h{horizon}.csv",
        PRED_DIR / f"predictions_test_h{horizon}.csv",
        PRED_DIR / f"lstm_predictions_test_h{horizon}.csv",
    ]
    return next((p for p in candidates if p.exists()), None)


def metrics_at_band(obs, pred, lower_pct: float, upper_pct: float | None = None):
    """
    Compute bias, MAE, RMSE for events within a percentile band.
    lower_pct = 95 means "top 5% of flows"
    """
    threshold_low = np.percentile(obs, lower_pct)
    if upper_pct is None:
        mask = obs >= threshold_low
    else:
        threshold_high = np.percentile(obs, upper_pct)
        mask = (obs >= threshold_low) & (obs < threshold_high)

    if mask.sum() == 0:
        return None

    o = obs[mask]
    p = pred[mask]
    return {
        "n_events": int(mask.sum()),
        "obs_mean": float(o.mean()),
        "pred_mean": float(p.mean()),
        "bias_m3s": float(p.mean() - o.mean()),
        "bias_pct": float(100 * (p.mean() - o.mean()) / o.mean()),
        "mae": float(np.mean(np.abs(p - o))),
        "rmse": float(np.sqrt(np.mean((p - o) ** 2))),
        "obs_max": float(o.max()),
        "pred_max": float(p.max()),
    }


# ─── Collect diagnostics per horizon ────────────────────────────────────────
all_rows = []
horizons_with_data = []
horizon_data = {}  # horizon → (obs, pred) for plotting

for h in HORIZONS:
    pred_file = find_test_predictions_file(h)
    if pred_file is None:
        print(f"  h={h}: no predictions file found, skipping")
        continue

    df = pd.read_csv(pred_file, parse_dates=["date"])
    if "observed" not in df.columns or "predicted" not in df.columns:
        print(f"  h={h}: unexpected columns in {pred_file.name}: {df.columns.tolist()}")
        continue

    obs = df["observed"].values
    pred = df["predicted"].values
    horizons_with_data.append(h)
    horizon_data[h] = (obs, pred, df["date"].values)
    print(f"\nh={h}: {pred_file.name}")
    print(
        f"  {len(obs)} test-set days, "
        f"obs range [{obs.min():.2f}, {obs.max():.2f}] m³/s"
    )

    # Bias across percentile bands
    bands = [
        ("Top 1%", 99, None),
        ("Top 5%", 95, None),
        ("Top 10%", 90, None),
        ("Top 25%", 75, None),
        ("All flows", 0, None),
    ]
    for label, lower, upper in bands:
        m = metrics_at_band(obs, pred, lower, upper)
        if m is None:
            continue
        m["horizon"] = h
        m["band"] = label
        all_rows.append(m)
        print(
            f"    {label:<12} n={m['n_events']:>4}  "
            f"obs_mean={m['obs_mean']:6.2f}  pred_mean={m['pred_mean']:6.2f}  "
            f"bias={m['bias_pct']:+6.1f}%  obs_max={m['obs_max']:.1f}  pred_max={m['pred_max']:.1f}"
        )

    # The 10 largest events
    top_idx = np.argsort(obs)[-10:][::-1]  # largest first
    print(f"\n  Top 10 observed events vs predictions:")
    print(f"    {'Date':<12} {'Observed':>10} {'Predicted':>10} {'Bias':>10}")
    for i in top_idx:
        date = pd.Timestamp(df["date"].iloc[i]).date()
        o = obs[i]
        p = pred[i]
        bias = 100 * (p - o) / o if o > 0 else float("nan")
        print(f"    {str(date):<12} {o:>9.2f}  {p:>9.2f}  {bias:>+8.1f}%")


# ─── Save summary CSV ────────────────────────────────────────────────────────
if all_rows:
    summary = pd.DataFrame(all_rows)
    summary = summary[
        [
            "horizon",
            "band",
            "n_events",
            "obs_mean",
            "pred_mean",
            "bias_m3s",
            "bias_pct",
            "mae",
            "rmse",
            "obs_max",
            "pred_max",
        ]
    ]
    out_csv = OUT_DIR / "peak_bias_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\nSaved summary → {out_csv.relative_to(ROOT)}\n")


# ─── Scatter plot: observed vs predicted across horizons ─────────────────────
if horizons_with_data:
    n = len(horizons_with_data)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, h in zip(axes, horizons_with_data):
        obs, pred, _ = horizon_data[h]
        # All points light grey
        ax.scatter(obs, pred, s=6, c="#888", alpha=0.35, label="All test days")
        # Top 5% highlighted
        thr = np.percentile(obs, 95)
        mask = obs >= thr
        ax.scatter(
            obs[mask],
            pred[mask],
            s=12,
            c="#B8412A",
            alpha=0.85,
            label=f"Top 5% (Q ≥ {thr:.1f})",
        )
        # 1:1 line
        lo = 0
        hi = max(float(obs.max()), float(pred.max())) * 1.05
        ax.plot([lo, hi], [lo, hi], "--", color="#444", lw=1, label="1:1 line")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"h={h}")
        ax.set_xlabel("Observed (m³/s)")
        if h == horizons_with_data[0]:
            ax.set_ylabel("Predicted (m³/s)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
    plt.suptitle(
        "Test-set predictions: observed vs predicted, top-5% highlighted", fontsize=12
    )
    plt.tight_layout()
    out_fig = OUT_DIR / "peak_scatter_per_horizon.png"
    plt.savefig(out_fig, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved scatter → {out_fig.relative_to(ROOT)}")


# ─── Bias-by-band bar plot ──────────────────────────────────────────────────
if all_rows:
    fig, ax = plt.subplots(figsize=(11, 5))
    summary_plot = summary[summary["band"] != "All flows"].copy()
    bands_order = ["Top 1%", "Top 5%", "Top 10%", "Top 25%"]
    horizons_present = sorted(summary_plot["horizon"].unique())
    width = 0.16
    x = np.arange(len(bands_order))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, h in enumerate(horizons_present):
        sub = summary_plot[summary_plot["horizon"] == h].set_index("band")
        vals = [
            sub.loc[b, "bias_pct"] if b in sub.index else np.nan for b in bands_order
        ]
        ax.bar(
            x + i * width, vals, width, label=f"h={h}", color=colors[i % len(colors)]
        )
    ax.axhline(0, color="#444", lw=1)
    ax.set_xticks(x + width * (len(horizons_present) - 1) / 2)
    ax.set_xticklabels(bands_order)
    ax.set_ylabel("Peak bias (%)")
    ax.set_title("Peak bias on test set, by percentile band and forecast horizon")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    out_fig = OUT_DIR / "peak_bias_by_band.png"
    plt.savefig(out_fig, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved bias plot → {out_fig.relative_to(ROOT)}")

print(f"\nDone. Results in {OUT_DIR.relative_to(ROOT)}")
