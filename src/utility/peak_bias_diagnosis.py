import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(
    os.environ.get("WATERSHED_ROOT", "C:/Users/marck/Downloads/nahr_ibrahim_watershed")
)
PRED_DIR = ROOT / "results" / "predictions"
OUT_DIR = ROOT / "results" / "peak_bias"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 3, 14]


def find_all_prediction_files(horizon: int, model_tag: str = None):
    if model_tag:
        patterns = [
            f"*{model_tag}*predictions_test*h{horizon}*.csv",
            f"*{model_tag}*predictions_test*.csv",
        ]
        files = []
        for pat in patterns:
            files.extend(PRED_DIR.glob(pat))
        files = sorted(set(files))
        return [(f, model_tag) for f in files if f.exists()]
    else:
        pattern = f"*predictions_test*h{horizon}*.csv"
        files = sorted(PRED_DIR.glob(pattern))
        results = []
        for f in files:
            stem = f.stem
            parts = stem.split("_predictions_test_")
            model_name = parts[0] if len(parts) > 0 else "unknown"
            results.append((f, model_name))
        return results


def metrics_at_band(obs, pred, lower_pct: float, upper_pct: float | None = None):
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_tag",
        type=str,
        default=None,
        help="Optional: analyze only one model (e.g. tcn, gr4j_tcn, lstm)",
    )
    p.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    p.add_argument(
        "--aggregate_seeds",
        action="store_true",
        default=True,
        help="Average duplicate model+horizon+band entries (e.g. multiple seeds)",
    )
    args = p.parse_args()

    all_rows = []
    plot_data = []

    for h in args.horizons:
        files = find_all_prediction_files(h, args.model_tag)
        if not files:
            print(f"\n  h={h}: no prediction files found, skipping")
            continue

        print(f"\n{'='*60}\n  Horizon h={h}\n{'='*60}")

        for pred_file, model_name in files:
            df = pd.read_csv(pred_file, parse_dates=["date"])
            if "observed" not in df.columns or "predicted" not in df.columns:
                print(f"    {model_name}: unexpected columns, skipping")
                continue

            obs = df["observed"].values
            pred = df["predicted"].values
            print(f"\n  {model_name}: {pred_file.name}")
            print(
                f"    {len(obs)} test days, obs range [{obs.min():.2f}, {obs.max():.2f}]"
            )

            plot_data.append((h, model_name, obs, pred, df["date"].values))

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
                m["model"] = model_name
                m["band"] = label
                all_rows.append(m)
                print(
                    f"    {label:<12} n={m['n_events']:>4}  "
                    f"obs={m['obs_mean']:6.2f}  pred={m['pred_mean']:6.2f}  "
                    f"bias={m['bias_pct']:+6.1f}%  obs_max={m['obs_max']:.1f}  pred_max={m['pred_max']:.1f}"
                )

            top_idx = np.argsort(obs)[-10:][::-1]
            print(f"\n    Top 10 observed events:")
            for i in top_idx:
                date = pd.Timestamp(df["date"].iloc[i]).date()
                o = obs[i]
                p = pred[i]
                bias = 100 * (p - o) / o if o > 0 else float("nan")
                print(f"      {str(date):<12} {o:>9.2f}  {p:>9.2f}  {bias:>+8.1f}%")

    if not all_rows:
        print("No prediction files processed. Exiting.")
        return

    summary = pd.DataFrame(all_rows)

    if args.aggregate_seeds:
        numeric_cols = summary.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in ["horizon"]]
        summary = summary.groupby(["model", "horizon", "band"], as_index=False)[
            numeric_cols
        ].mean()
        print(f"\n  Aggregated duplicate entries (multiple seeds) by mean.")

    cols = [
        "model",
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
    summary = summary[[c for c in cols if c in summary.columns]]
    out_csv = OUT_DIR / "peak_bias_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\n{'='*60}\nSaved summary → {out_csv.relative_to(ROOT)}")

    if plot_data:
        for h in args.horizons:
            h_data = [(m, o, p, d) for (hh, m, o, p, d) in plot_data if hh == h]
            if not h_data:
                continue

            n_models = len(h_data)
            fig, axes = plt.subplots(
                1, n_models, figsize=(4.5 * n_models, 4.5), sharex=True, sharey=True
            )
            if n_models == 1:
                axes = [axes]

            for ax, (model_name, obs, pred, dates) in zip(axes, h_data):
                ax.scatter(obs, pred, s=6, c="#888", alpha=0.35, label="All")
                thr = np.percentile(obs, 95)
                mask = obs >= thr
                ax.scatter(
                    obs[mask],
                    pred[mask],
                    s=12,
                    c="#B8412A",
                    alpha=0.85,
                    label=f"Top 5% (≥{thr:.1f})",
                )
                lo = 0
                hi = max(float(obs.max()), float(pred.max())) * 1.05
                ax.plot([lo, hi], [lo, hi], "--", color="#444", lw=1)
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                ax.set_title(f"{model_name}\nh={h}")
                ax.set_xlabel("Observed (m³/s)")
                ax.grid(alpha=0.3)
                if ax == axes[0]:
                    ax.set_ylabel("Predicted (m³/s)")
                ax.legend(loc="upper left", fontsize=8)

            plt.suptitle(f"Peak bias diagnosis — h={h}", fontsize=12)
            plt.tight_layout()
            out_fig = OUT_DIR / f"peak_scatter_h{h}.png"
            plt.savefig(out_fig, dpi=140, bbox_inches="tight")
            plt.close()
            print(f"Saved scatter → {out_fig.relative_to(ROOT)}")

    models_present = sorted(summary["model"].unique())
    n_models = len(models_present)
    if n_models == 0:
        print("No models to plot. Done.")
        return

    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models_present):
        sub = summary[(summary["model"] == model) & (summary["band"] != "All flows")]
        if sub.empty:
            ax.set_title(f"{model}\n(no data)")
            continue

        bands_order = ["Top 1%", "Top 5%", "Top 10%", "Top 25%"]
        horizons_present = sorted(sub["horizon"].unique())
        width = 0.18
        x = np.arange(len(bands_order))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for i, h in enumerate(horizons_present):
            hsub = sub[sub["horizon"] == h].set_index("band")
            vals = []
            for b in bands_order:
                if b in hsub.index:
                    v = hsub.loc[b, "bias_pct"]
                    if isinstance(v, (pd.Series, np.ndarray, list)):
                        v = float(np.mean(v))
                    else:
                        v = float(v)
                    vals.append(v)
                else:
                    vals.append(np.nan)

            vals = np.array(vals, dtype=float)
            ax.bar(
                x + i * width,
                vals,
                width,
                label=f"h={h}",
                color=colors[i % len(colors)],
            )

        ax.axhline(0, color="#444", lw=1)
        ax.set_xticks(x + width * (len(horizons_present) - 1) / 2)
        ax.set_xticklabels(bands_order)
        ax.set_ylabel("Peak bias (%)" if ax == axes[0] else "")
        ax.set_title(model)
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3, axis="y")

    plt.suptitle("Peak bias by model, percentile band, and horizon", fontsize=12)
    plt.tight_layout()
    out_fig = OUT_DIR / "peak_bias_by_model.png"
    plt.savefig(out_fig, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved bias plot → {out_fig.relative_to(ROOT)}")

    print(f"\nDone. Results in {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
