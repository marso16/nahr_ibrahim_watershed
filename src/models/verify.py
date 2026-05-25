"""
verify.py — Sanity-check LSTM/TCN test results against simple baselines.

Computes, on the same test period your model was evaluated on:
  1. Persistence baseline:  Q_hat(t) = Q_obs(t - horizon)
  2. Climatology baseline:  mean discharge by day-of-year, from data BEFORE
                            the test period (no leakage)
  3. Per-year metrics for the model (NSE, KGE, MAE, RMSE, PBIAS)
  4. Residual autocorrelation at lags 1, 7, 30 days

Reads only the saved predictions CSV — no retraining needed.

Usage:
  python verify.py                          # verifies lstm_predictions_test.csv
  python verify.py --model tcn              # verifies tcn_predictions_test.csv
  python verify.py --horizon 1              # persistence lag (default 1 day)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════
def get_config():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=str,
        default=os.environ.get(
            "WATERSHED_ROOT", "C:/Users/marck/Downloads/nahr_ibrahim_watershed"
        ),
    )
    p.add_argument(
        "--model",
        type=str,
        default="lstm",
        choices=["lstm", "tcn"],
        help="Which model's predictions to verify",
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Forecast horizon in days for persistence baseline (default: 1)",
    )
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics (standalone; mirrors definitions in lstm.py)
# ═══════════════════════════════════════════════════════════════════════════════
def safe_kge(obs, sim, eps=1e-8):
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    obs_std, sim_std = obs.std(), sim.std()
    obs_mean = obs.mean()
    if obs_std < eps or sim_std < eps or abs(obs_mean) < eps:
        return np.nan, np.nan, np.nan, np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = sim_std / obs_std
    beta = sim.mean() / obs_mean
    if np.isnan(r):
        return np.nan, np.nan, alpha, beta
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge, r, alpha, beta


def nse(obs, sim):
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom < 1e-12:
        return np.nan
    return 1 - np.sum((obs - sim) ** 2) / denom


def compute_metrics(obs, sim, name=""):
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs, sim = obs[mask], sim[mask]
    if len(obs) == 0:
        return {"name": name, "n": 0}
    kge, r, alpha, beta = safe_kge(obs, sim)
    return {
        "name": name,
        "n": int(len(obs)),
        "NSE": round(float(nse(obs, sim)), 4),
        "KGE": round(float(kge), 4),
        "r": round(float(r), 4),
        "alpha": round(float(alpha), 4),
        "beta": round(float(beta), 4),
        "MAE": round(float(mean_absolute_error(obs, sim)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(obs, sim))), 4),
        "PBIAS_%": round(
            (
                float(100 * np.sum(sim - obs) / np.sum(obs))
                if np.sum(obs) != 0
                else np.nan
            ),
            2,
        ),
    }


def print_table(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        print("  (no rows)")
        return
    cols = [
        c
        for c in ["name", "n", "NSE", "KGE", "MAE", "RMSE", "PBIAS_%"]
        if c in df.columns
    ]
    print(df[cols].to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# Raw timeseries lookup (for climatology baseline)
# ═══════════════════════════════════════════════════════════════════════════════
def find_raw_timeseries(root: Path):
    """Search data/{processed,raw,interim,.} for a CSV with discharge_m3s + a date column."""
    candidates = []
    for sub in ["processed", "raw", "interim", ""]:
        d = root / "data" / sub if sub else root / "data"
        if d.exists():
            candidates += sorted(d.glob("*.csv"))

    for c in candidates:
        try:
            head = pd.read_csv(c, nrows=2)
            cols_lower = {col.lower(): col for col in head.columns}
            if "discharge_m3s" not in cols_lower:
                continue
            date_cols = [orig for low, orig in cols_lower.items() if "date" in low]
            if not date_cols:
                continue
            full = pd.read_csv(c, parse_dates=[date_cols[0]])
            full = full.rename(
                columns={
                    date_cols[0]: "date",
                    cols_lower["discharge_m3s"]: "discharge_m3s",
                }
            )
            full = (
                full[["date", "discharge_m3s"]]
                .dropna()
                .sort_values("date")
                .reset_index(drop=True)
            )
            return full, c
        except Exception:
            continue
    return None, None


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    cfg = get_config()
    ROOT = Path(cfg.root)
    PRED_PATH = ROOT / "results" / "predictions" / f"{cfg.model}_predictions_test.csv"

    if not PRED_PATH.exists():
        raise FileNotFoundError(
            f"Predictions not found: {PRED_PATH}\n"
            f"Run {cfg.model}.py first to generate predictions."
        )

    print("=" * 72)
    print(f"  Verification report — {cfg.model.upper()}")
    print(f"  Source: {PRED_PATH}")
    print("=" * 72)

    df = (
        pd.read_csv(PRED_PATH, parse_dates=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    print(f"\nTest period: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Test samples: {len(df)}")

    # ── Baseline 1: persistence ───────────────────────────────────────────────
    print(f"\nPersistence baseline: Q_hat(t) = Q_obs(t - {cfg.horizon})")
    df["persistence"] = df["observed"].shift(cfg.horizon)
    print(
        f"  → {df['persistence'].notna().sum()} valid predictions "
        f"(lost first {cfg.horizon} sample(s) to lag)"
    )

    # ── Baseline 2: climatology from pre-test data ────────────────────────────
    print("\nClimatology baseline: mean discharge by day-of-year, training period only")
    raw_df, raw_path = find_raw_timeseries(ROOT)
    if raw_df is None:
        print(
            "  Could not find a raw discharge CSV under data/{processed,raw,interim,.}"
        )
        print("  → climatology baseline will be skipped")
        df["climatology"] = np.nan
    else:
        print(f"  Found raw timeseries: {raw_path.name} ({len(raw_df):,} rows)")
        pre_test = raw_df[raw_df["date"] < df["date"].min()].copy()
        if len(pre_test) < 365:
            print(
                f"  ⚠ Only {len(pre_test)} pre-test observations — climatology unreliable"
            )
        pre_test["doy"] = pre_test["date"].dt.dayofyear
        clim_map = pre_test.groupby("doy")["discharge_m3s"].mean()
        df["climatology"] = df["date"].dt.dayofyear.map(clim_map)
        print(
            f"  → Climatology built from {len(pre_test):,} pre-test days "
            f"(period: {pre_test['date'].min().date()} → {pre_test['date'].max().date()})"
        )

    # ── Overall metrics comparison ────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  OVERALL METRICS — model vs baselines")
    print("=" * 72 + "\n")

    rows = []
    rows.append(compute_metrics(df["observed"], df["predicted"], cfg.model.upper()))

    pm = df["persistence"].notna()
    rows.append(
        compute_metrics(
            df.loc[pm, "observed"],
            df.loc[pm, "persistence"],
            f"Persistence(t-{cfg.horizon})",
        )
    )

    if df["climatology"].notna().any():
        cm = df["climatology"].notna()
        rows.append(
            compute_metrics(
                df.loc[cm, "observed"], df.loc[cm, "climatology"], "Climatology(DoY)"
            )
        )

    print_table(rows)

    # Verdict vs persistence
    model_nse = rows[0]["NSE"]
    pers_nse = rows[1]["NSE"]
    if not np.isnan(model_nse) and not np.isnan(pers_nse):
        delta = model_nse - pers_nse
        print(f"\n  ΔNSE vs persistence: {delta:+.4f}")
        if delta < 0.0:
            print("  ✗ Model is WORSE than persistence — something is off")
        elif delta < 0.05:
            print(
                "  ⚠ Model barely beats persistence — likely learned mostly autocorrelation"
            )
        elif delta < 0.15:
            print("  ✓ Modest improvement over persistence")
        else:
            print("  ✓✓ Clear improvement over persistence")

    if df["climatology"].notna().any():
        clim_nse = rows[2]["NSE"]
        if not np.isnan(clim_nse) and not np.isnan(model_nse):
            print(f"  ΔNSE vs climatology: {model_nse - clim_nse:+.4f}")

    # ── Per-year metrics for the model ────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"  PER-YEAR METRICS — {cfg.model.upper()}")
    print("=" * 72 + "\n")

    df["year"] = df["date"].dt.year
    yearly = []
    for y, g in df.groupby("year"):
        if len(g) < 30:
            continue
        yearly.append(compute_metrics(g["observed"], g["predicted"], str(y)))
    print_table(yearly)

    nses = [r["NSE"] for r in yearly if not np.isnan(r.get("NSE", np.nan))]
    kges = [r["KGE"] for r in yearly if not np.isnan(r.get("KGE", np.nan))]
    if len(nses) >= 2:
        print(
            f"\n  NSE across years: min={min(nses):.3f}  max={max(nses):.3f}  "
            f"spread={max(nses) - min(nses):.3f}"
        )
        print(
            f"  KGE across years: min={min(kges):.3f}  max={max(kges):.3f}  "
            f"spread={max(kges) - min(kges):.3f}"
        )
        if max(nses) - min(nses) > 0.3:
            print(
                "  ⚠ Large year-to-year variation — headline NSE is masking instability"
            )

    # ── Residual autocorrelation ──────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"  RESIDUAL AUTOCORRELATION — {cfg.model.upper()}")
    print("=" * 72 + "\n")

    resid = (df["observed"] - df["predicted"]).dropna().reset_index(drop=True)
    for lag in [1, 7, 30]:
        if len(resid) > lag:
            ac = resid.autocorr(lag=lag)
            flag = (
                "  ← high; systematic structure model didn't capture"
                if abs(ac) > 0.3
                else ""
            )
            print(f"  Lag {lag:>2}d:  {ac:+.3f}{flag}")

    # ── Save augmented predictions for further inspection ─────────────────────
    out_path = ROOT / "results" / "metrics" / f"{cfg.model}_verification.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[["date", "observed", "predicted", "persistence", "climatology"]].to_csv(
        out_path, index=False
    )
    print(f"\nSaved augmented predictions → {out_path}")

    # ── Reading guide ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  HOW TO READ THIS")
    print("=" * 72)
    print("""
  ΔNSE vs persistence is the single most informative number here:
    < 0.05   →  model is essentially reproducing temporal autocorrelation.
    0.05–0.15 → modest skill above the naive baseline.
    > 0.15   →  clear skill beyond persistence.

  Per-year spread > 0.3 means the headline metric hides regime instability —
  examine the worst year's residuals to see what broke.

  Residual autocorrelation > 0.3 at lag 1 means the model is leaving
  predictable temporal structure on the table.

  CAVEAT: persistence Q(t-1) uses the previous day's *observation*. If your
  model's features don't include lagged discharge, persistence has an
  information advantage — interpret the comparison accordingly.
""")


if __name__ == "__main__":
    main()
