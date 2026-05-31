"""
GR4J conceptual rainfall-runoff model with calibration.

Reference: Perrin, C., Michel, C., Andréassian, V. (2003). Improvement of a
parsimonious model for streamflow simulation. Journal of Hydrology, 279(1-4).

Four parameters calibrated against GloFAS discharge on the training period:
  x1 = production store capacity (mm)        typical range 100-1200
  x2 = inter-catchment exchange coef. (mm)   typical range -5 to +3
  x3 = routing store capacity (mm)           typical range 20-300
  x4 = unit hydrograph time base (days)      typical range 1.1-2.9

Inputs needed per day: precipitation (mm), potential evapotranspiration (mm).
Output: simulated discharge (mm/day), converted to m³/s using catchment area.

Usage:
  python src/models/gr4j.py --horizon 1 --run_tag gr4j_h1
"""

import os
import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import differential_evolution
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ─────────────────────────────────────────────────────────────────────────────
# GR4J model equations
# ─────────────────────────────────────────────────────────────────────────────
def _unit_hydrograph_ordinates(x4: float, n: int):
    """
    Generate unit hydrograph ordinates for GR4J's two unit hydrographs.
    UH1 has time base x4, UH2 has time base 2*x4. Standard GR4J formulation.
    """

    def s_curve_1(t):
        if t <= 0:
            return 0.0
        if t < x4:
            return (t / x4) ** 2.5
        return 1.0

    def s_curve_2(t):
        if t <= 0:
            return 0.0
        if t < x4:
            return 0.5 * (t / x4) ** 2.5
        if t < 2 * x4:
            return 1.0 - 0.5 * (2 - t / x4) ** 2.5
        return 1.0

    uh1 = np.array([s_curve_1(i + 1) - s_curve_1(i) for i in range(n)])
    uh2 = np.array([s_curve_2(i + 1) - s_curve_2(i) for i in range(n)])
    return uh1, uh2


def gr4j_simulate(
    precip: np.ndarray, pet: np.ndarray, params, S0_frac=0.6, R0_frac=0.7
):
    """
    Run GR4J for a single time series.

    precip, pet: daily mm arrays of equal length
    params: (x1, x2, x3, x4)
    S0_frac, R0_frac: initial fill fractions for production and routing stores

    Returns simulated discharge in mm/day.
    """
    x1, x2, x3, x4 = params
    n = len(precip)

    # Unit hydrograph ordinates
    nh = max(int(np.ceil(2 * x4)) + 1, 2)
    uh1, uh2 = _unit_hydrograph_ordinates(x4, nh)

    # State variables
    S = S0_frac * x1  # production store (mm)
    R = R0_frac * x3  # routing store (mm)
    q_uh1 = np.zeros(nh)  # 90% pathway delay buffer
    q_uh2 = np.zeros(nh)  # 10% pathway delay buffer

    Q_sim = np.zeros(n)

    for t in range(n):
        P = max(precip[t], 0.0)
        E = max(pet[t], 0.0)

        # ── 1. Net rainfall / evaporation ────────────────────────────────────
        if P >= E:
            Pn = P - E
            En = 0.0
        else:
            Pn = 0.0
            En = E - P

        # ── 2. Production store ──────────────────────────────────────────────
        if Pn > 0:
            ws = np.tanh(Pn / x1)
            Ps = x1 * (1 - (S / x1) ** 2) * ws / (1 + S / x1 * ws)
        else:
            Ps = 0.0

        if En > 0:
            ws = np.tanh(En / x1)
            Es = S * (2 - S / x1) * ws / (1 + (1 - S / x1) * ws)
        else:
            Es = 0.0

        S = S - Es + Ps

        # Percolation from production store
        perc = S * (1 - (1 + (4.0 * S / (9.0 * x1)) ** 4) ** (-0.25))
        S = S - perc

        # Total effective rainfall reaching UH
        Pr = perc + (Pn - Ps)

        # ── 3. Split between UH1 (90%) and UH2 (10%) ─────────────────────────
        Pr_uh1 = 0.9 * Pr
        Pr_uh2 = 0.1 * Pr

        # Convolve with unit hydrographs (shift buffer and add)
        q_uh1[:-1] = q_uh1[1:]
        q_uh1[-1] = 0.0
        q_uh1 = q_uh1 + uh1 * Pr_uh1

        q_uh2[:-1] = q_uh2[1:]
        q_uh2[-1] = 0.0
        q_uh2 = q_uh2 + uh2 * Pr_uh2

        # Outflow from UH at this timestep
        Q9 = q_uh1[0]
        Q1 = q_uh2[0]

        # ── 4. Inter-catchment exchange ──────────────────────────────────────
        F = x2 * (R / x3) ** 3.5

        # ── 5. Routing store ─────────────────────────────────────────────────
        R = max(0.0, R + Q9 + F)
        Qr = R * (1 - (1 + (R / x3) ** 4) ** (-0.25))
        R = R - Qr

        # ── 6. Direct runoff (Q1 + F, clipped to 0) ──────────────────────────
        Qd = max(0.0, Q1 + F)

        # ── 7. Total discharge (mm/day) ──────────────────────────────────────
        Q_sim[t] = Qr + Qd

    return Q_sim


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def nse(obs, sim):
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    denom = np.sum((obs - obs.mean()) ** 2)
    return np.nan if denom < 1e-12 else 1 - np.sum((obs - sim) ** 2) / denom


def kge_full(obs, sim, eps=1e-8):
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    if obs.std() < eps or sim.std() < eps or abs(obs.mean()) < eps:
        return np.nan, np.nan, np.nan, np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    if np.isnan(r):
        return np.nan, np.nan, alpha, beta
    return (
        1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2),
        r,
        alpha,
        beta,
    )


def log_nse(obs, sim, eps=1e-3):
    obs = np.maximum(obs, 0) + eps
    sim = np.maximum(sim, 0) + eps
    lo, ls = np.log(obs), np.log(sim)
    denom = np.sum((lo - lo.mean()) ** 2)
    return np.nan if denom < 1e-12 else 1 - np.sum((lo - ls) ** 2) / denom


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────
def calibrate_gr4j(precip, pet, q_obs_mmday, bounds, seed=42, maxiter=80):
    """
    Calibrate GR4J using differential evolution. Objective: maximize NSE on
    training period (equivalent to minimizing 1 - NSE).
    """
    # Skip first year for warm-up (states stabilize)
    warmup = 365

    def objective(params):
        try:
            q_sim = gr4j_simulate(precip, pet, params)
            val = nse(q_obs_mmday[warmup:], q_sim[warmup:])
            if np.isnan(val):
                return 1.0
            return 1.0 - val  # minimize
        except Exception:
            return 1.0

    print("  Calibrating GR4J (differential evolution)...")
    t0 = time.time()
    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=20,
        tol=1e-4,
        mutation=(0.5, 1.5),
        recombination=0.7,
        workers=1,
        polish=True,
        disp=False,
    )
    elapsed = time.time() - t0
    print(f"  Calibration finished in {elapsed/60:.1f} min")
    print(f"  Best NSE on training (post-warmup): {1 - result.fun:.4f}")
    return result.x, 1 - result.fun


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
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
        "--horizon",
        type=int,
        default=1,
        help="Prediction horizon in days (GR4J shifts simulated Q forward by this much)",
    )
    p.add_argument(
        "--catchment_area_km2",
        type=float,
        default=326.0,
        help="Used to convert mm/day to m³/s",
    )
    p.add_argument(
        "--maxiter",
        type=int,
        default=80,
        help="Max iterations for differential evolution",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_tag", type=str, default=None)
    return p.parse_args()


def main():
    cfg = get_config()
    tag = cfg.run_tag if cfg.run_tag else f"gr4j_h{cfg.horizon}"

    ROOT = Path(cfg.root)
    MASTER = ROOT / "data" / "master"
    MET_DIR = ROOT / "results" / "metrics"
    PRED_DIR = ROOT / "results" / "predictions"
    FIG_DIR = ROOT / "results" / "figures"
    MODEL_DIR = ROOT / "models" / "trained"
    for d in [MET_DIR, PRED_DIR, FIG_DIR, MODEL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Load master CSV ──────────────────────────────────────────────────────
    df = pd.read_csv(MASTER / "nahr_ibrahim_master_model.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(
        f"Loaded master: {len(df)} rows ({df.date.min().date()} → {df.date.max().date()})\n"
    )

    # ── Convert discharge from m³/s to mm/day for calibration ────────────────
    # Q (mm/day) = Q (m³/s) × 86400 / (Area in m²) × 1000
    A_m2 = cfg.catchment_area_km2 * 1e6
    SEC_TO_MMDAY = 86400.0 * 1000.0 / A_m2  # m³/s → mm/day
    print(f"Catchment area: {cfg.catchment_area_km2} km²")
    print(f"Conversion factor: 1 m³/s = {SEC_TO_MMDAY:.4f} mm/day\n")

    df["q_obs_mmday"] = df["discharge_m3s"] * SEC_TO_MMDAY

    # ── Train/val/test split (must match split.py) ───────────────────────────
    TRAIN_END = pd.Timestamp("2017-12-31")
    VAL_END = pd.Timestamp("2020-12-31")

    train_idx = df["date"] <= TRAIN_END
    val_idx = (df["date"] > TRAIN_END) & (df["date"] <= VAL_END)
    test_idx = df["date"] > VAL_END

    print(
        f"Train: {train_idx.sum()} days  Val: {val_idx.sum()} days  Test: {test_idx.sum()} days\n"
    )

    # ── Calibrate on TRAIN period only ───────────────────────────────────────
    # Use the same arrays (precip, pet) for the full period, but the calibration
    # only evaluates the training portion. State variables warm up naturally
    # since GR4J needs continuous data.
    precip = df["precip_mm_day"].values
    pet = df["pet_mm_day"].values
    q_obs = df["q_obs_mmday"].values

    # GR4J parameter bounds (standard, slightly widened for karst tolerance)
    bounds = [
        (50.0, 5000.0),  # x1 — wider for karst storage
        (-10.0, 10.0),  # x2
        (10.0, 1500.0),  # x3 — wider for slow routing
        (0.5, 10.0),  # x4 — wider for delayed response
    ]

    # Calibrate only against the training portion
    train_precip = precip[: train_idx.sum()]
    train_pet = pet[: train_idx.sum()]
    train_q = q_obs[: train_idx.sum()]

    best_params, train_nse = calibrate_gr4j(
        train_precip,
        train_pet,
        train_q,
        bounds,
        seed=cfg.seed,
        maxiter=cfg.maxiter,
    )
    x1, x2, x3, x4 = best_params
    print(f"\nCalibrated parameters:")
    print(f"  x1 (production store capacity): {x1:.1f} mm")
    print(f"  x2 (inter-catchment exchange):  {x2:+.2f} mm")
    print(f"  x3 (routing store capacity):    {x3:.1f} mm")
    print(f"  x4 (UH time base):              {x4:.2f} days\n")

    # ── Run calibrated model on FULL series ──────────────────────────────────
    q_sim_mmday = gr4j_simulate(precip, pet, best_params)
    q_sim_m3s = q_sim_mmday / SEC_TO_MMDAY

    df["q_sim_mmday"] = q_sim_mmday
    df["q_sim_m3s"] = q_sim_m3s

    # ── Apply horizon shift ──────────────────────────────────────────────────
    # Horizon=1 means "predict tomorrow from today's inputs". For GR4J, we shift
    # the simulated discharge forward by `horizon` days and compare against the
    # observation `horizon` days ahead. This matches the LSTM setup.
    h = cfg.horizon
    if h > 0:
        # Predict Q at time t using GR4J output at time t (the model already
        # represents physical lag; we shift predictions forward by h to align
        # with the LSTM's "h-day-ahead" framing).
        # Predicted Q at date t = GR4J simulated Q at date t-h
        df["q_pred_m3s"] = df["q_sim_m3s"].shift(h)
    else:
        df["q_pred_m3s"] = df["q_sim_m3s"]

    # Drop rows where shift created NaN
    df_test = df.loc[test_idx & df["q_pred_m3s"].notna()].copy()

    # ── Evaluate on test set ─────────────────────────────────────────────────
    y_obs = df_test["discharge_m3s"].values
    y_pred = df_test["q_pred_m3s"].values

    r2 = r2_score(y_obs, y_pred)
    mae_v = mean_absolute_error(y_obs, y_pred)
    rmse_v = float(np.sqrt(mean_squared_error(y_obs, y_pred)))
    nse_v = nse(y_obs, y_pred)
    kge_v, kge_r, kge_alpha, kge_beta = kge_full(y_obs, y_pred)
    lognse_v = log_nse(y_obs, y_pred)
    pbias = 100 * np.sum(y_pred - y_obs) / np.sum(y_obs)

    peak_mask = y_obs >= np.percentile(y_obs, 95)
    if peak_mask.sum() > 0:
        peak_bias = (
            100
            * (y_pred[peak_mask].mean() - y_obs[peak_mask].mean())
            / y_obs[peak_mask].mean()
        )
        peak_mae = mean_absolute_error(y_obs[peak_mask], y_pred[peak_mask])
        peak_rmse = float(
            np.sqrt(mean_squared_error(y_obs[peak_mask], y_pred[peak_mask]))
        )
    else:
        peak_bias = peak_mae = peak_rmse = np.nan

    print(f"{'=' * 72}\nGR4J test evaluation (horizon={h})\n{'=' * 72}")
    print(f"  R²:     {r2:.4f}")
    print(f"  NSE:    {nse_v:.4f}")
    print(
        f"  KGE:    {kge_v:.4f}  (r={kge_r:.3f}, α={kge_alpha:.3f}, β={kge_beta:.3f})"
    )
    print(f"  logNSE: {lognse_v:.4f}")
    print(f"  MAE:    {mae_v:.3f} m³/s")
    print(f"  RMSE:   {rmse_v:.3f} m³/s")
    print(f"  PBIAS:  {pbias:+.2f}%")
    print(f"  Peak Bias: {peak_bias:+.2f}%  |  Peak MAE: {peak_mae:.3f}")

    # ── Save metrics (same schema as lstm_metrics_*.csv) ─────────────────────
    pd.DataFrame(
        [
            {
                "split": "Test",
                "NSE": round(float(nse_v), 4),
                "KGE": round(float(kge_v), 4),
                "RMSE": round(rmse_v, 4),
                "MAE": round(mae_v, 4),
                "R2": round(float(r2), 4),
                "PBIAS_%": round(float(pbias), 2),
                "Peak_Bias_%": round(float(peak_bias), 2),
                "Log_NSE": round(float(lognse_v), 4),
                "Peak_MAE": round(float(peak_mae), 4),
                "Peak_RMSE": round(float(peak_rmse), 4),
                "KGE_r": round(float(kge_r), 4),
                "KGE_alpha": round(float(kge_alpha), 4),
                "KGE_beta": round(float(kge_beta), 4),
                "x1": float(x1),
                "x2": float(x2),
                "x3": float(x3),
                "x4": float(x4),
                "train_nse": float(train_nse),
            }
        ]
    ).to_csv(MET_DIR / f"gr4j_metrics_{tag}.csv", index=False)

    # Predictions
    pd.DataFrame(
        {
            "date": df_test["date"].values,
            "observed": y_obs,
            "predicted": y_pred,
            "residual": y_obs - y_pred,
        }
    ).to_csv(PRED_DIR / f"gr4j_predictions_test_{tag}.csv", index=False)

    # Save the full simulated series too (needed later for the hybrid model)
    pd.DataFrame(
        {
            "date": df["date"].values,
            "q_sim_mmday": df["q_sim_mmday"].values,
            "q_sim_m3s": df["q_sim_m3s"].values,
        }
    ).to_csv(MODEL_DIR / f"gr4j_full_simulation_{tag}.csv", index=False)

    # Save calibrated params (so the hybrid can reuse them)
    pd.DataFrame(
        [
            {
                "x1": x1,
                "x2": x2,
                "x3": x3,
                "x4": x4,
                "train_nse": train_nse,
                "catchment_area_km2": cfg.catchment_area_km2,
            }
        ]
    ).to_csv(MODEL_DIR / f"gr4j_params_{tag}.csv", index=False)

    # ── Quick plot ───────────────────────────────────────────────────────────
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(df_test["date"], y_obs, color="#1f77b4", lw=1.0, label="Observed")
    ax.plot(
        df_test["date"],
        y_pred,
        color="#ff7f0e",
        lw=1.0,
        alpha=0.85,
        label="GR4J Predicted",
    )
    ax.set_title(
        f"GR4J {tag} — NSE={nse_v:.3f} | KGE={kge_v:.3f} | MAE={mae_v:.2f} m³/s"
    )
    ax.set_ylabel("Discharge (m³/s)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"gr4j_results_{tag}.png", dpi=130, bbox_inches="tight")
    plt.close()

    print(f"\nMetrics      → results/metrics/gr4j_metrics_{tag}.csv")
    print(f"Predictions  → results/predictions/gr4j_predictions_test_{tag}.csv")
    print(f"Full sim     → models/trained/gr4j_full_simulation_{tag}.csv")
    print(f"Parameters   → models/trained/gr4j_params_{tag}.csv")


if __name__ == "__main__":
    main()
