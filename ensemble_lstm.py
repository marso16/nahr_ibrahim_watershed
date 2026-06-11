import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ROOT = Path("C:/Users/marck/Downloads/nahr_ibrahim_watershed")
PRED_DIR = ROOT / "results" / "predictions"
MET_DIR = ROOT / "results" / "metrics"

horizons = [1, 3, 14]
seeds = [42, 69, 2024]

for horizon in horizons:
    print(f"\n{'='*70}")
    print(f"LSTM HORIZON {horizon} DAY ENSEMBLE")
    print(f"{'='*70}")

    preds = []

    for seed in seeds:
        tag = f"lstm_h{horizon}_s{seed}"
        file = PRED_DIR / f"lstm_predictions_test_{tag}.csv"

        if not file.exists():
            print(f"Missing: {file.name}")
            continue

        df = pd.read_csv(file, parse_dates=["date"])
        preds.append(
            df.set_index("date")[["predicted"]].rename(
                columns={"predicted": f"pred_{seed}"}
            )
        )

    if len(preds) == 0:
        print("No prediction files found.")
        continue

    ensemble = preds[0].join(preds[1:])
    ensemble["predicted"] = ensemble.mean(axis=1)

    obs = pd.read_csv(
        PRED_DIR / f"lstm_predictions_test_lstm_h{horizon}_s{seeds[0]}.csv",
        parse_dates=["date"],
    )
    obs = obs.set_index("date")[["observed"]]

    final = obs.join(ensemble[["predicted"]])
    final["residual"] = final["observed"] - final["predicted"]

    obs_vals = final["observed"].values
    sim_vals = final["predicted"].values

    r2 = r2_score(obs_vals, sim_vals)
    mae = mean_absolute_error(obs_vals, sim_vals)
    rmse = np.sqrt(mean_squared_error(obs_vals, sim_vals))
    nse = 1 - np.sum((obs_vals - sim_vals) ** 2) / np.sum(
        (obs_vals - obs_vals.mean()) ** 2
    )

    r = np.corrcoef(obs_vals, sim_vals)[0, 1]
    alpha = sim_vals.std() / obs_vals.std()
    beta = sim_vals.mean() / obs_vals.mean()
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    pbias = 100 * np.sum(sim_vals - obs_vals) / np.sum(obs_vals)

    peak_mask = obs_vals >= np.percentile(obs_vals, 95)
    peak_bias = (
        100
        * (sim_vals[peak_mask].mean() - obs_vals[peak_mask].mean())
        / obs_vals[peak_mask].mean()
    )
    peak_mae = mean_absolute_error(obs_vals[peak_mask], sim_vals[peak_mask])

    print(f"NSE:       {nse:.4f}")
    print(f"KGE:       {kge:.4f}")
    print(f"R²:        {r2:.4f}")
    print(f"MAE:       {mae:.3f}")
    print(f"RMSE:      {rmse:.3f}")
    print(f"PBIAS:     {pbias:+.2f}%")
    print(f"Peak Bias: {peak_bias:+.2f}%")
    print(f"Peak MAE:  {peak_mae:.3f}")

    final.to_csv(PRED_DIR / f"lstm_ensemble_h{horizon}.csv")

    pd.DataFrame(
        [
            {
                "Horizon": horizon,
                "NSE": nse,
                "KGE": kge,
                "R2": r2,
                "MAE": mae,
                "RMSE": rmse,
                "PBIAS": pbias,
                "Peak_Bias": peak_bias,
                "Peak_MAE": peak_mae,
            }
        ]
    ).to_csv(MET_DIR / f"lstm_ensemble_metrics_h{horizon}.csv", index=False)

    print(f"Saved: lstm_ensemble_h{horizon}.csv")
