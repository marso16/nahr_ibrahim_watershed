import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(
    os.environ.get("WATERSHED_ROOT", "C:/Users/marck/Downloads/nahr_ibrahim_watershed")
)
SPLIT_DIR = ROOT / "data" / "splits"
MET_DIR = ROOT / "results" / "metrics"
MET_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(SPLIT_DIR / "train_raw.csv", parse_dates=["date"])
test = pd.read_csv(SPLIT_DIR / "test_raw.csv", parse_dates=["date"])


def nse(obs, sim):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    denom = np.sum((obs - obs.mean()) ** 2)
    return np.nan if denom < 1e-12 else 1 - np.sum((obs - sim) ** 2) / denom


def kge(obs, sim):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    if obs.std() < 1e-8 or sim.std() < 1e-8:
        return np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    return 1 - np.sqrt(
        (r - 1) ** 2
        + (sim.std() / obs.std() - 1) ** 2
        + (sim.mean() / obs.mean() - 1) ** 2
    )


# Climatology lookup from training
train["doy"] = train["date"].dt.dayofyear
clim = train.groupby("doy")["discharge_m3s_raw"].mean()

results = []
for h in [1, 3, 7, 14, 30]:
    q = test["discharge_m3s_raw"].values

    # Persistence at h-day lag: predict Q(t) = Q(t-h)
    if len(q) <= h:
        continue
    y_obs = q[h:]
    y_persist = q[:-h]

    # Climatology (independent of horizon)
    test["doy"] = test["date"].dt.dayofyear
    y_clim_full = test["doy"].map(clim).values
    y_clim = y_clim_full[h:]

    for name, y_pred in [("Persistence", y_persist), ("Climatology", y_clim)]:
        results.append(
            {
                "horizon_days": h,
                "model": name,
                "NSE": round(nse(y_obs, y_pred), 4),
                "KGE": round(kge(y_obs, y_pred), 4),
                "MAE": round(mean_absolute_error(y_obs, y_pred), 4),
                "RMSE": round(float(np.sqrt(mean_squared_error(y_obs, y_pred))), 4),
            }
        )

df = pd.DataFrame(results)
print(df.to_string(index=False))
df.to_csv(MET_DIR / "baselines_multi_horizon.csv", index=False)
