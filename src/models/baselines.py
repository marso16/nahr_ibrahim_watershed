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

# Use RAW splits (real m³/s, not log-transformed or normalised)
train = pd.read_csv(SPLIT_DIR / "train_raw.csv", parse_dates=["date"])
test = pd.read_csv(SPLIT_DIR / "test_raw.csv", parse_dates=["date"])

# IMPORTANT: train_raw.csv currently has discharge_m3s in log-space because
# split.py overwrites it. You need to save the original column too.
# See "small split.py fix" below.


def nse(obs, sim):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - obs.mean()) ** 2)


def kge(obs, sim):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


# --- Baseline 1: Persistence ---
# Predicts Q(t) = Q(t-1). Crude but very hard to beat at short horizons.
y_obs = test["discharge_m3s_raw"].values[1:]
y_pred_persist = test["discharge_m3s_raw"].values[:-1]

# --- Baseline 2: Climatology (day-of-year mean from train) ---
train["doy"] = train["date"].dt.dayofyear
climatology = train.groupby("doy")["discharge_m3s_raw"].mean()
test["doy"] = test["date"].dt.dayofyear
y_pred_clim = test["doy"].map(climatology).values

# --- Baseline 3: Mean of training ---
y_pred_mean = np.full(len(test), train["discharge_m3s_raw"].mean())

results = []
for name, pred, obs in [
    ("Persistence", y_pred_persist, y_obs),
    ("Climatology", y_pred_clim, test["discharge_m3s_raw"].values),
    ("Train_Mean", y_pred_mean, test["discharge_m3s_raw"].values),
]:
    results.append(
        {
            "model": name,
            "NSE": round(nse(obs, pred), 4),
            "KGE": round(kge(obs, pred), 4),
            "MAE": round(mean_absolute_error(obs, pred), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(obs, pred))), 4),
        }
    )

df = pd.DataFrame(results)
print(df.to_string(index=False))
df.to_csv(MET_DIR / "baselines.csv", index=False)
