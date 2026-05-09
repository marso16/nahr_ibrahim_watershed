import xgboost as xgb
import joblib
import numpy as np
import matplotlib.pyplot as plt

model = joblib.load(
    r"C:\Users\marck\Downloads\nahr_ibrahim_watershed\models\trained\xgboost_final.pkl"
)

# Top 20 most important features
importances = model.feature_importances_
top20 = np.argsort(importances)[-20:]

# Feature names — lag_day_feature
FEATURE_COLS = [
    "precip_mm_day",
    "precip_3day",
    "precip_7day",
    "temp_mean_c",
    "temp_max_c",
    "temp_min_c",
    "temp_range_c",
    "swe_mm",
    "swe_delta",
    "snow_cover_pct",
    "month_sin",
    "month_cos",
    "soil_moisture_mm",
    "sm_7day_mean",
    "sm_anomaly",
    "pet_mm_day",
    "spi_3month",
    "spei_3month",
]
n_feat = len(FEATURE_COLS)

names = []
for idx in top20:
    lag = idx // n_feat
    feat = FEATURE_COLS[idx % n_feat]
    names.append(f"t-{30 - lag:02d}  {feat}")

plt.figure(figsize=(10, 8))
plt.barh(names, importances[top20], color="#3b9eff", alpha=0.85)
plt.title("XGBoost — Top 20 Feature Importances")
plt.tight_layout()
plt.savefig(
    r"C:\Users\marck\Downloads\nahr_ibrahim_watershed\results\figures\xgboost_feature_importance.png",
    dpi=150,
)
plt.show()
