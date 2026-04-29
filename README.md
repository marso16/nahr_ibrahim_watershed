# 🌊 AI Rainfall–Runoff Modeling — Nahr Ibrahim Watershed, Lebanon

> **MSc Thesis** · Saint Joseph University Beirut (ESIB)  
> *Testing AI Models for Climate-Resilient Rainfall–Runoff Modeling in the Nahr Ibrahim Watershed*

---

## 📋 Table of Contents

- [Overview](#overview)
- [Watershed](#watershed)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Pipeline](#pipeline)
- [Models](#models)
- [Results](#results)
- [Climate Scenarios](#climate-scenarios)
- [Data Validation](#data-validation)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Known Limitations](#known-limitations)
- [References](#references)

---

## Overview

This repository contains the full data pipeline, model training code, and climate scenario analysis for an MSc thesis benchmarking six AI rainfall–runoff architectures on the **Nahr Ibrahim watershed**, Lebanon, under historical (2000–2025) and projected (2015–2100) climate conditions.

**Key contributions:**
- First AI rainfall–runoff benchmarking study for the Nahr Ibrahim watershed
- Comparison of 4 pure data-driven + 2 physics-informed hybrid architectures
- Climate scenario projection under SSP2-4.5 and SSP5-8.5 using NEX-GDDP-CMIP6
- Physics-informed water balance constraint showing stronger climate signal detection
- Complete multi-source satellite data pipeline for a data-scarce Mediterranean karstic basin

**Main finding:** Under SSP5-8.5, physics-informed models project a **48% steeper discharge decline** (−0.0096 m³/s/decade) compared to pure AI counterparts (−0.0065 m³/s/decade), attributable to explicit evapotranspiration representation in the water balance constraint.

---

## Watershed

| Parameter | Value |
|---|---|
| Name | Nahr Ibrahim (Abraham River) |
| Location | Mount Lebanon, Lebanon |
| Coordinates | 33.90°–34.30°N, 35.60°–36.10°E |
| Area | ~326 km² |
| Centroid | 34.093°N, 35.878°E |
| Elevation range | ~50–2,900 m asl |
| Mean annual precipitation | 900–1,400 mm |
| Hydrology | Karstic, snowmelt-driven |
| Primary springs | Afqa (1,200 m) · Roueiss (1,265 m) |
| Climate zone | Mediterranean mountain |

---

## Project Structure

```
nahr_ibrahim_watershed/
│
├── data/
│   ├── raw/
│   │   ├── giovanni/          # Raw CSV downloads from NASA Giovanni
│   │   ├── glofas/            # GloFAS GRIB files + extracted CSVs
│   │   ├── modis/             # MODIS AppEEARS GeoTIFF chunks
│   │   ├── cmip6/             # NEX-GDDP-CMIP6 clipped NetCDF files
│   │   │   ├── ssp245/        # SSP2-4.5 scenario (pr, tas, tasmin, tasmax)
│   │   │   └── ssp585/        # SSP5-8.5 scenario
│   │   └── shapefiles/        # Watershed boundary GeoJSON
│   ├── master/
│   │   ├── nahr_ibrahim_master_full.csv   # All features (9,497 × 16)
│   │   └── nahr_ibrahim_master_model.csv  # Model-ready subset
│   ├── sequences/
│   │   ├── X_train.npy        # (6,545, 30, 12)
│   │   ├── X_val.npy          # (1,066, 30, 12)
│   │   ├── X_test.npy         # (1,796, 30, 12)
│   │   ├── y_train.npy        # (6,545,)
│   │   ├── y_val.npy          # (1,066,)
│   │   ├── y_test.npy         # (1,796,)
│   │   └── dates_*.npy        # Date arrays for each split
│   └── splits/
│       └── scaler_params.csv  # Min-max scaler parameters (train only)
│
├── models/
│   ├── trained/               # Saved .keras model files
│   ├── checkpoints/           # Best epoch checkpoints
│   └── configs/               # Training logs and hyperparameter CSVs
│
├── results/
│   ├── figures/               # All plots (PNG, 150 DPI)
│   ├── metrics/               # Performance metric CSVs
│   ├── predictions/           # Model predictions per split (CSV)
│   └── scenarios/             # Climate scenario discharge CSVs
│
├── src/
│   ├── preprocess.py          # Data loading, cleaning, feature engineering
│   ├── split.py               # Normalisation and chronological splitting
│   ├── windowing.py           # Sequence creation (30-day lookback)
│   ├── download_cmip6.py      # NEX-GDDP-CMIP6 THREDDS download
│   ├── climate_scenarios.py   # Full 6-model climate projection pipeline
│   ├── data_validation.py     # Giovanni vs Open-Meteo ERA5 validation
│   ├── data_validation_meteostat.py  # Giovanni vs Meteostat station validation
│   └── models/
│       ├── lstm.py            # LSTM v1 baseline
│       ├── cnn_lstm.py        # CNN-LSTM hybrid
│       ├── transformer.py     # Transformer encoder
│       ├── tft.py             # Temporal Fusion Transformer
│       └── pi_models.py       # PI-LSTM and PI-Transformer (hybrid)
│
├── app.py                     # Streamlit dashboard
├── data_validation.py         # Root-level validation script
├── download_meteostat.py      # Meteostat station data download
└── README.md
```

---

## Data Sources

### Historical Inputs (2000–2025)

| Variable | Product | Resolution | Platform | URL |
|---|---|---|---|---|
| Precipitation | GPM IMERG Final Run | 0.1° / daily | NASA Giovanni | https://giovanni.gsfc.nasa.gov |
| Mean temperature | MERRA-2 T2MMEAN | 0.5°×0.625° / daily | NASA Giovanni | https://giovanni.gsfc.nasa.gov |
| Min temperature | MERRA-2 T2MMIN | 0.5°×0.625° / daily | NASA Giovanni | https://giovanni.gsfc.nasa.gov |
| Max temperature | MERRA-2 T2MMAX | 0.5°×0.625° / daily | NASA Giovanni | https://giovanni.gsfc.nasa.gov |
| Snow water equivalent | GLDAS Noah SWE_inst | 0.25° / 3-hourly | NASA Giovanni | https://giovanni.gsfc.nasa.gov |
| Snow cover area | MODIS MOD10A1.061 | 500 m / daily | NASA AppEEARS | https://appeears.earthdatacloud.nasa.gov |
| River discharge (surrogate) | GloFAS-ERA5 v4.0 | 0.05° / daily | Copernicus CDS | https://cds.climate.copernicus.eu |

> **Note on discharge:** Real in-situ gauge data for Nahr Ibrahim is held by the Lebanese Directorate General of Hydraulic and Electrical Resources (DGHER) under the Ministry of Energy and Water. GloFAS-ERA5 is used as a physically consistent surrogate — results represent a lower-bound performance benchmark.

### Climate Projections (2015–2100)

| Product | Model | Scenarios | Variables | Access |
|---|---|---|---|---|
| NEX-GDDP-CMIP6 | MPI-ESM1-2-HR | SSP2-4.5, SSP5-8.5 | pr, tas, tasmin, tasmax | AWS S3: `s3://nex-gddp-cmip6/` |

Spatial subsetting performed via NASA NCCS THREDDS:
```
https://ds.nccs.nasa.gov/thredds/ncss/grid/AMES/NEX/GDDP-CMIP6/
```

### Validation Sources

| Source | Type | Coverage | URL |
|---|---|---|---|
| Beirut Airport (WMO 40100) | Station observations | 2000–2025 | https://meteostat.net/en/station/40100 |
| Rayack (WMO 40102) | Station observations | Partial (958 days) | https://meteostat.net/en/station/40102 |
| Open-Meteo ERA5 | Reanalysis reference | 2000–2025 | https://archive-api.open-meteo.com |

---

## Pipeline

Run scripts in this order:

```bash
# 1. Download CMIP6 data (run once)
python src/download_cmip6.py

# 2. Preprocess all raw data → master dataset
python src/preprocess.py

# 3. Normalise and split
python src/split.py

# 4. Create 30-day sequences
python src/windowing.py

# 5. Train models (run each independently)
python src/models/lstm.py
python src/models/cnn_lstm.py
python src/models/transformer.py
python src/models/tft.py
python src/models/pi_models.py   # trains PI-LSTM and PI-Transformer

# 6. Run climate scenario projections
python src/climate_scenarios.py

# 7. Validate input data (optional)
python src/data_validation.py
python src/data_validation_meteostat.py
```

---

## Feature Engineering

12 input features are engineered from raw data products:

| # | Feature | Description | Source |
|---|---|---|---|
| 0 | `precip_mm_day` | Daily precipitation | GPM IMERG |
| 1 | `precip_3day` | 3-day antecedent precipitation sum | Computed |
| 2 | `precip_7day` | 7-day antecedent precipitation sum | Computed |
| 3 | `temp_mean_c` | Daily mean temperature | MERRA-2 |
| 4 | `temp_max_c` | Daily maximum temperature | MERRA-2 |
| 5 | `temp_min_c` | Daily minimum temperature | MERRA-2 |
| 6 | `temp_range_c` | Daily temperature range | Computed |
| 7 | `swe_mm` | Snow water equivalent | GLDAS Noah |
| 8 | `swe_delta` | Daily SWE change (melt proxy) | Computed |
| 9 | `snow_cover_pct` | MODIS snow cover (NDSI ≥ 40) | MOD10A1 |
| 10 | `month_sin` | Cyclical month encoding — sine | Computed |
| 11 | `month_cos` | Cyclical month encoding — cosine | Computed |

**Data split (chronological, no shuffle):**

| Split | Period | Sequences |
|---|---|---|
| Training | 2000–2017 | 6,545 |
| Validation | 2018–2020 | 1,066 |
| Test | 2021–2025 | 1,796 |

Input shape: `(samples, 30, 12)` — 30-day lookback, 1-day ahead prediction.

---

## Models

### Pure AI Models

| Model | Architecture | Params | Training time (CPU) |
|---|---|---|---|
| **LSTM** | 2-layer stacked LSTM (128→64) + Dense(32) | 123,713 | ~15 min |
| **CNN-LSTM** | Conv1D(64,32) + LSTM(128,64) + Dense(32) | 142,497 | ~7 min |
| **Transformer** | 3-block encoder, d=64, 4 heads, FFN=128 | ~108,000 | ~11 min |
| **TFT** | VSN + GRN + LSTM encoder + interpretable attention | 299,773 | ~15 min |

### Physics-Informed Hybrid Models

Both models use the water balance loss:

```
L_PI = MSE(Q_pred, Q_obs) + λ · E[(P − ET − Q_pred − ΔS)²]
```

Where `λ = 0.05`, ET computed via Hamon (1961) PET method.

| Model | Base architecture | Params | Training time |
|---|---|---|---|
| **PI-LSTM** | LSTM + water balance constraint | 123,713 | ~3 min |
| **PI-Transformer** | Transformer + water balance constraint | 107,649 | ~2 min |

---

## Results

### Historical Benchmark (Test period: 2021–2025)

| Rank | Model | Type | NSE | KGE | Peak Bias | PBIAS |
|---|---|---|---|---|---|---|
| 1 | **Transformer** | Pure AI | **0.603** | **0.632** | −41.2% | −3.6% |
| 2 | CNN-LSTM | Pure AI | 0.534 | 0.599 | −39.5% | +7.9% |
| 3 | PI-Transformer | Hybrid | 0.528 | **0.660** | **−37.8%** | +4.1% |
| 4 | LSTM | Pure AI | 0.518 | 0.503 | −43.3% | +3.8% |
| 5 | PI-LSTM | Hybrid | 0.512 | 0.486 | −47.2% | +2.6% |
| 6 | TFT | Pure AI | 0.131 | 0.181 | −69.1% | — |

> **Key findings:**
> - Transformer achieves best NSE (0.603) and lowest RMSE
> - PI-Transformer achieves best KGE (0.660) and best Peak Bias (−37.8%)
> - TFT failed to converge (best epoch = 3) — insufficient training data for variable selection networks
> - All models show systematic peak flow underestimation attributable to GloFAS target smoothing

---

## Climate Scenarios

**Model:** MPI-ESM1-2-HR · **Period:** 2015–2100 · **Dataset:** NEX-GDDP-CMIP6

### Projected Discharge Trends

| Scenario | Mean Q 2015–2040 | Mean Q 2075–2100 | Change | Trend | Significant |
|---|---|---|---|---|---|
| **SSP2-4.5** | 0.562 m³/s | 0.544 m³/s | **−3.3%** | −0.0031 m³/s/decade | ✓ |
| **SSP5-8.5** | 0.567 m³/s | 0.518 m³/s | **−8.5%** | −0.0080 m³/s/decade | ✓ |

### Pure AI vs Physics-Informed Ensemble (SSP5-8.5)

| Ensemble | Trend |
|---|---|
| Pure AI (LSTM + CNN-LSTM + Transformer) | −0.0065 m³/s/decade |
| Physics-Informed (PI-LSTM + PI-Transformer) | **−0.0096 m³/s/decade** |

Physics-informed models detect a **48% steeper decline** under high emissions — attributable to the explicit ET term in the water balance constraint propagating the thermodynamic warming signal into discharge projections.

### Generated Figures

| Figure | Description |
|---|---|
| `scenario_annual_discharge.png` | Projected annual discharge 2015–2100, all models |
| `scenario_pure_vs_pi.png` | Pure AI vs physics-informed comparison |
| `scenario_seasonal_shift.png` | Seasonal discharge shift by 30-year period |
| `scenario_trend_summary.png` | Trend bar chart, all models and scenarios |
| `scenario_climate_forcing.png` | CMIP6 precipitation and temperature forcing |
| `scenario_pi_vs_pure_divergence.png` | Long-term ensemble divergence |

---

## Data Validation

### Giovanni vs Meteostat (Beirut Airport, WMO 40100)

| Variable | r (daily) | r (monthly) | Bias | Quality |
|---|---|---|---|---|
| Precipitation | 0.513 | 0.653 | −6.2% | Moderate — orographic effect |
| Mean temperature | 0.975 | 0.994 | −2.4°C | Excellent |
| Min temperature | 0.953 | 0.994 | −2.4°C | Excellent |
| Max temperature | 0.955 | 0.994 | −2.5°C | Excellent |

> Temperature bias of ~−2.4°C is physically expected from the 871 m elevation difference between Beirut Airport (29 m) and the watershed mean (~900 m). Monthly precipitation r of 0.653 confirms IMERG captures the correct seasonal regime.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/nahr-ibrahim-ai-hydrology.git
cd nahr-ibrahim-ai-hydrology

# Create conda environment
conda create -n thesis python=3.11 -y
conda activate thesis

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
tensorflow==2.21.0
numpy>=1.26.0
pandas>=2.2.0
xarray>=2024.0
scipy>=1.11.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
netCDF4>=1.6.0
cfgrib>=0.9.10
eccodes>=1.6.0
requests>=2.31.0
boto3>=1.28.0
tqdm>=4.65.0
```

---

## Usage

### Quick Start — Run Climate Scenarios on Pretrained Models

```python
# Load a trained model
import tensorflow as tf
model = tf.keras.models.load_model(
    "models/trained/transformer_final.keras",
    custom_objects={"nse_metric": nse_metric}
)

# Predict on test sequences
y_pred = model.predict(X_test, batch_size=256)
```

### Download CMIP6 Data for a Custom Bounding Box

```python
# Edit src/download_cmip6.py
NORTH = 34.25   # your bounding box
SOUTH = 33.99
WEST  = 35.60
EAST  = 36.05
MODEL    = "MPI-ESM1-2-HR"
SCENARIOS = ["ssp245", "ssp585"]
VARIABLES = ["pr", "tas", "tasmin", "tasmax"]
```

### Streamlit Dashboard

```bash
streamlit run app.py
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| TensorFlow | 2.21.0 | Model training and inference |
| NumPy | ≥1.26.0 | Array operations |
| Pandas | ≥2.2.0 | Data handling |
| xarray | ≥2024.0 | NetCDF4 processing (CMIP6) |
| SciPy | ≥1.11.0 | Mann-Kendall trend test |
| Matplotlib | ≥3.7.0 | Visualisation |
| cfgrib / eccodes | latest | GloFAS GRIB file reading |
| boto3 | ≥1.28.0 | AWS S3 access (CMIP6) |
| requests | ≥2.31.0 | THREDDS API download |

---

## Known Limitations

**1. GloFAS discharge ceiling**
All models are trained on GloFAS-ERA5 reanalysis discharge rather than real in-situ gauge data. This introduces a systematic performance ceiling (NSE ~0.60 vs ~0.80+ with real gauge data) due to GloFAS smoothing of karstic spring pulses. Real discharge data for Nahr Ibrahim is held by DGHER (Lebanese Ministry of Energy and Water).

**2. TFT convergence**
The Temporal Fusion Transformer failed to converge on the 6,545-sample training set (best epoch = 3, NSE = 0.131), consistent with the known data requirements of variable selection networks. TFT is excluded from the climate scenario ensemble.

**3. Single GCM**
Climate projections use only MPI-ESM1-2-HR. Multi-model ensemble analysis with additional CMIP6 models would provide uncertainty bounds on projected discharge trends.

**4. Single watershed**
Results are specific to Nahr Ibrahim. Transfer to other Lebanese or Eastern Mediterranean watersheds requires retraining or transfer learning.

---

## References

- Andraos, C., & Najem, W. (2020). Multi-model approach for reducing uncertainties in rainfall–runoff models. *Advances in Hydroinformatics*, Springer.
- Andraos, C. (2024a). Enhancing low-flow forecasts: A multi-model approach. *Hydrology*, 11(3), 35.
- Andraos, C. (2024b). Breaking uncertainty barriers: ABC advances in rainfall–runoff modeling. *Water*, 16(23), 3499.
- Bhasme, P., Vagadiya, J., & Bhatia, U. (2022). Enhancing predictive skills in physically-consistent way. *Journal of Hydrology*, 613, 128038.
- Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error. *Journal of Hydrology*, 377(1–2), 80–91.
- Hamon, W. R. (1961). Estimating potential evapotranspiration. *Journal of the Hydraulics Division*, 87(3), 107–120.
- Harrigan, S., et al. (2020). GloFAS-ERA5 operational global river discharge reanalysis. *Earth System Science Data*, 12(3), 2043–2060.
- Hoedt, P.-J., et al. (2021). MC-LSTM: Mass-conserving LSTM. *ICML 2021*, PMLR 139, 4275–4286.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
- Kratzert, F., et al. (2018). Rainfall–runoff modelling using LSTM. *Hydrology and Earth System Sciences*, 22(11), 6005–6022.
- Kratzert, F., et al. (2019). Towards learning universal, regional, and local hydrological behaviors. *Hydrology and Earth System Sciences*, 23(12), 5089–5110.
- Lim, B., et al. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4), 1748–1764.
- Nash, J. E., & Sutcliffe, J. V. (1970). River flow forecasting through conceptual models. *Journal of Hydrology*, 10(3), 282–290.
- Reichstein, M., et al. (2019). Deep learning and process understanding for data-driven Earth system science. *Nature*, 566(7743), 195–204.
- Tekeli, A. E., et al. (2005). Using MODIS snow cover maps in modeling snowmelt runoff. *Remote Sensing of Environment*, 97(2), 216–230.
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS 2017*.
- Zhang, Y., et al. (2025). CNN-LSTM with attention for streamflow forecasting. *Journal of Hydrology*, 632, 130901.

---

## License

This repository is part of an academic thesis. Code is provided for research and educational purposes.

---

## Contact

**Institution:** Saint Joseph University Beirut — Faculty of Engineering (ESIB)  
**Supervisor:** Dr. Cynthia Andraos  
**Watershed:** Nahr Ibrahim, Lebanon (34.09°N, 35.88°E)