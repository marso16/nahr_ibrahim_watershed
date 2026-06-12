$json = @'
{
  "calibration_window": {
    "start": "2000-01-01",
    "end": "2017-12-31",
    "n_days": 6557
  },
  "snow_model": {
    "T_snow_C": 0.06,
    "melt_factor_mm_per_C_per_day": 4.28,
    "nse": -0.68,
    "correlation": 0.73,
    "mae_mm": 1.87
  },
  "bucket_model": {
    "field_capacity_mm": 281.9,
    "wilting_point_mm": 117.2,
    "drainage_rate_per_day": 0.9124,
    "ET_scale": 0.178,
    "nse": 0.55,
    "correlation": 0.75,
    "mae_mm": 24.47,
    "calibrated_against": "ERA5-Land sm_28_100cm_mm"
  },
  "notes": [
    "Snow model: degree-day calibrated against ERA5-Land swe_mm. NSE is negative because snow is intermittent in this rain-dominated Mediterranean catchment; correlation (0.73) is the meaningful metric.",
    "Bucket: single-layer water balance against the 28-100cm soil moisture layer. Drainage rate = 0.9124 day^-1 reflects karstic geology.",
    "Inflow to bucket is rain (excluding snowfall) plus snowmelt.",
    "Target = sm_28_100cm_mm because that's what the LSTM uses as 'soil_moisture_mm'."
  ]
}
'@

$path = "C:/Users/marck/Downloads/nahr_ibrahim_watershed/models/trained/landsurface_params.json"
$json | Out-File -Encoding utf8 $path
Write-Host "Fixed. Contents:" -ForegroundColor Green
Get-Content $path