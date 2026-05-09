import ee
import pandas as pd
from pathlib import Path

ee.Initialize(project="final-project-490411")

OUTPUT_DIR = Path("C:/Users/marck/Downloads/nahr_ibrahim_watershed/data/raw/era5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

geometry = ee.Geometry.Rectangle([35.84, 34.02, 35.96, 34.16])

all_records = []

for year in range(2000, 2026):
    out_csv = OUTPUT_DIR / f"era5_precip_{year}.csv"
    if out_csv.exists():
        print(f"  [{year}] skipped")
        df = pd.read_csv(out_csv, parse_dates=["date"])
        all_records.append(df)
        continue

    print(f"  [{year}] downloading ...", end=" ", flush=True)
    try:
        collection = (
            ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .select("total_precipitation_sum")
        )

        def extract(image):
            mean = image.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=geometry, scale=11132, maxPixels=1e9
            )
            return ee.Feature(
                None,
                {
                    "date": image.date().format("YYYY-MM-dd"),
                    "precip_m": mean.get("total_precipitation_sum"),
                },
            )

        fc = ee.FeatureCollection(collection.map(extract))
        data = fc.getInfo()

        records = []
        for f in data["features"]:
            p = f["properties"]
            records.append(
                {
                    "date": p["date"],
                    "precip_mm_day": float(p["precip_m"] or 0) * 1000,  # m → mm
                }
            )

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df["precip_mm_day"] = df["precip_mm_day"].clip(lower=0)
        df = df.sort_values("date").reset_index(drop=True)
        df.to_csv(out_csv, index=False)
        all_records.append(df)
        print(f"OK  mean={df['precip_mm_day'].mean():.2f} mm/day  days={len(df)}")

    except Exception as e:
        print(f"FAILED — {e}")

# Combine
print("\n  Combining all years ...")
df_all = pd.concat(all_records, ignore_index=True)
df_all = df_all.sort_values("date").reset_index(drop=True)
out = OUTPUT_DIR / "era5_precip_2000_2025_daily.csv"
df_all.to_csv(out, index=False)
print(f"  Combined : {len(df_all)} days")
print(f"  Period   : {df_all.date.min().date()} → {df_all.date.max().date()}")
print(f"  Mean     : {df_all.precip_mm_day.mean():.3f} mm/day")
print(f"  Max      : {df_all.precip_mm_day.max():.3f} mm/day")
print(f"  Saved    → {out}")
print("=" * 55)
