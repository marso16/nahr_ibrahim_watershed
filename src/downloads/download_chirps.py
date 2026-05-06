import ee
import pandas as pd
from pathlib import Path

ee.Initialize(project='final-project-490411')

OUTPUT = Path(r"data\raw\chirps")
OUTPUT.mkdir(parents=True, exist_ok=True)

geometry = ee.Geometry.Rectangle([35.84, 34.02, 35.96, 34.16])

NORTH = 34.16
SOUTH = 34.02
WEST  = 35.84
EAST  = 35.96

all_records = []

for year in range(2000, 2026):
    print(f"  [{year}] ...", end=" ", flush=True)

    chirps = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                .filterDate(f"{year}-01-01", f"{year}-12-31")
                .filterBounds(geometry)
                .select("precipitation"))

    def extract_mean(image):
        mean = image.reduceRegion(
            reducer   = ee.Reducer.mean(),
            geometry  = geometry,
            scale     = 5000,
            maxPixels = 1e9
        )
        return ee.Feature(None, {
            "date"          : image.date().format("YYYY-MM-dd"),
            "precip_mm_day" : mean.get("precipitation")
        })

    fc   = ee.FeatureCollection(chirps.map(extract_mean))
    data = fc.getInfo()  

    records = []
    for f in data["features"]:
        props = f["properties"]
        records.append({
            "date"          : props.get("date"),
            "precip_mm_day" : props.get("precip_mm_day")
        })

    df_year = pd.DataFrame(records)
    df_year["date"]          = pd.to_datetime(df_year["date"])
    df_year["precip_mm_day"] = pd.to_numeric(
        df_year["precip_mm_day"], errors="coerce"
    ).clip(lower=0)

    mean = df_year["precip_mm_day"].mean()
    print(f"{len(df_year)} days | mean={mean:.2f} mm/day")
    all_records.append(df_year)

df = pd.concat(all_records, ignore_index=True)
df = df.sort_values("date").drop_duplicates(subset="date")
df = df.reset_index(drop=True)

print(f"\n  Total : {len(df):,} days")
print(f"  Period: {df.date.min().date()} → {df.date.max().date()}")
print(f"  Mean  : {df['precip_mm_day'].mean():.3f} mm/day")
print(f"  Max   : {df['precip_mm_day'].max():.2f} mm/day")

df["month"] = df["date"].dt.month
monthly = df.groupby("month")["precip_mm_day"].mean()
names   = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
print(f"\n  Monthly means:")
for m, name in enumerate(names, 1):
    bar = "█" * int(monthly.get(m, 0) / 2)
    print(f"    {name}: {monthly.get(m, 0):5.2f} mm/day  {bar}")

out = OUTPUT / "chirps_nahr_ibrahim_2000_2025_daily.csv"
df.drop(columns=["month"]).to_csv(out, index=False)