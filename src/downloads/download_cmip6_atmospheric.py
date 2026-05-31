import os
import sys
import time
from pathlib import Path
import ee
import geopandas as gpd
import pandas as pd

# ─── Configuration ──────────────────────────────────────────────────────────
ROOT = Path(
    os.environ.get("WATERSHED_ROOT", "C:/Users/marck/Downloads/nahr_ibrahim_watershed")
)
GEOJSON = ROOT / "data" / "raw" / "shapefiles" / "nahr_ibrahim_watershed.geojson"
OUT_DIR = ROOT / "data" / "raw" / "cmip6" / "atmospheric"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GEE_PROJECT = "final-project-490411"

# All 7 verified models. Comment out any you don't want.
GCMS = [
    "MPI-ESM1-2-HR",
    "EC-Earth3",
    "ACCESS-CM2",
    "NorESM2-MM",
    "MRI-ESM2-0",
    "CMCC-ESM2",
    "INM-CM5-0",
]

SCENARIOS = ["historical", "ssp245", "ssp585"]

# Core atmospheric variables. We get all 4 in one request per year.
VARIABLES = ["pr", "tas", "tasmin", "tasmax"]

# Year ranges per scenario.
# Historical: 1995-2014 is plenty for bias correction (20-yr overlap with ERA5).
# Future: full 2015-2100 to get mid-century (2041-60) AND end-century (2081-2100).
SCENARIO_YEARS = {
    "historical": range(1995, 2015),
    "ssp245": range(2015, 2101),
    "ssp585": range(2015, 2101),
}

SCALE_M = 25000  # NASA GDDP native resolution ~0.25° ≈ 27.8 km. 25 km is a safe value.


# ─── Init Earth Engine ──────────────────────────────────────────────────────
try:
    ee.Initialize(project=GEE_PROJECT)
    print(f"GEE initialized (project={GEE_PROJECT})\n")
except Exception as e:
    print(f"GEE init failed: {e}")
    print("Run: earthengine authenticate")
    sys.exit(1)


# ─── Watershed geometry ─────────────────────────────────────────────────────
ws = gpd.read_file(GEOJSON)
if ws.crs is None or ws.crs.to_epsg() != 4326:
    ws = ws.to_crs(epsg=4326)
geom_coords = ws.__geo_interface__["features"][0]["geometry"]
watershed = ee.Geometry(geom_coords)
print(f"Watershed loaded from {GEOJSON.name}")
print(f"Bounds (lon/lat): {ws.total_bounds}\n")


# ─── Helper: one year of one GCM × scenario ─────────────────────────────────
def image_to_feature(image):
    """Reduce one daily image to watershed mean for all selected variables."""
    means = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=watershed,
        scale=SCALE_M,
        bestEffort=True,
        maxPixels=1e9,
    )
    props = {"date": image.date().format("YYYY-MM-dd")}
    for var in VARIABLES:
        props[var] = means.get(var)
    return ee.Feature(None, props)


def fetch_year(gcm: str, scenario: str, year: int) -> pd.DataFrame:
    """Fetch a single year of daily watershed-averaged variables."""
    coll = (
        ee.ImageCollection("NASA/GDDP-CMIP6")
        .filter(ee.Filter.eq("model", gcm))
        .filter(ee.Filter.eq("scenario", scenario))
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .select(VARIABLES)
    )

    features = coll.map(image_to_feature)
    fc = features.getInfo()
    rows = [f["properties"] for f in fc["features"]]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ─── Convert units for downstream use ───────────────────────────────────────
def add_converted_units(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns in the units our pipeline expects."""
    # pr from kg/m²/s → mm/day  (1 kg/m²/s × 86400 s/day = 86400 mm/day)
    df["precip_mm_day"] = df["pr"] * 86400.0
    # Temperatures from K to °C
    df["temp_mean_c"] = df["tas"] - 273.15
    df["temp_max_c"] = df["tasmax"] - 273.15
    df["temp_min_c"] = df["tasmin"] - 273.15
    # Sanity clips
    df["precip_mm_day"] = df["precip_mm_day"].clip(lower=0)
    return df


# ─── Main loop ──────────────────────────────────────────────────────────────
total_jobs = sum(len(SCENARIO_YEARS[s]) for s in SCENARIOS) * len(GCMS)

print(f"Total year-files to download: {total_jobs}")
print(f"  GCMs: {len(GCMS)}")
print(f"  Scenarios: {SCENARIOS}")
print(f"  Variables per file: {VARIABLES}")
print(f"  Spatial scale: {SCALE_M} m\n")

t_start = time.time()
done = 0
skipped = 0
ok = 0
failed = 0
fails = []

for gcm in GCMS:
    for scenario in SCENARIOS:
        out_dir = OUT_DIR / gcm / scenario
        out_dir.mkdir(parents=True, exist_ok=True)
        years = SCENARIO_YEARS[scenario]

        print(f"  [{gcm} / {scenario}] {len(years)} years to do")

        for year in years:
            done += 1
            out_file = out_dir / f"{year}.csv"

            if out_file.exists() and out_file.stat().st_size > 0:
                skipped += 1
                continue

            t_year = time.time()
            try:
                df = fetch_year(gcm, scenario, year)
                if df.empty:
                    raise RuntimeError("Empty result returned")
                df = add_converted_units(df)
                df.to_csv(out_file, index=False)
                elapsed = time.time() - t_year
                ok += 1
                if year % 5 == 0 or year == years[0] or year == years[-1]:
                    pct = 100 * done / total_jobs
                    print(
                        f"    [{done}/{total_jobs}  {pct:.1f}%]  "
                        f"{year}  ok ({len(df)} days, {elapsed:.1f}s)"
                    )
            except Exception as e:
                failed += 1
                fails.append((gcm, scenario, year, str(e)[:150]))
                print(f"    [{done}/{total_jobs}]  {year}  FAILED: {str(e)[:120]}")

elapsed_total = (time.time() - t_start) / 60
print(f"\n{'=' * 60}")
print(f"Done. {elapsed_total:.1f} minutes total.")
print(f"  Newly downloaded: {ok}")
print(f"  Already existed:  {skipped}")
print(f"  Failed:           {failed}")
print(f"{'=' * 60}")

if failed > 0:
    print(f"\nFailed jobs (rerun the script to retry — it's resumable):")
    for gcm, scen, yr, err in fails[:30]:
        print(f"  {gcm}/{scen}/{yr}: {err}")
    if len(fails) > 30:
        print(f"  ... and {len(fails) - 30} more")
