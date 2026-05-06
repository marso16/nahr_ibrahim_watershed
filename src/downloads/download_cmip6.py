import ee
import pandas as pd
from pathlib import Path

ee.Initialize(project='final-project-490411')

# =============================================================================
# CONFIGURATION
# =============================================================================
NORTH = 34.16
SOUTH = 34.02
WEST  = 35.84
EAST  = 35.96

MODEL     = "MPI-ESM1-2-HR"
SCENARIOS = ["ssp245", "ssp585"]
VARIABLES = ["pr", "tas", "tasmin", "tasmax"]
YEARS     = range(2015, 2101)

OUTPUT_DIR = Path(r"data\raw\cmip6")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

geometry = ee.Geometry.Rectangle([WEST, SOUTH, EAST, NORTH])

print("=" * 55)
print("  Nahr Ibrahim — CMIP6 Download (Google Earth Engine)")
print("=" * 55)
print(f"\n  Model    : {MODEL}")
print(f"  Scenarios: {', '.join(SCENARIOS)}")
print(f"  Variables: {', '.join(VARIABLES)}")
print(f"  Period   : {min(YEARS)}–{max(YEARS)}")
print(f"  Bbox     : {SOUTH}–{NORTH}°N, {WEST}–{EAST}°E")
print(f"  Output   : {OUTPUT_DIR}\n")

# =============================================================================
# GEE BAND NAME MAP
# =============================================================================
# GEE uses same names: pr, tas, tasmin, tasmax — confirmed from band check
BAND_MAP = {
    "pr"    : "pr",
    "tas"   : "tas",
    "tasmin": "tasmin",
    "tasmax": "tasmax",
}

# =============================================================================
# DOWNLOAD LOOP
# =============================================================================
total   = len(SCENARIOS) * len(VARIABLES) * len(YEARS)
count   = 0
results = {"ok": 0, "skipped": 0, "failed": 0}
failed  = []

for scenario in SCENARIOS:
    for variable in VARIABLES:
        band = BAND_MAP[variable]
        out_dir = OUTPUT_DIR / scenario / variable
        out_dir.mkdir(parents=True, exist_ok=True)

        for year in YEARS:
            count += 1
            out_file = out_dir / f"{year}.csv"

            if out_file.exists():
                print(f"[{count:>4}/{total}] {scenario}/{variable}/{year} → skipped")
                results["skipped"] += 1
                continue

            try:
                collection = (
                    ee.ImageCollection("NASA/GDDP-CMIP6")
                    .filter(ee.Filter.eq("model", MODEL))
                    .filter(ee.Filter.eq("scenario", scenario))
                    .filterDate(f"{year}-01-01", f"{year}-12-31")
                    .select(band)
                )

                def extract(image):
                    mean = image.reduceRegion(
                        reducer  = ee.Reducer.mean(),
                        geometry = geometry,
                        scale    = 25000,
                        maxPixels= 1e9
                    )
                    return ee.Feature(None, {
                        "date" : image.date().format("YYYY-MM-dd"),
                        "value": mean.get(band)
                    })

                fc   = ee.FeatureCollection(collection.map(extract))
                data = fc.getInfo()

                records = []
                for f in data["features"]:
                    records.append({
                        "date" : f["properties"]["date"],
                        "value": f["properties"]["value"]
                    })

                df = pd.DataFrame(records)
                df["date"]  = pd.to_datetime(df["date"])
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                df = df.sort_values("date").reset_index(drop=True)
                df.to_csv(out_file, index=False)

                print(f"[{count:>4}/{total}] {scenario}/{variable}/{year} → "
                      f"ok ({len(df)} days | mean={df['value'].mean():.8f})")
                results["ok"] += 1

            except Exception as e:
                print(f"[{count:>4}/{total}] {scenario}/{variable}/{year} → FAILED — {e}")
                results["failed"] += 1
                failed.append(f"{scenario}/{variable}/{year}")

# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{'='*55}")
print(f"  DOWNLOAD COMPLETE")
print(f"{'='*55}")
print(f"  Downloaded : {results['ok']}")
print(f"  Skipped    : {results['skipped']} (already existed)")
print(f"  Failed     : {results['failed']}")

if failed:
    print(f"\n  Failed files ({len(failed)}):")
    for f in failed[:10]:
        print(f"    {f}")
    if len(failed) > 10:
        print(f"    ... and {len(failed)-10} more")
    print(f"\n  Rerun to retry failed files automatically.")

print(f"\n  Verifying completeness ...")
print(f"  {'Scenario':<10} {'Variable':<10} {'Files':>8} {'Expected':>10} {'Status':>10}")
print(f"  {'-'*52}")
expected    = len(YEARS)
all_complete = True
for scenario in SCENARIOS:
    for variable in VARIABLES:
        folder = OUTPUT_DIR / scenario / variable
        found  = len(list(folder.glob("*.csv"))) if folder.exists() else 0
        status = "Complete" if found == expected else f"{found}/{expected}"
        if found != expected:
            all_complete = False
        print(f"  {scenario:<10} {variable:<10} {found:>8} {expected:>10} {status:>10}")

if all_complete:
    print(f"\n  All files downloaded.")
else:
    print(f"\n  Some files missing. Rerun to retry.")
print(f"{'='*55}")