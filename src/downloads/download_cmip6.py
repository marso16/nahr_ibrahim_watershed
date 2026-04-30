import requests
import xarray as xr
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
NORTH = 34.2120
SOUTH = 33.9929
WEST  = 35.6429
EAST  = 36.0487

MODEL    = "MPI-ESM1-2-HR"
REALIZ   = "r1i1p1f1"
VERSION  = "v2.0"

SCENARIOS = ["ssp245", "ssp585"]
VARIABLES = ["pr", "tas", "tasmin", "tasmax"]  
YEARS     = range(2015, 2101)

BASE_URL   = "https://ds.nccs.nasa.gov/thredds/ncss/grid/AMES/NEX/GDDP-CMIP6"
OUTPUT_DIR = Path("C:/Users/marck/Downloads/nahr_ibrahim_watershed/data/raw/cmip6")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# BUILD THREDDS SUBSET URL
# =============================================================================
def build_url(scenario: str, variable: str, year: int) -> str:
    filename = (f"{variable}_day_{MODEL}_{scenario}_"
                f"{REALIZ}_gn_{year}_{VERSION}.nc")
    base = (f"{BASE_URL}/{MODEL}/{scenario}/"
            f"{REALIZ}/{variable}/{filename}")
    params = (f"?var={variable}"
              f"&north={NORTH}&south={SOUTH}"
              f"&west={WEST}&east={EAST}"
              f"&horizStride=1"
              f"&time_start={year}-01-01T00:00:00Z"
              f"&time_end={year}-12-31T23:59:59Z"
              f"&accept=netcdf4"
              f"&addLatLon=true")
    return base + params

# =============================================================================
# DOWNLOAD ONE FILE
# =============================================================================

def download_subset(scenario: str, variable: str, year: int) -> str:
    out_path = OUTPUT_DIR / scenario / variable / f"{year}.nc"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        return "skipped"

    url = build_url(scenario, variable, year)

    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()

        with open(out_path, "wb") as f:
            f.write(response.content)

        size_kb = len(response.content) / 1024
        return f"ok ({size_kb:.0f} KB)"

    except Exception as e:
        return f"failed: {e}"

# =============================================================================
# VERIFY ONE FILE BEFORE FULL DOWNLOAD
# =============================================================================
print("=" * 55)
print("  Nahr Ibrahim — CMIP6 Download (THREDDS Subsetting)")
print("=" * 55)
print(f"\n  Model    : {MODEL}")
print(f"  Scenarios: {', '.join(SCENARIOS)}")
print(f"  Variables: {', '.join(VARIABLES)}")
print(f"  Period   : {min(YEARS)}–{max(YEARS)}")
print(f"  Bbox     : {SOUTH}–{NORTH}°N, {WEST}–{EAST}°E")
print(f"  Output   : {OUTPUT_DIR}")

print("\n  Running test download (ssp245/pr/2015) ...")
test_result = download_subset("ssp245", "pr", 2015)
print(f"  Test result: {test_result}")

if test_result.startswith("ok") or test_result == "skipped":
    # Verify the file
    test_path = OUTPUT_DIR / "ssp245" / "pr" / "2015.nc"
    if test_path.exists():
        ds = xr.open_dataset(test_path)
        print(f"  Variables : {list(ds.data_vars)}")
        print(f"  Lat range : {ds.lat.values}")
        print(f"  Lon range : {ds.lon.values}")
        print(f"  Time steps: {len(ds.time)} days")
        ds.close()
    print("\n  ✅ Test passed — starting full download ...\n")
else:
    print(f"\n  ❌ Test failed: {test_result}")
    print("  Check your internet connection and try again.")
    exit(1)

# =============================================================================
# FULL DOWNLOAD LOOP
# =============================================================================
total   = len(SCENARIOS) * len(VARIABLES) * len(YEARS)
count   = 0
results = {"ok": 0, "skipped": 0, "failed": 0}
failed_files = []

print(f"  Total files : {total}")
print(f"  Est. size   : ~{total * 36 / 1024:.0f} MB")
print(f"  Est. time   : ~{total * 30 / 3600:.1f} hours\n")

for scenario in SCENARIOS:
    for variable in VARIABLES:
        for year in YEARS:
            count += 1
            result = download_subset(scenario, variable, year)

            if result.startswith("ok"):
                results["ok"] += 1
            elif result == "skipped":
                results["skipped"] += 1
            else:
                results["failed"] += 1
                failed_files.append(f"{scenario}/{variable}/{year}")

            print(f"[{count:>4}/{total}] {scenario}/{variable}/{year} → {result}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print(f"\n{'='*55}")
print(f"  DOWNLOAD COMPLETE")
print(f"{'='*55}")
print(f"  Downloaded : {results['ok']}")
print(f"  Skipped    : {results['skipped']} (already existed)")
print(f"  Failed     : {results['failed']}")

if failed_files:
    print(f"\n  Failed files ({len(failed_files)}):")
    for f in failed_files[:10]:
        print(f"    {f}")
    if len(failed_files) > 10:
        print(f"    ... and {len(failed_files)-10} more")
    print(f"\n  To retry failed files, simply rerun this script.")
    print(f"  Already downloaded files will be skipped automatically.")

# =============================================================================
# VERIFY COMPLETE DOWNLOAD
# =============================================================================
print(f"\n  Verifying download completeness ...")
print(f"  {'Scenario':<10} {'Variable':<10} {'Files':>8} {'Expected':>10} {'Status':>10}")
print(f"  {'-'*52}")

expected = len(YEARS)
all_complete = True

for scenario in SCENARIOS:
    for variable in VARIABLES:
        folder = OUTPUT_DIR / scenario / variable
        found  = len(list(folder.glob("*.nc"))) if folder.exists() else 0
        status = "✅ Complete" if found == expected else f"⚠ {found}/{expected}"
        if found != expected:
            all_complete = False
        print(f"  {scenario:<10} {variable:<10} {found:>8} {expected:>10} {status:>10}")

if all_complete:
    print(f"\n  ✅ All files downloaded successfully.")
    print(f"     Ready to run: python src/climate_scenarios.py")
else:
    print(f"\n  ⚠ Some files missing. Rerun this script to retry.")
print(f"{'='*55}")