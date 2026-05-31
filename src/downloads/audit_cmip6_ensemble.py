import sys
import ee

GEE_PROJECT = "final-project-490411"

# Candidate ensemble — 7 well-known CMIP6 GCMs. We'll pick 5 from whichever
# pass the full availability check. Including extras gives us fallbacks.
CANDIDATE_GCMS = [
    "MPI-ESM1-2-HR",  # Germany, MPI
    "EC-Earth3",  # European consortium
    "ACCESS-CM2",  # Australia
    "NorESM2-MM",  # Norway
    "MRI-ESM2-0",  # Japan
    "CMCC-ESM2",  # Italy (backup)
    "INM-CM5-0",  # Russia (backup)
]

# Scenarios we need
SCENARIOS = ["historical", "ssp245", "ssp585"]

# Atmospheric variables we want from NASA/GDDP-CMIP6
# pr, tas, tasmin, tasmax are required.
# hurs, rsds are nice-to-have for PET if we improve it later.
REQUIRED_VARIABLES = ["pr", "tas", "tasmin", "tasmax"]
OPTIONAL_VARIABLES = ["hurs", "rsds"]

# Expected date ranges
EXPECTED_HISTORICAL_DAYS = 65 * 365  # 1950-2014, with leap years
EXPECTED_FUTURE_DAYS = 86 * 365  # 2015-2100, with leap years


def banner(text):
    print(f"\n{'=' * 78}\n  {text}\n{'=' * 78}")


def init_gee():
    try:
        ee.Initialize(project=GEE_PROJECT)
        print(f"GEE initialized (project={GEE_PROJECT})")
    except Exception as e:
        print(f"GEE init failed: {e}\nRun: earthengine authenticate")
        sys.exit(1)


def check_collection_meta():
    """Confirm the collection exists and report its bands."""
    coll = ee.ImageCollection("NASA/GDDP-CMIP6")
    first = coll.first()
    bands = first.bandNames().getInfo()
    print(f"\n  Collection: NASA/GDDP-CMIP6")
    print(f"  Available bands: {bands}")

    # Check that all required variables exist as bands
    missing = [v for v in REQUIRED_VARIABLES if v not in bands]
    if missing:
        print(f"  ERROR: required variables missing as bands: {missing}")
        sys.exit(1)

    return bands


def check_gcm_scenario(gcm: str, scenario: str):
    """Return (n_days, first_date, last_date) for a GCM × scenario, or None if not found."""
    coll = (
        ee.ImageCollection("NASA/GDDP-CMIP6")
        .filter(ee.Filter.eq("model", gcm))
        .filter(ee.Filter.eq("scenario", scenario))
    )
    try:
        n = coll.size().getInfo()
    except Exception as e:
        return None, f"size query failed: {e}"

    if n == 0:
        return None, "no images"

    try:
        first_dt = (
            ee.Date(coll.aggregate_min("system:time_start"))
            .format("YYYY-MM-dd")
            .getInfo()
        )
        last_dt = (
            ee.Date(coll.aggregate_max("system:time_start"))
            .format("YYYY-MM-dd")
            .getInfo()
        )
    except Exception as e:
        return n, f"date query failed: {e}"

    return n, (first_dt, last_dt)


def main():
    init_gee()
    banner("Step 1: collection metadata")
    bands = check_collection_meta()

    # Identify which optional variables are also present
    optional_present = [v for v in OPTIONAL_VARIABLES if v in bands]
    optional_missing = [v for v in OPTIONAL_VARIABLES if v not in bands]
    if optional_missing:
        print(f"  Optional variables missing: {optional_missing}")
    if optional_present:
        print(f"  Optional variables present: {optional_present}")

    banner("Step 2: per-GCM scenario coverage")

    # Matrix: rows = GCMs, columns = scenarios, value = day count or "missing"
    print(
        f"  {'GCM':<20} "
        f"{'historical':>13} {'ssp245':>13} {'ssp585':>13}   "
        f"verdict"
    )
    print(f"  {'-' * 80}")

    full_ensemble = []
    partial = []
    failed = []

    for gcm in CANDIDATE_GCMS:
        row_results = {}
        for scen in SCENARIOS:
            n, info = check_gcm_scenario(gcm, scen)
            row_results[scen] = (n, info)

        # Render the row
        cells = []
        for scen in SCENARIOS:
            n, info = row_results[scen]
            if n is None:
                cells.append(f"{'MISSING':>13}")
            else:
                cells.append(f"{n:>13,}")

        # Determine verdict
        all_present = all(row_results[s][0] is not None for s in SCENARIOS)
        if all_present:
            # Also check the date ranges look right
            hist_n = row_results["historical"][0]
            fut_n = row_results["ssp245"][0]
            hist_ok = hist_n > 20000  # rough: at least 55 years
            fut_ok = fut_n > 28000  # rough: at least 78 years
            if hist_ok and fut_ok:
                verdict = "OK — full"
                full_ensemble.append(gcm)
            else:
                verdict = "partial date coverage"
                partial.append(gcm)
        else:
            verdict = "missing scenario(s)"
            failed.append(gcm)

        print(f"  {gcm:<20} " + " ".join(cells) + f"   {verdict}")

    banner("Step 3: date-range detail (best 5 candidates)")
    pick_for_detail = (
        full_ensemble[:5]
        if len(full_ensemble) >= 5
        else full_ensemble + partial[: 5 - len(full_ensemble)]
    )
    for gcm in pick_for_detail:
        print(f"\n  {gcm}")
        for scen in SCENARIOS:
            n, info = check_gcm_scenario(gcm, scen)
            if n is None:
                print(f"    {scen:<12} MISSING")
            else:
                if isinstance(info, tuple):
                    print(f"    {scen:<12} {n:>6,} days   {info[0]} → {info[1]}")
                else:
                    print(f"    {scen:<12} {n:>6,} days   ({info})")

    banner("Recommendation")

    print(f"\n  GCMs with full coverage: {len(full_ensemble)} of {len(CANDIDATE_GCMS)}")
    for g in full_ensemble:
        print(f"    {g}")

    if partial:
        print(f"\n  GCMs with partial coverage: {len(partial)}")
        for g in partial:
            print(f"    {g}")

    if failed:
        print(f"\n  GCMs missing scenarios: {len(failed)}")
        for g in failed:
            print(f"    {g}")

    print()
    if len(full_ensemble) >= 5:
        print(f"  → Use the first 5 from 'full coverage' as your ensemble:")
        print(f"    {full_ensemble[:5]}")
    elif len(full_ensemble) >= 3:
        print(f"  → Only {len(full_ensemble)} GCMs have full coverage.")
        print(f"    Use these; report ensemble size honestly in methodology.")
    else:
        print(f"  → Only {len(full_ensemble)} GCM(s) with full coverage — investigate")
        print(f"    why the others are missing before downloading.")

    print(f"\n  Land-surface note: NASA/GDDP-CMIP6 has NO soil moisture or SWE.")
    print(f"  Those will come from Pangeo for the same GCMs (next script).")


if __name__ == "__main__":
    main()
