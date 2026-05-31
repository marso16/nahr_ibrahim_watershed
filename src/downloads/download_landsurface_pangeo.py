import os
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(
    os.environ.get("WATERSHED_ROOT", "C:/Users/marck/Downloads/nahr_ibrahim_watershed")
)
GEOJSON = ROOT / "data" / "raw" / "shapefiles" / "nahr_ibrahim_watershed.geojson"
OUT_DIR = ROOT / "data" / "raw" / "cmip6" / "landsurface"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Configuration ──────────────────────────────────────────────────────────
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
MEMBER = "r1i1p1f1"  # first ensemble member, common standard

# Variables we want (Pangeo CMIP6 names)
# mrsos = soil moisture in top 10 cm (kg/m²)
# snw   = snow water equivalent (kg/m²)
VARIABLES = ["mrsos", "snw"]

# Catalog URL (Pangeo CMIP6, hosted by Google Cloud)
CATALOG_URL = "https://storage.googleapis.com/cmip6/pangeo-cmip6.csv"

# Watershed bounding box (lon/lat). We grab a small bbox around it and
# spatially average. CMIP6 native resolution varies 50-250 km, so for our
# 330 km² watershed we'll often get only 1 grid cell.
BBOX_BUFFER_DEG = 1.0  # widen bbox by 1° in each direction to ensure at least one cell


# ─── Init ───────────────────────────────────────────────────────────────────
print("Loading dependencies...")
try:
    import xarray as xr
    import gcsfs
    import geopandas as gpd
except ImportError as e:
    print(f"\nMissing dependency: {e}")
    print("Install with:  pip install xarray zarr gcsfs geopandas")
    sys.exit(1)

print("  xarray OK, gcsfs OK, geopandas OK\n")


# ─── Watershed bounds ──────────────────────────────────────────────────────
ws = gpd.read_file(GEOJSON)
if ws.crs is None or ws.crs.to_epsg() != 4326:
    ws = ws.to_crs(epsg=4326)
minx, miny, maxx, maxy = ws.total_bounds
print(f"Watershed bounds (lon/lat): {minx:.3f}, {miny:.3f} → {maxx:.3f}, {maxy:.3f}")

# Widen bbox to ensure we have grid cells even with coarse GCMs
WEST = minx - BBOX_BUFFER_DEG
EAST = maxx + BBOX_BUFFER_DEG
SOUTH = miny - BBOX_BUFFER_DEG
NORTH = maxy + BBOX_BUFFER_DEG
print(
    f"Buffered bbox for Pangeo:   {WEST:.2f}, {SOUTH:.2f} → {EAST:.2f}, {NORTH:.2f}\n"
)


# ─── Load Pangeo catalog ────────────────────────────────────────────────────
print(f"Loading Pangeo catalog from {CATALOG_URL}...")
t0 = time.time()
try:
    cat = pd.read_csv(CATALOG_URL)
except Exception as e:
    print(f"  Failed: {e}")
    print("  Check your internet connection and that the catalog URL is current.")
    sys.exit(1)
print(f"  Loaded {len(cat):,} entries in {time.time()-t0:.1f}s\n")


# ─── Helper: find zarr store for one combination ──────────────────────────
def find_zstore(
    gcm: str, scenario: str, variable: str, member: str = MEMBER
) -> str | None:
    """Search the catalog for a matching daily zarr store. Returns URL or None."""
    matches = cat[
        (cat.source_id == gcm)
        & (cat.experiment_id == scenario)
        & (cat.variable_id == variable)
        & (cat.member_id == member)
        & (cat.table_id == "day")
    ]
    if len(matches) == 0:
        # Try without member constraint as fallback
        matches = cat[
            (cat.source_id == gcm)
            & (cat.experiment_id == scenario)
            & (cat.variable_id == variable)
            & (cat.table_id == "day")
        ]
        if len(matches) == 0:
            return None

    # If multiple grid_label or version variants, prefer "gn" (native grid) first row
    if "grid_label" in matches.columns:
        gn_matches = matches[matches.grid_label == "gn"]
        if len(gn_matches) > 0:
            matches = gn_matches
    return matches.iloc[0]["zstore"]


# ─── Helper: open zarr, subset, mean, save CSV ─────────────────────────────
def fetch_variable(gcm: str, scenario: str, variable: str) -> pd.DataFrame:
    zstore_url = find_zstore(gcm, scenario, variable)
    if zstore_url is None:
        return pd.DataFrame()  # empty signals "not available"

    ds = xr.open_zarr(
        zstore_url,
        consolidated=True,
        storage_options={"token": "anon"},
    )

    # Pangeo CMIP6 lon is typically 0-360. Convert our bbox if needed.
    if float(ds.lon.min()) >= 0 and WEST < 0:
        west_q = WEST % 360
        east_q = EAST % 360
    else:
        west_q = WEST
        east_q = EAST

    # Latitude may be ascending or descending — try both
    sub = ds.sel(lat=slice(SOUTH, NORTH), lon=slice(west_q, east_q))
    if sub.lat.size == 0:
        sub = ds.sel(lat=slice(NORTH, SOUTH), lon=slice(west_q, east_q))
    if sub.lat.size == 0 or sub.lon.size == 0:
        ds.close()
        raise RuntimeError(
            f"Bbox missed grid for {gcm}/{scenario}/{variable}. "
            f"Available lat: {float(ds.lat.min()):.2f}–{float(ds.lat.max()):.2f}, "
            f"lon: {float(ds.lon.min()):.2f}–{float(ds.lon.max()):.2f}"
        )

    # Spatial mean over the subset cells
    da = sub[variable].mean(dim=["lat", "lon"], skipna=True)

    # Materialize (forces lazy reads to happen)
    values = da.values
    times = pd.to_datetime(da.time.values).strftime("%Y-%m-%d")

    df = pd.DataFrame({"date": times, variable: values})
    ds.close()
    return df


# ─── Main loop ──────────────────────────────────────────────────────────────
print(f"{'=' * 70}")
print(
    f"  Downloading {len(GCMS)} GCMs × {len(SCENARIOS)} scenarios × {len(VARIABLES)} vars"
)
print(f"  = {len(GCMS) * len(SCENARIOS) * len(VARIABLES)} combinations")
print(f"{'=' * 70}\n")

t_start = time.time()
ok = 0
skipped = 0
missing = 0
failed = 0
fails = []

for gcm in GCMS:
    for scenario in SCENARIOS:
        out_path = OUT_DIR / gcm / scenario / "merged_daily.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Resumable check
        if out_path.exists() and out_path.stat().st_size > 0:
            try:
                existing = pd.read_csv(out_path, nrows=2)
                has_all_vars = all(v in existing.columns for v in VARIABLES)
                if has_all_vars:
                    skipped += 1
                    print(f"  [skip] {gcm}/{scenario}: already complete")
                    continue
            except Exception:
                pass  # corrupt file — re-download

        print(f"  [fetch] {gcm}/{scenario}")
        t_combo = time.time()

        per_var_df = {}
        any_failed = False
        any_missing = False

        for variable in VARIABLES:
            try:
                df = fetch_variable(gcm, scenario, variable)
                if df.empty:
                    print(f"           {variable}: NOT AVAILABLE in catalog")
                    any_missing = True
                    missing += 1
                else:
                    per_var_df[variable] = df
                    print(f"           {variable}: {len(df)} days")
            except Exception as e:
                msg = str(e)[:120]
                print(f"           {variable}: FAILED — {msg}")
                fails.append((gcm, scenario, variable, msg))
                failed += 1
                any_failed = True

        # Merge variables into a single CSV if we got at least one
        if per_var_df:
            merged = None
            for var, df in per_var_df.items():
                merged = (
                    df if merged is None else merged.merge(df, on="date", how="outer")
                )
            merged = merged.sort_values("date").reset_index(drop=True)
            merged.to_csv(out_path, index=False)
            ok += 1
            elapsed = time.time() - t_combo
            print(
                f"           → saved {out_path.name} ({len(merged)} rows, {elapsed:.1f}s)"
            )
        else:
            print(f"           → no variables retrieved; nothing saved")

elapsed_total = (time.time() - t_start) / 60
print(f"\n{'=' * 70}")
print(f"Done. {elapsed_total:.1f} minutes total.")
print(f"  Combinations saved (≥1 variable): {ok}")
print(f"  Already complete:                 {skipped}")
print(f"  Variables not in catalog:         {missing}")
print(f"  Variables failed to download:     {failed}")
print(f"{'=' * 70}")

if fails:
    print(f"\nDetailed failures:")
    for gcm, scen, var, err in fails[:30]:
        print(f"  {gcm}/{scen}/{var}: {err}")

# Final inventory
print(f"\n{'=' * 70}\n  Inventory\n{'=' * 70}")
print(f"  {'GCM':<20} {'Scenario':<14} {'Variables':<25} {'Rows':>8}")
print(f"  {'-' * 72}")
for gcm in GCMS:
    for scen in SCENARIOS:
        p = OUT_DIR / gcm / scen / "merged_daily.csv"
        if p.exists():
            df = pd.read_csv(p, nrows=10000)
            vars_present = [v for v in VARIABLES if v in df.columns]
            n = len(pd.read_csv(p))
            print(f"  {gcm:<20} {scen:<14} {', '.join(vars_present):<25} {n:>8}")
        else:
            print(f"  {gcm:<20} {scen:<14} {'(none)':<25} {'-':>8}")
