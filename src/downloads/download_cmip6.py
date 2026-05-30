import sys
import time
import zlib
import pandas as pd
import numpy as np
from pathlib import Path

# CONFIGURATION =============================================================================
# Watershed bounding box
NORTH = 34.16
SOUTH = 34.02
WEST = 35.84
EAST = 35.96

# CMIP6 selection
MODEL_GEE = "MPI-ESM1-2-HR"
MODEL_PANGEO = "MPI-ESM1-2-HR"
MEMBER = "r1i1p1f1"
SCENARIOS = ["ssp245", "ssp585"]
YEARS = range(2015, 2101)

GEE_VARIABLES = ["pr", "tas", "tasmin", "tasmax"]
PANGEO_VARIABLES = ["snw", "mrsos"]
DERIVED = ["snc"]

GEE_PROJECT = "final-project-490411"

SWE_THRESHOLD_MM = 25.0

OUTPUT_DIR = Path(r"data\raw\cmip6")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PANGEO_CATALOG_CSV = "https://storage.googleapis.com/cmip6/pangeo-cmip6.csv"


def banner(text, char="="):
    line = char * 65
    print(f"\n{line}\n  {text}\n{line}")


# Download atmospheric variables from GEE =============================================================================
def download_from_gee():
    banner("Atmospheric variables from NASA/GDDP-CMIP6 (GEE)")
    try:
        import ee
    except ImportError:
        print("earthengine-api not installed. Run: pip install earthengine-api")
        return

    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception as e:
        print(f"GEE init failed: {e}")
        print("Run `earthengine authenticate` first.")
        return

    geometry = ee.Geometry.Rectangle([WEST, SOUTH, EAST, NORTH])

    total = len(SCENARIOS) * len(GEE_VARIABLES) * len(YEARS)
    count = 0
    ok = skipped = failed = 0
    fails = []

    for scenario in SCENARIOS:
        for variable in GEE_VARIABLES:
            out_dir = OUTPUT_DIR / scenario / variable
            out_dir.mkdir(parents=True, exist_ok=True)

            for year in YEARS:
                count += 1
                out_file = out_dir / f"{year}.csv"

                if out_file.exists():
                    skipped += 1
                    if count % 50 == 0:
                        print(f"  [{count:>4}/{total}] ...skipping existing files")
                    continue

                try:
                    coll = (
                        ee.ImageCollection("NASA/GDDP-CMIP6")
                        .filter(ee.Filter.eq("model", MODEL_GEE))
                        .filter(ee.Filter.eq("scenario", scenario))
                        .filterDate(f"{year}-01-01", f"{year}-12-31")
                        .select(variable)
                    )

                    def extract(image):
                        mean = image.reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=geometry,
                            scale=25000,
                            maxPixels=1e9,
                        )
                        return ee.Feature(
                            None,
                            {
                                "date": image.date().format("YYYY-MM-dd"),
                                "value": mean.get(variable),
                            },
                        )

                    fc = ee.FeatureCollection(coll.map(extract))
                    data = fc.getInfo()

                    records = [
                        {
                            "date": f["properties"]["date"],
                            "value": f["properties"]["value"],
                        }
                        for f in data["features"]
                    ]
                    df = pd.DataFrame(records)
                    df["date"] = pd.to_datetime(df["date"])
                    df["value"] = pd.to_numeric(df["value"], errors="coerce")
                    df = df.sort_values("date").reset_index(drop=True)
                    df.to_csv(out_file, index=False)

                    ok += 1
                    print(
                        f"  [{count:>4}/{total}] {scenario}/{variable}/{year} "
                        f"→ ok (mean={df['value'].mean():.6f})"
                    )
                except Exception as e:
                    failed += 1
                    fails.append(f"{scenario}/{variable}/{year}: {e}")
                    print(
                        f"  [{count:>4}/{total}] {scenario}/{variable}/{year} "
                        f"→ FAILED: {e}"
                    )

    print(f"\n  GEE summary: ok={ok}, skipped={skipped}, failed={failed}")
    if fails:
        print(f"  Failed files (first 10):")
        for f in fails[:10]:
            print(f"    {f}")


# Download land-surface variables from Pangeo =============================================================================
def download_from_pangeo():
    banner("Land-surface variables from Pangeo CMIP6 (Zarr)")
    try:
        import xarray as xr
    except ImportError:
        print("xarray not installed. Run: pip install xarray zarr gcsfs")
        return

    print("  Reading Pangeo catalog...")
    try:
        cat = pd.read_csv(PANGEO_CATALOG_CSV)
    except Exception as e:
        print(f"  Failed to read catalog: {e}")
        return

    selected = cat[
        (cat.source_id == MODEL_PANGEO)
        & (cat.experiment_id.isin(SCENARIOS))
        & (cat.variable_id.isin(PANGEO_VARIABLES))
        & (cat.table_id == "day")
        & (cat.member_id == MEMBER)
    ]

    if len(selected) == 0:
        print(f"  No datasets found in Pangeo for the requested combination.")
        return

    print(f"  Found {len(selected)} datasets to download.\n")

    for _, row in selected.iterrows():
        var = row.variable_id
        scen = row.experiment_id
        zstore_url = row.zstore

        out_folder = OUTPUT_DIR / scen / var
        out_folder.mkdir(parents=True, exist_ok=True)
        out_csv = out_folder / "merged_daily.csv"

        if out_csv.exists():
            print(f"  {scen}/{var} → already exists, skipped")
            continue

        print(f"  {scen}/{var}")
        print(f"    Opening Zarr...")
        try:
            ds = xr.open_zarr(
                zstore_url, consolidated=True, storage_options={"token": "anon"}
            )
        except Exception as e:
            print(f"    FAILED to open: {e}")
            continue

        west_q = WEST % 360
        east_q = EAST % 360

        sub = ds.sel(lat=slice(SOUTH, NORTH), lon=slice(west_q, east_q))
        if sub.lat.size == 0 or sub.lon.size == 0:
            sub = ds.sel(lat=slice(NORTH, SOUTH), lon=slice(west_q, east_q))

        if sub.lat.size == 0 or sub.lon.size == 0:
            print(
                f"    FAILED: bbox missed grid. Available lat: "
                f"{ds.lat.values.min():.2f} to {ds.lat.values.max():.2f}"
            )
            ds.close()
            continue

        print(f"    Subset grid: {sub.lat.size} × {sub.lon.size} cells")
        da = sub[var].mean(dim=["lat", "lon"], skipna=True)
        print(f"    Loading {da.size} daily values...")
        t0 = time.time()
        values = da.values
        dates = pd.to_datetime(da.time.values).strftime("%Y-%m-%d")
        elapsed = time.time() - t0

        df_out = pd.DataFrame({"date": dates, "value": values})
        df_out.to_csv(out_csv, index=False)

        print(f"    Saved → {out_csv}")
        print(
            f"      Rows: {len(df_out)} | "
            f"Range: {df_out.date.min()} → {df_out.date.max()}"
        )
        print(
            f"      Values: mean={df_out.value.mean():.3f} "
            f"max={df_out.value.max():.3f}"
        )
        print(f"      Download time: {elapsed:.1f} s")
        ds.close()


# Derive snc from snw =============================================================================
def derive_snc():
    banner("Derive snc from snw")
    print(f"  Formula: snc = tanh(snw / {SWE_THRESHOLD_MM} mm)")
    print(f"  Reasoning: standard parameterisation used in many land-surface")
    print(f"  schemes when snc isn't published.\n")

    for scen in SCENARIOS:
        snw_csv = OUTPUT_DIR / scen / "snw" / "merged_daily.csv"
        snc_folder = OUTPUT_DIR / scen / "snc"
        snc_folder.mkdir(parents=True, exist_ok=True)
        snc_csv = snc_folder / "merged_daily.csv"

        if not snw_csv.exists():
            print(f"  {scen}/snw missing — cannot derive snc")
            continue
        if snc_csv.exists():
            print(f"  {scen}/snc → already exists, skipped")
            continue

        snw_df = pd.read_csv(snw_csv)
        snc_values = np.tanh(snw_df["value"] / SWE_THRESHOLD_MM)
        snc_df = pd.DataFrame({"date": snw_df["date"], "value": snc_values})
        snc_df.to_csv(snc_csv, index=False)
        print(f"  {scen}/snc → {snc_csv}")
        print(f"    mean={snc_df.value.mean():.3f} max={snc_df.value.max():.3f}")


# Final inventory =============================================================================
def inventory():
    banner("Inventory of downloaded files")
    all_vars = GEE_VARIABLES + PANGEO_VARIABLES + DERIVED

    print(
        f"  {'Scenario':<10} {'Variable':<10} {'Files':>8} "
        f"{'Format':>12} {'Status':>10}"
    )
    print(f"  {'-' * 55}")

    all_complete = True
    for scen in SCENARIOS:
        for var in all_vars:
            folder = OUTPUT_DIR / scen / var
            if not folder.exists():
                print(f"  {scen:<10} {var:<10} {'-':>8} {'-':>12} {'MISSING':>10}")
                all_complete = False
                continue

            csv_files = list(folder.glob("*.csv"))
            n = len(csv_files)

            if var in GEE_VARIABLES:
                expected = len(YEARS)
                fmt = "yearly"
                status = "OK" if n == expected else f"{n}/{expected}"
                if n != expected:
                    all_complete = False
            else:
                fmt = "merged"
                merged = folder / "merged_daily.csv"
                status = "OK" if merged.exists() else "MISSING"
                if not merged.exists():
                    all_complete = False
                n = 1 if merged.exists() else 0

            print(f"  {scen:<10} {var:<10} {n:>8} {fmt:>12} {status:>10}")

    print()
    if all_complete:
        print("  All data ready. Next step: merge into a single per-scenario CSV.")
    else:
        print("  Some files missing. Rerun script to retry.")


def main():
    print("=" * 65)
    print("  Nahr Ibrahim — CMIP6 Unified Download")
    print("=" * 65)
    print(f"  Model:     {MODEL_GEE}")
    print(f"  Member:    {MEMBER}")
    print(f"  Scenarios: {', '.join(SCENARIOS)}")
    print(f"  Period:    {min(YEARS)}-{max(YEARS)}")
    print(f"  Bbox:      {SOUTH}-{NORTH}°N, {WEST}-{EAST}°E")
    print(f"  Output:    {OUTPUT_DIR.resolve()}")

    download_from_gee()
    download_from_pangeo()
    derive_snc()
    inventory()


if __name__ == "__main__":
    main()
