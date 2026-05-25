import os
import pandas as pd
from pathlib import Path

ROOT = Path(
    os.environ.get("WATERSHED_ROOT", "C:/Users/marck/Downloads/nahr_ibrahim_watershed")
)
CMIP6 = ROOT / "data" / "raw" / "cmip6"
OUT_DIR = CMIP6
YEARS = range(2015, 2101)


def merge_variable(scenario, var_folder):
    """Read all yearly CSVs for one variable and concatenate."""
    folder = CMIP6 / scenario / var_folder
    frames = []
    for yr in YEARS:
        f = folder / f"{yr}.csv"
        if f.exists():
            frames.append(pd.read_csv(f, parse_dates=["date"]))
    if not frames:
        raise FileNotFoundError(f"No files found in {folder}")
    df = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    return df


for scenario in ["ssp245", "ssp585"]:
    print(f"\nProcessing {scenario} ...")

    pr = merge_variable(scenario, "pr")
    tas = merge_variable(scenario, "tas")
    tasmax = merge_variable(scenario, "tasmax")
    tasmin = merge_variable(scenario, "tasmin")

    # Convert units
    # pr    : kg/m²/s  → mm/day  (× 86400)
    # tas   : Kelvin   → Celsius  (− 273.15)
    # tasmax: Kelvin   → Celsius  (− 273.15)
    # tasmin: Kelvin   → Celsius  (− 273.15)
    out = pd.DataFrame(
        {
            "date": pr["date"],
            "precip_mm_day": pr["value"] * 86400,
            "temp_mean_c": tas["value"] - 273.15,
            "temp_max_c": tasmax["value"] - 273.15,
            "temp_min_c": tasmin["value"] - 273.15,
        }
    )

    # Sanity checks
    neg_precip = (out["precip_mm_day"] < 0).sum()
    if neg_precip:
        print(f"  {neg_precip} negative precip values clipped to 0")
        out["precip_mm_day"] = out["precip_mm_day"].clip(lower=0)

    print(f"  Days        : {len(out):,}")
    print(f"  Period      : {out['date'].min().date()} → {out['date'].max().date()}")
    print(f"  Precip mean : {out['precip_mm_day'].mean():.2f} mm/day")
    print(f"  Precip max  : {out['precip_mm_day'].max():.1f} mm/day")
    print(f"  Temp mean   : {out['temp_mean_c'].mean():.1f} °C")
    print(
        f"  Temp range  : {out['temp_min_c'].min():.1f} – {out['temp_max_c'].max():.1f} °C"
    )

    out_path = OUT_DIR / f"cmip6_{scenario}_daily.csv"
    out.to_csv(out_path, index=False)
    print(f"  Saved → {out_path.name}")

print("\nDone — ready for climate_scenarios.py")
