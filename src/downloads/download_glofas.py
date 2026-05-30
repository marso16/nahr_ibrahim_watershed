import os
import glob
import warnings
import pandas as pd
import xarray as xr
from pathlib import Path
from tqdm import tqdm

# config
INPUT_DIR = os.path.join("data", "raw", "glofas")
OUTPUT_FILE = os.path.join(INPUT_DIR, "glofas_discharge.csv")


def main():
    grib_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.grib")))

    if not grib_files:
        print(f"No .grib files found in {INPUT_DIR}")
        return

    print(f"Found {len(grib_files)} .grib files\n")

    all_records = []

    for fpath in tqdm(grib_files, desc="Extracting"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds = xr.open_dataset(fpath, engine="cfgrib")

            dis = ds[
                "dis24"
            ]  # here the shape of the data is (time, latitude, longitude)

            # for each day we compute mean and max discharge over all grid points
            for t in range(len(dis.time)):
                daily = dis.isel(time=t)
                date = pd.Timestamp(dis.time.values[t]).strftime("%Y-%m-%d")

                all_records.append(
                    {
                        "date": date,
                        "dis24_mean": float(
                            daily.mean()
                        ),  # mean over grid points (m3/s)
                        "dis24_max": float(daily.max()),  # max over grid points(m3/s)
                    }
                )

            ds.close()

        except Exception as e:
            print(f"\n  X Failed {os.path.basename(fpath)}: {e}")

    if not all_records:
        print("No data extracted.")
        return

    # build d then sort by date and remove duplicates
    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'='*50}")
    print(f"Done. {len(df)} daily records extracted.")
    print(f"Date range : {df['date'].min()} to {df['date'].max()}")
    print(f"Saved to   : {os.path.abspath(OUTPUT_FILE)}")
    print(f"{'='*50}")
    print(f"Sample data :")
    print(df.head())


if __name__ == "__main__":
    main()
