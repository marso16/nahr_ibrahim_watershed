import cdsapi
import os
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==================================================
# CONFIG
# ==================================================
DATASET = "cems-glofas-historical" 
OUTPUT_DIR = r"data\raw\glofas"
STATE_FILE = "download_state.json"

YEARS = list(range(2000, 2026))
MONTHS = list(range(1, 13))

MAX_WORKERS = 4
MAX_RETRIES = 6

os.makedirs(OUTPUT_DIR, exist_ok=True)

client = cdsapi.Client()

# ==================================================
# STATE MANAGEMENT
# ==================================================
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"done": []}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


state = load_state()
done_set = set(state["done"])

# ==================================================
# REQUEST BUILDER
# ==================================================
def build_request(year, month):
    return {
        "system_version": ["version_4_0"],
        "hydrological_model": ["lisflood"],
        "product_type": ["consolidated"],
        "variable": ["river_discharge_in_the_last_24_hours"],
        "hyear": [str(year)],
        "hmonth": [f"{month:02d}"],
        "hday": [f"{d:02d}" for d in range(1, 32)],
        "download_format": "zip",
        "area": [34.2120, 33.9929, 35.6429, 36.0487],
    }

# ==================================================
# DOWNLOAD FUNCTION
# ==================================================
def download_chunk(year, month):
    key = f"{year}-{month:02d}"
    output_file = os.path.join(OUTPUT_DIR, f"glofas_{key}.zip")

    # Skip if already done in state
    if key in done_set:
        return f"[SKIP] {key} (state)"

    # Skip if file exists and looks valid
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        done_set.add(key)
        return f"[SKIP] {key} (file exists)"

    request = build_request(year, month)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[START] {key} attempt {attempt}")

            client.retrieve(DATASET, request).download(output_file)

            # mark success
            done_set.add(key)
            state["done"] = list(done_set)
            save_state(state)

            return f"[OK] {key}"

        except Exception as e:
            wait = min(120, (2 ** attempt) + random.random() * 2)
            print(f"[RETRY] {key} attempt {attempt} failed: {e}")
            print(f"         waiting {wait:.1f}s")

            time.sleep(wait)

    return f"[FAILED] {key}"


# ==================================================
# WORK LIST
# ==================================================
tasks = [
    (y, m)
    for y in YEARS
    for m in MONTHS
    if f"{y}-{m:02d}" not in done_set
]

# ==================================================
# RUN PARALLEL
# ==================================================
def main():
    print(f"Total remaining chunks: {len(tasks)}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_chunk, y, m): (y, m)
            for (y, m) in tasks
        }

        for f in as_completed(futures):
            y, m = futures[f]
            try:
                result = f.result()
                print(result)
            except Exception as e:
                print(f"[CRASH] {y}-{m:02d}: {e}")

    print("\nDONE")

if __name__ == "__main__":
    main()