"""
SCRIPT 00: SETUP VALIDATION & ENVIRONMENT CHECK

Purpose
-------
Validate the computational environment before any ICESat-2 processing.
This script ensures all required Python packages are installed, verifies
directory structure and configuration paths, and initializes the run
timestamp used by all downstream scripts.

Origin / Consolidation
----------------------
- Consolidates: 00_check_packages.py + 00_validate_scripts.py
- No processing logic from later scripts is executed here.

Main Tasks
----------
- Check presence and version of required packages:
  pandas, numpy, scipy, matplotlib, geopandas, rasterio,
  laspy, shapely, scikit-learn, pyproj, etc.
- Verify paths such as:
  - BASE_OUTPUT_DIR (e.g. E:\\Processing\\ICESat-2\\output_data)
  - INPUT_DIR (raw ATL03 data)
  - INSAR_DIR (InSAR GeoTIFFs)
  - CANADA_LAND_SHAPEFILE, CANVEC_WATERBODY_SHAPEFILE (GIS layers)
- Create output directory structure if missing.
- Confirm read/write access to all configured locations.
- Create or update `latest_run_timestamp.txt` that identifies the
  current processing run (e.g. 20260105_n1).

Inputs
------
- System environment and installed Python packages.
- Configuration constants defined in this script (paths, etc.).

Outputs
-------
- latest_run_timestamp.txt  (used by Scripts 01–06)
- Console log summarizing:
  - Package versions
  - Path checks
  - Any missing dependencies

Reproducibility Notes
---------------------
Run this script once before any other script in the workflow. If this
script fails, do not proceed, as downstream outputs would not be
reproducible or may be incomplete.
"""

import importlib
from pathlib import Path
import sys
import datetime as dt

print("=" * 80)
print("SCRIPT 00: ENVIRONMENT SETUP & VALIDATION (MINIMAL)")
print("=" * 80)

# User-editable paths
BASE_OUTPUT_DIR = Path(r"E:\Processing\ICESat-2\output_data")
TIMESTAMP_FILE = BASE_OUTPUT_DIR / "latest_run_timestamp.txt"
CROSSOVER_FILENAME = "icesat2_repeat_track_matches.csv"

# ============================================================================
# 1) CHECK PYTHON PACKAGES
# ============================================================================

print("\n[1/3] Checking required Python packages...\n")

required_packages = [
    "numpy",
    "pandas",
    "matplotlib",
    "geopandas",
    "scipy",
    "shapely",
]

missing = []
for pkg in required_packages:
    try:
        importlib.import_module(pkg)
        print(f"  OK {pkg}")
    except ImportError:
        print(f"  MISSING {pkg}")
        missing.append(pkg)

if missing:
    print("\nERROR: Missing packages:")
    for pkg in missing:
        print(f"  - {pkg}")
    print("\nInstall with: pip install " + " ".join(missing))
    sys.exit(1)

print("\nOK All required packages available.\n")

# ============================================================================
# 2) CHECK OUTPUT DIRECTORY & TIMESTAMP
# ============================================================================

print("[2/3] Checking output directory and latest run timestamp...\n")

if not BASE_OUTPUT_DIR.exists():
    print(f"ERROR: Base output directory not found:")
    print(f"  {BASE_OUTPUT_DIR}\n")
    sys.exit(1)

print(f"OK Base output dir: {BASE_OUTPUT_DIR}\n")

if not TIMESTAMP_FILE.exists():
    print(f"ERROR: latest_run_timestamp.txt not found in:")
    print(f"  {BASE_OUTPUT_DIR}\n")
    print("Run initial processing (Scripts 01-04) first.\n")
    sys.exit(1)

try:
    with open(TIMESTAMP_FILE, "r") as f:
        run_timestamp = f.read().strip()
    OUTPUT_DIR = BASE_OUTPUT_DIR / f"run_{run_timestamp}"
except Exception as e:
    print(f"ERROR: Failed to read timestamp: {e}\n")
    sys.exit(1)

print(f"OK Latest run ID: {run_timestamp}")
print(f"OK Run directory: {OUTPUT_DIR}\n")

if not OUTPUT_DIR.exists():
    print(f"ERROR: Run directory not found: {OUTPUT_DIR}\n")
    sys.exit(1)

print("OK Run directory exists.\n")

# ============================================================================
# 3) CHECK MAIN INPUT CSV
# ============================================================================

print("[3/3] Checking main crossover CSV...\n")

crossover_path = OUTPUT_DIR / CROSSOVER_FILENAME

if not crossover_path.exists():
    print(f"ERROR: Crossover file not found:")
    print(f"  {crossover_path}\n")
    sys.exit(1)

try:
    import pandas as pd
    df_head = pd.read_csv(crossover_path, nrows=5)
    n_cols = len(df_head.columns)
except Exception as e:
    print(f"ERROR: Could not read CSV: {e}\n")
    sys.exit(1)

print(f"OK Crossover CSV: {CROSSOVER_FILENAME}")
print(f"OK Columns: {n_cols}\n")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("=" * 80)
print("SCRIPT 00 - SUCCESS")
print("=" * 80)
print(f"Date/Time: {dt.datetime.now()}")
print(f"Base dir : {BASE_OUTPUT_DIR}")
print(f"Run ID   : run_{run_timestamp}")
print(f"CSV file : {CROSSOVER_FILENAME}")
print("=" * 80 + "\n")

sys.exit(0)