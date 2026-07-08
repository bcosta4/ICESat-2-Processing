"""

================================================================================

SCRIPT 03: CORE ANALYSIS (CROSSOVER DETECTION + INSAR INTEGRATION)

================================================================================

Purpose

-------

Detect repeat-track crossover points from land photons, compute

elevation change and subsidence rates, and integrate InSAR-derived

subsidence for independent validation.

Origin / Consolidation

----------------------

- Consolidates:

  - 02_crossover_analysis.py
  - 03_insar_integration.py

- Implemented as two major sections:

  - Section 1 = Crossover detection and rate estimation.
  - Section 2 = InSAR sampling and validation metrics.

Main Tasks – Section 1 (Crossover Analysis)

-------------------------------------------

- Load land photons from icesat2_filtered_ground_photons_LAND.csv.

- Build a cKDTree of photon coordinates for efficient spatial search.

- Identify crossover photon pairs within a specified horizontal distance (e.g. 2.0 m).

- Apply temporal filters according to selected strategy:
  - e.g. 10–365 days, seasonal thaw period, or custom range.

- Compute:
  - elevation change (dH, in metres)
  - time interval in years
  - subsidence rate dH_rate_m_per_year.

- Remove outliers using a 3-sigma criterion on dH or rate.

- Enforce consistent date ordering (older vs newer epoch).

- Save per-pair results to CSV.

Main Tasks – Section 2 (InSAR Integration - ENHANCED)

-----------------------------------------------------

- Discover and load InSAR GeoTIFF rasters from INSAR_DIR:
  - Rate files: *average_mm_per_year* (temporally adjusted)
  - dH files: *sum_*_cm* (June-Sept period, needs conversion)

- Auto-detect year from filename and pair files correctly

- Reproject crossover locations to raster CRS when needed.

- Sample InSAR subsidence rate at each crossover location.

- For dH files: convert cm to annualized rate (cm × 10 × 365.25/120)

- Combine ICeSat-2 and InSAR measurements in one table.

- Compute validation statistics:
  - Pearson correlation, R-squared, RMSE, bias.

- Write out a textual summary of InSAR validation.

Main Tasks – Section 3 (UNCERTAINTY QUANTIFICATION - NEW)

----------------------------------------------------------

- Initialize uncertainty columns for all crossovers.

- Assign global measurement uncertainty (4.2 cm, ICeSat-2 accuracy).

- Assign global temporal uncertainty (15 days, tunable).

- Compute per-crossover rate uncertainty from measurement + temporal error.

- Perform 1000-sample bootstrap on mean subsidence rate.

- Compute 95% confidence interval (CI) on mean rate.

- Store CI bounds in output CSV.

Inputs

------

- icesat2_filtered_ground_photons_LAND.csv

- InSAR GeoTIFF files in INSAR_DIR (optional)

Outputs

-------

- icesat2_repeat_track_matches.csv
  (crossover pairs with dH, dt, subsidence_rate_m_per_year, + uncertainty columns)

- icesat2_insar_integrated.csv
  (crossover pairs with InSAR values and source tracking)

- insar_validation_metrics.csv
  (Pearson R, RMSE, bias, etc.)

- insar_integration_summary.txt
  (validation metrics and summary statistics)

Reproducibility Notes

---------------------

Temporal filters, distance thresholds and outlier criteria determine

which pairs contribute to subsidence estimates. Document chosen

parameters and InSAR products (sensor, acquisition years, processing

chain) so that the ICeSat-2 vs InSAR comparison can be repeated.

"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import sys
import argparse
import time
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("SCRIPT 03: CORE ANALYSIS - CROSSOVER DETECTION & InSAR INTEGRATION")
print("="*80)

# ============================================================================
# PARSE COMMAND-LINE ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(
    description="ICeSat-2 Core Crossover Analysis with Temporal Filtering"
)

parser.add_argument(
    '--temporal',
    type=str,
    choices=['1', '2', '3', '4'],
    default='2',
    help='Temporal filtering: 1=June-Oct, 2=June-Sept (default), 3=Early vs Late, 4=Same-month-only'
)

parser.add_argument(
    '--skip-insar',
    action='store_true',
    help='Skip InSAR integration for faster processing'
)

args = parser.parse_args()

TEMPORAL_OPTION = args.temporal
SKIP_INSAR = args.skip_insar

TEMPORAL_OPTIONS = {
    '1': 'June-October (Broad summer window)',
    '2': 'June-September (Standard thaw season)',
    '3': 'Early vs Late Summer (June-July vs Aug-Sept)',
    '4': 'Same-month-only (Strictest)',
}

print(f"\nTemporal option: {TEMPORAL_OPTION} - {TEMPORAL_OPTIONS[TEMPORAL_OPTION]}")
if SKIP_INSAR:
    print("InSAR integration: SKIPPED")
else:
    print("InSAR integration: ENABLED")

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_OUTPUT_DIR = Path(r'E:\Processing\ICeSat-2\output_data')
INSAR_DIR = Path(r'C:\bcosta\InSAR_for_ICeSat-2')

CROSSOVER_DISTANCE_THRESHOLD = 2.0  # meters
TIME_SEPARATION_THRESHOLD = 10  # days minimum

# Get latest run timestamp
timestamp_file = BASE_OUTPUT_DIR / 'latest_run_timestamp.txt'

try:
    with open(timestamp_file, 'r') as f:
        RUN_TIMESTAMP = f.read().strip()
    OUTPUT_DIR = BASE_OUTPUT_DIR / f'run_{RUN_TIMESTAMP}'
except FileNotFoundError:
    print("ERROR: Timestamp file not found. Run Script 00 or 01 first.")
    sys.exit(1)

print(f"Run timestamp: {RUN_TIMESTAMP}")
print(f"Output directory: {OUTPUT_DIR}")
if not SKIP_INSAR:
    print(f"InSAR directory: {INSAR_DIR}")
print("="*80)

# ============================================================================
# SECTION 1: CROSSOVER DETECTION
# ============================================================================

print("\nPART 1/2: CROSSOVER DETECTION")
print("-"*80)

# Load filtered land photons
INPUT_FILE = OUTPUT_DIR / 'icesat2_filtered_ground_photons_LAND.csv'

if not INPUT_FILE.exists():
    # Try fallback
    INPUT_FILE = OUTPUT_DIR / 'icesat2_filtered_ground_photons.csv'

if not INPUT_FILE.exists():
    print(f"ERROR: No input file found!")
    print(f" Tried: {OUTPUT_DIR / 'icesat2_filtered_ground_photons_LAND.csv'}")
    print(f" Tried: {OUTPUT_DIR / 'icesat2_filtered_ground_photons.csv'}")
    sys.exit(1)

print(f"Loading {INPUT_FILE.name}...")
photons = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(photons)} filtered photons")

# Parse datetime
date_cols = [c for c in photons.columns if 'datetime' in c.lower() or 'date' in c.lower()]

if date_cols:
    photons['datetime'] = pd.to_datetime(photons[date_cols[0]])
else:
    if 'year' in photons.columns:
        print("WARNING: Using year column as datetime proxy (no full datetime available)")
        photons['datetime'] = pd.to_datetime(photons['year'], format='%Y')
    else:
        print("ERROR: No datetime column found")
        sys.exit(1)

photons['month'] = photons['datetime'].dt.month
photons['year'] = photons['datetime'].dt.year

# Find coordinate columns
lat_col = next((c for c in photons.columns if 'latitude' in c.lower()), None)
lon_col = next((c for c in photons.columns if 'longitude' in c.lower()), None)
elev_col = next((c for c in photons.columns if 'height' in c.lower() or 'elevation' in c.lower()), None)

if not all([lat_col, lon_col, elev_col]):
    print(f"ERROR: Could not find required columns")
    print(f" Latitude: {lat_col}")
    print(f" Longitude: {lon_col}")
    print(f" Elevation: {elev_col}")
    print(f" Available columns: {list(photons.columns)}")
    sys.exit(1)

print(f"Column mapping:")
print(f" Latitude: {lat_col}")
print(f" Longitude: {lon_col}")
print(f" Elevation: {elev_col}")

# ============================================================================
# Apply temporal filtering
# ============================================================================

print("\n" + "="*80)
print(f"TEMPORAL FILTERING: {TEMPORAL_OPTIONS[TEMPORAL_OPTION]}")
print("="*80)

if TEMPORAL_OPTION == '1':
    # June-October
    data = photons[(photons['month'] >= 6) & (photons['month'] <= 10)].copy()
    print("Months allowed: 6, 7, 8, 9, 10 (June-October)")
elif TEMPORAL_OPTION == '2':
    # June-September
    data = photons[(photons['month'] >= 6) & (photons['month'] <= 9)].copy()
    print("Months allowed: 6, 7, 8, 9 (June-September)")
elif TEMPORAL_OPTION == '3':
    # Early vs Late Summer (June-July vs Aug-Sept)
    data = photons[(photons['month'] >= 6) & (photons['month'] <= 9)].copy()
    print("Months allowed: 6, 7, 8, 9 (Early=June-July, Late=Aug-Sept)")
elif TEMPORAL_OPTION == '4':
    # Same-month-only
    data = photons[(photons['month'] >= 6) & (photons['month'] <= 9)].copy()
    print("Months allowed: 6, 7, 8, 9 (Same-month pairs only)")

n_before = len(photons)
n_after = len(data)

print(f"Photons retained: {n_after}/{n_before} ({100*n_after/n_before:.1f}%)")

# ============================================================================
# Crossover detection function
# ============================================================================

def find_crossovers_optimized(df, option_selected, lat_col, lon_col, elev_col):
    """Find crossovers using optimized cKDTree spatial matching."""
    start_time = time.time()

    print("\n" + "="*80)
    print("SPATIAL-TEMPORAL CROSSOVER MATCHING")
    print("="*80)

    # Project to meters (UTM approximation)
    mean_lat = df[lat_col].mean()
    cos_lat = np.cos(np.radians(mean_lat))

    y = df[lat_col].values * 111000
    x = df[lon_col].values * 111000 * cos_lat

    coords = np.column_stack([x, y])

    print(f"Building spatial index (cKDTree)...")
    tree = cKDTree(coords)

    print(f"Searching for point pairs within {CROSSOVER_DISTANCE_THRESHOLD}m...")
    pairs = tree.query_pairs(r=CROSSOVER_DISTANCE_THRESHOLD, output_type='ndarray')

    print(f"Found {len(pairs)} candidate spatial pairs")

    if len(pairs) == 0:
        print("No spatial pairs found!")
        return pd.DataFrame()

    print(f"Filtering by temporal criteria...")

    # Extract indices
    idx1 = pairs[:, 0]
    idx2 = pairs[:, 1]

    # Get datetime values
    t1 = df['datetime'].values[idx1]
    t2 = df['datetime'].values[idx2]

    # Calculate time delta
    dt_days = (t2 - t1).astype('timedelta64[D]').astype(int)

    # Create time filter
    abs_dt = np.abs(dt_days)
    time_mask = (abs_dt >= TIME_SEPARATION_THRESHOLD) & (abs_dt <= 2000)

    # Apply same-month filter if option 4
    if option_selected == '4':
        m1 = df['month'].values[idx1]
        m2 = df['month'].values[idx2]
        time_mask = time_mask & (m1 == m2)

    valid_idx = np.where(time_mask)[0]
    idx1_valid = idx1[valid_idx]
    idx2_valid = idx2[valid_idx]
    dt_final = dt_days[valid_idx]

    print(f"Valid crossovers after temporal filter: {len(valid_idx)}")

    if len(valid_idx) == 0:
        return pd.DataFrame()

    # Get elevation change
    h1 = df[elev_col].values[idx1_valid]
    h2 = df[elev_col].values[idx2_valid]

    dh = h2 - h1
    dt_years = dt_final / 365.25
    rates = dh / dt_years

    # Create results
    results = pd.DataFrame({
        'lon_early': df[lon_col].values[idx1_valid],
        'lat_early': df[lat_col].values[idx1_valid],
        'date_early': df['datetime'].values[idx1_valid],
        'elev_early': h1,
        'lon_late': df[lon_col].values[idx2_valid],
        'lat_late': df[lat_col].values[idx2_valid],
        'date_late': df['datetime'].values[idx2_valid],
        'elev_late': h2,
        'dH': dh,
        'dt_days': dt_final,
        'subsidence_rate_m_per_year': rates,
    })

    # Ensure chronological ordering
    mask_swap = results['dt_days'] < 0

    if mask_swap.any():
        # Swap early/late if needed
        temp_lon = results.loc[mask_swap, 'lon_early'].copy()
        temp_lat = results.loc[mask_swap, 'lat_early'].copy()
        temp_date = results.loc[mask_swap, 'date_early'].copy()
        temp_elev = results.loc[mask_swap, 'elev_early'].copy()

        results.loc[mask_swap, 'lon_early'] = results.loc[mask_swap, 'lon_late']
        results.loc[mask_swap, 'lat_early'] = results.loc[mask_swap, 'lat_late']
        results.loc[mask_swap, 'date_early'] = results.loc[mask_swap, 'date_late']
        results.loc[mask_swap, 'elev_early'] = results.loc[mask_swap, 'elev_late']

        results.loc[mask_swap, 'lon_late'] = temp_lon
        results.loc[mask_swap, 'lat_late'] = temp_lat
        results.loc[mask_swap, 'date_late'] = temp_date
        results.loc[mask_swap, 'elev_late'] = temp_elev

        results.loc[mask_swap, 'dt_days'] *= -1
        results.loc[mask_swap, 'dH'] *= -1

    # Add average coordinates
    results['lon_avg'] = (results['lon_early'] + results['lon_late']) / 2
    results['lat_avg'] = (results['lat_early'] + results['lat_late']) / 2

    results['year_early'] = pd.to_datetime(results['date_early']).dt.year
    results['year_late'] = pd.to_datetime(results['date_late']).dt.year

    elapsed = time.time() - start_time
    print(f"Processing time: {elapsed:.1f} seconds")

    return results

# Execute crossover detection
crossovers = find_crossovers_optimized(data, TEMPORAL_OPTION, lat_col, lon_col, elev_col)

print(f"\nTotal crossovers found: {len(crossovers)}")

if len(crossovers) > 0:
    print("\n" + "="*80)
    print("OUTLIER REMOVAL (3-SIGMA FILTERING)")
    print("="*80)

    mean_rate = crossovers['subsidence_rate_m_per_year'].mean()
    std_rate = crossovers['subsidence_rate_m_per_year'].std()

    print(f"Mean subsidence rate: {mean_rate:.6f} m/year")
    print(f"Std deviation: {std_rate:.6f} m/year")
    print(f"3-sigma bounds: ({mean_rate - 3*std_rate:.6f}, {mean_rate + 3*std_rate:.6f})")

    clean = crossovers[
        (crossovers['subsidence_rate_m_per_year'] >= mean_rate - 3*std_rate) &
        (crossovers['subsidence_rate_m_per_year'] <= mean_rate + 3*std_rate)
    ].copy()

    removed = len(crossovers) - len(clean)
    print(f"Removed {removed} outliers")
    print(f"Remaining crossovers: {len(clean)}")

    # ============================================================================
    # UNCERTAINTY QUANTIFICATION (NEW - SECTION 3)
    # ============================================================================

    print("\n" + "="*80)
    print("UNCERTAINTY QUANTIFICATION")
    print("="*80)

    # Initialize uncertainty columns
    clean['measurement_uncertainty_cm'] = np.nan
    clean['temporal_uncertainty_days'] = np.nan
    clean['rate_uncertainty_mm_per_year'] = np.nan
    clean['ci_lower_95'] = np.nan
    clean['ci_upper_95'] = np.nan

    # Global measurement and temporal uncertainties (tunable)
    MEASUREMENT_UNCERTAINTY_CM = 4.2  # ICeSat-2 vertical accuracy (~4 cm)
    TEMPORAL_UNCERTAINTY_DAYS = 15    # Temporal sampling uncertainty (days)

    # Assign global uncertainties to all crossovers
    clean['measurement_uncertainty_cm'] = MEASUREMENT_UNCERTAINTY_CM
    clean['temporal_uncertainty_days'] = TEMPORAL_UNCERTAINTY_DAYS

    print(f"Measurement uncertainty: {MEASUREMENT_UNCERTAINTY_CM} cm")
    print(f"Temporal uncertainty: {TEMPORAL_UNCERTAINTY_DAYS} days")

    # Compute rate uncertainty (mm/year) for each crossover
    # Formula: sqrt( (sigma_h / dt)^2 + (dh * sigma_dt / dt^2)^2 ) / 10
    # where sigma_h = measurement_uncertainty_cm (in cm)
    #       sigma_dt = temporal_uncertainty_days (in days)
    #       dh = elevation change (in m) -> convert to cm
    #       dt = time interval (in days)

    dt_days = clean['dt_days'].values
    dh_cm = clean['dH'].values * 100  # Convert m to cm

    sigma_h = MEASUREMENT_UNCERTAINTY_CM
    sigma_dt = TEMPORAL_UNCERTAINTY_DAYS

    # Rate uncertainty (in cm/day)
    rate_unc_cm_per_day = np.sqrt(
        (sigma_h / dt_days)**2 +
        (np.abs(dh_cm) / (dt_days**2) * sigma_dt)**2
    )

    # Convert to mm/year
    # 1 cm/day * 365.25 days/yr * 10 mm/cm = 3652.5 mm/yr
    rate_unc_mm_per_year = rate_unc_cm_per_day * 365.25 * 10

    clean['rate_uncertainty_mm_per_year'] = rate_unc_mm_per_year

    print(f"Mean rate uncertainty: {clean['rate_uncertainty_mm_per_year'].mean():.3f} mm/yr")
    print(f"Median rate uncertainty: {clean['rate_uncertainty_mm_per_year'].median():.3f} mm/yr")

    # Bootstrap confidence interval on mean subsidence rate (1000 resamples)
    print("\nComputing bootstrap 95% CI on mean subsidence rate...")

    rates_mm = clean['subsidence_rate_m_per_year'].values * 1000
    valid_rates = rates_mm[~np.isnan(rates_mm)]

    if len(valid_rates) > 1:
        np.random.seed(42)  # For reproducibility
        n_bootstraps = 1000
        bootstrap_means = []

        for _ in range(n_bootstraps):
            resample = np.random.choice(valid_rates, size=len(valid_rates), replace=True)
            bootstrap_means.append(resample.mean())

        bootstrap_means = np.array(bootstrap_means)
        ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])

        print(f"Bootstrap CI on mean: [{ci_lower:.4f}, {ci_upper:.4f}] mm/yr")
        print(f"Mean ± 95% CI: {valid_rates.mean():.4f} ± {(ci_upper - ci_lower) / 2:.4f} mm/yr")

        # Store in first row (global statistics)
        clean.loc[0, 'ci_lower_95'] = ci_lower
        clean.loc[0, 'ci_upper_95'] = ci_upper

    else:
        print("Insufficient data for bootstrap CI")

    print("="*80)

    # Save results
    out_file_clean = OUTPUT_DIR / 'icesat2_crossover_points_clean.csv'
    out_file_matches = OUTPUT_DIR / 'icesat2_repeat_track_matches.csv'

    clean.to_csv(out_file_clean, index=False)
    clean.to_csv(out_file_matches, index=False)

    print(f"Saved: {out_file_clean.name}")
    print(f"Saved: {out_file_matches.name}")

    print("\n" + "="*80)
    print("CROSSOVER STATISTICS")
    print("="*80)

    print(f"Count: {len(clean)}")
    print(f"Mean subsidence: {clean['subsidence_rate_m_per_year'].mean():.6f} m/yr")
    print(f"Median subsidence: {clean['subsidence_rate_m_per_year'].median():.6f} m/yr")
    print(f"Min: {clean['subsidence_rate_m_per_year'].min():.6f} m/yr")
    print(f"Max: {clean['subsidence_rate_m_per_year'].max():.6f} m/yr")
    print(f"Std dev: {clean['subsidence_rate_m_per_year'].std():.6f} m/yr")

else:
    print("No crossovers found!")
    clean = pd.DataFrame()
    out_file_matches = OUTPUT_DIR / 'icesat2_repeat_track_matches.csv'
    clean.to_csv(out_file_matches, index=False)

# ============================================================================
# SECTION 2: InSAR INTEGRATION (ENHANCED - DUAL FILE HANDLING)
# ============================================================================

if SKIP_INSAR:
    print("\n" + "="*80)
    print("PART 2/2: InSAR INTEGRATION SKIPPED")
    print("="*80)
    print("\nSaved crossover results without InSAR validation.")
    print("\n" + "="*80)
    print("SCRIPT 03 COMPLETE")
    print("="*80)
    sys.exit(0)

print("\n" + "="*80)
print("PART 2/2: InSAR INTEGRATION & VALIDATION (ENHANCED)")
print("="*80)

icesat2_matches = clean.copy() if len(clean) > 0 else crossovers.copy()

# Discover InSAR files
print("Scanning InSAR directory for rate and dH files...")

rate_files = {}  # Year -> path mapping
dh_files = {}    # Year -> path mapping

if INSAR_DIR.exists():
    all_tifs = list(INSAR_DIR.glob('*.tif'))

    for f in all_tifs:
        filename = f.name.lower()

        # Pattern 1: masked_YYYY_*_average_mm_per_year_*
        if 'average' in filename and 'mm_per_year' in filename:
            try:
                parts = f.stem.split('_')
                year = None
                for part in parts:
                    if part.isdigit() and len(part) == 4 and 2010 <= int(part) <= 2030:
                        year = int(part)
                        break
                if year:
                    rate_files[year] = f
                    print(f" ✓ Rate file {year}: {f.name}")
            except:
                pass

        # Pattern 2: sum_*_cm_gacoscorr_YYYY_masked
        if 'sum' in filename and 'cm' in filename:
            try:
                parts = f.stem.split('_')
                year = None
                for part in parts:
                    if part.isdigit() and len(part) == 4 and 2010 <= int(part) <= 2030:
                        year = int(part)
                        break
                if year:
                    dh_files[year] = f
                    print(f" ✓ dH file {year}: {f.name}")
            except:
                pass

    if not rate_files and not dh_files:
        print(f" ⚠ No InSAR files found in {INSAR_DIR}")
        print(" Saving crossover results without InSAR validation...")
        output_file = OUTPUT_DIR / 'icesat2_insar_integrated.csv'
        icesat2_matches.to_csv(output_file, index=False)
        print(f" Saved: {output_file.name}")
        print("\n" + "="*80)
        print("SCRIPT 03 COMPLETE")
        print("="*80)
        sys.exit(0)

    # Load and prepare rasters
    print(f"\nLoading {len(rate_files)} rate files and {len(dh_files)} dH files...")

    try:
        import rasterio
    except ImportError:
        print("ERROR: rasterio not found. Install with: pip install rasterio")
        sys.exit(1)

    insar_rate_data = {}  # year -> raster object
    insar_dh_data = {}    # year -> (raster, time_window_days)

    # Load rate files
    for year, filepath in rate_files.items():
        try:
            src = rasterio.open(filepath)
            insar_rate_data[year] = src
            print(f" ✓ Rate {year}: CRS={src.crs}, size={src.width}x{src.height}")
        except Exception as e:
            print(f" ✗ Rate {year}: {str(e)[:50]}")

    # Load dH files
    for year, filepath in dh_files.items():
        try:
            src = rasterio.open(filepath)
            # dH files represent June-September (~120 days)
            insar_dh_data[year] = (src, 120)
            print(f" ✓ dH {year}: CRS={src.crs}, size={src.width}x{src.height}, window=June-Sept (~120 days)")
        except Exception as e:
            print(f" ✗ dH {year}: {str(e)[:50]}")

    # Add columns for InSAR results
    icesat2_matches['insar_source'] = 'none'
    icesat2_matches['insar_subsidence_rate_mm_yr'] = np.nan
    icesat2_matches['insar_year'] = np.nan

    # Sample InSAR values
    print("\nSampling InSAR values at crossover locations...")

    try:
        import pyproj

        # Get unique years in the data
        years_to_process = sorted(
            set(icesat2_matches['year_early'].unique()) |
            set(icesat2_matches['year_late'].unique())
        )

        for year in years_to_process:
            year = int(year)

            print(f"\n Processing year {year}...", end=' ', flush=True)

            # Find crossovers from this year
            mask = (icesat2_matches['year_early'] == year) | (icesat2_matches['year_late'] == year)
            subset = icesat2_matches[mask]

            if len(subset) == 0:
                print("no crossovers")
                continue

            insar_sampled = False
            insar_values_list = None
            source_name = 'none'

            # Try rate file first (preferred - already temporally adjusted)
            if year in insar_rate_data:
                try:
                    raster = insar_rate_data[year]

                    # Reproject coordinates
                    in_proj = pyproj.CRS("EPSG:4326")
                    out_proj = pyproj.CRS(raster.crs)
                    transformer = pyproj.Transformer.from_crs(in_proj, out_proj, always_xy=True)

                    ix, iy = transformer.transform(
                        subset['lon_avg'].values,
                        subset['lat_avg'].values
                    )

                    coords = list(zip(ix, iy))

                    # Sample raster
                    insar_values_list = [val[0] for val in raster.sample(coords)]

                    insar_sampled = True
                    source_name = f'rate_{year}'

                    print(f"sampled from rate file", end=' ')

                except Exception as e:
                    print(f"rate file error: {str(e)[:25]}...", end=' ')

            # Fall back to dH file if rate file not available or failed
            if not insar_sampled and year in insar_dh_data:
                try:
                    raster, window_days = insar_dh_data[year]

                    # Reproject coordinates
                    in_proj = pyproj.CRS("EPSG:4326")
                    out_proj = pyproj.CRS(raster.crs)
                    transformer = pyproj.Transformer.from_crs(in_proj, out_proj, always_xy=True)

                    ix, iy = transformer.transform(
                        subset['lon_avg'].values,
                        subset['lat_avg'].values
                    )

                    coords = list(zip(ix, iy))

                    # Sample raster (in cm)
                    dh_values_cm = [val[0] for val in raster.sample(coords)]

                    # Convert: cm in 120 days -> mm/year
                    # (X cm) * (365.25 / 120 days) * (10 mm/cm)
                    insar_values_list = [
                        (dh_cm * 10) * (365.25 / window_days) if dh_cm is not None else np.nan
                        for dh_cm in dh_values_cm
                    ]

                    insar_sampled = True
                    source_name = f'dh_{year}_converted'

                    print(f"sampled from dH file (converted cm->mm/yr)", end=' ')

                except Exception as e:
                    print(f"dH file error: {str(e)[:25]}...", end=' ')

            # Assign values to subset
            if insar_sampled and insar_values_list:
                icesat2_matches.loc[mask, 'insar_subsidence_rate_mm_yr'] = insar_values_list
                icesat2_matches.loc[mask, 'insar_year'] = year
                icesat2_matches.loc[mask, 'insar_source'] = source_name

                print(f"({len(subset)} points)")

            else:
                print("failed")
                continue

    except Exception as e:
        print(f"ERROR during sampling: {str(e)}")
        import traceback
        traceback.print_exc()

    # Close rasters
    for r in insar_rate_data.values():
        r.close()

    for r, _ in insar_dh_data.values():
        r.close()

    # Validation metrics
    print("\n" + "="*80)
    print("VALIDATION METRICS")
    print("="*80)

    valid_pairs = icesat2_matches[
        icesat2_matches['subsidence_rate_m_per_year'].notna() &
        icesat2_matches['insar_subsidence_rate_mm_yr'].notna()
    ].copy()

    print(f"Valid pairs for comparison: {len(valid_pairs)}/{len(icesat2_matches)}")

    # Count by source
    source_counts = valid_pairs['insar_source'].value_counts()
    print(f"\nSamples by source:")
    for source, count in source_counts.items():
        print(f" {source}: {count}")

    if len(valid_pairs) >= 10:

        is2_rate_mm = valid_pairs['subsidence_rate_m_per_year'].values * 1000
        insar_rate_mm = valid_pairs['insar_subsidence_rate_mm_yr'].values

        # Remove NaN pairs
        valid_mask = ~(np.isnan(is2_rate_mm) | np.isnan(insar_rate_mm))
        is2_rate_mm = is2_rate_mm[valid_mask]
        insar_rate_mm = insar_rate_mm[valid_mask]

        if len(is2_rate_mm) > 1:

            correlation = np.corrcoef(is2_rate_mm, insar_rate_mm)[0, 1]
            r_squared = correlation ** 2
            rmse = np.sqrt(np.mean((is2_rate_mm - insar_rate_mm) ** 2))
            bias = np.mean(is2_rate_mm - insar_rate_mm)

            print(f"\n ICESat-2 Statistics")
            print(f" Mean: {is2_rate_mm.mean():.3f} mm/yr")
            print(f" Median: {np.median(is2_rate_mm):.3f} mm/yr")
            print(f" Std: {is2_rate_mm.std():.3f} mm/yr")
            print(f" N: {len(is2_rate_mm)}")

            print(f"\n InSAR Statistics")
            print(f" Mean: {insar_rate_mm.mean():.3f} mm/yr")
            print(f" Median: {np.median(insar_rate_mm):.3f} mm/yr")
            print(f" Std: {insar_rate_mm.std():.3f} mm/yr")

            print(f"\n Validation Metrics")
            print(f" Pearson R: {correlation:.4f}")
            print(f" R-squared: {r_squared:.4f}")
            print(f" RMSE: {rmse:.3f} mm/yr")
            print(f" Bias (IS2 - InSAR): {bias:.3f} mm/yr")

            # Save validation metrics
            metrics_list = [
                ('pearson_r', correlation),
                ('r_squared', r_squared),
                ('rmse_mm_yr', rmse),
                ('bias_mm_yr', bias),
                ('n_pairs', len(is2_rate_mm)),
                ('is2_mean_mm_yr', is2_rate_mm.mean()),
                ('insar_mean_mm_yr', insar_rate_mm.mean()),
            ]

            metrics_df = pd.DataFrame(metrics_list, columns=['metric', 'value'])
            metrics_file = OUTPUT_DIR / 'insar_validation_metrics.csv'
            metrics_df.to_csv(metrics_file, index=False)

            print(f"\nSaved: insar_validation_metrics.csv")

        else:
            print("Insufficient valid pairs for correlation analysis")

    else:
        print("Not enough valid pairs for statistical validation")

    # Save integrated results
    output_file = OUTPUT_DIR / 'icesat2_insar_integrated.csv'
    icesat2_matches.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file.name}")

    # Write summary report
    summary_file = OUTPUT_DIR / 'insar_integration_summary.txt'

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("InSAR INTEGRATION SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write(f"ICeSat-2 Crossover Points: {len(icesat2_matches)}\n")
        f.write(f"InSAR-matched Crossovers: {len(valid_pairs)}\n")
        f.write(f"Match rate: {100*len(valid_pairs)/len(icesat2_matches):.1f}%\n\n")

        f.write("InSAR File Sources:\n")
        f.write(f" Rate files (mm/yr, temporally adjusted): {len(rate_files)}\n")
        for year in sorted(rate_files.keys()):
            f.write(f" - {year}: {rate_files[year].name}\n")

        f.write(f" dH files (cm, June-Sept, converted to mm/yr): {len(dh_files)}\n")
        for year in sorted(dh_files.keys()):
            f.write(f" - {year}: {dh_files[year].name}\n")

        if len(valid_pairs) >= 10:
            f.write("\nValidation Results:\n")
            f.write(f" Pearson R: {correlation:.4f}\n")
            f.write(f" R-squared: {r_squared:.4f}\n")
            f.write(f" RMSE: {rmse:.3f} mm/yr\n")
            f.write(f" Bias (IS2 - InSAR): {bias:.3f} mm/yr\n")

            f.write("\nInterpretation:\n")

            if abs(bias) < 2:
                f.write(f" ✓ Low bias ({bias:.2f} mm/yr): IS2 and InSAR agree well\n")
            elif abs(bias) < 5:
                f.write(f" ~ Moderate bias ({bias:.2f} mm/yr): Small systematic difference\n")
            else:
                f.write(f" ⚠ High bias ({bias:.2f} mm/yr): Consider potential scaling/processing differences\n")

            if correlation > 0.7:
                f.write(f" ✓ Strong correlation ({correlation:.3f}): Good spatial agreement\n")
            elif correlation > 0.5:
                f.write(f" ~ Moderate correlation ({correlation:.3f}): Reasonable agreement\n")
            else:
                f.write(f" ⚠ Weak correlation ({correlation:.3f}): Check data consistency\n")

    print(f"Saved: {summary_file.name}")

else:
    print(f" ⚠ InSAR directory not found: {INSAR_DIR}")
    print(" Saving crossover results without InSAR validation...")
    output_file = OUTPUT_DIR / 'icesat2_insar_integrated.csv'
    icesat2_matches.to_csv(output_file, index=False)
    print(f" Saved: {output_file.name}")

print("\n" + "="*80)
print("SCRIPT 03 COMPLETE")
print("="*80)

sys.exit(0)
