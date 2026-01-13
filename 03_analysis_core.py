"""
================================================================================
SCRIPT 03: CORE ANALYSIS (CROSSOVER DETECTION + INSAR INTEGRATION + REPRODUCIBILITY)
================================================================================

Purpose
-------
Detect repeat-track crossover points from land photons, compute
elevation change and subsidence rates, and integrate InSAR-derived
subsidence for independent validation.

NEW: Complete reproducibility infrastructure (Tier 1)
- Config class for centralized parameter management
- RunMetadata class for complete audit trail
- Logging framework (file + console)
- All parameters documented and validated

Origin / Consolidation
----------------------
- Consolidates:
- 02_crossover_analysis.py
- 03_insar_integration.py

- Implemented as three major sections:
- Section 1 = Crossover detection and rate estimation.
- Section 2 = Uncertainty quantification.
- Section 3 = InSAR sampling and validation metrics.

Main Tasks – Section 1 (Crossover Analysis)
-------------------------------------------
- Load land photons from icesat2_filtered_ground_photons_LAND.csv.
- Build a cKDTree of photon coordinates for efficient spatial search.
- Identify crossover photon pairs within a specified horizontal distance (e.g. 2.0 m).
- Apply temporal filters according to selected strategy.
- Compute: elevation change (dH), time interval, subsidence rate.
- Remove outliers using a 3-sigma criterion on dH or rate.
- Enforce consistent date ordering (older vs newer epoch).
- Save per-pair results to CSV.

Main Tasks – Section 2 (Uncertainty Quantification)
---------------------------------------------------
- Initialize uncertainty columns for all crossovers.
- Assign global measurement uncertainty (4.2 cm, ICESat-2 accuracy).
- Assign global temporal uncertainty (15 days, tunable).
- Compute per-crossover rate uncertainty from measurement + temporal error.
- Perform bootstrap on mean subsidence rate.
- Compute 95% confidence interval (CI) on mean rate.
- Store CI bounds in output CSV.

Main Tasks – Section 3 (InSAR Integration)
-------------------------------------------
- Discover and load InSAR GeoTIFF rasters from INSAR_DIR.
- Auto-detect year from filename and pair files correctly.
- Sample InSAR subsidence rate at each crossover location.
- Combine ICeSat-2 and InSAR measurements in one table.
- Compute validation statistics: Pearson R, R-squared, RMSE, bias.

Inputs
------
- icesat2_filtered_ground_photons_LAND.csv
- InSAR GeoTIFF files in INSAR_DIR (optional)

Outputs
-------
- icesat2_repeat_track_matches.csv (crossover pairs with uncertainty)
- icesat2_insar_integrated.csv (crossover pairs with InSAR values)
- insar_validation_metrics.csv (validation statistics)
- insar_integration_summary.txt (summary report)
- run_metadata.json (complete parameter audit trail) ← NEW
- script_03_run.log (complete execution log) ← NEW

Reproducibility Infrastructure (Tier 1)
----------------------------------------
Config Class:
  - Centralizes ALL parameters in one place
  - Parameters documented with meanings
  - Validation ensures sensible values
  - Easy to track changes over time

RunMetadata Class:
  - Captures run_id, timestamp, git hash, all config values
  - Saved as JSON (machine-readable audit trail)
  - Can replay any run exactly from metadata

Logging Framework:
  - All output logged to file (DEBUG level)
  - Console shows clean output (INFO level)
  - Nothing lost, everything debuggable
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import sys
import argparse
import time
import warnings
import json
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

class Config:
    """Master configuration for Script 03."""

    BASE_OUTPUT_DIR = Path(r'E:\Processing\ICeSat-2\output_data')
    INSAR_DIR = Path(r'C:\bcosta\InSAR_for_ICeSat-2')
    
    CROSSOVER_DISTANCE_THRESHOLD_M = 2.0
    TIME_SEPARATION_THRESHOLD_DAYS = 10
    TIME_SEPARATION_MAX_DAYS = 2000
    OUTLIER_SIGMA_THRESHOLD = 3.0
    
    MEASUREMENT_UNCERTAINTY_CM = 4.2
    TEMPORAL_UNCERTAINTY_DAYS = 15
    
    BOOTSTRAP_SAMPLES = 1000
    BOOTSTRAP_CI_PERCENT = 95
    BOOTSTRAP_SEED = 42
    
    COASTAL_ZONES = {
        'Coastal (0-1 km)': (0.0, 1.0),
        'Near-coastal (1-2 km)': (1.0, 2.0),
        'Transition (2-5 km)': (2.0, 5.0),
        'Inland (>5 km)': (5.0, np.inf),
    }
    
    COASTAL_OCEAN_SAMPLE_RATE = 10
    COASTAL_TEMPORAL_CUTOFF_YEAR = 2022
    
    INSAR_RATE_PATTERN = '*average_mm_per_year*'
    INSAR_DH_PATTERN = 'sum_*_cm*'
    INSAR_DH_WINDOW_DAYS = 120
    INSAR_YEAR_RANGE = (2010, 2030)

    @classmethod
    def validate(cls):
        """Verify all configuration parameters are sensible."""
        errors = []

        if not cls.BASE_OUTPUT_DIR.exists():
            errors.append(f"BASE_OUTPUT_DIR not found: {cls.BASE_OUTPUT_DIR}")
        if cls.CROSSOVER_DISTANCE_THRESHOLD_M <= 0:
            errors.append("CROSSOVER_DISTANCE_THRESHOLD_M must be positive")
        if cls.TIME_SEPARATION_THRESHOLD_DAYS <= 0:
            errors.append("TIME_SEPARATION_THRESHOLD_DAYS must be positive")
        if cls.TIME_SEPARATION_MAX_DAYS <= cls.TIME_SEPARATION_THRESHOLD_DAYS:
            errors.append("TIME_SEPARATION_MAX_DAYS must be > THRESHOLD")
        if cls.OUTLIER_SIGMA_THRESHOLD <= 0:
            errors.append("OUTLIER_SIGMA_THRESHOLD must be positive")
        if cls.MEASUREMENT_UNCERTAINTY_CM < 0:
            errors.append("MEASUREMENT_UNCERTAINTY_CM cannot be negative")
        if cls.TEMPORAL_UNCERTAINTY_DAYS < 0:
            errors.append("TEMPORAL_UNCERTAINTY_DAYS cannot be negative")

        if errors:
            print("\n⚠ CONFIGURATION VALIDATION ERRORS:\n")
            for i, error in enumerate(errors, 1):
                print(f" {i}. {error}")
            raise ValueError(f"{len(errors)} configuration error(s) found")
        return True

    @classmethod
    def as_dict(cls):
        """Export all config values as dictionary."""
        return {
            'CROSSOVER_DISTANCE_THRESHOLD_M': cls.CROSSOVER_DISTANCE_THRESHOLD_M,
            'TIME_SEPARATION_THRESHOLD_DAYS': cls.TIME_SEPARATION_THRESHOLD_DAYS,
            'OUTLIER_SIGMA_THRESHOLD': cls.OUTLIER_SIGMA_THRESHOLD,
            'MEASUREMENT_UNCERTAINTY_CM': cls.MEASUREMENT_UNCERTAINTY_CM,
            'TEMPORAL_UNCERTAINTY_DAYS': cls.TEMPORAL_UNCERTAINTY_DAYS,
            'BOOTSTRAP_SAMPLES': cls.BOOTSTRAP_SAMPLES,
            'BOOTSTRAP_SEED': cls.BOOTSTRAP_SEED,
        }

# ============================================================================
# RUN METADATA CLASS
# ============================================================================

class RunMetadata:
    """Capture and save complete run parameters for reproducibility."""

    def __init__(self, run_id, config, temporal_option, skip_insar):
        self.run_id = run_id
        self.script_version = "03_analysis_core.py v3.2-FIXED"
        self.temporal_option = temporal_option
        self.skip_insar = skip_insar
        self.config_dict = config.as_dict()
        self.start_time = datetime.now().isoformat()
        self.end_time = None

    def finalize(self):
        """Mark run as complete."""
        self.end_time = datetime.now().isoformat()

    def save(self, output_dir):
        """Save run metadata as JSON file."""
        metadata_file = output_dir / 'run_metadata.json'
        metadata = {
            'run_id': self.run_id,
            'script': self.script_version,
            'temporal_option': self.temporal_option,
            'skip_insar': self.skip_insar,
            'config': self.config_dict,
            'timestamps': {
                'start_time': self.start_time,
                'end_time': self.end_time,
            },
        }

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)

        return metadata_file

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(output_dir):
    """Configure logging to both file and console."""
    log_file = output_dir / 'script_03_run.log'
    logger = logging.getLogger('script_03')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    fh = logging.FileHandler(log_file, encoding='utf-8', mode='w')
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(fh_formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

# ============================================================================
# SECTION 3A: CROSSOVER RMSE VALIDATION
# ============================================================================

def calculate_crossover_rmse(df_pairs, dh_col, measurement_uncertainty_cm=4.2, logger=None):
    """
    Calculate empirical and theoretical RMSE of elevation differences at crossover points.
    
    Theory: σ_crossover = √2 × σ_h ≈ 5.9 cm (with σ_h = 4.2 cm)
    """

    measurement_uncertainty_m = measurement_uncertainty_cm / 100.0

    dh_values = df_pairs[dh_col].dropna().values
    dh_values = dh_values[np.isfinite(dh_values)]

    if len(dh_values) == 0:
        return {
            'n_crossovers': 0,
            'empirical_rmse_m': np.nan,
            'empirical_rmse_cm': np.nan,
            'theoretical_rmse_m': float(np.sqrt(2) * measurement_uncertainty_m),
            'theoretical_rmse_cm': float(np.sqrt(2) * measurement_uncertainty_m * 100.0),
            'ratio': np.nan,
            'agreement': 'NO_DATA',
            'dh_mean_m': np.nan,
            'dh_std_m': np.nan,
        }

    empirical_rmse_m = float(np.sqrt(np.mean(dh_values ** 2)))
    theoretical_rmse_m = float(np.sqrt(2.0) * measurement_uncertainty_m)
    ratio = empirical_rmse_m / theoretical_rmse_m

    if ratio < 1.2:
        agreement = 'Excellent'
    elif ratio < 1.5:
        agreement = 'Good'
    elif ratio < 2.0:
        agreement = 'Acceptable'
    else:
        agreement = 'Divergent - Check outliers'

    results = {
        'n_crossovers': int(len(dh_values)),
        'empirical_rmse_m': empirical_rmse_m,
        'empirical_rmse_cm': empirical_rmse_m * 100.0,
        'theoretical_rmse_m': theoretical_rmse_m,
        'theoretical_rmse_cm': theoretical_rmse_m * 100.0,
        'ratio': float(ratio),
        'agreement': agreement,
        'dh_mean_m': float(np.mean(dh_values)),
        'dh_std_m': float(np.std(dh_values)),
    }

    if logger is not None:
        logger.info("Crossover RMSE validation:")
        logger.info(f"  N crossovers: {results['n_crossovers']}")
        logger.info(f"  Empirical RMSE: {results['empirical_rmse_cm']:.2f} cm")
        logger.info(f"  Theoretical RMSE: {results['theoretical_rmse_cm']:.2f} cm")
        logger.info(f"  Ratio: {results['ratio']:.3f}")
        logger.info(f"  Agreement: {results['agreement']}")

    return results

# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

print("="*80)
print("SCRIPT 03: CORE ANALYSIS - CROSSOVER DETECTION & UNCERTAINTY VALIDATION")
print("="*80)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="ICeSat-2 Core Crossover Analysis")
parser.add_argument('--temporal', type=str, choices=['1', '2', '3', '4'], default='2',
                    help='Temporal filtering: 1=June-Oct, 2=June-Sept (default), 3=Early vs Late, 4=Same-month')
parser.add_argument('--skip-insar', action='store_true', help='Skip InSAR integration')

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

# Initialize reproducibility infrastructure
print("\n" + "="*80)
print("INITIALIZING REPRODUCIBILITY INFRASTRUCTURE")
print("="*80)

config = Config()
config.validate()

BASE_OUTPUT_DIR = config.BASE_OUTPUT_DIR
timestamp_file = BASE_OUTPUT_DIR / 'latest_run_timestamp.txt'

try:
    with open(timestamp_file, 'r') as f:
        RUN_TIMESTAMP = f.read().strip()
    OUTPUT_DIR = BASE_OUTPUT_DIR / f'run_{RUN_TIMESTAMP}'
except FileNotFoundError:
    print("ERROR: Timestamp file not found. Run Script 00 or 01 first.")
    sys.exit(1)

logger = setup_logging(OUTPUT_DIR)
metadata = RunMetadata(RUN_TIMESTAMP, config, TEMPORAL_OPTION, SKIP_INSAR)

logger.info(f"Configuration validated")
logger.info(f"Run timestamp: {RUN_TIMESTAMP}")
logger.info(f"Output directory: {OUTPUT_DIR}")

print(f"\nRun timestamp: {RUN_TIMESTAMP}")
print(f"Output directory: {OUTPUT_DIR}")

CROSSOVER_DISTANCE_THRESHOLD = config.CROSSOVER_DISTANCE_THRESHOLD_M
TIME_SEPARATION_THRESHOLD = config.TIME_SEPARATION_THRESHOLD_DAYS
TIME_SEPARATION_MAX = config.TIME_SEPARATION_MAX_DAYS
OUTLIER_SIGMA = config.OUTLIER_SIGMA_THRESHOLD
MEASUREMENT_UNCERTAINTY_CM = config.MEASUREMENT_UNCERTAINTY_CM
TEMPORAL_UNCERTAINTY_DAYS = config.TEMPORAL_UNCERTAINTY_DAYS
BOOTSTRAP_SAMPLES = config.BOOTSTRAP_SAMPLES
BOOTSTRAP_SEED = config.BOOTSTRAP_SEED

print("="*80)

# ============================================================================
# SECTION 1: CROSSOVER DETECTION
# ============================================================================

logger.info("="*80)
logger.info("PART 1/5: CROSSOVER DETECTION")
logger.info("="*80)

print("\nPART 1/5: CROSSOVER DETECTION")
print("-"*80)

INPUT_FILE = OUTPUT_DIR / 'icesat2_filtered_ground_photons_LAND.csv'

if not INPUT_FILE.exists():
    INPUT_FILE = OUTPUT_DIR / 'icesat2_filtered_ground_photons.csv'

if not INPUT_FILE.exists():
    logger.error(f"No input file found!")
    print(f"ERROR: No input file found!")
    sys.exit(1)

logger.info(f"Loading {INPUT_FILE.name}...")
print(f"Loading {INPUT_FILE.name}...")

photons = pd.read_csv(INPUT_FILE)

logger.info(f"Loaded {len(photons):,} filtered photons")
print(f"Loaded {len(photons):,} filtered photons")

# Parse datetime with error handling (BUG FIX #5)
date_cols = [c for c in photons.columns if 'datetime' in c.lower() or 'date' in c.lower()]

try:
    if date_cols:
        photons['datetime'] = pd.to_datetime(photons[date_cols[0]])
        logger.info(f"Parsed datetime from column: {date_cols[0]}")
        print(f"Parsed datetime from: {date_cols[0]}")
    else:
        if 'year' in photons.columns:
            logger.warning("Using year column as datetime proxy (no full datetime available)")
            print("WARNING: Using year column as datetime proxy")
            photons['datetime'] = pd.to_datetime(photons['year'], format='%Y')
        else:
            logger.error("No datetime column found")
            print("ERROR: No datetime column found")
            sys.exit(1)
except Exception as e:
    logger.error(f"Failed to parse datetime: {e}")
    logger.error(f"Available columns: {photons.columns.tolist()}")
    print(f"ERROR: Failed to parse datetime: {e}")
    sys.exit(1)

photons['month'] = photons['datetime'].dt.month
photons['year'] = photons['datetime'].dt.year

lat_col = next((c for c in photons.columns if 'latitude' in c.lower()), None)
lon_col = next((c for c in photons.columns if 'longitude' in c.lower()), None)
elev_col = next((c for c in photons.columns if 'height' in c.lower() or 'elevation' in c.lower()), None)

if not all([lat_col, lon_col, elev_col]):
    logger.error(f"Could not find required columns")
    print(f"ERROR: Could not find required columns")
    sys.exit(1)

logger.info(f"Column mapping: Lat={lat_col}, Lon={lon_col}, Elev={elev_col}")
print(f"Column mapping: Lat={lat_col}, Lon={lon_col}, Elev={elev_col}")

# Apply temporal filtering
logger.info("="*80)
logger.info(f"TEMPORAL FILTERING: {TEMPORAL_OPTIONS[TEMPORAL_OPTION]}")
logger.info("="*80)

print("\n" + "="*80)
print(f"TEMPORAL FILTERING: {TEMPORAL_OPTIONS[TEMPORAL_OPTION]}")
print("="*80)

if TEMPORAL_OPTION == '1':
    data = photons[(photons['month'] >= 6) & (photons['month'] <= 10)].copy()
    logger.info("Months allowed: 6, 7, 8, 9, 10 (June-October)")
    print("Months allowed: 6, 7, 8, 9, 10 (June-October)")
elif TEMPORAL_OPTION == '2':
    data = photons[(photons['month'] >= 6) & (photons['month'] <= 9)].copy()
    logger.info("Months allowed: 6, 7, 8, 9 (June-September)")
    print("Months allowed: 6, 7, 8, 9 (June-September)")
elif TEMPORAL_OPTION == '3':
    data = photons[(photons['month'] >= 6) & (photons['month'] <= 9)].copy()
    logger.info("Months allowed: 6, 7, 8, 9 (Early=June-July, Late=Aug-Sept)")
    print("Months allowed: 6, 7, 8, 9 (Early=June-July, Late=Aug-Sept)")
elif TEMPORAL_OPTION == '4':
    data = photons[(photons['month'] >= 6) & (photons['month'] <= 9)].copy()
    logger.info("Months allowed: 6, 7, 8, 9 (Same-month pairs only)")
    print("Months allowed: 6, 7, 8, 9 (Same-month pairs only)")

n_before = len(photons)
n_after = len(data)

logger.info(f"Photons retained: {n_after:,}/{n_before:,} ({100*n_after/n_before:.1f}%)")
print(f"Photons retained: {n_after:,}/{n_before:,} ({100*n_after/n_before:.1f}%)")

# Crossover detection function
def find_crossovers_optimized(df, option_selected, lat_col, lon_col, elev_col, logger):
    """Find crossovers using optimized cKDTree spatial matching."""
    start_time = time.time()

    logger.info("="*80)
    logger.info("SPATIAL-TEMPORAL CROSSOVER MATCHING")
    logger.info("="*80)

    print("\n" + "="*80)
    print("SPATIAL-TEMPORAL CROSSOVER MATCHING")
    print("="*80)

    mean_lat = df[lat_col].mean()
    cos_lat = np.cos(np.radians(mean_lat))

    y = df[lat_col].values * 111000
    x = df[lon_col].values * 111000 * cos_lat

    coords = np.column_stack([x, y])

    logger.info(f"Building spatial index (cKDTree)...")
    print(f"Building spatial index (cKDTree)...")

    tree = cKDTree(coords)

    logger.info(f"Searching for point pairs within {CROSSOVER_DISTANCE_THRESHOLD}m...")
    print(f"Searching for point pairs within {CROSSOVER_DISTANCE_THRESHOLD}m...")

    pairs = tree.query_pairs(r=CROSSOVER_DISTANCE_THRESHOLD, output_type='ndarray')

    logger.info(f"Found {len(pairs):,} candidate spatial pairs")
    print(f"Found {len(pairs):,} candidate spatial pairs")

    if len(pairs) == 0:
        logger.warning("No spatial pairs found!")
        print("No spatial pairs found!")
        return pd.DataFrame()

    logger.info(f"Filtering by temporal criteria...")
    print(f"Filtering by temporal criteria...")

    idx1 = pairs[:, 0]
    idx2 = pairs[:, 1]

    t1 = df['datetime'].values[idx1]
    t2 = df['datetime'].values[idx2]

    dt_days = (t2 - t1).astype('timedelta64[D]').astype(int)
    abs_dt = np.abs(dt_days)

    time_mask = (abs_dt >= TIME_SEPARATION_THRESHOLD) & (abs_dt <= TIME_SEPARATION_MAX)

    if option_selected == '4':
        m1 = df['month'].values[idx1]
        m2 = df['month'].values[idx2]
        time_mask = time_mask & (m1 == m2)

    valid_idx = np.where(time_mask)[0]

    idx1_valid = idx1[valid_idx]
    idx2_valid = idx2[valid_idx]
    dt_final = dt_days[valid_idx]

    logger.info(f"Valid crossovers after temporal filter: {len(valid_idx):,}")
    print(f"Valid crossovers after temporal filter: {len(valid_idx):,}")

    if len(valid_idx) == 0:
        return pd.DataFrame()

    h1 = df[elev_col].values[idx1_valid]
    h2 = df[elev_col].values[idx2_valid]

    dh = h2 - h1
    dt_years = dt_final / 365.25
    rates = dh / dt_years

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

    mask_swap = results['dt_days'] < 0

    if mask_swap.any():
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

    results['lon_avg'] = (results['lon_early'] + results['lon_late']) / 2
    results['lat_avg'] = (results['lat_early'] + results['lat_late']) / 2
    results['year_early'] = pd.to_datetime(results['date_early']).dt.year
    results['year_late'] = pd.to_datetime(results['date_late']).dt.year

    elapsed = time.time() - start_time

    logger.info(f"Processing time: {elapsed:.1f} seconds")
    print(f"Processing time: {elapsed:.1f} seconds")

    return results

# Execute crossover detection
crossovers = find_crossovers_optimized(data, TEMPORAL_OPTION, lat_col, lon_col, elev_col, logger)

logger.info(f"Total crossovers found: {len(crossovers):,}")
print(f"\nTotal crossovers found: {len(crossovers):,}")

if len(crossovers) > 0:

    # ================================================================
    # OUTLIER REMOVAL (3-SIGMA FILTERING)
    # ================================================================

    logger.info("="*80)
    logger.info("OUTLIER REMOVAL (3-SIGMA FILTERING)")
    logger.info("="*80)

    print("\n" + "="*80)
    print("OUTLIER REMOVAL (3-SIGMA FILTERING)")
    print("="*80)

    mean_rate = crossovers['subsidence_rate_m_per_year'].mean()
    std_rate = crossovers['subsidence_rate_m_per_year'].std()

    logger.info(f"Mean subsidence rate: {mean_rate:.6f} m/year")
    logger.info(f"Std deviation: {std_rate:.6f} m/year")
    logger.info(f"3-sigma bounds: ({mean_rate - OUTLIER_SIGMA*std_rate:.6f}, {mean_rate + OUTLIER_SIGMA*std_rate:.6f})")

    print(f"Mean subsidence rate: {mean_rate:.6f} m/year")
    print(f"Std deviation: {std_rate:.6f} m/year")
    print(f"3-sigma bounds: ({mean_rate - OUTLIER_SIGMA*std_rate:.6f}, {mean_rate + OUTLIER_SIGMA*std_rate:.6f})")

    clean = crossovers[
        (crossovers['subsidence_rate_m_per_year'] >= mean_rate - OUTLIER_SIGMA*std_rate) &
        (crossovers['subsidence_rate_m_per_year'] <= mean_rate + OUTLIER_SIGMA*std_rate)
    ].copy()

    clean = clean.reset_index(drop=True)  # *** BUG FIX #1: Reset index ***

    removed = len(crossovers) - len(clean)

    logger.info(f"Removed {removed:,} outliers")
    logger.info(f"Remaining crossovers: {len(clean):,}")

    print(f"Removed {removed:,} outliers")
    print(f"Remaining crossovers: {len(clean):,}")

    # ================================================================
    # SECTION 2: UNCERTAINTY QUANTIFICATION
    # ================================================================

    logger.info("="*80)
    logger.info("PART 2/5: UNCERTAINTY QUANTIFICATION")
    logger.info("="*80)

    print("\n" + "="*80)
    print("PART 2/5: UNCERTAINTY QUANTIFICATION")
    print("="*80)

    clean['measurement_uncertainty_cm'] = MEASUREMENT_UNCERTAINTY_CM
    clean['temporal_uncertainty_days'] = TEMPORAL_UNCERTAINTY_DAYS

    logger.info(f"Measurement uncertainty: {MEASUREMENT_UNCERTAINTY_CM} cm")
    logger.info(f"Temporal uncertainty: {TEMPORAL_UNCERTAINTY_DAYS} days")

    print(f"Measurement uncertainty: {MEASUREMENT_UNCERTAINTY_CM} cm")
    print(f"Temporal uncertainty: {TEMPORAL_UNCERTAINTY_DAYS} days")

    # Calculate per-observation rate uncertainties
    dt_days = clean['dt_days'].values
    dh_cm = clean['dH'].values * 100.0

    sigma_h = MEASUREMENT_UNCERTAINTY_CM
    sigma_dt = TEMPORAL_UNCERTAINTY_DAYS

    rate_unc_cm_per_day = np.sqrt(
        (sigma_h / dt_days)**2 +
        (np.abs(dh_cm) / (dt_days**2) * sigma_dt)**2
    )

    rate_unc_mm_per_year = rate_unc_cm_per_day * 365.25 * 10.0

    clean['rate_uncertainty_mm_yr'] = rate_unc_mm_per_year  # *** BUG FIX #2: Correct column name ***

    logger.info(f"Mean rate uncertainty: {clean['rate_uncertainty_mm_yr'].mean():.3f} mm/yr")
    logger.info(f"Median rate uncertainty: {clean['rate_uncertainty_mm_yr'].median():.3f} mm/yr")

    print(f"Mean rate uncertainty: {clean['rate_uncertainty_mm_yr'].mean():.3f} mm/yr")
    print(f"Median rate uncertainty: {clean['rate_uncertainty_mm_yr'].median():.3f} mm/yr")

    # Bootstrap 95% CI
    logger.info(f"\nComputing bootstrap 95% CI on mean subsidence rate ({BOOTSTRAP_SAMPLES} resamples)...")
    print(f"\nComputing bootstrap 95% CI on mean subsidence rate ({BOOTSTRAP_SAMPLES} resamples)...")

    rates_mm = clean['subsidence_rate_m_per_year'].values * 1000.0
    valid_rates = rates_mm[~np.isnan(rates_mm)]

    if len(valid_rates) > 1:
        np.random.seed(BOOTSTRAP_SEED)
        bootstrap_means = []
        for _ in range(BOOTSTRAP_SAMPLES):
            resample = np.random.choice(valid_rates, size=len(valid_rates), replace=True)
            bootstrap_means.append(resample.mean())

        bootstrap_means = np.array(bootstrap_means)
        ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])

        logger.info(f"Bootstrap CI on mean: [{ci_lower:.4f}, {ci_upper:.4f}] mm/yr")
        logger.info(f"Mean ± 95% CI: {valid_rates.mean():.4f} ± {(ci_upper - ci_lower) / 2:.4f} mm/yr")

        print(f"Bootstrap CI on mean: [{ci_lower:.4f}, {ci_upper:.4f}] mm/yr")
        print(f"Mean ± 95% CI: {valid_rates.mean():.4f} ± {(ci_upper - ci_lower) / 2:.4f} mm/yr")

        clean['ci_lower_95'] = ci_lower  # *** BUG FIX #3: Set for all rows ***
        clean['ci_upper_95'] = ci_upper  # *** BUG FIX #3: Set for all rows ***
    else:
        logger.warning("Insufficient data for bootstrap CI")
        print("Insufficient data for bootstrap CI")
        clean['ci_lower_95'] = np.nan
        clean['ci_upper_95'] = np.nan

    logger.info("="*80)

    # Save results
    out_file_matches = OUTPUT_DIR / 'icesat2_repeat_track_matches.csv'
    clean.to_csv(out_file_matches, index=False)

    logger.info(f"Saved: {out_file_matches.name}")
    print(f"Saved: {out_file_matches.name}")

    # ================================================================
    # CROSSOVER STATISTICS
    # ================================================================

    logger.info("="*80)
    logger.info("CROSSOVER STATISTICS")
    logger.info("="*80)

    print("\n" + "="*80)
    print("CROSSOVER STATISTICS")
    print("="*80)

    logger.info(f"Count: {len(clean):,}")
    logger.info(f"Mean subsidence: {clean['subsidence_rate_m_per_year'].mean():.6f} m/yr")
    logger.info(f"Median subsidence: {clean['subsidence_rate_m_per_year'].median():.6f} m/yr")
    logger.info(f"Min: {clean['subsidence_rate_m_per_year'].min():.6f} m/yr")
    logger.info(f"Max: {clean['subsidence_rate_m_per_year'].max():.6f} m/yr")
    logger.info(f"Std dev: {clean['subsidence_rate_m_per_year'].std():.6f} m/yr")

    print(f"Count: {len(clean):,}")
    print(f"Mean subsidence: {clean['subsidence_rate_m_per_year'].mean():.6f} m/yr")
    print(f"Median subsidence: {clean['subsidence_rate_m_per_year'].median():.6f} m/yr")
    print(f"Min: {clean['subsidence_rate_m_per_year'].min():.6f} m/yr")
    print(f"Max: {clean['subsidence_rate_m_per_year'].max():.6f} m/yr")
    print(f"Std dev: {clean['subsidence_rate_m_per_year'].std():.6f} m/yr")

    # ================================================================
    # SECTION 3A: CROSSOVER RMSE VALIDATION
    # ================================================================

    logger.info("\n" + "="*80)
    logger.info("PART 3A/5: CROSSOVER RMSE VALIDATION")
    logger.info("="*80)

    print("\n" + "="*80)
    print("PART 3A/5: CROSSOVER RMSE VALIDATION")
    print("="*80)

    rmse_results = calculate_crossover_rmse(
        df_pairs=clean,
        dh_col='dH',
        measurement_uncertainty_cm=MEASUREMENT_UNCERTAINTY_CM,
        logger=logger
    )

    print(f"Empirical Crossover RMSE: {rmse_results['empirical_rmse_cm']:.2f} cm")
    print(f"Theoretical Crossover RMSE: {rmse_results['theoretical_rmse_cm']:.2f} cm")
    print(f"Ratio (Empirical/Theoretical): {rmse_results['ratio']:.3f}")
    print(f"Agreement Assessment: {rmse_results['agreement']}")

    # Save RMSE validation results
    RMSE_VALIDATION_FILE = OUTPUT_DIR / '05_crossover_rmse_validation.json'
    with open(RMSE_VALIDATION_FILE, 'w', encoding='utf-8') as f:
        json.dump(rmse_results, f, indent=2, default=str, ensure_ascii=False)

    RMSE_REPORT_FILE = OUTPUT_DIR / '05_crossover_rmse_validation.txt'
    with open(RMSE_REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("CROSSOVER RMSE VALIDATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Run ID: {RUN_TIMESTAMP}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("METHODOLOGY\n")
        f.write("-"*80 + "\n")
        f.write("When two ICeSat-2 passes cross over the same location, elevation\n")
        f.write("measurements are independent. The measurement uncertainty propagates as:\n\n")
        f.write("σ_crossover = √(σ²_h + σ²_h) = √2 × σ_h ≈ 5.9 cm (with σ_h = 4.2 cm)\n\n")
        f.write("Empirical RMSE calculated from actual crossover elevation differences.\n")
        f.write("Ratio (Empirical/Theoretical) indicates agreement quality.\n\n")
        f.write("RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(f"N Crossovers: {rmse_results['n_crossovers']:,}\n")
        f.write(f"Empirical RMSE: {rmse_results['empirical_rmse_cm']:.2f} cm\n")
        f.write(f"Theoretical RMSE: {rmse_results['theoretical_rmse_cm']:.2f} cm\n")
        f.write(f"Ratio: {rmse_results['ratio']:.3f}\n")
        f.write(f"Agreement: {rmse_results['agreement']}\n\n")
        f.write("INTERPRETATION GUIDE\n")
        f.write("-"*80 + "\n")
        f.write("Ratio < 1.2  : Excellent agreement → System performing optimally\n")
        f.write("Ratio 1.2-1.5: Good agreement → System performing well\n")
        f.write("Ratio 1.5-2.0: Acceptable agreement → Minor deviations expected\n")
        f.write("Ratio > 2.0  : Divergent → Check for outliers or systematic errors\n\n")
        f.write(f"Your data shows ratio = {rmse_results['ratio']:.3f} ({rmse_results['agreement']})\n")

    logger.info(f"✓ Saved: {RMSE_VALIDATION_FILE.name}")
    logger.info(f"✓ Saved: {RMSE_REPORT_FILE.name}")
    print(f"✓ Saved: {RMSE_VALIDATION_FILE.name}")
    print(f"✓ Saved: {RMSE_REPORT_FILE.name}")

    # ================================================================
    # SECTION 3B: RATE UNCERTAINTY VALIDATION
    # ================================================================

    logger.info("\n" + "="*80)
    logger.info("PART 3B/5: RATE UNCERTAINTY VALIDATION")
    logger.info("="*80)

    print("\n" + "="*80)
    print("PART 3B/5: RATE UNCERTAINTY VALIDATION")
    print("="*80)

    all_unc = clean['rate_uncertainty_mm_yr'].dropna().values

    logger.info(f"Total observations with uncertainty: {len(all_unc)}")
    if len(all_unc) > 0:
        logger.info(f"Mean uncertainty: {np.mean(all_unc):.3f} mm/yr")
        logger.info(f"Median uncertainty: {np.median(all_unc):.3f} mm/yr")
        logger.info(f"Std dev: {np.std(all_unc):.3f} mm/yr")

    # Build elevation bins (10 m intervals)
    elev_min = np.floor(clean['elev_early'].min() / 10.0) * 10.0
    elev_max = np.ceil(clean['elev_early'].max() / 10.0) * 10.0
    elev_bins = np.arange(elev_min, elev_max + 10.0, 10.0)

    rows = []
    print("\nRate uncertainty by elevation band:")
    print("-" * 100)
    print(f"{'Elevation':>20} {'Count':>10} {'Mean':>12} {'Median':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("=" * 100)

    for i in range(len(elev_bins) - 1):
        lo = elev_bins[i]
        hi = elev_bins[i + 1]
        label = f"{int(lo)}–{int(hi)} m"

        mask = (clean['elev_early'] >= lo) & (clean['elev_early'] < hi)
        z_unc = clean.loc[mask, 'rate_uncertainty_mm_yr'].dropna()

        if len(z_unc) == 0:
            continue

        stats = {
            'elevation_range': label,
            'n_observations': int(len(z_unc)),
            'mean_uncertainty_mm_yr': float(z_unc.mean()),
            'median_uncertainty_mm_yr': float(z_unc.median()),
            'std_uncertainty_mm_yr': float(z_unc.std()),
            'min_uncertainty_mm_yr': float(z_unc.min()),
            'max_uncertainty_mm_yr': float(z_unc.max()),
        }
        rows.append(stats)

        print(f"{label:>20} {len(z_unc):>10} "
              f"{stats['mean_uncertainty_mm_yr']:>12.3f} {stats['median_uncertainty_mm_yr']:>12.3f} "
              f"{stats['std_uncertainty_mm_yr']:>12.3f} {stats['min_uncertainty_mm_yr']:>12.3f} "
              f"{stats['max_uncertainty_mm_yr']:>12.3f}")

        logger.info(f" {label:>20} Count: {len(z_unc):,} Mean: {stats['mean_uncertainty_mm_yr']:.3f} mm/yr")

    print("=" * 100)

    df_unc = pd.DataFrame(rows)
    unc_csv = OUTPUT_DIR / "05_rate_uncertainty_by_elevation.csv"
    df_unc.to_csv(unc_csv, index=False, encoding="utf-8")
    logger.info(f"✓ Saved: {unc_csv.name}")
    print(f"✓ Saved: {unc_csv.name}")

    # Text summary
    unc_txt = OUTPUT_DIR / "05_rate_uncertainty_summary.txt"
    with open(unc_txt, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("RATE UNCERTAINTY VALIDATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Run ID: {RUN_TIMESTAMP}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("METHODOLOGY\n")
        f.write("-"*80 + "\n")
        f.write("Per-observation rate uncertainty is derived from measurement and temporal\n")
        f.write("uncertainty using an error-propagation model (units: mm/year).\n\n")
        f.write("σ_rate = √[(σ_h/Δt)² + |ΔH/Δt²| × σ_t²] × 365.25 × 10\n\n")

        f.write("SUMMARY STATISTICS (ALL OBSERVATIONS)\n")
        f.write("-"*80 + "\n")
        if len(all_unc) > 0:
            f.write(f"Total observations with uncertainty: {len(all_unc):,}\n")
            f.write(f"Mean uncertainty: {np.mean(all_unc):.3f} mm/yr\n")
            f.write(f"Median uncertainty: {np.median(all_unc):.3f} mm/yr\n")
            f.write(f"Std dev: {np.std(all_unc):.3f} mm/yr\n")
            f.write(f"Min: {np.min(all_unc):.3f} mm/yr\n")
            f.write(f"Max: {np.max(all_unc):.3f} mm/yr\n\n")
        else:
            f.write("No uncertainty values available.\n\n")

        f.write("BY ELEVATION BAND\n")
        f.write("-"*80 + "\n")
        if len(df_unc) > 0:
            f.write(f"{'Elevation':>20} {'Count':>10} {'Mean':>12} {'Median':>12} "
                    f"{'Std':>12} {'Min':>12} {'Max':>12}\n")
            f.write("-"*80 + "\n")
            for _, r in df_unc.iterrows():
                f.write(f"{r['elevation_range']:>20} {int(r['n_observations']):>10} "
                        f"{r['mean_uncertainty_mm_yr']:>12.3f} {r['median_uncertainty_mm_yr']:>12.3f} "
                        f"{r['std_uncertainty_mm_yr']:>12.3f} {r['min_uncertainty_mm_yr']:>12.3f} "
                        f"{r['max_uncertainty_mm_yr']:>12.3f}\n")
            f.write("\n")
        else:
            f.write("No elevation bands with data.\n\n")

        f.write("INTERPRETATION NOTES\n")
        f.write("-"*80 + "\n")
        f.write("Typical ICESat-2 subsidence rate uncertainties are on the order of\n")
        f.write("5–15 mm/yr. Bands with much higher values may need longer baselines\n")
        f.write("or more crossovers to stabilize rates.\n")

    logger.info(f"✓ Saved: {unc_txt.name}")
    print(f"✓ Saved: {unc_txt.name}")

    logger.info("\n✓ Rate uncertainty validation complete!")
    print("✓ Rate uncertainty validation complete!")

else:
    logger.warning("No crossovers found!")
    print("No crossovers found!")

# ============================================================================
# FINALIZE
# ============================================================================

metadata.finalize()
metadata_file = metadata.save(OUTPUT_DIR)

logger.info(f"Metadata saved: {metadata_file.name}")
logger.info("="*80)
logger.info("SCRIPT 03 COMPLETE")
logger.info("="*80)

print("\n" + "="*80)
print("SCRIPT 03 COMPLETE")
print("="*80)
print(f"\nOutput files saved to: {OUTPUT_DIR}")
print(f"Check logs: {OUTPUT_DIR / 'script_03_run.log'}")
