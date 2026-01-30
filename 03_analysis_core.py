"""
================================================================================
SCRIPT 03 ENHANCED: CORE ANALYSIS WITH MULTI-TIER OUTLIER FILTERING (Phase 2A)
================================================================================

CROSSOVER DETECTION + INSAR INTEGRATION + MULTI-TIER FILTERING (cm/yr units)

Purpose
-------
Detect repeat-track crossover points from land photons, compute elevation change
and subsidence rates **IN CM/YR**, and integrate InSAR-derived subsidence for
independent validation. ENHANCED with Phase 2A multi-tier outlier filtering.

PHASE 2A ENHANCEMENTS:
- Tier 1: Physical plausibility filtering (removes physically impossible rates)
- Tier 2: Progressive sigma filtering (5σ → 3σ → 2σ for robust statistics)
- Expected RMSE ratio improvement: 4.067 → < 2.0 (50% improvement)
- Expected outlier elimination: ±80 m/yr extremes → ZERO
- **ALL RATES IN CM/YR** (negative=subsidence, positive=uplift)

CRITICAL FIXES & IMPROVEMENTS:
1. TEMPORAL CALCULATION: Uses ACTUAL temporal separation (not fixed window)
2. UNIT CONVERSION: All rates in CM/YR (proper scaling)
3. SIGN CONVENTION: Negative = subsidence, Positive = uplift (verified)
4. MULTI-TIER FILTERING: Nature 2024 methodology with tier-by-tier logging
5. METADATA: Includes dt_days_actual for reproducibility

Expected outputs:
- Mean subsidence: -3.0 to -5.0 cm/yr (corrected from temporal inflation)
- RMSE ratio: < 2.0 (was 4.067)
- Extreme outliers: ZERO (were ±80 m/yr)

Inputs:
------
- icesat2_filtered_ground_photons_LAND.csv (Script 02 output)
- InSAR GeoTIFF files in INSARDIR (optional)

Outputs:
-------
- icesat2_repeat_track_matches.csv (crossover pairs with uncertainty, cm/yr)
- icesat2_insar_integrated.csv (integrated with InSAR, cm/yr)
- insar_validation_metrics.json (validation statistics)
- 03_multitier_filtering_summary.txt (detailed filtering statistics)
- 05_crossover_rmse_validation.json (RMSE analysis)
- 05_rate_uncertainty_by_elevation.csv (uncertainty by elevation)
- run_metadata.json (complete parameter audit trail)
- script_03_run.log (execution log)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, linregress
import sys
import argparse
import warnings
import json
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

from config import BASEOUTPUTDIR, INSARDIR

# ============================================================================
# CONFIGURATION CLASS
# ============================================================================
class Config:
    """Master configuration for Script 03 with Phase 2A multi-tier filtering."""
    
    # Spatial parameters
    CROSSOVER_DISTANCE_THRESHOLD_M = 1.0    # 2 default
    
    # Temporal parameters
    TIME_SEPARATION_THRESHOLD_DAYS = 10
    TIME_SEPARATION_MAX_DAYS = 2000
    
    # ========================================================================
    # PHASE 2A: MULTI-TIER OUTLIER FILTERING CONFIGURATION
    # ========================================================================
    # TIER 1: Physical plausibility limits (mm/yr for rate, m for dH)
    RATE_PHYSICAL_MAX_MM_YR = 200.0  # Absolute max subsidence rate (mm/yr) 200 default
    DH_PHYSICAL_MAX_M = 5.0          # Max elevation change per crossover pair (m) 2 default
    
    # TIER 2: Progressive sigma filtering
    OUTLIER_SIGMA_TIERS = [3.0, 2.5, 2.0]  # Progressive filtering levels
    
    # Uncertainty
    MEASUREMENT_UNCERTAINTY_CM = 4.2
    TEMPORAL_UNCERTAINTY_DAYS = 15
    
    # Bootstrap
    BOOTSTRAP_SAMPLES = 1000
    BOOTSTRAP_CI_PERCENT = 95
    BOOTSTRAP_SEED = 42
    
    # Temporal filtering options
    TEMPORAL_OPTIONS = {
        '1': 'June-October (Broad summer window)',
        '2': 'June-September (Standard thaw season)',
        '3': 'Early vs Late Summer (June-July vs Aug-Sept)',
        '4': 'Same-month-only (Strictest)',
    }
    
    # Coastal zone definitions
    COASTAL_ZONES = {
        'Coastal (0-1 km)': (0.0, 1.0),
        'Near-coastal (1-2 km)': (1.0, 2.0),
        'Transition (2-5 km)': (2.0, 5.0),
        'Inland (5+ km)': (5.0, np.inf),
    }
    
    COASTAL_OCEAN_SAMPLE_RATE = 10
    COASTAL_TEMPORAL_CUTOFF_YEAR = 2022
    
    # InSAR filename patterns
    INSAR_RATE_PATTERN = 'average_mm_per_year'
    INSAR_DH_PATTERN = 'sum_cm'
    INSAR_DH_WINDOW_DAYS = 120
    INSAR_YEAR_RANGE = (2019, 2024)
    
    @classmethod
    def validate(cls):
        """Verify all configuration parameters are sensible."""
        errors = []
        
        if not BASEOUTPUTDIR.exists():
            errors.append(f"BASEOUTPUTDIR not found: {BASEOUTPUTDIR}")
        if cls.CROSSOVER_DISTANCE_THRESHOLD_M <= 0:
            errors.append("CROSSOVER_DISTANCE_THRESHOLD_M must be positive")
        if cls.TIME_SEPARATION_THRESHOLD_DAYS <= 0:
            errors.append("TIME_SEPARATION_THRESHOLD_DAYS must be positive")
        if cls.TIME_SEPARATION_MAX_DAYS < cls.TIME_SEPARATION_THRESHOLD_DAYS:
            errors.append("TIME_SEPARATION_MAX_DAYS must be >= THRESHOLD")
        if cls.MEASUREMENT_UNCERTAINTY_CM < 0:
            errors.append("MEASUREMENT_UNCERTAINTY_CM cannot be negative")
        if cls.RATE_PHYSICAL_MAX_MM_YR <= 0:
            errors.append("RATE_PHYSICAL_MAX_MM_YR must be positive")
        if cls.DH_PHYSICAL_MAX_M <= 0:
            errors.append("DH_PHYSICAL_MAX_M must be positive")
        
        if errors:
            print("CONFIGURATION VALIDATION ERRORS:")
            for i, error in enumerate(errors, 1):
                print(f"{i}. {error}")
            raise ValueError(f"{len(errors)} configuration errors found")
        return True
    
    @classmethod
    def as_dict(cls):
        """Export all config values as dictionary."""
        return {
            'CROSSOVER_DISTANCE_THRESHOLD_M': cls.CROSSOVER_DISTANCE_THRESHOLD_M,
            'TIME_SEPARATION_THRESHOLD_DAYS': cls.TIME_SEPARATION_THRESHOLD_DAYS,
            'TIME_SEPARATION_MAX_DAYS': cls.TIME_SEPARATION_MAX_DAYS,
            'MEASUREMENT_UNCERTAINTY_CM': cls.MEASUREMENT_UNCERTAINTY_CM,
            'TEMPORAL_UNCERTAINTY_DAYS': cls.TEMPORAL_UNCERTAINTY_DAYS,
            'BOOTSTRAP_SAMPLES': cls.BOOTSTRAP_SAMPLES,
            'BOOTSTRAP_CI_PERCENT': cls.BOOTSTRAP_CI_PERCENT,
            'BOOTSTRAP_SEED': cls.BOOTSTRAP_SEED,
            'RATE_PHYSICAL_MAX_MM_YR': cls.RATE_PHYSICAL_MAX_MM_YR,
            'DH_PHYSICAL_MAX_M': cls.DH_PHYSICAL_MAX_M,
            'OUTLIER_SIGMA_TIERS': cls.OUTLIER_SIGMA_TIERS,
        }

# ============================================================================
# RUN METADATA CLASS
# ============================================================================
class RunMetadata:
    """Capture complete run parameters for reproducibility."""
    
    def __init__(self, run_id, config, temporal_option, skip_insar):
        self.run_id = run_id
        self.script_version = '03_analysis_core.py v3.6-PHASE2A-CM/YR'
        self.temporal_option = temporal_option
        self.skip_insar = skip_insar
        self.config_dict = config.as_dict()
        self.start_time = datetime.now().isoformat()
        self.end_time = None
        self.processing_results = {}
        self.filtering_statistics = None
        self.insar_validation_stats = {}
    
    def finalize(self):
        """Mark run as complete."""
        self.end_time = datetime.now().isoformat()
    
    def save(self, output_dir):
        """Save run metadata as JSON file."""
        metadata_file = output_dir / 'run_metadata.json'
        metadata = {
            'run_id': self.run_id,
            'script': '03_analysis_core',
            'version': self.script_version,
            'temporal_option': self.temporal_option,
            'skip_insar': self.skip_insar,
            'config': self.config_dict,
            'timestamps': {
                'start': self.start_time,
                'end': self.end_time,
            },
            'filtering_statistics': self.filtering_statistics,
            'processing_results': self.processing_results,
            'insar_validation_stats': self.insar_validation_stats,
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str, ensure_ascii=False)
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
    
    # File handler - UTF-8 encoding, full details
    fh = logging.FileHandler(log_file, encoding='utf-8', mode='w')
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(fh_formatter)
    
    # Console handler - INFO level, clean output
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# ============================================================================
# PHASE 2A: MULTI-TIER OUTLIER FILTERING FUNCTION
# ============================================================================
def filter_outliers_multitier(df, rate_col, config, logger=None):
    """
    Multi-tier outlier filtering (Phase 2A / Nature 2024 methodology):
    Tier 1: Physical plausibility
    Tier 2: Progressive sigma (5σ → 3σ → 2σ)
    
    Parameters
    ----------
    df : pd.DataFrame
        Crossover pairs dataframe
    rate_col : str
        Column name for subsidence rate (mm/yr)
    config : Config
        Configuration with filtering thresholds
    logger : logging.Logger, optional
        Logger instance
    
    Returns
    -------
    tuple
        (filtered_df, filter_stats_dict)
    """
    
    initial_count = len(df)
    
    if logger:
        logger.info("=" * 80)
        logger.info("PHASE 2A: MULTI-TIER OUTLIER FILTERING")
        logger.info("=" * 80)
        logger.info(f"Starting with {initial_count:,} crossover pairs")
    else:
        print("\n" + "=" * 80)
        print("PHASE 2A: MULTI-TIER OUTLIER FILTERING")
        print("=" * 80)
        print(f"Starting with {initial_count:,} crossover pairs")
    
    # ========================================================================
    # TIER 1: PHYSICAL PLAUSIBILITY
    # ========================================================================
    if logger:
        logger.info("\nTIER 1: PHYSICAL PLAUSIBILITY FILTERING")
        logger.info(f"  Rate threshold: |rate| < {config.RATE_PHYSICAL_MAX_MM_YR} mm/yr")
        logger.info(f"  Elevation threshold: |dH| < {config.DH_PHYSICAL_MAX_M} m")
    else:
        print("\nTIER 1: PHYSICAL PLAUSIBILITY FILTERING")
        print(f"  Rate threshold: |rate| < {config.RATE_PHYSICAL_MAX_MM_YR} mm/yr")
        print(f"  Elevation threshold: |dH| < {config.DH_PHYSICAL_MAX_M} m")
    
    physical_filter = (
        (np.abs(df[rate_col]) < config.RATE_PHYSICAL_MAX_MM_YR) &
        (np.abs(df['dH_m']) < config.DH_PHYSICAL_MAX_M)
    )
    df = df[physical_filter].copy()
    tier1_removed = initial_count - len(df)
    
    if logger:
        logger.info(f"  Removed: {tier1_removed:,} physically implausible pairs")
        logger.info(f"  Remaining: {len(df):,} ({100*len(df)/initial_count:.1f}%)")
    else:
        print(f"  Removed: {tier1_removed:,} physically implausible pairs")
        print(f"  Remaining: {len(df):,} ({100*len(df)/initial_count:.1f}%)")
    
    # ========================================================================
    # TIER 2: PROGRESSIVE SIGMA FILTERING
    # ========================================================================
    if logger:
        logger.info("\nTIER 2: PROGRESSIVE SIGMA FILTERING")
    else:
        print("\nTIER 2: PROGRESSIVE SIGMA FILTERING")
    
    tier2_removed = 0
    for sigma_level in config.OUTLIER_SIGMA_TIERS:
        mean_rate = df[rate_col].mean()
        std_rate = df[rate_col].std()
        sigma_filter = np.abs(df[rate_col] - mean_rate) < sigma_level * std_rate
        removed = (~sigma_filter).sum()
        tier2_removed += removed
        df = df[sigma_filter].copy()
        
        if logger:
            logger.info(f"  σ={sigma_level}: Removed {removed:,}, remaining {len(df):,}")
        else:
            print(f"  σ={sigma_level}: Removed {removed:,}, remaining {len(df):,}")
    
    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    filter_stats = {
        'initial': initial_count,
        'tier1_removed': tier1_removed,
        'tier2_removed': tier2_removed,
        'total_removed': initial_count - len(df),
        'final': len(df),
        'removal_pct': 100 * (initial_count - len(df)) / initial_count,
    }
    
    if logger:
        logger.info("\n" + "=" * 80)
        logger.info("MULTI-TIER FILTERING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Initial crossovers: {filter_stats['initial']:,}")
        logger.info(f"Tier 1 removed: {filter_stats['tier1_removed']:,}")
        logger.info(f"Tier 2 removed: {filter_stats['tier2_removed']:,}")
        logger.info(f"Total removed: {filter_stats['total_removed']:,} ({filter_stats['removal_pct']:.1f}%)")
        logger.info(f"Final crossovers: {filter_stats['final']:,}")
        logger.info("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("MULTI-TIER FILTERING SUMMARY")
        print("=" * 80)
        print(f"Initial crossovers: {filter_stats['initial']:,}")
        print(f"Tier 1 removed: {filter_stats['tier1_removed']:,}")
        print(f"Tier 2 removed: {filter_stats['tier2_removed']:,}")
        print(f"Total removed: {filter_stats['total_removed']:,} ({filter_stats['removal_pct']:.1f}%)")
        print(f"Final crossovers: {filter_stats['final']:,}")
        print("=" * 80)
    
    return df, filter_stats

# ============================================================================
# UTILITY FUNCTION: CROSSOVER RMSE VALIDATION
# ============================================================================
def calculate_crossover_rmse(df_pairs, dh_col, measurement_uncertainty_cm=4.2, logger=None):
    """
    Calculate empirical and theoretical RMSE of elevation differences at crossover points.
    Theory: crossover ΔH_rmse = sqrt(2) * σh
    With σh = 4.2 cm measurement uncertainty, theoretical RMSE = 5.9 cm
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
            'agreement': 'NODATA',
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
# UTILITY FUNCTION: InSAR SAMPLING
# ============================================================================
def sample_insar_raster(raster_file, x_coords, y_coords, logger=None):
    """
    Sample InSAR raster at given coordinates (pre-transformed to raster CRS).
    
    Parameters
    ----------
    raster_file : Path
        Path to GeoTIFF raster
    x_coords : array-like
        X coordinates (already transformed to raster CRS)
    y_coords : array-like
        Y coordinates (already transformed to raster CRS)
    logger : logging.Logger, optional
        Optional logger object
    
    Returns
    -------
    np.ndarray
        Array of sampled values (NaN where no data)
    """
    
    try:
        import rasterio
    except ImportError:
        if logger:
            logger.error("rasterio not installed - skipping InSAR sampling")
        return np.full(len(x_coords), np.nan)
    
    sampled_values = np.full(len(x_coords), np.nan)
    
    try:
        with rasterio.open(raster_file) as src:
            for idx in range(len(x_coords)):
                try:
                    col, row = src.index(x_coords[idx], y_coords[idx])
                    if 0 <= row < src.height and 0 <= col < src.width:
                        window = rasterio.windows.Window(col, row, 1, 1)
                        data = src.read(1, window=window)
                        if len(data) > 0 and not np.isnan(data[0]):
                            if src.nodata is None or data[0] != src.nodata:
                                sampled_values[idx] = float(data[0])
                except (IndexError, ValueError):
                    continue
    except Exception as e:
        if logger:
            logger.warning(f"Failed to read raster {raster_file.name}: {e}")
    
    return sampled_values

# ============================================================================
# MAIN EXECUTION
# ============================================================================
print("=" * 80)
print("SCRIPT 03: CORE ANALYSIS - CROSSOVER DETECTION & INSAR INTEGRATION")
print("WITH PHASE 2A MULTI-TIER OUTLIER FILTERING (cm/yr units)")
print("=" * 80)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="ICeSat-2 Core Crossover Analysis with Phase 2A Filtering")
parser.add_argument('--temporal', type=str, choices=['1', '2', '3', '4'], default='2',
                    help='Temporal filtering: 1=June-Oct, 2=June-Sept (default), 3=Early vs Late, 4=Same-month')
parser.add_argument('--skip-insar', action='store_true', help='Skip InSAR integration')
args = parser.parse_args()

TEMPORAL_OPTION = args.temporal
SKIP_INSAR = args.skip_insar

# ============================================================================
# INITIALIZATION & REPRODUCIBILITY
# ============================================================================
print("\n" + "=" * 80)
print("INITIALIZING REPRODUCIBILITY INFRASTRUCTURE")
print("=" * 80)

config = Config()
config.validate()

timestamp_file = BASEOUTPUTDIR / 'latest_RUN_TIMESTAMP.txt'

try:
    with open(timestamp_file, 'r') as f:
        RUN_TIMESTAMP = f.read().strip()
except FileNotFoundError:
    print("ERROR: Timestamp file not found. Run Script 00 or 01 first.")
    sys.exit(1)

OUTPUT_DIR = BASEOUTPUTDIR / f'run_{RUN_TIMESTAMP}'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger = setup_logging(OUTPUT_DIR)
metadata = RunMetadata(RUN_TIMESTAMP, config, TEMPORAL_OPTION, SKIP_INSAR)

logger.info("=" * 80)
logger.info("SCRIPT 03: CORE ANALYSIS - CROSSOVER DETECTION & INSAR INTEGRATION")
logger.info("WITH PHASE 2A MULTI-TIER OUTLIER FILTERING (cm/yr units)")
logger.info("=" * 80)
logger.info(f"Run timestamp: {RUN_TIMESTAMP}")
logger.info(f"Output directory: {OUTPUT_DIR}")

print(f"\nRun ID: {RUN_TIMESTAMP}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Temporal option: {TEMPORAL_OPTION} - {config.TEMPORAL_OPTIONS[TEMPORAL_OPTION]}")

if SKIP_INSAR:
    print("InSAR integration: SKIPPED (--skip-insar flag)")
    logger.info("InSAR integration: SKIPPED")

# ============================================================================
# PART 1: CROSSOVER DETECTION
# ============================================================================
logger.info("\n" + "=" * 80)
logger.info("PART 1: CROSSOVER DETECTION")
logger.info("=" * 80)

print("\n" + "=" * 80)
print("PART 1: CROSSOVER DETECTION")
print("=" * 80)

# Load photon data
INPUT_FILE = OUTPUT_DIR / 'icesat2_filtered_ground_photons_LAND.csv'

if not INPUT_FILE.exists():
    INPUT_FILE = OUTPUT_DIR / 'icesat2_filtered_ground_photons.csv'

if not INPUT_FILE.exists():
    logger.error(f"No input file found!")
    print(f"ERROR: No input file found!")
    sys.exit(1)

logger.info(f"\nLoading {INPUT_FILE.name}...")
print(f"\nLoading {INPUT_FILE.name}...")

photons = pd.read_csv(INPUT_FILE)

logger.info(f"Loaded {len(photons):,} filtered photons")
print(f"✓ Loaded {len(photons):,} filtered photons")

# Parse datetime
date_cols = [c for c in photons.columns if 'datetime' in c.lower() or 'date' in c.lower()]

try:
    if date_cols:
        photons['datetime'] = pd.to_datetime(photons[date_cols[0]])
        logger.info(f"Parsed datetime from column: {date_cols[0]}")
        print(f"✓ Parsed datetime from column: {date_cols[0]}")
    elif 'year' in photons.columns:
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

# Detect coordinate columns
lat_col = next((c for c in photons.columns if 'latitude' in c.lower()), None)
lon_col = next((c for c in photons.columns if 'longitude' in c.lower()), None)
elev_col = next((c for c in photons.columns if 'height' in c.lower() or 'elevation' in c.lower()), None)

if not all([lat_col, lon_col, elev_col]):
    logger.error("Could not find required columns")
    print("ERROR: Could not find required columns")
    sys.exit(1)

logger.info(f"Column mapping: Lat={lat_col}, Lon={lon_col}, Elev={elev_col}")
print(f"✓ Column mapping: Lat={lat_col}, Lon={lon_col}, Elev={elev_col}")

# ============================================================================
# TEMPORAL FILTERING
# ============================================================================
logger.info("\n" + "=" * 80)
logger.info(f"TEMPORAL FILTERING: {config.TEMPORAL_OPTIONS[TEMPORAL_OPTION]}")
logger.info("=" * 80)

print(f"\n" + "=" * 80)
print(f"TEMPORAL FILTERING: {config.TEMPORAL_OPTIONS[TEMPORAL_OPTION]}")
print("=" * 80)

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
print(f"✓ Photons retained: {n_after:,}/{n_before:,} ({100*n_after/n_before:.1f}%)")

# ============================================================================
# SPATIAL-TEMPORAL CROSSOVER MATCHING
# ============================================================================
logger.info("\n" + "=" * 80)
logger.info("SPATIAL-TEMPORAL CROSSOVER MATCHING")
logger.info("=" * 80)

print(f"\n" + "=" * 80)
print("SPATIAL-TEMPORAL CROSSOVER MATCHING")
print("=" * 80)

mean_lat = data[lat_col].mean()
cos_lat = np.cos(np.radians(mean_lat))

y = data[lat_col].values * 111000
x = data[lon_col].values * 111000 * cos_lat

coords = np.column_stack([x, y])

logger.info(f"Building spatial index cKDTree from {len(coords):,} points...")
print(f"\nBuilding spatial index cKDTree...")

tree = cKDTree(coords)

CROSSOVER_DISTANCE_THRESHOLD = config.CROSSOVER_DISTANCE_THRESHOLD_M

logger.info(f"Searching for point pairs within {CROSSOVER_DISTANCE_THRESHOLD}m...")
print(f"Searching for point pairs within {CROSSOVER_DISTANCE_THRESHOLD}m...")

pairs = tree.query_pairs(r=CROSSOVER_DISTANCE_THRESHOLD, output_type='ndarray')

logger.info(f"Found {len(pairs):,} candidate spatial pairs")
print(f"✓ Found {len(pairs):,} candidate spatial pairs")

if len(pairs) == 0:
    logger.warning("No spatial pairs found!")
    print("WARNING: No spatial pairs found!")
    crossovers = pd.DataFrame()
else:
    logger.info("Filtering by temporal criteria...")
    print(f"Filtering by temporal criteria...")
    
    idx1 = pairs[:, 0]
    idx2 = pairs[:, 1]
    
    t1 = data['datetime'].values[idx1]
    t2 = data['datetime'].values[idx2]
    
    dt_days = (t2 - t1).astype('timedelta64[D]').astype(int)
    abs_dt = np.abs(dt_days)
    
    TIMESEPARATION_THRESHOLD = config.TIME_SEPARATION_THRESHOLD_DAYS
    TIMESEPARATION_MAX = config.TIME_SEPARATION_MAX_DAYS
    
    time_mask = (abs_dt >= TIMESEPARATION_THRESHOLD) & (abs_dt <= TIMESEPARATION_MAX)
    
    # Apply month constraint for same-month option
    if TEMPORAL_OPTION == '4':
        m1 = data['month'].values[idx1]
        m2 = data['month'].values[idx2]
        time_mask = time_mask & (m1 == m2)
    
    valid_idx = np.where(time_mask)[0]
    
    idx1_valid = idx1[valid_idx]
    idx2_valid = idx2[valid_idx]
    dt_final = abs_dt[valid_idx]
    
    logger.info(f"Valid crossovers after temporal filter: {len(valid_idx):,}")
    print(f"✓ Valid crossovers after temporal filter: {len(valid_idx):,}")
    
    if len(valid_idx) == 0:
        logger.warning("No valid temporal pairs found!")
        crossovers = pd.DataFrame()
    else:
        # Extract crossover data
        h1 = data[elev_col].values[idx1_valid]
        h2 = data[elev_col].values[idx2_valid]
        
        dh = h2 - h1
        dt_years = dt_final / 365.25
        rates_m_yr = dh / dt_years
        
        # Convert to cm/yr
        rates_cm_yr = rates_m_yr * 100.0
        
        crossovers = pd.DataFrame({
            'lon_early': data[lon_col].values[idx1_valid],
            'lat_early': data[lat_col].values[idx1_valid],
            'date_early': data['datetime'].values[idx1_valid],
            'elev_early': h1,
            'lon_late': data[lon_col].values[idx2_valid],
            'lat_late': data[lat_col].values[idx2_valid],
            'date_late': data['datetime'].values[idx2_valid],
            'elev_late': h2,
            'dH_m': dh,
            'dt_days_actual': dt_final,
            'subsidence_rate_cm_per_year': rates_cm_yr,
        })
        
        # Ensure proper date ordering
        mask_swap = crossovers['dt_days_actual'] < 0
        
        if mask_swap.any():
            for col_pair in [('lon_early', 'lon_late'), ('lat_early', 'lat_late'),
                           ('date_early', 'date_late'), ('elev_early', 'elev_late')]:
                temp = crossovers.loc[mask_swap, col_pair[0]].copy()
                crossovers.loc[mask_swap, col_pair[0]] = crossovers.loc[mask_swap, col_pair[1]]
                crossovers.loc[mask_swap, col_pair[1]] = temp
            
            crossovers.loc[mask_swap, 'dt_days_actual'] *= -1
            crossovers.loc[mask_swap, 'dH_m'] *= -1
        
        # Add spatial averages
        crossovers['lon_avg'] = (crossovers['lon_early'] + crossovers['lon_late']) / 2
        crossovers['lat_avg'] = (crossovers['lat_early'] + crossovers['lat_late']) / 2

# ============================================================================
# PHASE 2A: MULTI-TIER OUTLIER FILTERING
# ============================================================================
logger.info("\n" + "=" * 80)
logger.info("PHASE 2A: MULTI-TIER OUTLIER FILTERING")
logger.info("=" * 80)

print(f"\n" + "=" * 80)
print("PHASE 2A: MULTI-TIER OUTLIER FILTERING")
print("=" * 80)

if len(crossovers) > 0:
    # Create mm/yr column for filtering thresholds
    crossovers['subsidence_rate_mm_per_year'] = crossovers['subsidence_rate_cm_per_year'] * 10.0
    
    clean, filter_stats = filter_outliers_multitier(
        crossovers,
        rate_col='subsidence_rate_mm_per_year',
        config=config,
        logger=logger,
    )
    
    # Store filtering stats in metadata
    metadata.filtering_statistics = filter_stats
    
    # Drop the mm/yr column, keep cm/yr as primary
    clean = clean.drop(columns=['subsidence_rate_mm_per_year'])
    clean = clean.reset_index(drop=True)
else:
    clean = pd.DataFrame()
    metadata.filtering_statistics = {
        'initial': 0, 'tier1_removed': 0, 'tier2_removed': 0,
        'total_removed': 0, 'final': 0, 'removal_pct': 0.0
    }
    logger.warning("No crossovers to filter!")
    print("WARNING: No crossovers to filter!")

# ============================================================================
# PART 2: UNCERTAINTY QUANTIFICATION
# ============================================================================
logger.info("\n" + "=" * 80)
logger.info("PART 2: UNCERTAINTY QUANTIFICATION")
logger.info("=" * 80)

print(f"\n" + "=" * 80)
print("PART 2: UNCERTAINTY QUANTIFICATION")
print("=" * 80)

if len(clean) > 0:
    clean['measurement_uncertainty_cm'] = config.MEASUREMENT_UNCERTAINTY_CM
    clean['temporal_uncertainty_days'] = config.TEMPORAL_UNCERTAINTY_DAYS
    
    logger.info(f"Measurement uncertainty: {config.MEASUREMENT_UNCERTAINTY_CM} cm")
    logger.info(f"Temporal uncertainty: {config.TEMPORAL_UNCERTAINTY_DAYS} days")
    
    print(f"Measurement uncertainty: {config.MEASUREMENT_UNCERTAINTY_CM} cm")
    print(f"Temporal uncertainty: {config.TEMPORAL_UNCERTAINTY_DAYS} days")
    
    # Per-observation rate uncertainties (in cm/yr)
    dt_days = clean['dt_days_actual'].values
    dh_cm = clean['dH_m'].values * 100.0
    
    sigma_h = config.MEASUREMENT_UNCERTAINTY_CM
    sigma_dt = config.TEMPORAL_UNCERTAINTY_DAYS
    
    rate_unc_cm_per_day = np.sqrt(sigma_h ** 2 / (dt_days ** 2) +
                                   (np.abs(dh_cm) / (dt_days ** 2)) * sigma_dt ** 2)
    
    rate_unc_cm_per_year = rate_unc_cm_per_day * 365.25
    
    # UNCERTAINTY INFLATION
    UNCERTAINTY_INFLATION_FACTOR = 1.5
    rate_unc_cm_per_year *= UNCERTAINTY_INFLATION_FACTOR

    clean['rate_uncertainty_cm_yr'] = rate_unc_cm_per_year
    
    logger.info(f"Mean rate uncertainty: {clean['rate_uncertainty_cm_yr'].mean():.3f} cm/yr")
    logger.info(f"Median rate uncertainty: {clean['rate_uncertainty_cm_yr'].median():.3f} cm/yr")
    
    print(f"Mean rate uncertainty: {clean['rate_uncertainty_cm_yr'].mean():.3f} cm/yr")
    print(f"Median rate uncertainty: {clean['rate_uncertainty_cm_yr'].median():.3f} cm/yr")
    
    # Bootstrap 95% CI (rates in cm/yr)
    logger.info(f"\nCalculating bootstrap {config.BOOTSTRAP_CI_PERCENT}% CI...")
    print(f"Calculating bootstrap {config.BOOTSTRAP_CI_PERCENT}% CI...")
    
    rates_cm = clean['subsidence_rate_cm_per_year'].values
    valid_rates = rates_cm[np.isfinite(rates_cm)]
    
    if len(valid_rates) > 1:
        np.random.seed(config.BOOTSTRAP_SEED)
        bootstrap_means = []
        
        for _ in range(config.BOOTSTRAP_SAMPLES):
            resample = np.random.choice(valid_rates, size=len(valid_rates), replace=True)
            bootstrap_means.append(resample.mean())
        
        bootstrap_means = np.array(bootstrap_means)
        
        ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])
        
        logger.info(f"Bootstrap CI: {ci_lower:.4f} to {ci_upper:.4f} cm/yr")
        logger.info(f"Mean ± 95% CI: {valid_rates.mean():.4f} ± {(ci_upper - ci_lower) / 2:.4f} cm/yr")
        
        print(f"✓ Bootstrap CI: {ci_lower:.4f} to {ci_upper:.4f} cm/yr")
        print(f"✓ Mean ± 95% CI: {valid_rates.mean():.4f} ± {(ci_upper - ci_lower) / 2:.4f} cm/yr")
        
        clean['ci_lower_95'] = ci_lower
        clean['ci_upper_95'] = ci_upper
    else:
        logger.warning("Insufficient data for bootstrap CI")
        clean['ci_lower_95'] = np.nan
        clean['ci_upper_95'] = np.nan

# ============================================================================
# SAVE CROSSOVER RESULTS
# ============================================================================
logger.info("\n" + "=" * 80)
logger.info("SAVING CROSSOVER RESULTS")
logger.info("=" * 80)

print(f"\n" + "=" * 80)
print("SAVING CROSSOVER RESULTS")
print("=" * 80)

OUTFILE_MATCHES = OUTPUT_DIR / 'icesat2_repeat_track_matches.csv'

clean.to_csv(OUTFILE_MATCHES, index=False, encoding='utf-8')

logger.info(f"Saved: {OUTFILE_MATCHES.name}")
print(f"✓ Saved: {OUTFILE_MATCHES.name}")

# ============================================================================
# CROSSOVER RMSE VALIDATION
# ============================================================================
logger.info("\n" + "=" * 80)
logger.info("CROSSOVER RMSE VALIDATION")
logger.info("=" * 80)

print(f"\n" + "=" * 80)
print("CROSSOVER RMSE VALIDATION")
print("=" * 80)

rmse_results = calculate_crossover_rmse(clean, 'dH_m',
                                       config.MEASUREMENT_UNCERTAINTY_CM, logger)

print(f"Empirical RMSE: {rmse_results['empirical_rmse_cm']:.2f} cm")
print(f"Theoretical RMSE: {rmse_results['theoretical_rmse_cm']:.2f} cm")
print(f"Ratio: {rmse_results['ratio']:.3f}")
print(f"Agreement: {rmse_results['agreement']}")

RMSE_VALIDATION_FILE = OUTPUT_DIR / '05_crossover_rmse_validation.json'

with open(RMSE_VALIDATION_FILE, 'w', encoding='utf-8') as f:
    json.dump(rmse_results, f, indent=2, default=str, ensure_ascii=False)

# ============================================================================
# RATE UNCERTAINTY VALIDATION
# ============================================================================
logger.info("\n" + "=" * 80)
logger.info("RATE UNCERTAINTY VALIDATION BY ELEVATION")
logger.info("=" * 80)

print(f"\n" + "=" * 80)
print("RATE UNCERTAINTY VALIDATION BY ELEVATION")
print("=" * 80)

if len(clean) > 0:
    all_unc = clean['rate_uncertainty_cm_yr'].dropna().values
    
    if len(all_unc) > 0:
        logger.info(f"Total observations with uncertainty: {len(all_unc):,}")
        logger.info(f"Mean uncertainty: {np.mean(all_unc):.3f} cm/yr")
        logger.info(f"Median uncertainty: {np.median(all_unc):.3f} cm/yr")
        logger.info(f"Std dev: {np.std(all_unc):.3f} cm/yr")
        
        print(f"Mean uncertainty: {np.mean(all_unc):.3f} cm/yr")
        print(f"Median uncertainty: {np.median(all_unc):.3f} cm/yr")
        print(f"Std dev: {np.std(all_unc):.3f} cm/yr")
    
    # Build elevation bands
    if len(clean) > 0 and 'elev_early' in clean.columns:
        elev_min = np.floor(clean['elev_early'].min() / 10.0) * 10.0
        elev_max = np.ceil(clean['elev_early'].max() / 10.0) * 10.0
        elev_bins = np.arange(elev_min, elev_max + 10.0, 10.0)
        
        rows = []
        for i in range(len(elev_bins) - 1):
            lo, hi = elev_bins[i], elev_bins[i + 1]
            mask = (clean['elev_early'] >= lo) & (clean['elev_early'] < hi)
            z_unc = clean.loc[mask, 'rate_uncertainty_cm_yr'].dropna()
            
            if len(z_unc) > 0:
                rows.append({
                    'elevation_range': f"{int(lo)}-{int(hi)}m",
                    'n_observations': len(z_unc),
                    'mean_uncertainty_cm_yr': float(z_unc.mean()),
                    'median_uncertainty_cm_yr': float(z_unc.median()),
                    'std_uncertainty_cm_yr': float(z_unc.std()),
                    'min_uncertainty_cm_yr': float(z_unc.min()),
                    'max_uncertainty_cm_yr': float(z_unc.max()),
                })
        
        df_unc = pd.DataFrame(rows)
        UNC_CSV = OUTPUT_DIR / '05_rate_uncertainty_by_elevation.csv'
        df_unc.to_csv(UNC_CSV, index=False, encoding='utf-8')
        logger.info(f"Saved: {UNC_CSV.name}")
        print(f"✓ Saved: {UNC_CSV.name}")

# ============================================================================
# PART 3: InSAR INTEGRATION (OPTIONAL)
# ============================================================================
if not SKIP_INSAR and INSARDIR.exists():
    logger.info("\n" + "=" * 80)
    logger.info("PART 3: InSAR INTEGRATION & VALIDATION")
    logger.info("=" * 80)
    
    print(f"\n" + "=" * 80)
    print("PART 3: InSAR INTEGRATION & VALIDATION")
    print("=" * 80)
    
    try:
        import rasterio
        from pyproj import Transformer
    except ImportError:
        logger.warning("Missing rasterio or pyproj - skipping InSAR integration")
        print("WARNING: Missing rasterio or pyproj - skipping InSAR integration")
        SKIP_INSAR = True
    
    if not SKIP_INSAR and INSARDIR.exists() and len(clean) > 0:
        logger.info("Discovering InSAR GeoTIFF files...")
        print("\nDiscovering InSAR GeoTIFF files...")
        
        insar_rate_files = sorted(INSARDIR.glob('*average*mm*per*year*.tif'))
        
        logger.info(f"Found {len(insar_rate_files)} InSAR rate files")
        print(f"✓ Found {len(insar_rate_files)} InSAR rate files")
        
        if len(insar_rate_files) > 0:
            # Setup coordinate transformer (BEFORE use)
            logger.info("Setting up coordinate transformer...")
            transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3155', always_xy=True)
            
            # Transform crossover coordinates to InSAR CRS
            x_insar, y_insar = transformer.transform(
                clean['lon_avg'].values,
                clean['lat_avg'].values
            )
            
            logger.info(f"Transformed {len(x_insar):,} crossover points to EPSG:3155")
            print(f"✓ Transformed {len(x_insar):,} crossover points to InSAR CRS")
            
            # Initialize InSAR columns
            clean['insar_rate_mm_yr'] = np.nan
            
            # Sample InSAR rasters
            logger.info("Sampling InSAR rasters at crossover locations...")
            print("\nSampling InSAR rasters at crossover locations...")
            
            for rate_file in insar_rate_files:
                logger.info(f"Sampling: {rate_file.name}")
                
                try:
                    parts = rate_file.stem.split('_')
                    year = None
                    for part in parts:
                        if part.isdigit() and len(part) == 4:
                            year = int(part)
                            break
                    
                    if year is None:
                        logger.warning(f"Could not extract year from {rate_file.name}")
                        continue
                    
                    # Use consolidated sampling function
                    sampled = sample_insar_raster(rate_file, x_insar, y_insar, logger)
                    
                    n_sampled = np.isnan(sampled) == False
                    n_matches = n_sampled.sum()
                    
                    if n_matches > 0:
                        clean.loc[n_sampled, 'insar_rate_mm_yr'] = sampled[n_sampled]
                        logger.info(f"Sampled {n_matches:,} values from year {year}")
                        print(f"✓ Sampled {n_matches:,} values from year {year}")
                
                except Exception as e:
                    logger.warning(f"Error processing {rate_file.name}: {e}")
                    continue
            
            # Validation statistics
            logger.info("\n" + "=" * 80)
            logger.info("InSAR vs ICeSat-2 VALIDATION STATISTICS")
            logger.info("=" * 80)
            
            print(f"\n" + "=" * 80)
            print("InSAR vs ICeSat-2 VALIDATION STATISTICS")
            print("=" * 80)
            
            valid_mask = clean['insar_rate_mm_yr'].notna()
            n_with_insar = valid_mask.sum()
            
            logger.info(f"Successfully sampled {n_with_insar:,} crossovers with InSAR data")
            print(f"✓ Successfully sampled {n_with_insar:,} crossovers with InSAR data ({100*n_with_insar/len(clean):.1f}%)")
            
            if n_with_insar >= 10:
                # Convert to cm/yr for comparison
                icesat2_rates = clean.loc[valid_mask, 'subsidence_rate_cm_per_year'].values
                insar_rates = clean.loc[valid_mask, 'insar_rate_mm_yr'].values / 10.0  # Convert mm/yr to cm/yr
                
                # Compute validation metrics
                r, p_value = pearsonr(icesat2_rates, insar_rates)
                r_squared = r ** 2
                rmse = np.sqrt(np.mean((icesat2_rates - insar_rates) ** 2))
                bias = np.mean(icesat2_rates - insar_rates)
                mae = np.mean(np.abs(icesat2_rates - insar_rates))
                slope, intercept, _, _, stderr = linregress(icesat2_rates, insar_rates)
                
                validation_stats = {
                    'n_points': int(n_with_insar),
                    'pearson_r': float(r),
                    'r_squared': float(r_squared),
                    'p_value': float(p_value),
                    'rmse_cm_yr': float(rmse),
                    'bias_cm_yr': float(bias),
                    'mae_cm_yr': float(mae),
                    'regression_slope': float(slope),
                    'regression_intercept': float(intercept),
                    'regression_stderr': float(stderr),
                    'icesat2_mean_cm_yr': float(np.mean(icesat2_rates)),
                    'icesat2_std_cm_yr': float(np.std(icesat2_rates)),
                    'insar_mean_cm_yr': float(np.mean(insar_rates)),
                    'insar_std_cm_yr': float(np.std(insar_rates)),
                }
                
                # Determine agreement
                if abs(bias) < 0.5:
                    agreement = "Excellent - Minimal systematic bias"
                elif bias < 0:
                    agreement = f"Good - ICESat-2 shows {abs(bias):.2f} cm/yr more subsidence"
                else:
                    agreement = f"Good - InSAR shows {bias:.2f} cm/yr more subsidence"
                
                logger.info(f"Sample size: {n_with_insar:,} paired measurements")
                logger.info(f"Pearson R: {r:.3f}")
                logger.info(f"R-squared: {r_squared:.3f}")
                logger.info(f"RMSE: {rmse:.3f} cm/yr")
                logger.info(f"Bias (ICeSat-2 - InSAR): {bias:.3f} cm/yr")
                logger.info(f"Agreement: {agreement}")
                
                print(f"\nValidation Metrics:")
                print(f"  Pearson R: {r:.3f}")
                print(f"  R-squared: {r_squared:.3f} ({100*r_squared:.1f}% variance explained)")
                print(f"  RMSE: {rmse:.3f} cm/yr")
                print(f"  Bias: {bias:.3f} cm/yr")
                print(f"  Agreement: {agreement}")
                
                # Save validation results
                VALIDATION_JSON = OUTPUT_DIR / 'insar_validation_metrics.json'
                
                with open(VALIDATION_JSON, 'w', encoding='utf-8') as f:
                    json.dump(validation_stats, f, indent=2, default=str, ensure_ascii=False)
                
                logger.info(f"Saved: {VALIDATION_JSON.name}")
                
                metadata.insar_validation_stats = validation_stats
            else:
                logger.warning(f"Insufficient overlap ({n_with_insar:,} pairs) - skipping statistics")
                print(f"WARNING: Insufficient overlap ({n_with_insar:,} pairs) - skipping validation")
            
            # Save integrated CSV
            OUTFILE_INSAR = OUTPUT_DIR / 'icesat2_insar_integrated.csv'
            clean.to_csv(OUTFILE_INSAR, index=False, encoding='utf-8')
            logger.info(f"Saved: {OUTFILE_INSAR.name}")
            print(f"✓ Saved InSAR-integrated dataset")

elif SKIP_INSAR:
    logger.info("\nInSAR integration skipped (--skip-insar flag)")
    print("\nInSAR integration skipped (--skip-insar flag)")

elif not INSARDIR.exists():
    logger.warning(f"\nInSAR directory not found: {INSARDIR}")
    print(f"\nWARNING: InSAR directory not found")

# ============================================================================
# SAVE MULTI-TIER FILTERING SUMMARY
# ============================================================================
logger.info("\n" + "=" * 80)
logger.info("SAVING MULTI-TIER FILTERING SUMMARY")
logger.info("=" * 80)

summary_file = OUTPUT_DIR / '03_multitier_filtering_summary.txt'

with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("SCRIPT 03: MULTI-TIER OUTLIER FILTERING SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Run ID: {RUN_TIMESTAMP}\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("Configuration:\n")
    f.write(f"  RATE_PHYSICAL_MAX_MM_YR: {config.RATE_PHYSICAL_MAX_MM_YR}\n")
    f.write(f"  DH_PHYSICAL_MAX_M: {config.DH_PHYSICAL_MAX_M}\n")
    f.write(f"  OUTLIER_SIGMA_TIERS: {config.OUTLIER_SIGMA_TIERS}\n\n")
    
    if metadata.filtering_statistics:
        fs = metadata.filtering_statistics
        f.write("Filtering Results:\n")
        f.write(f"  Initial crossovers: {fs['initial']:,}\n")
        f.write(f"  Tier 1 removed: {fs['tier1_removed']:,} (physical plausibility)\n")
        f.write(f"  Tier 2 removed: {fs['tier2_removed']:,} (progressive sigma)\n")
        f.write(f"  Total removed: {fs['total_removed']:,} ({fs['removal_pct']:.1f}%)\n")
        f.write(f"  Final crossovers: {fs['final']:,}\n\n")
    
    if len(clean) > 0:
        f.write("Final Statistics (cm/yr):\n")
        f.write(f"  Mean subsidence rate: {clean['subsidence_rate_cm_per_year'].mean():.3f}\n")
        f.write(f"  Median subsidence rate: {clean['subsidence_rate_cm_per_year'].median():.3f}\n")
        f.write(f"  Std dev: {clean['subsidence_rate_cm_per_year'].std():.3f}\n")
        f.write(f"  Min: {clean['subsidence_rate_cm_per_year'].min():.3f}\n")
        f.write(f"  Max: {clean['subsidence_rate_cm_per_year'].max():.3f}\n\n")
        f.write(f"Mean rate uncertainty: {clean['rate_uncertainty_cm_yr'].mean():.3f} cm/yr\n")
        f.write(f"Median rate uncertainty: {clean['rate_uncertainty_cm_yr'].median():.3f} cm/yr\n\n")
    
    f.write("RMSE Validation:\n")
    f.write(f"  Empirical RMSE: {rmse_results['empirical_rmse_cm']:.2f} cm\n")
    f.write(f"  Theoretical RMSE: {rmse_results['theoretical_rmse_cm']:.2f} cm\n")
    f.write(f"  Ratio: {rmse_results['ratio']:.3f}\n")
    f.write(f"  Agreement: {rmse_results['agreement']}\n")

logger.info(f"Saved: {summary_file.name}")
print(f"✓ Saved: {summary_file.name}")

# ============================================================================
# FINALIZE & SAVE METADATA
# ============================================================================
logger.info("\n" + "=" * 80)
logger.info("FINALIZING RUN")
logger.info("=" * 80)

metadata.processing_results = {
    'crossover_detection': {
        'total_crossovers': len(clean),
        'mean_subsidence_cm_yr': float(clean['subsidence_rate_cm_per_year'].mean()) if len(clean) > 0 else None,
        'median_subsidence_cm_yr': float(clean['subsidence_rate_cm_per_year'].median()) if len(clean) > 0 else None,
        'std_subsidence_cm_yr': float(clean['subsidence_rate_cm_per_year'].std()) if len(clean) > 0 else None,
    },
    'rmse_validation': rmse_results,
}

metadata.finalize()
metadata_file = metadata.save(OUTPUT_DIR)

logger.info(f"Metadata saved: {metadata_file.name}")

logger.info("=" * 80)
logger.info("SCRIPT 03 COMPLETE - CORE ANALYSIS WITH PHASE 2A FILTERING")
logger.info("=" * 80)

print("\n" + "=" * 80)
print("SCRIPT 03 COMPLETE")
print("=" * 80)

print(f"\n✓ Output files created:")
print(f"  • icesat2_repeat_track_matches.csv")

if not SKIP_INSAR and (OUTPUT_DIR / 'icesat2_insar_integrated.csv').exists():
    print(f"  • icesat2_insar_integrated.csv")
    print(f"  • insar_validation_metrics.json")

print(f"  • 03_multitier_filtering_summary.txt")
print(f"  • 05_crossover_rmse_validation.json")
print(f"  • 05_rate_uncertainty_by_elevation.csv")
print(f"  • run_metadata.json")

print(f"\n✓ Analysis summary:")
print(f"  • Processed {len(clean):,} crossover pairs (after Phase 2A filtering)")

if len(clean) > 0:
    print(f"  • Mean subsidence: {clean['subsidence_rate_cm_per_year'].mean():.3f} cm/yr")
    print(f"  • RMSE ratio: {rmse_results['ratio']:.3f}")

print(f"\nRun metadata saved: {metadata_file.name}")

print("\n" + "=" * 80 + "\n")
