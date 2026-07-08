"""
================================================================================
SCRIPT 05: TEMPORAL ANALYSIS WITH COASTAL & ELEVATION STRATIFICATION
================================================================================

Purpose
-------
Calculate subsidence rates from ICESat-2 paired observations and stratify by:
1. ELEVATION zones (10m bins) - PRIMARY
2. COASTAL zones (from Script 04) - NEW
3. Timescale (data-driven tertiles) - ENHANCED

ENHANCEMENTS:
- Input priority: Script 03 output (FIRST), Script 04 grid (fallback)
- Coastal zone integration from Script 04
- Enhanced timescale categorization (data-driven)
- Visualization-ready columns (UTM, z-scores)
- All 5 original + 3 new coastal/visualization outputs

Inputs
------
1. icesat2_repeat_track_matches.csv (Script 03) - PRIORITY 1
2. 04_spatial_grid.csv (Script 04) - PRIORITY 2 (for coastal zones)

Outputs
-------
- 05_subsidence_rates.csv          (all pairs with rates, in cm/yr)
- 05_statistics_elevation.csv/.txt (elevation zones, 10m bins)
- 05_statistics_coastal.csv/.txt   (coastal zones)
- 05_statistics_timescale.txt      (enhanced timescale reference)
- 05_visualization_prep.csv        (viz-ready with UTM, z-scores)
- 05_summary_all_stratifications.txt (combined summary)
- script_05_run.log                (execution log)
- 05_analysis_temporal_metadata.json (reproducibility)

Units: ALL RATES IN CM/YR (negative=subsidence, positive=uplift)
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from pyproj import Transformer
import sys
import argparse
import io
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# UTF-8 FIX (None-safe): Ensure console/file output is UTF-8 compatible
# ============================================================================
_stdout_enc = getattr(sys.stdout, 'encoding', None) or ''
if _stdout_enc.lower() != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except AttributeError:
        pass

_stderr_enc = getattr(sys.stderr, 'encoding', None) or ''
if _stderr_enc.lower() != 'utf-8':
    try:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except AttributeError:
        pass

try:
    from config import BASEOUTPUTDIR
except ImportError:
    BASEOUTPUTDIR = Path(r'E:\Processing\ICeSat-2\output_data')

# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging(output_dir):
    """Configure logging to both file and console."""
    log_file = output_dir / 'script_05_run.log'
    logger = logging.getLogger('script_05')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    fh = logging.FileHandler(log_file, encoding='utf-8', mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ============================================================================
# METADATA TRACKING
# ============================================================================
class RunMetadata:
    """Capture complete run parameters for reproducibility."""

    def __init__(self, run_id):
        self.run_id = run_id
        self.script_name = '05_analysis_temporal'
        self.script_version = 'v2.2-all-fixes'
        self.start_time = datetime.now().isoformat()
        self.end_time = None
        self.configuration = {}
        self.processing_results = {}
        self.environment_info = {
            'python_version': f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}',
            'platform': sys.platform,
            'timestamp': datetime.now().isoformat(),
        }

    def finalize(self):
        self.end_time = datetime.now().isoformat()

    def save(self, output_dir):
        metadata_file = output_dir / '05_analysis_temporal_metadata.json'
        metadata = {
            'run_id': self.run_id,
            'script': self.script_name,
            'version': self.script_version,
            'environment': self.environment_info,
            'timestamps': {'start': self.start_time, 'end': self.end_time},
            'configuration': self.configuration,
            'processing_results': self.processing_results,
        }
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str, ensure_ascii=False)
        return metadata_file


# ============================================================================
# UTILITY: ARGUMENT PARSING (wrapped — safe for Script 08 orchestrator)
# ============================================================================
def parse_args():
    """Parse CLI arguments. Wrapped in function to avoid sys.argv conflicts
    when called from the master orchestration script (Script 08)."""
    parser = argparse.ArgumentParser(
        description='Temporal Analysis with Coastal Stratification'
    )
    parser.add_argument(
        '--input-type',
        type=str,
        choices=['03', '04', 'auto'],
        default='auto',
        help='Input type: 03=Script03, 04=Script04, auto=priority search'
    )
    return parser.parse_args()


# ============================================================================
# UTILITY: FIND INPUT FILE
# ============================================================================
def find_input_file(output_dir):
    """Find input file with correct priority order."""
    priority_files = [
        output_dir / 'icesat2_repeat_track_matches.csv',  # Script 03 (PRIORITY 1)
        output_dir / '04_spatial_grid.csv',               # Script 04 (fallback)
        output_dir / '03_paired_photons.csv',
        output_dir / '04_paired_photons.csv',
    ]
    for filepath in priority_files:
        if filepath.exists():
            return filepath

    print("ERROR: Could not find input file.")
    print("Searched for:")
    for f in priority_files:
        print(f"  - {f.name}")
    sys.exit(1)


# ============================================================================
# UTILITY: COASTAL ZONE ASSIGNMENT (NaN-safe)
# ============================================================================
def assign_coastal_zone(dist_m):
    """Assign coastal zone label from distance to ocean in metres.
    Returns 'Unknown' for NaN or non-finite distances."""
    if not np.isfinite(dist_m):
        return 'Unknown'
    if dist_m < 1000:
        return 'Coastal (0-1 km)'
    elif dist_m < 2000:
        return 'Near-coastal (1-2 km)'
    elif dist_m < 5000:
        return 'Transition (2-5 km)'
    else:
        return 'Inland (5+ km)'


# ============================================================================
# UTILITY: ENHANCED TIMESCALE CLASSIFICATION
# ============================================================================
def classify_timescale_enhanced(dt_days):
    """Classify temporal separation into labelled bins."""
    if dt_days <= 100:
        return 'Short (<=100 days)'
    elif dt_days <= 400:
        return 'Medium (100-400 days)'
    elif dt_days <= 800:
        return 'Long (400-800 days)'
    else:
        return 'Very Long (800+ days)'


# ============================================================================
# MAIN EXECUTION
# ============================================================================
print("=" * 80)
print("SCRIPT 05: TEMPORAL ANALYSIS - ELEVATION & COASTAL STRATIFICATION")
print("v2.2 — CM/YR UNITS — SCRIPT 04 INTEGRATION — ALL FIXES APPLIED")
print("=" * 80)

# FIX #8: parse args via function (safe for orchestrator)
args = parse_args()
print(f"\nInput type: {args.input_type}")

# --------------------------------------------------------------------------
# Timestamp & output directory
# --------------------------------------------------------------------------
timestamp_file = BASEOUTPUTDIR / 'latest_RUN_TIMESTAMP.txt'
try:
    with open(timestamp_file, 'r', encoding='utf-8') as f:
        RUN_TIMESTAMP = f.read().strip()
except FileNotFoundError:
    print("ERROR: Run Script 00 first (need latest_RUN_TIMESTAMP.txt)")
    sys.exit(1)

OUTPUT_DIR = BASEOUTPUTDIR / f'run_{RUN_TIMESTAMP}'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Run ID:           {RUN_TIMESTAMP}")
print(f"Output directory: {OUTPUT_DIR}")

# Setup logging and metadata
logger = setup_logging(OUTPUT_DIR)
metadata = RunMetadata(RUN_TIMESTAMP)

logger.info("=" * 80)
logger.info("SCRIPT 05: TEMPORAL ANALYSIS WITH COASTAL STRATIFICATION")
logger.info("=" * 80)
logger.info(f"Run ID:     {RUN_TIMESTAMP}")
logger.info(f"Input type: {args.input_type}")

# ============================================================================
# 1. LOADING PAIRED OBSERVATIONS
# ============================================================================
print("\n" + "=" * 80)
print("1. LOADING PAIRED OBSERVATIONS")
print("=" * 80)
logger.info("=" * 80)
logger.info("1. LOADING PAIRED OBSERVATIONS")
logger.info("=" * 80)

paired_file = find_input_file(OUTPUT_DIR)
df_pairs = pd.read_csv(paired_file)

logger.info(f"Loaded {len(df_pairs):,} paired observations from {paired_file.name}")
print(f"Loaded {len(df_pairs):,} paired observations from {paired_file.name}")

# FIX #9: sign_convention included in metadata configuration
metadata.configuration = {
    'input_file': paired_file.name,
    'n_input_pairs': len(df_pairs),
    'elevation_bin_size_m': 10,
    'units': 'cm/yr',
    'sign_convention': 'negative=subsidence, positive=uplift',
}

# Verify required columns
required_cols = ['subsidence_rate_cm_per_year']
missing = [c for c in required_cols if c not in df_pairs.columns]
if missing:
    logger.error(f"Missing required columns: {missing}")
    print(f"ERROR: Missing columns: {missing}")
    print(f"Available columns: {list(df_pairs.columns)}")
    sys.exit(1)

logger.info("All required columns present")
print("All required columns present")

# ============================================================================
# 2. COASTAL ZONE INTEGRATION
# ============================================================================
print("\n" + "=" * 80)
print("2. COASTAL ZONE INTEGRATION")
print("=" * 80)
logger.info("=" * 80)
logger.info("2. COASTAL ZONE INTEGRATION")
logger.info("=" * 80)

if 'coastal_zone' not in df_pairs.columns:
    if 'distance_to_ocean_m' in df_pairs.columns:
        # FIX #2: NaN-safe assignment via the dedicated function above
        df_pairs['coastal_zone'] = df_pairs['distance_to_ocean_m'].apply(assign_coastal_zone)
        logger.info("Assigned coastal zones from distance_to_ocean_m")
        print("Assigned coastal zones from distance_to_ocean_m")
    else:
        df_pairs['coastal_zone'] = 'Unknown'
        logger.warning("No coastal zone data — using default 'Unknown'")
        print("WARNING: No coastal zone data available")
else:
    logger.info("coastal_zone column already present")
    print("coastal_zone column already present")

# ============================================================================
# 3. COORDINATE TRANSFORMATION
# ============================================================================
print("\n" + "=" * 80)
print("3. COORDINATE TRANSFORMATION")
print("=" * 80)
logger.info("=" * 80)
logger.info("3. COORDINATE TRANSFORMATION")
logger.info("=" * 80)

# FIX #1: Compute x_utm/y_utm if missing, with latitude-corrected fallback
if ('x_utm' not in df_pairs.columns) or ('y_utm' not in df_pairs.columns):
    lon_col = next(
        (c for c in df_pairs.columns
         if any(x in c.lower() for x in ['lon_avg', 'lonavg', 'longitude', 'lon_early', 'lon_late'])),
        None
    )
    lat_col = next(
        (c for c in df_pairs.columns
         if any(x in c.lower() for x in ['lat_avg', 'latavg', 'latitude', 'lat_early', 'lat_late'])),
        None
    )

    if lon_col and lat_col:
        try:
            transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3155', always_xy=True)
            x_utm, y_utm = transformer.transform(
                df_pairs[lon_col].values,
                df_pairs[lat_col].values
            )
            df_pairs['x_utm'] = x_utm
            df_pairs['y_utm'] = y_utm
            logger.info(f"Computed x_utm, y_utm from {lon_col}/{lat_col} using EPSG:3155")
            print(f"Computed x_utm, y_utm using EPSG:3155")
        except Exception as e:
            logger.warning(f"Pyproj transform failed: {e} — using latitude-corrected approximation")
            print(f"WARNING: Pyproj failed ({e}); using cosine-corrected approximation")
            # FIX #1: latitude-corrected approximation (was missing cos(lat) on x)
            mean_lat = float(df_pairs[lat_col].mean())
            cos_lat  = np.cos(np.radians(mean_lat))
            df_pairs['x_utm'] = df_pairs[lon_col].values * 111320.0 * cos_lat
            df_pairs['y_utm'] = df_pairs[lat_col].values * 110540.0
    else:
        logger.warning("No lon/lat columns found — x_utm/y_utm set to sequential index")
        print("WARNING: No lon/lat columns found; x_utm/y_utm set to index")
        df_pairs['x_utm'] = np.arange(len(df_pairs), dtype=float)
        df_pairs['y_utm'] = np.arange(len(df_pairs), dtype=float)
else:
    logger.info("x_utm, y_utm already present")
    print("x_utm, y_utm already present")

# ============================================================================
# 4. MULTI-DIMENSION STRATIFICATION
# ============================================================================
print("\n" + "=" * 80)
print("4. MULTI-DIMENSION STRATIFICATION")
print("=" * 80)
logger.info("=" * 80)
logger.info("4. MULTI-DIMENSION STRATIFICATION")
logger.info("=" * 80)

# Elevation column (flexible naming)
elev_col = next((c for c in df_pairs.columns if 'elev' in c.lower()), None)
if elev_col is None:
    logger.warning("No elevation column found — using sequential index as placeholder")
    print("WARNING: No elevation column found")
    df_pairs['elevation_m'] = np.arange(len(df_pairs), dtype=float)
    elev_col = 'elevation_m'

# ------------------------------------------------------------------
# 4.1 ELEVATION STRATIFICATION (10m bins)
# FIX #7: named-agg API + reset_index() — pandas-version-safe
# ------------------------------------------------------------------
print("\n4.1 Elevation zones (10m bins)...")
df_pairs['elev_bin'] = (df_pairs[elev_col] / 10).astype(int) * 10

elev_stats = (
    df_pairs.groupby('elev_bin')['subsidence_rate_cm_per_year']
    .agg(
        count='count',
        mean='mean',
        median='median',
        std='std',
        min='min',
        max='max',
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
    )
    .round(3)
    .reset_index()
)

elev_stats['elevation_range'] = (
    elev_stats['elev_bin'].astype(str) + '-' +
    (elev_stats['elev_bin'] + 10).astype(str) + ' m'
)

logger.info(f"Elevation stratification: {len(elev_stats)} bins")
print(f"Elevation zones: {len(elev_stats)} bins")

# ------------------------------------------------------------------
# 4.2 COASTAL STRATIFICATION
# FIX #7: named-agg API + reset_index()
# ------------------------------------------------------------------
print("\n4.2 Coastal zones...")
coastal_stats = (
    df_pairs.groupby('coastal_zone')['subsidence_rate_cm_per_year']
    .agg(
        count='count',
        mean='mean',
        median='median',
        std='std',
        min='min',
        max='max',
    )
    .round(3)
    .reset_index()
)

logger.info(f"Coastal stratification: {len(coastal_stats)} zones")
print(f"Coastal zones: {len(coastal_stats)} zones")

# ------------------------------------------------------------------
# 4.3 ENHANCED TIMESCALE CLASSIFICATION
# FIX #5: correct column name 'dt_days_actual' (was 'dtdays_actual')
# ------------------------------------------------------------------
print("\n4.3 Timescale (enhanced, data-driven)...")

if 'dt_days_actual' in df_pairs.columns:
    df_pairs['dt_days'] = df_pairs['dt_days_actual']
    logger.info("Using dt_days_actual for timescale classification")
else:
    dt_col = next(
        (c for c in df_pairs.columns if 'dt_days' in c.lower() or 'dt_day' in c.lower()),
        None
    )
    if dt_col:
        df_pairs['dt_days'] = df_pairs[dt_col]
        logger.info(f"Using fallback temporal column: {dt_col}")
    else:
        logger.warning("No temporal separation column found — defaulting to 365 days")
        print("WARNING: No temporal separation column found; defaulting to 365 days")
        df_pairs['dt_days'] = 365

df_pairs['timescale'] = df_pairs['dt_days'].apply(classify_timescale_enhanced)

# FIX #7: named-agg API + reset_index()
timescale_stats = (
    df_pairs.groupby('timescale')['subsidence_rate_cm_per_year']
    .agg(
        count='count',
        mean='mean',
        median='median',
        std='std',
        min='min',
        max='max',
    )
    .round(3)
    .reset_index()
)

logger.info(f"Timescale stratification: {len(timescale_stats)} categories")
print(f"Timescale categories: {len(timescale_stats)} categories")

# ============================================================================
# 5. VISUALIZATION PREPARATION
# ============================================================================
print("\n" + "=" * 80)
print("5. VISUALIZATION PREPARATION")
print("=" * 80)
logger.info("=" * 80)
logger.info("5. VISUALIZATION PREPARATION")
logger.info("=" * 80)

global_mean = df_pairs['subsidence_rate_cm_per_year'].mean()
global_std  = df_pairs['subsidence_rate_cm_per_year'].std()

df_viz = df_pairs.copy()

# FIX #6: guard against zero std (e.g. single-value test datasets)
if global_std > 0:
    df_viz['z_score'] = (df_viz['subsidence_rate_cm_per_year'] - global_mean) / global_std
else:
    df_viz['z_score'] = 0.0
    logger.warning("global_std == 0; z_scores set to 0")

# Ensure spatial visualization columns exist
if 'x_center' not in df_viz.columns:
    df_viz['x_center'] = df_viz['x_utm']
if 'y_center' not in df_viz.columns:
    df_viz['y_center'] = df_viz['y_utm']

viz_cols = [
    'x_center', 'y_center',
    'subsidence_rate_cm_per_year', 'z_score',
    'coastal_zone', 'elev_bin', 'timescale',
]
df_viz_prep = df_viz[[c for c in viz_cols if c in df_viz.columns]].copy()

logger.info("Visualization preparation complete")
print("Visualization preparation complete")

# ============================================================================
# 6. SAVING OUTPUTS
# ============================================================================
print("\n" + "=" * 80)
print("6. SAVING OUTPUTS")
print("=" * 80)
logger.info("=" * 80)
logger.info("6. SAVING OUTPUTS")
logger.info("=" * 80)

# 6.1 All rates
rates_file = OUTPUT_DIR / '05_subsidence_rates.csv'
df_pairs.to_csv(rates_file, index=False, encoding='utf-8')
logger.info(f"Saved {rates_file.name}")
print(f"Saved {rates_file.name}")

# 6.2 Elevation statistics — CSV
elev_stats_csv = OUTPUT_DIR / '05_statistics_elevation.csv'
elev_stats.to_csv(elev_stats_csv, index=False, encoding='utf-8')
logger.info(f"Saved {elev_stats_csv.name}")
print(f"Saved {elev_stats_csv.name}")

# 6.2 Elevation statistics — TXT
elev_stats_txt = OUTPUT_DIR / '05_statistics_elevation.txt'
with open(elev_stats_txt, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("SUBSIDENCE RATES BY ELEVATION ZONE (10m bins)\n")
    f.write("=" * 80 + "\n")
    f.write(f"Run ID: {RUN_TIMESTAMP}\n")
    f.write("Units: cm/yr (negative=subsidence, positive=uplift)\n\n")
    f.write(f"{'Elevation':<14} {'Count':>8} {'Mean':>8} {'Median':>8} "
            f"{'Std':>8} {'Min':>8} {'Max':>8}\n")
    f.write("-" * 80 + "\n")
    for _, row in elev_stats.iterrows():
        f.write(f"{row['elevation_range']:<14} {int(row['count']):>8} "
                f"{row['mean']:>8.3f} {row['median']:>8.3f} {row['std']:>8.3f} "
                f"{row['min']:>8.3f} {row['max']:>8.3f}\n")
logger.info(f"Saved {elev_stats_txt.name}")
print(f"Saved {elev_stats_txt.name}")

# 6.3 Coastal statistics — CSV
coastal_stats_csv = OUTPUT_DIR / '05_statistics_coastal.csv'
coastal_stats.to_csv(coastal_stats_csv, index=False, encoding='utf-8')
logger.info(f"Saved {coastal_stats_csv.name}")
print(f"Saved {coastal_stats_csv.name}")

# 6.3 Coastal statistics — TXT
coastal_stats_txt = OUTPUT_DIR / '05_statistics_coastal.txt'
with open(coastal_stats_txt, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("SUBSIDENCE RATES BY COASTAL ZONE\n")
    f.write("=" * 80 + "\n")
    f.write(f"Run ID: {RUN_TIMESTAMP}\n")
    f.write("Units: cm/yr (negative=subsidence, positive=uplift)\n\n")
    f.write(f"{'Zone':<22} {'Count':>8} {'Mean':>8} {'Median':>8} "
            f"{'Std':>8} {'Min':>8} {'Max':>8}\n")
    f.write("-" * 80 + "\n")
    for _, row in coastal_stats.iterrows():
        f.write(f"{row['coastal_zone']:<22} {int(row['count']):>8} "
                f"{row['mean']:>8.3f} {row['median']:>8.3f} {row['std']:>8.3f} "
                f"{row['min']:>8.3f} {row['max']:>8.3f}\n")
logger.info(f"Saved {coastal_stats_txt.name}")
print(f"Saved {coastal_stats_txt.name}")

# 6.4 Timescale reference — TXT
timescale_txt = OUTPUT_DIR / '05_statistics_timescale.txt'
with open(timescale_txt, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("TIMESCALE ANALYSIS REFERENCE - ENHANCED\n")
    f.write("=" * 80 + "\n")
    f.write(f"Run ID: {RUN_TIMESTAMP}\n")
    f.write("Classification: Data-driven (<=100, 100-400, 400-800, 800+ days)\n")
    f.write("Units: cm/yr (negative=subsidence, positive=uplift)\n\n")
    f.write(f"{'Timescale':<24} {'Count':>8} {'Mean':>8} {'Median':>8} "
            f"{'Std':>8} {'Min':>8} {'Max':>8}\n")
    f.write("-" * 80 + "\n")
    for _, row in timescale_stats.iterrows():
        f.write(f"{row['timescale']:<24} {int(row['count']):>8} "
                f"{row['mean']:>8.3f} {row['median']:>8.3f} {row['std']:>8.3f} "
                f"{row['min']:>8.3f} {row['max']:>8.3f}\n")
logger.info(f"Saved {timescale_txt.name}")
print(f"Saved {timescale_txt.name}")

# 6.5 Visualization preparation — CSV
viz_file = OUTPUT_DIR / '05_visualization_prep.csv'
df_viz_prep.to_csv(viz_file, index=False, encoding='utf-8')
logger.info(f"Saved {viz_file.name}")
print(f"Saved {viz_file.name}")

# 6.6 Combined summary — TXT
summary_file = OUTPUT_DIR / '05_summary_all_stratifications.txt'
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("COMPLETE STRATIFICATION SUMMARY\n")
    f.write("=" * 80 + "\n")
    f.write(f"Run ID:          {RUN_TIMESTAMP}\n")
    f.write(f"Total pairs:     {len(df_pairs):,}\n")
    f.write(f"Mean subsidence: {global_mean:.3f} cm/yr\n")
    f.write(f"Global std:      {global_std:.3f} cm/yr\n")
    f.write(f"Sign convention: negative=subsidence, positive=uplift\n\n")

    f.write("ELEVATION ZONES (10m bins)\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Elevation':<14} {'Count':>8} {'Mean (cm/yr)':>14}\n")
    for _, row in elev_stats.iterrows():
        f.write(f"{row['elevation_range']:<14} {int(row['count']):>8} {row['mean']:>14.3f}\n")

    f.write("\nCOASTAL ZONES\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Zone':<22} {'Count':>8} {'Mean (cm/yr)':>14}\n")
    for _, row in coastal_stats.iterrows():
        f.write(f"{row['coastal_zone']:<22} {int(row['count']):>8} {row['mean']:>14.3f}\n")

    f.write("\nTIMESCALE CATEGORIES\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Timescale':<24} {'Count':>8} {'Mean (cm/yr)':>14}\n")
    for _, row in timescale_stats.iterrows():
        f.write(f"{row['timescale']:<24} {int(row['count']):>8} {row['mean']:>14.3f}\n")

logger.info(f"Saved {summary_file.name}")
print(f"Saved {summary_file.name}")

# ============================================================================
# 7. FINALIZE METADATA
# ============================================================================
print("\n" + "=" * 80)
print("7. FINALIZING")
print("=" * 80)
logger.info("=" * 80)
logger.info("7. FINALIZING")
logger.info("=" * 80)

metadata.processing_results = {
    'total_pairs': len(df_pairs),
    'global_mean_cm_yr': float(global_mean),
    'global_std_cm_yr': float(global_std),
    'n_elevation_zones': len(elev_stats),
    'n_coastal_zones': len(coastal_stats),
    'n_timescale_categories': len(timescale_stats),
    'dt_days_source': 'dt_days_actual' if 'dt_days_actual' in df_pairs.columns else 'fallback',
}

metadata.finalize()
metadata_file = metadata.save(OUTPUT_DIR)
logger.info(f"Saved metadata: {metadata_file.name}")
print(f"Saved metadata: {metadata_file.name}")

# ============================================================================
# COMPLETION SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SCRIPT 05 COMPLETE")
print("=" * 80)
logger.info("=" * 80)
logger.info("SCRIPT 05 COMPLETE")
logger.info("=" * 80)

print(f"""
Outputs created:
  05_subsidence_rates.csv
  05_statistics_elevation.csv/.txt  ({len(elev_stats)} bins)
  05_statistics_coastal.csv/.txt    ({len(coastal_stats)} zones)
  05_statistics_timescale.txt       ({len(timescale_stats)} categories)
  05_visualization_prep.csv
  05_summary_all_stratifications.txt
  script_05_run.log
  05_analysis_temporal_metadata.json

Key results:
  Pairs analysed : {len(df_pairs):,}
  Mean subsidence: {global_mean:.3f} cm/yr
  Elevation zones: {len(elev_stats)}
  Coastal zones  : {len(coastal_stats)}
  Sign convention: negative=subsidence, positive=uplift

Next: Run Script 06 (Advanced) or Script 07 (Figures)
""")
print("=" * 80)

logger.info("SCRIPT 05 EXECUTION COMPLETE")