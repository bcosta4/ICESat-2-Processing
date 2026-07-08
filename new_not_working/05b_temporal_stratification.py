"""
================================================================================
SCRIPT 05b: TEMPORAL STRATIFICATION & ACCELERATION METRICS
================================================================================

Purpose
-------
Advanced temporal stratification with acceleration detection:
1. Early vs Late period comparison (before/after 2022)
2. Acceleration metrics (trend analysis, rate changes)
3. Temporal acceleration summary statistics

CRITICAL FIXES (v2.1):
✅ FIX #1: Remove parse_dates=['datetime']; construct datetime from existing columns
✅ FIX #2: Flexible date column detection (date, date_early, datetime)
✅ FIX #3: UTF-8 output wrapper for console compatibility
✅ FIX #4: All rates in cm/yr (negative=subsidence, positive=uplift)

ENHANCEMENTS:
- Robust date column discovery (flexible naming)
- Early/Late period stratification (before/after 2022)
- Acceleration trend analysis
- Uncertainty propagation
- Complete reproducibility metadata

Inputs
------
- 05_subsidence_rates.csv (Script 05 output)
  OR
- icesat2_repeat_track_matches.csv (Script 03 output)

Outputs
-------
- icesat2_early_period.csv (2019-2022 subsidence rates)
- icesat2_late_period.csv (2022-2024 subsidence rates)
- temporal_acceleration_metrics.csv (trend analysis)
- 05b_temporal_stratification_report.txt (summary)
- script_05b_run.log (execution log)
- 05b_analysis_temporal_stratification_metadata.json

Units: ALL RATES IN CM/YR (negative=subsidence, positive=uplift)
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from scipy.stats import linregress
import sys
import io
import warnings

warnings.filterwarnings('ignore')

# UTF-8 FIX: Ensure console/file output is UTF-8 compatible
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

try:
    from config import BASEOUTPUTDIR
except ImportError:
    BASEOUTPUTDIR = Path(r'E:\Processing\ICeSat-2\output_data')

# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging(output_dir):
    """Configure logging to both file and console."""
    log_file = output_dir / 'script_05b_run.log'
    logger = logging.getLogger('script_05b')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    # File handler
    fh = logging.FileHandler(log_file, encoding='utf-8', mode='w')
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(fh_formatter)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    
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
        self.script_name = '05b_analysis_temporal_stratification'
        self.script_version = 'v2.1-fixes-datetime'
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
        """Mark run as complete."""
        self.end_time = datetime.now().isoformat()
    
    def save(self, output_dir):
        """Save metadata as JSON."""
        metadata_file = output_dir / '05b_analysis_temporal_stratification_metadata.json'
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
# UTILITY: FIND INPUT FILE
# ============================================================================
def find_input_file(output_dir):
    """Find input file with priority order."""
    priority_files = [
        output_dir / '05_subsidence_rates.csv',             # Script 05 output (PRIORITY 1)
        output_dir / 'icesat2_repeat_track_matches.csv',    # Script 03 output (fallback)
        output_dir / '04_spatial_grid.csv',                 # Script 04 (fallback)
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
# UTILITY: FLEXIBLE DATE COLUMN DETECTION
# ============================================================================
def extract_datetime_from_df(df, logger=None):
    """
    FIX #1: Flexibly detect and extract datetime from CSV columns.
    Tries: 'datetime', 'date', 'date_early', 'date_late', 'year'
    """
    
    # Priority order for datetime detection
    datetime_col_candidates = ['datetime', 'date', 'date_early', 'date_late']
    
    for col in datetime_col_candidates:
        if col in df.columns:
            try:
                dt = pd.to_datetime(df[col])
                if logger:
                    logger.info(f"Extracted datetime from column: '{col}'")
                print(f"Extracted datetime from column: '{col}'")
                return dt
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to parse '{col}' as datetime: {e}")
                continue
    
    # Fallback: try year column
    if 'year' in df.columns:
        try:
            dt = pd.to_datetime(df['year'], format='%Y')
            if logger:
                logger.warning("Using 'year' column as datetime (year-only, no month/day)")
            print("WARNING: Using 'year' column as datetime (year-only)")
            return dt
        except:
            pass
    
    # If all else fails, raise error
    if logger:
        logger.error(f"Could not find datetime column. Available columns: {list(df.columns)}")
    print(f"ERROR: Could not find datetime column. Available columns: {list(df.columns)}")
    sys.exit(1)

# ============================================================================
# MAIN EXECUTION
# ============================================================================
print("=" * 80)
print("SCRIPT 05b: TEMPORAL STRATIFICATION & ACCELERATION METRICS")
print("WITH EARLY/LATE PERIOD COMPARISON")
print("=" * 80)

# Get timestamp
timestamp_file = BASEOUTPUTDIR / 'latest_RUN_TIMESTAMP.txt'
try:
    with open(timestamp_file, 'r', encoding='utf-8') as f:
        RUN_TIMESTAMP = f.read().strip()
except FileNotFoundError:
    print("ERROR: Run Script 00 first (need latestRUNTIMESTAMP.txt)")
    sys.exit(1)

OUTPUT_DIR = BASEOUTPUTDIR / f'run_{RUN_TIMESTAMP}'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Run ID: {RUN_TIMESTAMP}")
print(f"Output directory: {OUTPUT_DIR}")

# Setup logging
logger = setup_logging(OUTPUT_DIR)
metadata = RunMetadata(RUN_TIMESTAMP)

logger.info("=" * 80)
logger.info("SCRIPT 05b: TEMPORAL STRATIFICATION & ACCELERATION METRICS")
logger.info("=" * 80)
logger.info(f"Run ID: {RUN_TIMESTAMP}")

# ============================================================================
# 1. LOAD DATA
# ============================================================================
logger.info("\n" + "=" * 80)
logger.info("1. LOADING DATA")
logger.info("=" * 80)

print("\n" + "=" * 80)
print("1. LOADING DATA")
print("=" * 80)

input_file = find_input_file(OUTPUT_DIR)
df = pd.read_csv(input_file)

logger.info(f"Loaded {len(df):,} observations from {input_file.name}")
print(f"Loaded {len(df):,} observations from {input_file.name}")

# Verify rate column
if 'subsidence_rate_cm_per_year' not in df.columns:
    logger.error("Missing 'subsidence_rate_cm_per_year' column")
    print("ERROR: Missing 'subsidence_rate_cm_per_year' column")
    sys.exit(1)

logger.info("Rate column verified: 'subsidence_rate_cm_per_year'")
print("Rate column verified")

metadata.configuration = {
    'input_file': input_file.name,
    'n_observations': len(df),
    'units': 'cm/yr',
    'early_period_year': 2022,
}

# ============================================================================
# 2. EXTRACT AND PARSE DATETIME (FIX #1)
# ============================================================================
logger.info("\n" + "=" * 80)
logger.info("2. EXTRACTING DATETIME")
logger.info("=" * 80)

print("\n" + "=" * 80)
print("2. EXTRACTING DATETIME")
print("=" * 80)

df['datetime'] = extract_datetime_from_df(df, logger)
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month

logger.info(f"Year range: {df['year'].min()} - {df['year'].max()}")
print(f"Year range: {df['year'].min()} - {df['year'].max()}")

# ============================================================================
# 3. EARLY/LATE PERIOD STRATIFICATION
# ============================================================================
logger.info("\n" + "=" * 80)
logger.info("3. EARLY/LATE PERIOD STRATIFICATION")
logger.info("=" * 80)

print("\n" + "=" * 80)
print("3. EARLY/LATE PERIOD STRATIFICATION")
print("=" * 80)

early_cutoff_year = 2022

# Early period: before cutoff
df_early = df[df['year'] < early_cutoff_year].copy()
logger.info(f"Early period (<{early_cutoff_year}): {len(df_early):,} observations")
print(f"Early period (<{early_cutoff_year}): {len(df_early):,} observations")

# Late period: from cutoff onwards
df_late = df[df['year'] >= early_cutoff_year].copy()
logger.info(f"Late period (>={early_cutoff_year}): {len(df_late):,} observations")
print(f"Late period (>={early_cutoff_year}): {len(df_late):,} observations")

# Compute statistics for each period
def compute_period_stats(df_period, period_name, logger=None):
    """Compute comprehensive statistics for a period."""
    rates = df_period['subsidence_rate_cm_per_year'].dropna().values
    
    if len(rates) == 0:
        if logger:
            logger.warning(f"No valid rates in {period_name}")
        return None
    
    stats = {
        'period': period_name,
        'n_observations': len(rates),
        'mean_subsidence_cm_yr': float(np.mean(rates)),
        'median_subsidence_cm_yr': float(np.median(rates)),
        'std_subsidence_cm_yr': float(np.std(rates)),
        'std_error_cm_yr': float(np.std(rates) / np.sqrt(len(rates))),
        'ci_lower_95_cm_yr': float(np.mean(rates) - 1.96 * np.std(rates) / np.sqrt(len(rates))),
        'ci_upper_95_cm_yr': float(np.mean(rates) + 1.96 * np.std(rates) / np.sqrt(len(rates))),
        'min_subsidence_cm_yr': float(np.min(rates)),
        'max_subsidence_cm_yr': float(np.max(rates)),
        'q25_subsidence_cm_yr': float(np.percentile(rates, 25)),
        'q75_subsidence_cm_yr': float(np.percentile(rates, 75)),
    }
    
    if logger:
        logger.info(f"{period_name}: mean={stats['mean_subsidence_cm_yr']:.3f} cm/yr, "
                   f"n={len(rates):,}")
    
    return stats

early_stats = compute_period_stats(df_early, 'Early (2019-2021)', logger)
late_stats = compute_period_stats(df_late, 'Late (2022-2024)', logger)

print(f"\nEarly period mean: {early_stats['mean_subsidence_cm_yr']:.3f} cm/yr")
print(f"Late period mean: {late_stats['mean_subsidence_cm_yr']:.3f} cm/yr")

# ============================================================================
# 4. SAVE EARLY/LATE PERIOD DATA
# ============================================================================
logger.info("\n" + "=" * 80)
logger.info("4. SAVING EARLY/LATE PERIOD DATA")
logger.info("=" * 80)

print("\n" + "=" * 80)
print("4. SAVING EARLY/LATE PERIOD DATA")
print("=" * 80)

early_file = OUTPUT_DIR / 'icesat2_early_period.csv'
df_early.to_csv(early_file, index=False, encoding='utf-8')
logger.info(f"Saved {early_file.name}")
print(f"Saved {early_file.name}")

late_file = OUTPUT_DIR / 'icesat2_late_period.csv'
df_late.to_csv(late_file, index=False, encoding='utf-8')
logger.info(f"Saved {late_file.name}")
print(f"Saved {late_file.name}")

# ============================================================================
# 5. ACCELERATION METRICS (TREND ANALYSIS)
# ============================================================================
logger.info("\n" + "=" * 80)
logger.info("5. ACCELERATION METRICS & TREND ANALYSIS")
logger.info("=" * 80)

print("\n" + "=" * 80)
print("5. ACCELERATION METRICS & TREND ANALYSIS")
print("=" * 80)

acceleration_metrics = []

# Metric 1: Period comparison
if early_stats and late_stats:
    acceleration = late_stats['mean_subsidence_cm_yr'] - early_stats['mean_subsidence_cm_yr']
    acceleration_metrics.append({
        'Metric': 'Period Acceleration (Late - Early)',
        'Early Period': early_stats['mean_subsidence_cm_yr'],
        'Late Period': late_stats['mean_subsidence_cm_yr'],
        'Change_cm_yr': acceleration,
        'Relative_Change_Percent': (acceleration / abs(early_stats['mean_subsidence_cm_yr']) * 100
                                   if early_stats['mean_subsidence_cm_yr'] != 0 else 0),
    })
    
    logger.info(f"Period acceleration: {acceleration:.3f} cm/yr")
    print(f"Period acceleration: {acceleration:.3f} cm/yr "
          f"({(acceleration/abs(early_stats['mean_subsidence_cm_yr'])*100 if early_stats['mean_subsidence_cm_yr'] != 0 else 0):.1f}%)")

# Metric 2: Temporal trend (linear regression on all data)
if len(df) > 1 and 'datetime' in df.columns:
    # Convert datetime to ordinal (days since epoch)
    df_with_trend = df.dropna(subset=['subsidence_rate_cm_per_year', 'datetime']).copy()
    
    if len(df_with_trend) > 1:
        x_days = (df_with_trend['datetime'] - df_with_trend['datetime'].min()).dt.total_seconds() / 86400.0
        y_rates = df_with_trend['subsidence_rate_cm_per_year'].values
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x_days, y_rates)
        
        # Convert slope to annual rate (365.25 days/year)
        annual_trend = slope * 365.25
        
        acceleration_metrics.append({
            'Metric': 'Temporal Trend (All Data)',
            'Early Period': early_stats['mean_subsidence_cm_yr'] if early_stats else np.nan,
            'Late Period': late_stats['mean_subsidence_cm_yr'] if late_stats else np.nan,
            'Change_cm_yr': annual_trend,
            'Relative_Change_Percent': (annual_trend / abs(early_stats['mean_subsidence_cm_yr']) * 100
                                       if early_stats and early_stats['mean_subsidence_cm_yr'] != 0 else np.nan),
        })
        
        logger.info(f"Annual trend: {annual_trend:.3f} cm/yr/yr (R²={r_value**2:.3f}, p={p_value:.2e})")
        print(f"Annual trend: {annual_trend:.3f} cm/yr/yr (R²={r_value**2:.3f})")

# Metric 3: Variability comparison
if early_stats and late_stats:
    acceleration_metrics.append({
        'Metric': 'Std Dev Change',
        'Early Period': early_stats['std_subsidence_cm_yr'],
        'Late Period': late_stats['std_subsidence_cm_yr'],
        'Change_cm_yr': late_stats['std_subsidence_cm_yr'] - early_stats['std_subsidence_cm_yr'],
        'Relative_Change_Percent': ((late_stats['std_subsidence_cm_yr'] - early_stats['std_subsidence_cm_yr']) /
                                   early_stats['std_subsidence_cm_yr'] * 100 if early_stats['std_subsidence_cm_yr'] != 0 else 0),
    })

# Save acceleration metrics
accel_file = OUTPUT_DIR / 'temporal_acceleration_metrics.csv'
df_accel = pd.DataFrame(acceleration_metrics)
df_accel.to_csv(accel_file, index=False, encoding='utf-8')
logger.info(f"Saved {accel_file.name}")
print(f"Saved {accel_file.name}")

# ============================================================================
# 6. GENERATE REPORT
# ============================================================================
logger.info("\n" + "=" * 80)
logger.info("6. GENERATING TEMPORAL STRATIFICATION REPORT")
logger.info("=" * 80)

print("\n" + "=" * 80)
print("6. GENERATING TEMPORAL STRATIFICATION REPORT")
print("=" * 80)

report_file = OUTPUT_DIR / '05b_temporal_stratification_report.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("SCRIPT 05b: TEMPORAL STRATIFICATION & ACCELERATION METRICS\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Run ID: {RUN_TIMESTAMP}\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Data period: {df['year'].min()}-{df['year'].max()}\n")
    f.write(f"Total observations: {len(df):,}\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("EARLY PERIOD (2019-2021)\n")
    f.write("=" * 80 + "\n")
    if early_stats:
        f.write(f"Observations: {early_stats['n_observations']:,}\n")
        f.write(f"Mean subsidence: {early_stats['mean_subsidence_cm_yr']:.3f} cm/yr\n")
        f.write(f"Median: {early_stats['median_subsidence_cm_yr']:.3f} cm/yr\n")
        f.write(f"Std dev: {early_stats['std_subsidence_cm_yr']:.3f} cm/yr\n")
        f.write(f"95% CI: [{early_stats['ci_lower_95_cm_yr']:.3f}, {early_stats['ci_upper_95_cm_yr']:.3f}] cm/yr\n")
        f.write(f"Range: {early_stats['min_subsidence_cm_yr']:.3f} to {early_stats['max_subsidence_cm_yr']:.3f} cm/yr\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("LATE PERIOD (2022-2024)\n")
    f.write("=" * 80 + "\n")
    if late_stats:
        f.write(f"Observations: {late_stats['n_observations']:,}\n")
        f.write(f"Mean subsidence: {late_stats['mean_subsidence_cm_yr']:.3f} cm/yr\n")
        f.write(f"Median: {late_stats['median_subsidence_cm_yr']:.3f} cm/yr\n")
        f.write(f"Std dev: {late_stats['std_subsidence_cm_yr']:.3f} cm/yr\n")
        f.write(f"95% CI: [{late_stats['ci_lower_95_cm_yr']:.3f}, {late_stats['ci_upper_95_cm_yr']:.3f}] cm/yr\n")
        f.write(f"Range: {late_stats['min_subsidence_cm_yr']:.3f} to {late_stats['max_subsidence_cm_yr']:.3f} cm/yr\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("ACCELERATION METRICS\n")
    f.write("=" * 80 + "\n")
    for _, row in df_accel.iterrows():
        f.write(f"\n{row['Metric']}\n")
        f.write(f"  Early: {row['Early Period']:.3f} cm/yr\n")
        f.write(f"  Late: {row['Late Period']:.3f} cm/yr\n")
        f.write(f"  Change: {row['Change_cm_yr']:.3f} cm/yr ({row['Relative_Change_Percent']:.1f}%)\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("INTERPRETATION\n")
    f.write("=" * 80 + "\n")
    if early_stats and late_stats:
        if late_stats['mean_subsidence_cm_yr'] < early_stats['mean_subsidence_cm_yr']:
            f.write("Subsidence is ACCELERATING (more negative in late period)\n")
        elif late_stats['mean_subsidence_cm_yr'] > early_stats['mean_subsidence_cm_yr']:
            f.write("Subsidence is DECELERATING (less negative in late period)\n")
        else:
            f.write("No significant change between periods\n")

logger.info(f"Saved {report_file.name}")
print(f"Saved {report_file.name}")

# ============================================================================
# 7. FINALIZE METADATA
# ============================================================================
logger.info("\n" + "=" * 80)
logger.info("7. FINALIZING")
logger.info("=" * 80)

metadata.processing_results = {
    'early_period_observations': len(df_early) if early_stats else 0,
    'late_period_observations': len(df_late) if late_stats else 0,
    'early_mean_cm_yr': early_stats['mean_subsidence_cm_yr'] if early_stats else None,
    'late_mean_cm_yr': late_stats['mean_subsidence_cm_yr'] if late_stats else None,
    'n_acceleration_metrics': len(df_accel),
}

metadata.finalize()
metadata_file = metadata.save(OUTPUT_DIR)
logger.info(f"Saved metadata: {metadata_file.name}")
print(f"Saved metadata: {metadata_file.name}")

# ============================================================================
# COMPLETION
# ============================================================================
logger.info("\n" + "=" * 80)
logger.info("SCRIPT 05b COMPLETE")
logger.info("=" * 80)

print("\n" + "=" * 80)
print("SCRIPT 05b COMPLETE")
print("=" * 80)

print(f"\nOutputs created:")
print(f" • icesat2_early_period.csv ({len(df_early):,} observations)")
print(f" • icesat2_late_period.csv ({len(df_late):,} observations)")
print(f" • temporal_acceleration_metrics.csv ({len(df_accel)} metrics)")
print(f" • 05b_temporal_stratification_report.txt")
print(f" • script_05b_run.log")
print(f" • 05b_analysis_temporal_stratification_metadata.json")

if early_stats and late_stats:
    accel = late_stats['mean_subsidence_cm_yr'] - early_stats['mean_subsidence_cm_yr']
    print(f"\nKey results:")
    print(f" • Early period: {early_stats['mean_subsidence_cm_yr']:.3f} cm/yr")
    print(f" • Late period: {late_stats['mean_subsidence_cm_yr']:.3f} cm/yr")
    print(f" • Acceleration: {accel:.3f} cm/yr")

print(f"\nNext: Run Script 06 (Advanced) or Script 07 (Figures)")
print("=" * 80 + "\n")

logger.info("SCRIPT 05b EXECUTION COMPLETE")
