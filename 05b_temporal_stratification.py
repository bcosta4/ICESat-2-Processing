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

Inputs
------
- 05_subsidence_rates.csv (Script 05 output)  — PRIORITY 1
- icesat2_repeat_track_matches.csv (Script 03) — fallback
- 04_spatial_grid.csv (Script 04)              — fallback

Outputs
-------
- icesat2_early_period.csv              (2019-2021 subsidence rates)
- icesat2_late_period.csv               (2022-2024 subsidence rates)
- temporal_acceleration_metrics.csv     (trend analysis)
- 05b_temporal_stratification_report.txt (summary)
- script_05b_run.log                    (execution log)
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

# ============================================================================
# FIX #3: UTF-8 output wrapper — None-safe encoding check
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
    log_file = output_dir / 'script_05b_run.log'
    logger = logging.getLogger('script_05b')
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
        self.script_name = '05b_analysis_temporal_stratification'
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
        output_dir / '05_subsidence_rates.csv',           # Script 05 (PRIORITY 1)
        output_dir / 'icesat2_repeat_track_matches.csv',  # Script 03 (fallback)
        output_dir / '04_spatial_grid.csv',               # Script 04 (fallback)
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
# FIX #1 + FIX #6: flexible detection; bare except replaced with explicit catch
# ============================================================================
def extract_datetime_from_df(df, logger=None):
    """
    Flexibly detect and extract datetime from CSV columns.
    Tries in order: 'datetime', 'date', 'date_early', 'date_late', 'year'.
    Never uses a bare except clause.
    """
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

    # Fallback: year column
    if 'year' in df.columns:
        try:
            dt = pd.to_datetime(df['year'], format='%Y')
            if logger:
                logger.warning("Using 'year' column as datetime (year-only, no month/day)")
            print("WARNING: Using 'year' column as datetime (year-only)")
            return dt
        except Exception as e:
            if logger:
                logger.warning(f"Failed to parse 'year' column as datetime: {e}")

    # All attempts failed
    if logger:
        logger.error(f"Could not find datetime column. Available: {list(df.columns)}")
    print(f"ERROR: Could not find datetime column. Available: {list(df.columns)}")
    sys.exit(1)


# ============================================================================
# UTILITY: PERIOD STATISTICS
# ============================================================================
def compute_period_stats(df_period, period_name, logger=None):
    """Compute comprehensive statistics for a time period.
    Returns None (with a warning) if no valid rates exist."""
    rates = df_period['subsidence_rate_cm_per_year'].dropna().values

    if len(rates) == 0:
        if logger:
            logger.warning(f"No valid rates in {period_name} — returning None")
        print(f"WARNING: No valid rates in {period_name}")
        return None

    n = len(rates)
    stderr = np.std(rates) / np.sqrt(n)
    stats = {
        'period':                  period_name,
        'n_observations':          n,
        'mean_subsidence_cm_yr':   float(np.mean(rates)),
        'median_subsidence_cm_yr': float(np.median(rates)),
        'std_subsidence_cm_yr':    float(np.std(rates)),
        'std_error_cm_yr':         float(stderr),
        'ci_lower_95_cm_yr':       float(np.mean(rates) - 1.96 * stderr),
        'ci_upper_95_cm_yr':       float(np.mean(rates) + 1.96 * stderr),
        'min_subsidence_cm_yr':    float(np.min(rates)),
        'max_subsidence_cm_yr':    float(np.max(rates)),
        'q25_subsidence_cm_yr':    float(np.percentile(rates, 25)),
        'q75_subsidence_cm_yr':    float(np.percentile(rates, 75)),
    }

    if logger:
        logger.info(f"{period_name}: mean={stats['mean_subsidence_cm_yr']:.3f} cm/yr, "
                    f"n={n:,}")
    return stats


# ============================================================================
# UTILITY: SAFE RELATIVE CHANGE
# ============================================================================
def safe_relative_change(change, reference):
    """Return relative change % or 0.0 if reference is zero."""
    if reference != 0:
        return (change / abs(reference)) * 100.0
    return 0.0


# ============================================================================
# MAIN EXECUTION
# ============================================================================
print("=" * 80)
print("SCRIPT 05b: TEMPORAL STRATIFICATION & ACCELERATION METRICS")
print("WITH EARLY/LATE PERIOD COMPARISON")
print("=" * 80)

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

logger   = setup_logging(OUTPUT_DIR)
metadata = RunMetadata(RUN_TIMESTAMP)

logger.info("=" * 80)
logger.info("SCRIPT 05b: TEMPORAL STRATIFICATION & ACCELERATION METRICS")
logger.info("=" * 80)
logger.info(f"Run ID: {RUN_TIMESTAMP}")

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n" + "=" * 80)
print("1. LOADING DATA")
print("=" * 80)
logger.info("=" * 80)
logger.info("1. LOADING DATA")
logger.info("=" * 80)

input_file = find_input_file(OUTPUT_DIR)
df = pd.read_csv(input_file)

logger.info(f"Loaded {len(df):,} observations from {input_file.name}")
print(f"Loaded {len(df):,} observations from {input_file.name}")

if 'subsidence_rate_cm_per_year' not in df.columns:
    logger.error("Missing 'subsidence_rate_cm_per_year' column")
    print("ERROR: Missing 'subsidence_rate_cm_per_year' column")
    sys.exit(1)

logger.info("Rate column verified: 'subsidence_rate_cm_per_year'")
print("Rate column verified")

# FIX #9: sign_convention in metadata
metadata.configuration = {
    'input_file':         input_file.name,
    'n_observations':     len(df),
    'units':              'cm/yr',
    'sign_convention':    'negative=subsidence, positive=uplift',
    'early_period_year':  2022,
}

# ============================================================================
# 2. EXTRACT AND PARSE DATETIME
# ============================================================================
print("\n" + "=" * 80)
print("2. EXTRACTING DATETIME")
print("=" * 80)
logger.info("=" * 80)
logger.info("2. EXTRACTING DATETIME")
logger.info("=" * 80)

df['datetime'] = extract_datetime_from_df(df, logger)
df['year']     = df['datetime'].dt.year
df['month']    = df['datetime'].dt.month

logger.info(f"Year range: {df['year'].min()} - {df['year'].max()}")
print(f"Year range: {df['year'].min()} - {df['year'].max()}")

# ============================================================================
# 3. EARLY/LATE PERIOD STRATIFICATION
# ============================================================================
print("\n" + "=" * 80)
print("3. EARLY/LATE PERIOD STRATIFICATION")
print("=" * 80)
logger.info("=" * 80)
logger.info("3. EARLY/LATE PERIOD STRATIFICATION")
logger.info("=" * 80)

EARLY_CUTOFF_YEAR = 2022

df_early = df[df['year'] < EARLY_CUTOFF_YEAR].copy()
df_late  = df[df['year'] >= EARLY_CUTOFF_YEAR].copy()

logger.info(f"Early period (<{EARLY_CUTOFF_YEAR}):  {len(df_early):,} observations")
logger.info(f"Late period  (>={EARLY_CUTOFF_YEAR}): {len(df_late):,} observations")
print(f"Early period (<{EARLY_CUTOFF_YEAR}):  {len(df_early):,} observations")
print(f"Late period  (>={EARLY_CUTOFF_YEAR}): {len(df_late):,} observations")

early_stats = compute_period_stats(df_early, 'Early (2019-2021)', logger)
late_stats  = compute_period_stats(df_late,  'Late (2022-2024)',  logger)

# FIX #5: None-guard every direct access
if early_stats:
    print(f"\nEarly period mean: {early_stats['mean_subsidence_cm_yr']:.3f} cm/yr")
else:
    print("\nEarly period: no valid observations")

if late_stats:
    print(f"Late period mean:  {late_stats['mean_subsidence_cm_yr']:.3f} cm/yr")
else:
    print("Late period: no valid observations")

# ============================================================================
# 4. SAVE EARLY/LATE PERIOD DATA
# ============================================================================
print("\n" + "=" * 80)
print("4. SAVING EARLY/LATE PERIOD DATA")
print("=" * 80)
logger.info("=" * 80)
logger.info("4. SAVING EARLY/LATE PERIOD DATA")
logger.info("=" * 80)

early_file = OUTPUT_DIR / 'icesat2_early_period.csv'
df_early.to_csv(early_file, index=False, encoding='utf-8')
logger.info(f"Saved {early_file.name}")
print(f"Saved {early_file.name}")

late_file = OUTPUT_DIR / 'icesat2_late_period.csv'
df_late.to_csv(late_file, index=False, encoding='utf-8')
logger.info(f"Saved {late_file.name}")
print(f"Saved {late_file.name}")

# ============================================================================
# 5. ACCELERATION METRICS
# ============================================================================
print("\n" + "=" * 80)
print("5. ACCELERATION METRICS & TREND ANALYSIS")
print("=" * 80)
logger.info("=" * 80)
logger.info("5. ACCELERATION METRICS & TREND ANALYSIS")
logger.info("=" * 80)

acceleration_metrics = []

# Metric 1: Period mean comparison
if early_stats and late_stats:
    acceleration = (late_stats['mean_subsidence_cm_yr']
                    - early_stats['mean_subsidence_cm_yr'])
    rel_change   = safe_relative_change(acceleration,
                                        early_stats['mean_subsidence_cm_yr'])
    acceleration_metrics.append({
        'Metric':                  'Period Acceleration (Late - Early)',
        'Early_Period':            early_stats['mean_subsidence_cm_yr'],
        'Late_Period':             late_stats['mean_subsidence_cm_yr'],
        'Change_cm_yr':            acceleration,
        'Relative_Change_Percent': rel_change,
    })
    logger.info(f"Period acceleration: {acceleration:.3f} cm/yr ({rel_change:.1f}%)")
    print(f"Period acceleration: {acceleration:.3f} cm/yr ({rel_change:.1f}%)")

# Metric 2: Linear temporal trend on all data
# FIX #7: np.isfinite() mask applied before linregress
df_trend = df.dropna(subset=['subsidence_rate_cm_per_year', 'datetime']).copy()

if len(df_trend) > 1:
    x_days = ((df_trend['datetime'] - df_trend['datetime'].min())
               .dt.total_seconds() / 86400.0).values
    y_rates = df_trend['subsidence_rate_cm_per_year'].values

    finite_mask = np.isfinite(x_days) & np.isfinite(y_rates)
    x_clean     = x_days[finite_mask]
    y_clean     = y_rates[finite_mask]

    if len(x_clean) > 1 and np.std(x_clean) > 0:
        slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)
        annual_trend = slope * 365.25   # cm/yr per year

        rel_trend = (safe_relative_change(annual_trend,
                                          early_stats['mean_subsidence_cm_yr'])
                     if early_stats else np.nan)

        acceleration_metrics.append({
            'Metric':                  'Temporal Trend (All Data)',
            'Early_Period':            early_stats['mean_subsidence_cm_yr'] if early_stats else np.nan,
            'Late_Period':             late_stats['mean_subsidence_cm_yr']  if late_stats  else np.nan,
            'Change_cm_yr':            annual_trend,
            'Relative_Change_Percent': rel_trend,
        })
        logger.info(f"Annual trend: {annual_trend:.4f} cm/yr/yr "
                    f"(R²={r_value**2:.3f}, p={p_value:.2e})")
        print(f"Annual trend: {annual_trend:.4f} cm/yr/yr "
              f"(R²={r_value**2:.3f}, p={p_value:.2e})")
    else:
        logger.warning("Insufficient finite data points for temporal trend regression")
        print("WARNING: Insufficient finite data for temporal trend regression")

# Metric 3: Variability (std dev) comparison
if early_stats and late_stats:
    std_change  = (late_stats['std_subsidence_cm_yr']
                   - early_stats['std_subsidence_cm_yr'])
    rel_std     = safe_relative_change(std_change,
                                       early_stats['std_subsidence_cm_yr'])
    acceleration_metrics.append({
        'Metric':                  'Std Dev Change',
        'Early_Period':            early_stats['std_subsidence_cm_yr'],
        'Late_Period':             late_stats['std_subsidence_cm_yr'],
        'Change_cm_yr':            std_change,
        'Relative_Change_Percent': rel_std,
    })

# Save acceleration metrics
accel_file = OUTPUT_DIR / 'temporal_acceleration_metrics.csv'
df_accel   = pd.DataFrame(acceleration_metrics)
df_accel.to_csv(accel_file, index=False, encoding='utf-8')
logger.info(f"Saved {accel_file.name} ({len(df_accel)} metrics)")
print(f"Saved {accel_file.name} ({len(df_accel)} metrics)")

# ============================================================================
# 6. GENERATE REPORT
# ============================================================================
print("\n" + "=" * 80)
print("6. GENERATING TEMPORAL STRATIFICATION REPORT")
print("=" * 80)
logger.info("=" * 80)
logger.info("6. GENERATING TEMPORAL STRATIFICATION REPORT")
logger.info("=" * 80)

report_file = OUTPUT_DIR / '05b_temporal_stratification_report.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("SCRIPT 05b: TEMPORAL STRATIFICATION & ACCELERATION METRICS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Run ID:             {RUN_TIMESTAMP}\n")
    f.write(f"Generated:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Data period:        {df['year'].min()}-{df['year'].max()}\n")
    f.write(f"Total observations: {len(df):,}\n")
    f.write(f"Sign convention:    negative=subsidence, positive=uplift\n\n")

    # Early period
    f.write("=" * 80 + "\n")
    f.write("EARLY PERIOD (2019-2021)\n")
    f.write("=" * 80 + "\n")
    if early_stats:
        f.write(f"Observations: {early_stats['n_observations']:,}\n")
        f.write(f"Mean subsidence:  {early_stats['mean_subsidence_cm_yr']:.3f} cm/yr\n")
        f.write(f"Median:           {early_stats['median_subsidence_cm_yr']:.3f} cm/yr\n")
        f.write(f"Std dev:          {early_stats['std_subsidence_cm_yr']:.3f} cm/yr\n")
        f.write(f"95% CI:           [{early_stats['ci_lower_95_cm_yr']:.3f}, "
                f"{early_stats['ci_upper_95_cm_yr']:.3f}] cm/yr\n")
        f.write(f"Range:            {early_stats['min_subsidence_cm_yr']:.3f} to "
                f"{early_stats['max_subsidence_cm_yr']:.3f} cm/yr\n\n")
    else:
        f.write("No valid observations in early period.\n\n")

    # Late period
    f.write("=" * 80 + "\n")
    f.write("LATE PERIOD (2022-2024)\n")
    f.write("=" * 80 + "\n")
    if late_stats:
        f.write(f"Observations: {late_stats['n_observations']:,}\n")
        f.write(f"Mean subsidence:  {late_stats['mean_subsidence_cm_yr']:.3f} cm/yr\n")
        f.write(f"Median:           {late_stats['median_subsidence_cm_yr']:.3f} cm/yr\n")
        f.write(f"Std dev:          {late_stats['std_subsidence_cm_yr']:.3f} cm/yr\n")
        f.write(f"95% CI:           [{late_stats['ci_lower_95_cm_yr']:.3f}, "
                f"{late_stats['ci_upper_95_cm_yr']:.3f}] cm/yr\n")
        f.write(f"Range:            {late_stats['min_subsidence_cm_yr']:.3f} to "
                f"{late_stats['max_subsidence_cm_yr']:.3f} cm/yr\n\n")
    else:
        f.write("No valid observations in late period.\n\n")

    # Acceleration metrics
    f.write("=" * 80 + "\n")
    f.write("ACCELERATION METRICS\n")
    f.write("=" * 80 + "\n")
    # FIX #8: guard against empty df_accel
    if len(df_accel) > 0:
        for _, row in df_accel.iterrows():
            f.write(f"\n{row['Metric']}\n")
            f.write(f"  Early: {row['Early_Period']:.3f} cm/yr\n")
            f.write(f"  Late:  {row['Late_Period']:.3f} cm/yr\n")
            f.write(f"  Change: {row['Change_cm_yr']:.3f} cm/yr "
                    f"({row['Relative_Change_Percent']:.1f}%)\n")
    else:
        f.write("\nNo acceleration metrics computed (insufficient data).\n")

    # Interpretation
    f.write("\n" + "=" * 80 + "\n")
    f.write("INTERPRETATION\n")
    f.write("=" * 80 + "\n")
    if early_stats and late_stats:
        delta = (late_stats['mean_subsidence_cm_yr']
                 - early_stats['mean_subsidence_cm_yr'])
        if delta < 0:
            f.write("Subsidence is ACCELERATING (more negative in late period).\n")
        elif delta > 0:
            f.write("Subsidence is DECELERATING (less negative in late period).\n")
        else:
            f.write("No change detected between periods.\n")
        f.write(f"Late minus early: {delta:.3f} cm/yr\n")
    else:
        f.write("Insufficient data in one or both periods for interpretation.\n")

logger.info(f"Saved {report_file.name}")
print(f"Saved {report_file.name}")

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
    'early_period_observations': len(df_early),
    'late_period_observations':  len(df_late),
    'early_mean_cm_yr':          early_stats['mean_subsidence_cm_yr'] if early_stats else None,
    'late_mean_cm_yr':           late_stats['mean_subsidence_cm_yr']  if late_stats  else None,
    'n_acceleration_metrics':    len(df_accel),
}

metadata.finalize()
metadata_file = metadata.save(OUTPUT_DIR)
logger.info(f"Saved metadata: {metadata_file.name}")
print(f"Saved metadata: {metadata_file.name}")

# ============================================================================
# COMPLETION SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SCRIPT 05b COMPLETE")
print("=" * 80)
logger.info("=" * 80)
logger.info("SCRIPT 05b COMPLETE")
logger.info("=" * 80)

print(f"""
Outputs created:
  icesat2_early_period.csv             ({len(df_early):,} observations)
  icesat2_late_period.csv              ({len(df_late):,} observations)
  temporal_acceleration_metrics.csv    ({len(df_accel)} metrics)
  05b_temporal_stratification_report.txt
  script_05b_run.log
  05b_analysis_temporal_stratification_metadata.json""")

if early_stats and late_stats:
    accel = (late_stats['mean_subsidence_cm_yr']
             - early_stats['mean_subsidence_cm_yr'])
    print(f"""
Key results:
  Early period : {early_stats['mean_subsidence_cm_yr']:.3f} cm/yr
  Late period  : {late_stats['mean_subsidence_cm_yr']:.3f} cm/yr
  Acceleration : {accel:.3f} cm/yr
  Sign convention: negative=subsidence, positive=uplift""")

print("\nNext: Run Script 06 (Advanced) or Script 07 (Figures)")
print("=" * 80 + "\n")

logger.info("SCRIPT 05b EXECUTION COMPLETE")