"""
================================================================================
SCRIPT 05: TEMPORAL ANALYSIS - WITH DEBUGGING (FIND THE REAL ISSUE)
================================================================================

PURPOSE: Debug timescale classification to find why dt_days isn't being computed

STRATEGY:
1. Print all available columns
2. Print detected column names
3. Print sample of dt_days before classification
4. Show what's actually being classified

================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("SCRIPT 05: TEMPORAL ANALYSIS - DEBUG VERSION")
print("="*80)

# ============================================================================
# LOAD CONFIGURATION
# ============================================================================

BASE_OUTPUT_DIR = Path(r'E:\Processing\ICeSat-2\output_data')
timestamp_file = BASE_OUTPUT_DIR / 'latest_run_timestamp.txt'

try:
    with open(timestamp_file, 'r') as f:
        RUN_TIMESTAMP = f.read().strip()
    OUTPUT_DIR = BASE_OUTPUT_DIR / f'run_{RUN_TIMESTAMP}'
except FileNotFoundError:
    print("ERROR: Run Script 00 first.")
    sys.exit(1)

print(f"Run ID: {RUN_TIMESTAMP}")
print(f"Output directory: {OUTPUT_DIR}\n")

# ============================================================================
# LOAD DATA
# ============================================================================

print("="*80)
print("LOADING DATA")
print("="*80 + "\n")

try:
    try:
        data = pd.read_csv(OUTPUT_DIR / 'icesat2_repeat_track_matches.csv')
        source_file = 'icesat2_repeat_track_matches.csv'
    except FileNotFoundError:
        data = pd.read_csv(OUTPUT_DIR / 'icesat2_insar_integrated.csv')
        source_file = 'icesat2_insar_integrated.csv'
    
    print(f"✓ Loaded: {source_file} ({len(data):,} crossover points)\n")
except FileNotFoundError as e:
    print(f"ERROR: No crossover file found: {e}")
    sys.exit(1)

# ============================================================================
# DEBUG: PRINT ALL COLUMNS
# ============================================================================

print("="*80)
print("DEBUG: AVAILABLE COLUMNS")
print("="*80 + "\n")

print(f"Total columns: {len(data.columns)}")
print("\nColumn names:")
for i, col in enumerate(data.columns, 1):
    print(f"  {i:2d}. {col}")

print()

# ============================================================================
# COLUMN DETECTION & DATE PARSING
# ============================================================================

print("="*80)
print("COLUMN DETECTION")
print("="*80 + "\n")

def get_column(df, *possible_names):
    """Auto-detect column from multiple possibilities"""
    for name in possible_names:
        if name in df.columns:
            print(f"  ✓ Found '{name}'")
            return name
    print(f"  ✗ Tried: {', '.join(possible_names)} - NONE FOUND")
    return None

print("Detecting latitude column:")
lat_col = get_column(data, 'lat', 'latitude', 'lat_early', 'lat_avg', 'Latitude', 'LAT')

print("\nDetecting longitude column:")
lon_col = get_column(data, 'lon', 'longitude', 'lon_early', 'lon_avg', 'Longitude', 'LON')

print("\nDetecting subsidence rate column:")
rate_col = get_column(data, 'subsidence_rate_m_per_year', 'dh_rate_m_per_year', 'subsidence_rate', 'rate', 'height_change', 'dh')

print("\nDetecting date_early column:")
date_early_col = get_column(data, 'date_early', 'date', 'datetime', 'acquisition_date', 't1', 'time_1', 'start_time')

print("\nDetecting date_late column:")
date_late_col = get_column(data, 'date_late', 'date_2', 'datetime_late', 'measurement_date', 't2', 'time_2', 'end_time')

print()

if not lat_col or not lon_col or not rate_col:
    print("ERROR: Could not detect required columns")
    sys.exit(1)

print(f"✓ Latitude column: {lat_col}")
print(f"✓ Longitude column: {lon_col}")
print(f"✓ Rate column: {rate_col}")
print(f"✓ Date column (early): {date_early_col}")
print(f"✓ Date column (late): {date_late_col}\n")

# ============================================================================
# CHECK IF WE HAVE BOTH DATE COLUMNS
# ============================================================================

print("="*80)
print("DATE COLUMN CHECK")
print("="*80 + "\n")

if not date_early_col:
    print("ERROR: date_early_col is None!")
    print("Cannot proceed without early date column.")
    sys.exit(1)

if not date_late_col:
    print("ERROR: date_late_col is None!")
    print("Cannot proceed without late date column.")
    print("\nAvailable columns that might be the second date:")
    date_cols = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
    for col in date_cols:
        print(f"  - {col}")
    sys.exit(1)

print("✓ Both date columns detected\n")

# ============================================================================
# PARSE DATES AND COMPUTE dt_days
# ============================================================================

print("="*80)
print("PARSING DATES AND COMPUTING dt_days")
print("="*80 + "\n")

print(f"Parsing {date_early_col}...")
data['date_early'] = pd.to_datetime(data[date_early_col])
print(f"  Sample: {data['date_early'].iloc[0]} to {data['date_early'].iloc[-1]}")

print(f"\nParsing {date_late_col}...")
data['date_late'] = pd.to_datetime(data[date_late_col])
print(f"  Sample: {data['date_late'].iloc[0]} to {data['date_late'].iloc[-1]}")

print(f"\nComputing dt_days (difference)...")
data['dt_days'] = (data['date_late'] - data['date_early']).dt.days
print(f"  Sample dt_days: {data['dt_days'].iloc[:10].values}")

# ============================================================================
# STATISTICS ON dt_days
# ============================================================================

print("\n" + "="*80)
print("dt_days STATISTICS")
print("="*80 + "\n")

print(f"dt_days statistics (days between paired observations):")
print(f" Min: {data['dt_days'].min():.1f} days")
print(f" Max: {data['dt_days'].max():.1f} days")
print(f" Mean: {data['dt_days'].mean():.1f} days")
print(f" Median: {data['dt_days'].median():.1f} days")
print(f" Std: {data['dt_days'].std():.1f} days")

print(f"\ndt_days distribution:")
print(data['dt_days'].describe())

print(f"\ndt_days value counts (top 20):")
print(data['dt_days'].value_counts().head(20))

# ============================================================================
# TIMESCALE CLASSIFICATION
# ============================================================================

print("\n" + "="*80)
print("TIMESCALE CLASSIFICATION")
print("="*80 + "\n")

# Define timescale bins (in days)
TIMESCALE_BINS = {
    'short': (0, 30),           # 0-30 days
    'medium': (30, 100),        # 30-100 days
    'long': (100, 365),         # 100-365 days
    'seasonal': (365, np.inf),  # >1 year
}

print("Timescale bins:")
for scale_name, (min_days, max_days) in TIMESCALE_BINS.items():
    print(f"  {scale_name:10s}: {min_days:4d} to {max_days if max_days != np.inf else '∞':>4s} days")

print("\nApplying timescale classification...")

data['timescale'] = 'unknown'

for scale_name, (min_days, max_days) in TIMESCALE_BINS.items():
    mask = (data['dt_days'] >= min_days) & (data['dt_days'] < max_days)
    count = mask.sum()
    print(f"  {scale_name:10s}: {count:,} points")
    data.loc[mask, 'timescale'] = scale_name

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "="*80)
print("TIMESCALE DISTRIBUTION")
print("="*80 + "\n")

timescale_counts = data['timescale'].value_counts()
print("Final distribution:")
for scale in ['short', 'medium', 'long', 'seasonal', 'unknown']:
    count = timescale_counts.get(scale, 0)
    pct = 100 * count / len(data) if len(data) > 0 else 0
    print(f" {scale.capitalize():10s}: {count:,} ({pct:5.1f}%)")

print("\nExpected (for ICESat-2 repeat tracks):")
print("  Short (0-30d):   ~85% (repeat track pairs, typical interval)")
print("  Medium (30-100d): ~10% (different track passes)")
print("  Long (100-365d):  ~4% (seasonal comparisons)")
print("  Seasonal (365+d): ~1% (rare cross-year comparisons)")

# ============================================================================
# SAMPLE DATA
# ============================================================================

print("\n" + "="*80)
print("SAMPLE DATA (First 20 rows)")
print("="*80 + "\n")

sample = data[[date_early_col, date_late_col, 'dt_days', 'timescale']].head(20)
print(sample.to_string())

# ============================================================================
# SAVE DEBUG OUTPUT
# ============================================================================

print("\n" + "="*80)
print("SAVING DEBUG DATA")
print("="*80 + "\n")

debug_file = OUTPUT_DIR / 'icesat2_debug_temporal.csv'
data.to_csv(debug_file, index=False)
print(f"✓ Saved: {debug_file.name}\n")

# ============================================================================
# COMPLETION
# ============================================================================

print("="*80)
print("SCRIPT 05 DEBUG COMPLETE")
print("="*80)
print("\nIf timescale distribution looks correct (~85% short):")
print("  → The logic IS WORKING, just update the main Script 05")
print("\nIf timescale distribution looks wrong:")
print("  → Check dt_days statistics above")
print("  → Verify column detection")
print("  → Check if date_late values are different from date_early\n")

sys.exit(0)
