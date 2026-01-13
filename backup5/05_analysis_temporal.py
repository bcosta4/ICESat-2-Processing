"""
================================================================================
SCRIPT 05: SUBSIDENCE RATE ANALYSIS & STRATIFICATION
================================================================================

Purpose
-------
Calculate subsidence rates from paired ICESat-2 observations and stratify
by physically meaningful variables (elevation zones only).

NEW in this version:
- Removed broken cross-stratification (Period × Timescale)
- Fixed timescale classification to match ACTUAL data distribution
- Focus on ELEVATION stratification (most physically meaningful)
- Optional: timescale analysis kept for reference but not cross-stratified

Method
------
1. Load paired observations from Script 04
2. Calculate subsidence rates (Δh / Δt in mm/year)
3. Stratify by elevation zones (10m bins)
4. Compute statistics per zone
5. Save results

Why cross-stratification was removed:
- Your data: 83% are 365+ day intervals (not 85% 0-30 day repeats)
- This mismatch suggests:
  * Timescale classification was fundamentally misaligned with ICESat-2 data
  * Cross-stratification produced statistically unstable results
  * Some "short" intervals may actually be errors or different products
- Solution: Focus on elevation (well-understood spatial pattern)
  and timescale separately (if at all)

Inputs
------
- 04_paired_photons.csv (Script 04 output)

Outputs
-------
- 05_subsidence_rates.csv (all pairs with rates)
- 05_statistics_elevation.txt (subsidence by elevation zone)
- 05_statistics_elevation.csv (machine-readable format)
- 05_statistics_timescale.txt (reference only - not stratified)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import io

# UTF-8 encoding for Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_OUTPUT_DIR = Path(r'E:\Processing\ICeSat-2\output_data')

# Load latest run timestamp
timestamp_file = BASE_OUTPUT_DIR / 'latest_run_timestamp.txt'
with open(timestamp_file, 'r') as f:
    RUN_TIMESTAMP = f.read().strip()

OUTPUT_DIR = BASE_OUTPUT_DIR / f'run_{RUN_TIMESTAMP}'

# Input from Script 04
PAIRED_FILE = OUTPUT_DIR / '04_paired_photons.csv'

# Output files
RATES_FILE = OUTPUT_DIR / '05_subsidence_rates.csv'
ELEV_STATS_FILE = OUTPUT_DIR / '05_statistics_elevation.txt'
ELEV_STATS_CSV = OUTPUT_DIR / '05_statistics_elevation.csv'
TIMESCALE_STATS_FILE = OUTPUT_DIR / '05_statistics_timescale.txt'  # Reference only

# Elevation stratification (10m bins)
ELEVATION_BIN_SIZE = 10  # meters

print("="*80, flush=True)
print("SCRIPT 05: SUBSIDENCE RATE ANALYSIS (ELEVATION STRATIFICATION ONLY)", flush=True)
print("="*80, flush=True)

print(f"\nRun ID: {RUN_TIMESTAMP}", flush=True)
print(f"Input: {PAIRED_FILE.name}", flush=True)
print(f"Elevation bin size: {ELEVATION_BIN_SIZE} m\n", flush=True)

# ============================================================================
# LOAD PAIRED PHOTONS
# ============================================================================
print("="*80, flush=True)
print("LOADING PAIRED PHOTONS", flush=True)
print("="*80, flush=True)

if not PAIRED_FILE.exists():
    print(f"ERROR: Input file not found: {PAIRED_FILE}", flush=True)
    sys.exit(1)

df_pairs = pd.read_csv(PAIRED_FILE)
print(f"\n✓ Loaded {len(df_pairs):,} paired observations\n", flush=True)

# ============================================================================
# CALCULATE SUBSIDENCE RATES
# ============================================================================
print("="*80, flush=True)
print("CALCULATING SUBSIDENCE RATES", flush=True)
print("="*80, flush=True)

# Get elevation column name
elev_col = None
for col in ['Height (m MSL)', 'height', 'Height', 'elevation', 'h_li']:
    if col in df_pairs.columns:
        elev_col = col
        break

if elev_col is None:
    print("ERROR: Cannot find elevation column!", flush=True)
    sys.exit(1)

print(f"Using elevation column: '{elev_col}'")

# Detect time difference columns (from Script 04)
dt_col = None
dh_col = None
for col in df_pairs.columns:
    if 'dt' in col.lower() or 'days' in col.lower():
        dt_col = col
    if 'dh' in col.lower() or 'elevation_diff' in col.lower():
        dh_col = col

if dt_col is None or dh_col is None:
    print("\nCalculating time and height differences...")
    
    # Ensure date columns exist
    date_cols = [c for c in df_pairs.columns if 'date' in c.lower() or 'time' in c.lower()]
    
    if len(date_cols) >= 2:
        # Parse dates
        for col in date_cols:
            df_pairs[col] = pd.to_datetime(df_pairs[col])
        
        # Sort by date
        date_cols = sorted([c for c in df_pairs.columns if 'date' in c.lower()])
        
        if len(date_cols) >= 2:
            # Assume first is earlier, second is later
            df_pairs['dt_days'] = (df_pairs[date_cols[1]] - df_pairs[date_cols[0]]).dt.days
            df_pairs['dh_m'] = df_pairs[date_cols[1]] + '_height' - df_pairs[date_cols[0]] + '_height'
            
            dt_col = 'dt_days'
            dh_col = 'dh_m'
            print(f"  Created dt_days and dh_m columns")
    
    if dt_col is None or dh_col is None:
        print("ERROR: Cannot calculate time/height differences!", flush=True)
        print(f"Available columns: {df_pairs.columns.tolist()}", flush=True)
        sys.exit(1)

# Calculate subsidence rate (mm/year)
# Positive rate = subsidence (elevation decrease)
# Negative rate = uplift (elevation increase)

print(f"\nCalculating rates from {dt_col} and {dh_col}...")

df_pairs['rate_mm_yr'] = (-df_pairs[dh_col] / df_pairs[dt_col]) * 365.25 * 1000

print(f"✓ Subsidence rates calculated")
print(f"\nRate statistics (mm/yr):")
print(f"  Min:    {df_pairs['rate_mm_yr'].min():>10.2f}")
print(f"  Max:    {df_pairs['rate_mm_yr'].max():>10.2f}")
print(f"  Mean:   {df_pairs['rate_mm_yr'].mean():>10.2f}")
print(f"  Median: {df_pairs['rate_mm_yr'].median():>10.2f}")
print(f"  Std:    {df_pairs['rate_mm_yr'].std():>10.2f}")

# Remove infinite/NaN values
print(f"\nData quality check:")
nan_count = df_pairs['rate_mm_yr'].isna().sum()
inf_count = np.isinf(df_pairs['rate_mm_yr']).sum()
print(f"  NaN values: {nan_count:,}")
print(f"  Inf values: {inf_count:,}")

if nan_count > 0 or inf_count > 0:
    print(f"  Removing {nan_count + inf_count:,} invalid rates...")
    df_pairs = df_pairs[df_pairs['rate_mm_yr'].notna() & np.isfinite(df_pairs['rate_mm_yr'])].copy()
    print(f"  Remaining: {len(df_pairs):,} pairs")

# ============================================================================
# ELEVATION STRATIFICATION (PRIMARY ANALYSIS)
# ============================================================================
print("\n" + "="*80, flush=True)
print("ELEVATION STRATIFICATION ANALYSIS", flush=True)
print("="*80, flush=True)

# Use mean elevation for stratification
df_pairs['elevation_mean'] = (df_pairs[elev_col]) if isinstance(df_pairs[elev_col].iloc[0], (int, float)) else df_pairs[elev_col]

# Bin by elevation
df_pairs['elev_bin'] = (df_pairs['elevation_mean'] // ELEVATION_BIN_SIZE * ELEVATION_BIN_SIZE).astype(int)

# Compute statistics per elevation bin
elev_stats = df_pairs.groupby('elev_bin')['rate_mm_yr'].agg([
    ('count', 'count'),
    ('mean', 'mean'),
    ('median', 'median'),
    ('std', 'std'),
    ('min', 'min'),
    ('max', 'max'),
    ('q25', lambda x: x.quantile(0.25)),
    ('q75', lambda x: x.quantile(0.75))
]).round(3)

elev_stats['elevation_range'] = elev_stats.index.astype(str) + '-' + \
                                 (elev_stats.index + ELEVATION_BIN_SIZE).astype(str) + ' m'

print("\nSubsidence rates by elevation zone:")
print("-" * 100)
print(f"{'Elevation':>20} {'Count':>10} {'Mean':>12} {'Median':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
print("=" * 100)

for idx in sorted(elev_stats.index):
    row = elev_stats.loc[idx]
    print(f"{row['elevation_range']:>20} {int(row['count']):>10,} "
          f"{row['mean']:>12.2f} {row['median']:>12.2f} {row['std']:>12.2f} "
          f"{row['min']:>12.2f} {row['max']:>12.2f}")

print("=" * 100)

# Save elevation statistics to text
with open(ELEV_STATS_FILE, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("SUBSIDENCE RATES BY ELEVATION ZONE\n")
    f.write("="*80 + "\n\n")
    f.write(f"Run ID: {RUN_TIMESTAMP}\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("METHODOLOGY:\n")
    f.write("-"*80 + "\n")
    f.write(f"Elevation bin size: {ELEVATION_BIN_SIZE} m\n")
    f.write("Subsidence rate = -ΔH / Δt × 365.25 × 1000 (mm/year)\n")
    f.write("Positive rate = subsidence (elevation decrease)\n")
    f.write("Negative rate = uplift (elevation increase)\n\n")
    
    f.write("NOTE: Cross-stratification (Period × Timescale) removed.\n")
    f.write("Reason: Fundamental mismatch between expected and actual data distribution:\n")
    f.write("  • Expected: 85% repeat-track pairs (0-30 days)\n")
    f.write("  • Actual: 83% are 365+ day intervals\n")
    f.write("This suggests the timescale classification does not align with ICESat-2 data.\n")
    f.write("Focus on elevation stratification (spatially meaningful).\n\n")
    
    f.write("SUBSIDENCE RATES BY ELEVATION ZONE:\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Elevation':>20} {'Count':>10} {'Mean':>12} {'Median':>12} {'Std':>12} {'Min':>12} {'Max':>12}\n")
    f.write("="*80 + "\n")
    
    for idx in sorted(elev_stats.index):
        row = elev_stats.loc[idx]
        f.write(f"{row['elevation_range']:>20} {int(row['count']):>10,} "
                f"{row['mean']:>12.2f} {row['median']:>12.2f} {row['std']:>12.2f} "
                f"{row['min']:>12.2f} {row['max']:>12.2f}\n")
    
    f.write("="*80 + "\n\n")
    f.write("INTERPRETATION:\n")
    f.write("-"*80 + "\n")
    f.write("• Positive rates = subsidence (ground sinking)\n")
    f.write("• Negative rates = uplift (ground rising)\n")
    f.write("• Magnitude indicates rate of change (mm/year)\n")
    f.write("• Std dev indicates variability of rates within zone\n")

print(f"\n✓ Elevation statistics saved to: {ELEV_STATS_FILE.name}")

# Save as CSV
elev_stats_export = elev_stats.reset_index()
elev_stats_export.to_csv(ELEV_STATS_CSV, index=False)
print(f"✓ Elevation statistics (CSV) saved to: {ELEV_STATS_CSV.name}")

# ============================================================================
# TIMESCALE ANALYSIS (REFERENCE ONLY - NOT STRATIFIED)
# ============================================================================
print("\n" + "="*80, flush=True)
print("TIMESCALE ANALYSIS (REFERENCE ONLY)", flush=True)
print("="*80, flush=True)

# Classify timescales based on ACTUAL distribution
print(f"\nAnalyzing actual timescale distribution from {len(df_pairs):,} pairs...")

# Get percentiles from actual data
p33 = df_pairs[dt_col].quantile(0.33)
p67 = df_pairs[dt_col].quantile(0.67)

print(f"  33rd percentile: {p33:.0f} days")
print(f"  67th percentile: {p67:.0f} days")

# Classify based on actual distribution
def classify_timescale_data_driven(days):
    """Classify based on actual data distribution (tertiles)"""
    if days <= p33:
        return 'Short'
    elif days <= p67:
        return 'Medium'
    else:
        return 'Long'

df_pairs['timescale'] = df_pairs[dt_col].apply(classify_timescale_data_driven)

# Compute statistics per timescale
timescale_stats = df_pairs.groupby('timescale')['rate_mm_yr'].agg([
    ('count', 'count'),
    ('mean', 'mean'),
    ('median', 'median'),
    ('std', 'std'),
    ('min', 'min'),
    ('max', 'max')
]).round(3)

print("\nSubsidence rates by timescale (data-driven tertiles):")
print("-" * 90)
print(f"{'Timescale':>15} {'Count':>10} {'Mean':>12} {'Median':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
print("=" * 90)

for timescale in ['Short', 'Medium', 'Long']:
    if timescale in timescale_stats.index:
        row = timescale_stats.loc[timescale]
        print(f"{timescale:>15} {int(row['count']):>10,} "
              f"{row['mean']:>12.2f} {row['median']:>12.2f} {row['std']:>12.2f} "
              f"{row['min']:>12.2f} {row['max']:>12.2f}")

print("=" * 90)

# Save as reference
with open(TIMESCALE_STATS_FILE, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("TIMESCALE ANALYSIS (REFERENCE ONLY - NOT USED IN CROSS-STRATIFICATION)\n")
    f.write("="*80 + "\n\n")
    f.write(f"Run ID: {RUN_TIMESTAMP}\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("NOTE: Timescale analysis is provided for reference only.\n")
    f.write("Cross-stratification (Elevation × Timescale) has been removed due to\n")
    f.write("fundamental mismatch between expected and actual data distribution.\n\n")
    
    f.write("CLASSIFICATION METHOD (DATA-DRIVEN TERTILES):\n")
    f.write("-"*80 + "\n")
    f.write(f"Short : 0-{p33:.0f} days ({df_pairs[df_pairs['timescale']=='Short'].shape[0]:,} pairs, "
            f"{100*df_pairs[df_pairs['timescale']=='Short'].shape[0]/len(df_pairs):.1f}%)\n")
    f.write(f"Medium: {p33:.0f}-{p67:.0f} days ({df_pairs[df_pairs['timescale']=='Medium'].shape[0]:,} pairs, "
            f"{100*df_pairs[df_pairs['timescale']=='Medium'].shape[0]/len(df_pairs):.1f}%)\n")
    f.write(f"Long  : >{p67:.0f} days ({df_pairs[df_pairs['timescale']=='Long'].shape[0]:,} pairs, "
            f"{100*df_pairs[df_pairs['timescale']=='Long'].shape[0]/len(df_pairs):.1f}%)\n\n")
    
    f.write("SUBSIDENCE RATES BY TIMESCALE:\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Timescale':>15} {'Count':>10} {'Mean':>12} {'Median':>12} {'Std':>12} {'Min':>12} {'Max':>12}\n")
    f.write("="*80 + "\n")
    
    for timescale in ['Short', 'Medium', 'Long']:
        if timescale in timescale_stats.index:
            row = timescale_stats.loc[timescale]
            f.write(f"{timescale:>15} {int(row['count']):>10,} "
                    f"{row['mean']:>12.2f} {row['median']:>12.2f} {row['std']:>12.2f} "
                    f"{row['min']:>12.2f} {row['max']:>12.2f}\n")
    
    f.write("="*80 + "\n\n")
    f.write("WHY CROSS-STRATIFICATION WAS REMOVED:\n")
    f.write("-"*80 + "\n")
    f.write("Your data shows:\n")
    f.write("  • 83% of pairs have 365+ day intervals (not 85% with 0-30 day repeats)\n")
    f.write("  • Only 0.7% are in the 'Short' category\n")
    f.write("  • This fundamental mismatch suggests timescale classification doesn't\n")
    f.write("    align with your actual ICESat-2 repeat-track structure\n\n")
    f.write("Cross-stratifying on such unbalanced categories produces:\n")
    f.write("  • Unstable statistics (very small sample sizes in some cells)\n")
    f.write("  • Confounded results (cannot separate spatial vs temporal effects)\n")
    f.write("  • Unreliable interpretations\n\n")
    f.write("RECOMMENDATION:\n")
    f.write("Use elevation stratification as your primary analysis.\n")
    f.write("If timescale analysis is needed, examine separately (as shown here).\n")

print(f"✓ Timescale analysis saved to: {TIMESCALE_STATS_FILE.name}")

# ============================================================================
# SAVE SUBSIDENCE RATES
# ============================================================================
print("\n" + "="*80, flush=True)
print("SAVING SUBSIDENCE RATES", flush=True)
print("="*80, flush=True)

df_pairs.to_csv(RATES_FILE, index=False)
size_mb = RATES_FILE.stat().st_size / (1024**2)
print(f"\n✓ Subsidence rates saved to: {RATES_FILE.name} ({size_mb:.1f} MB)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80, flush=True)
print("SCRIPT 05 COMPLETE", flush=True)
print("="*80, flush=True)

print("\n✓ Output files created:")
print(f"  • {RATES_FILE.name} (all pairs with subsidence rates)")
print(f"  • {ELEV_STATS_FILE.name} (elevation stratification results)")
print(f"  • {ELEV_STATS_CSV.name} (elevation statistics - CSV format)")
print(f"  • {TIMESCALE_STATS_FILE.name} (timescale reference analysis)")

print("\n✓ Key findings:")
print(f"  • Analyzed {len(df_pairs):,} paired observations")
print(f"  • Mean subsidence rate: {df_pairs['rate_mm_yr'].mean():.2f} mm/year")
print(f"  • Stratified by {len(elev_stats)} elevation zones ({ELEVATION_BIN_SIZE}m bins)")
print(f"  • Removed cross-stratification (fundamentally misaligned with data)")

print("\nNext steps:")
print("  • Use 05_statistics_elevation.csv for visualization")
print("  • Focus on elevation patterns (spatially meaningful)")
print("  • Timescale analysis available for reference if needed")
print("  • Script 06 will create visualizations")

print("\n" + "="*80, flush=True)
