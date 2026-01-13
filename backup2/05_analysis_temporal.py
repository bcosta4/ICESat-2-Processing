"""
SCRIPT 05: TEMPORAL ANALYSIS - SUBSIDENCE & MULTI-SCALE

Purpose
-------
Classify crossover-derived subsidence rates into behavioral classes
(stable, subsiding, uplift, rapid subsidence) and compute spatially
aggregated statistics at multiple spatial scales.

Origin / Consolidation
----------------------
- Consolidates:
  - 06_interannual_subsidence_analysis.py
  - 07_multiscale_analysis.py

Main Tasks – Section 1 (Temporal / Rate Classification)
-------------------------------------------------------
- Load icesat2_repeat_track_matches.csv.
- Define classification thresholds (e.g. 1 cm/yr, 5 cm/yr).
- Assign each measurement to a class such as:
  - stable, subsiding, rapidly subsiding, rising/uplifting.
- Optionally group by time interval (short-, medium-, long-term).
- Produce summary statistics by class.

Main Tasks – Section 2 (Multi-Scale Spatial Aggregation)
--------------------------------------------------------
- For each of several spatial scales (for example 500 m, 5 km,
  50 km):
  - Define grid cells at that scale.
  - Aggregate crossover rates into grid statistics:
    mean, std, min, max, and count.
- Save one output file per scale for mapping and further analysis.

Inputs
------
- icesat2_repeat_track_matches.csv

Outputs
-------
- icesat2_classified_subsidence.csv
  (adds a subsidence_class column)
- spatial_aggregation_local_500m.csv
- spatial_aggregation_regional_5000m.csv
- spatial_aggregation_watershed_50000m.csv
- multiscale_analysis_summary.txt  (optional summary/report)

Reproducibility Notes
---------------------
Classification thresholds and chosen aggregation scales should be
justified based on permafrost or coastal-process literature. These
choices determine how "rapid subsidence" vs "stable" is defined and how
local variability is smoothed at regional scales.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import yaml
import sys

print("="*80)
print("SCRIPT 05: TEMPORAL ANALYSIS - MULTI-SCALE SUBSIDENCE")
print("="*80)

# ============================================================================
# LOAD CONFIGURATION
# ============================================================================

BASE_OUTPUT_DIR = Path(r'E:\Processing\ICeSat-2\output_data')

# Try to load CONFIG.yml
config_file = Path(r'E:\Processing\ICeSat-2\scripts\CONFIG.yml')

try:
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    print("✓ Loaded CONFIG.yml")
    OUTPUT_DIR = Path(config.get('paths', {}).get('base_output_dir', str(BASE_OUTPUT_DIR)))
except FileNotFoundError:
    print("⚠ CONFIG.yml not found, using defaults")
    config = {}
    OUTPUT_DIR = BASE_OUTPUT_DIR

# Get latest run
timestamp_file = OUTPUT_DIR / 'latest_run_timestamp.txt'
try:
    with open(timestamp_file, 'r') as f:
        RUN_TIMESTAMP = f.read().strip()
    OUTPUT_DIR = OUTPUT_DIR / f'run_{RUN_TIMESTAMP}'
except:
    print("ERROR: Could not find run timestamp")
    sys.exit(1)

print(f"Run ID: {RUN_TIMESTAMP}")
print(f"Output directory: {OUTPUT_DIR}")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\nLoading crossover data...")

input_file = OUTPUT_DIR / 'icesat2_repeat_track_matches.csv'
if not input_file.exists():
    print(f"ERROR: {input_file} not found!")
    sys.exit(1)

data = pd.read_csv(input_file)
print(f"✓ Loaded {len(data):,} crossover pairs")

# ============================================================================
# DETECT COLUMNS (flexible)
# ============================================================================

print("\nDetecting columns...")

# Try to find required columns with flexibility
rate_candidates = ['subsidence_rate_m_per_year', 'dH', 'rate', 'subsidence_rate_m_yr']
date_candidates = ['date_early', 'date_late', 'datetime', 'date']
lat_candidates = ['lat_avg', 'lat_early', 'latitude', 'lat']
lon_candidates = ['lon_avg', 'lon_early', 'longitude', 'lon']

rate_col = next((c for c in data.columns if any(x in c.lower() for x in rate_candidates)), None)
date_col = next((c for c in data.columns if any(x in c.lower() for x in date_candidates)), None)
lat_col = next((c for c in data.columns if any(x in c.lower() for x in lat_candidates)), None)
lon_col = next((c for c in data.columns if any(x in c.lower() for x in lon_candidates)), None)

if None in [rate_col, date_col, lat_col, lon_col]:
    print(f"⚠ Warning: Could not auto-detect all columns")
    print(f"  Rate: {rate_col}")
    print(f"  Date: {date_col}")
    print(f"  Latitude: {lat_col}")
    print(f"  Longitude: {lon_col}")
    print(f"  Available: {list(data.columns)}")
    
    # Provide defaults
    if rate_col is None:
        rate_col = 'subsidence_rate_m_per_year'
    if date_col is None:
        date_col = 'date_early'

print(f"✓ Using columns:")
print(f"  Rate: {rate_col}")
print(f"  Date: {date_col}")

# ============================================================================
# PARSE DATES
# ============================================================================

data['datetime'] = pd.to_datetime(data[date_col], errors='coerce')
data['year'] = data['datetime'].dt.year
data['month'] = data['datetime'].dt.month
data['doy'] = data['datetime'].dt.dayofyear

# ============================================================================
# TEMPORAL BINNING ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("TEMPORAL BINNING ANALYSIS")
print("="*80)

# Year-by-year analysis
yearly_stats = data.groupby('year')[rate_col].agg(['count', 'mean', 'std', 'median'])
print("\nAnnual subsidence rates:")
print(yearly_stats)

# Month-by-month analysis
monthly_stats = data.groupby('month')[rate_col].agg(['count', 'mean', 'std', 'median'])
print("\nMonthly subsidence rates:")
print(monthly_stats)

# ============================================================================
# MULTI-SCALE SPATIAL BINNING
# ============================================================================

print("\n" + "="*80)
print("MULTI-SCALE SPATIAL CLUSTERING")
print("="*80)

if lat_col and lon_col:
    # Create spatial bins (5 km × 5 km grid)
    lat_bins = pd.cut(data[lat_col], bins=10)
    lon_bins = pd.cut(data[lon_col], bins=10)
    
    spatial_stats = data.groupby([lat_bins, lon_bins])[rate_col].agg(['count', 'mean', 'std'])
    print(f"\nSpatial clustering ({10}×{10} grid):")
    print(f"Non-empty cells: {(spatial_stats['count'] > 0).sum()}")
    print(f"Mean cells: {spatial_stats['mean'].mean():.6f} m/yr")

# ============================================================================
# TREND ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("TEMPORAL TREND ANALYSIS")
print("="*80)

# Linear regression on year
years = data['year'].values
rates = data[rate_col].values

# Remove NaNs
valid_mask = ~np.isnan(rates)
years_valid = years[valid_mask]
rates_valid = rates[valid_mask]

if len(years_valid) > 1:
    slope, intercept, r_value, p_value, std_err = stats.linregress(years_valid, rates_valid)
    
    print(f"\nLinear trend (across all data):")
    print(f"  Slope: {slope:.6f} m/yr²  (acceleration/deceleration)")
    print(f"  Intercept: {intercept:.6f} m/yr")
    print(f"  R-squared: {r_value**2:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Std error: {std_err:.6f}")
    
    if p_value < 0.05:
        direction = "ACCELERATING (worsening)" if slope < 0 else "DECELERATING (improving)"
        print(f"  Significance: ✓ SIGNIFICANT at p<0.05 ({direction})")
    else:
        print(f"  Significance: ✗ Not significant (p>{0.05})")

# ============================================================================
# SAVE ANALYSIS RESULTS
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save yearly stats
yearly_file = OUTPUT_DIR / '05_temporal_yearly_stats.csv'
yearly_stats.to_csv(yearly_file)
print(f"✓ Saved: {yearly_file.name}")

# Save monthly stats
monthly_file = OUTPUT_DIR / '05_temporal_monthly_stats.csv'
monthly_stats.to_csv(monthly_file)
print(f"✓ Saved: {monthly_file.name}")

# Save detailed temporal report
report_file = OUTPUT_DIR / '05_temporal_analysis_report.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("TEMPORAL ANALYSIS REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Total crossovers: {len(data):,}\n")
    f.write(f"Date range: {data['datetime'].min().strftime('%Y-%m-%d')} to {data['datetime'].max().strftime('%Y-%m-%d')}\n")
    f.write(f"Time span: {(data['datetime'].max() - data['datetime'].min()).days / 365.25:.1f} years\n\n")
    
    f.write("ANNUAL STATISTICS\n")
    f.write("─" * 80 + "\n")
    f.write(yearly_stats.to_string())
    f.write("\n\n")
    
    f.write("MONTHLY STATISTICS\n")
    f.write("─" * 80 + "\n")
    f.write(monthly_stats.to_string())
    f.write("\n\n")
    
    if len(years_valid) > 1:
        f.write("LINEAR TREND (Year-based)\n")
        f.write("─" * 80 + "\n")
        f.write(f"Slope: {slope:.8f} m/yr²\n")
        f.write(f"R-squared: {r_value**2:.6f}\n")
        f.write(f"p-value: {p_value:.8f}\n")
        f.write(f"Significance: {'YES (p<0.05)' if p_value < 0.05 else 'NO (p>0.05)'}\n")

print(f"✓ Saved: {report_file.name}")

# ============================================================================
# GENERATE FIGURES
# ============================================================================

print("\nGenerating figures...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Scatter by year
axes[0, 0].scatter(data['year'], data[rate_col], alpha=0.3, s=1)
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Subsidence Rate (m/yr)')
axes[0, 0].set_title('Subsidence Rate Over Time')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Annual mean with error bars
yearly_mean = yearly_stats['mean']
yearly_std = yearly_stats['std']
axes[0, 1].errorbar(yearly_mean.index, yearly_mean.values, yerr=yearly_std.values, 
                    fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Year')
axes[0, 1].set_ylabel('Mean Subsidence Rate (m/yr)')
axes[0, 1].set_title('Annual Mean Rate with Std Dev')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Monthly distribution
monthly_mean = monthly_stats['mean']
axes[1, 0].bar(monthly_mean.index, monthly_mean.values, color='steelblue', alpha=0.7)
axes[1, 0].set_xlabel('Month')
axes[1, 0].set_ylabel('Mean Subsidence Rate (m/yr)')
axes[1, 0].set_title('Monthly Mean Subsidence Rate')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Distribution histogram
axes[1, 1].hist(data[rate_col].dropna(), bins=50, color='coral', alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Subsidence Rate (m/yr)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Distribution of All Subsidence Rates')
axes[1, 1].axvline(data[rate_col].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data[rate_col].mean():.4f}')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05_temporal_analysis_figures.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: 05_temporal_analysis_figures.png")

print("\n" + "="*80)
print("SCRIPT 05 COMPLETE")
print("="*80)

sys.exit(0)
