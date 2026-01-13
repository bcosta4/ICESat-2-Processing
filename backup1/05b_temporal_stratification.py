"""
================================================================================
SCRIPT 05B: TEMPORAL PERIOD STRATIFICATION & ACCELERATION ANALYSIS
================================================================================

Purpose: Split ICESat-2 data into early/late periods and quantify temporal
acceleration/deceleration (your novel finding: subsidence deceleration).

Input: icesat2_repeat_track_matches.csv (from Script 02)

Output:
 • icesat2_early_period.csv (early period data with period flag)
 • icesat2_late_period.csv (late period data with period flag)
 • temporal_acceleration_metrics.csv (slope, p-value, effect size, etc.)
 • temporal_period_statistics.txt (human-readable summary)
 • temporal_comparison_maps.png (spatial: early vs late)
 • temporal_comparison_distributions.png (histograms + boxplots)

Key Finding: Quantifies transition from subsiding to stabilizing (deceleration).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("SCRIPT 05B: TEMPORAL PERIOD STRATIFICATION & ACCELERATION ANALYSIS")
print("="*80)

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_OUTPUT_DIR = Path(r'E:\Processing\ICeSat-2\output_data')
MIN_NEIGHBOR_COUNT = 0  # Use all points (no quality filter, for this step)
DPI = 300
SPLIT_METHOD = 'MEDIAN'  # Options: 'MEDIAN', 'FIXED_DATE', 'EQUAL_SAMPLES'
FIXED_DATE = '2021-06-01'  # Only used if SPLIT_METHOD = 'FIXED_DATE'

# =============================================================================
# LOAD DATA
# =============================================================================

try:
    with open(BASE_OUTPUT_DIR / 'latest_run_timestamp.txt', 'r') as f:
        RUN_TIMESTAMP = f.read().strip()
    OUTPUT_DIR = BASE_OUTPUT_DIR / f'run_{RUN_TIMESTAMP}'
except:
    print("ERROR: Timestamp file not found. Run Script 00 first.")
    exit(1)

print(f"\nRun ID: {RUN_TIMESTAMP}")

try:
    data = pd.read_csv(OUTPUT_DIR / 'icesat2_repeat_track_matches.csv')
    print(f"✓ Loaded {len(data):,} crossover points")
except FileNotFoundError:
    print("ERROR: icesat2_repeat_track_matches.csv not found!")
    exit(1)

# =============================================================================
# AUTO-DETECT COLUMNS
# =============================================================================

possible_lat = ['latitude', 'lat', 'ref_lat', 'latitude_mean', 'y', 'lat_early', 'lat_avg']
possible_lon = ['longitude', 'lon', 'ref_lon', 'longitude_mean', 'x', 'lon_early', 'lon_avg']
possible_rate = ['dh_fit_dx', 'subsidence_rate_m_per_yr', 'rate', 'height_change', 'subsidence_rate_m_per_year']
possible_date = ['date', 'time', 'acquisition_date', 'ref_date', 'datetime', 'date_early']

lat_col = next((c for c in data.columns if any(x in c.lower() for x in possible_lat)), None)
lon_col = next((c for c in data.columns if any(x in c.lower() for x in possible_lon)), None)
rate_col = next((c for c in data.columns if any(x in c.lower() for x in possible_rate)), None)
date_col = next((c for c in data.columns if any(x in c.lower() for x in possible_date)), None)

if None in [lat_col, lon_col, rate_col, date_col]:
    print("\nERROR: Could not auto-detect required columns!")
    print(f"Available: {list(data.columns)}")
    exit(1)

print(f"\n✓ Detected columns:")
print(f"  Latitude: {lat_col}")
print(f"  Longitude: {lon_col}")
print(f"  Rate: {rate_col}")
print(f"  Date: {date_col}")

# =============================================================================
# CONVERT DATES & FILTER
# =============================================================================

data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
data = data.dropna(subset=[date_col])

if 'neighbor_count' in data.columns:
    data_filtered = data[data['neighbor_count'] >= MIN_NEIGHBOR_COUNT].copy()
    print(f"\n✓ Quality filtered: {len(data_filtered):,} points")
else:
    data_filtered = data.copy()
    print(f"\nℹ Using all {len(data_filtered):,} points (no neighbor_count column)")

# =============================================================================
# DETERMINE TEMPORAL SPLIT
# =============================================================================

date_min = data_filtered[date_col].min()
date_max = data_filtered[date_col].max()

print(f"\nData time range:")
print(f"  Start: {date_min.strftime('%Y-%m-%d')}")
print(f"  End: {date_max.strftime('%Y-%m-%d')}")
print(f"  Span: {(date_max - date_min).days / 365.25:.1f} years")

if SPLIT_METHOD == 'MEDIAN':
    split_date = data_filtered[date_col].median()
    print(f"\nSplit method: MEDIAN date")
elif SPLIT_METHOD == 'FIXED_DATE':
    split_date = pd.to_datetime(FIXED_DATE)
    print(f"\nSplit method: FIXED date ({FIXED_DATE})")
elif SPLIT_METHOD == 'EQUAL_SAMPLES':
    sorted_dates = data_filtered[date_col].sort_values()
    split_date = sorted_dates.iloc[len(sorted_dates) // 2]
    print(f"\nSplit method: EQUAL SAMPLES")
else:
    print(f"ERROR: Unknown SPLIT_METHOD '{SPLIT_METHOD}'")
    exit(1)

print(f"Split date: {split_date.strftime('%Y-%m-%d')}")

# =============================================================================
# SPLIT INTO EARLY & LATE PERIODS
# =============================================================================

early = data_filtered[data_filtered[date_col] < split_date].copy()
late = data_filtered[data_filtered[date_col] >= split_date].copy()

print(f"\n" + "="*80)
print(f"TEMPORAL PERIODS")
print(f"="*80)

print(f"\nEARLY: {early[date_col].min().strftime('%Y-%m-%d')} to {early[date_col].max().strftime('%Y-%m-%d')}")
print(f"  Points: {len(early):,}")
print(f"  Mean: {early[rate_col].mean()*1000:.2f} mm/yr")
print(f"  Median: {early[rate_col].median()*1000:.2f} mm/yr")
print(f"  Std: {early[rate_col].std()*1000:.2f} mm/yr")

print(f"\nLATE: {late[date_col].min().strftime('%Y-%m-%d')} to {late[date_col].max().strftime('%Y-%m-%d')}")
print(f"  Points: {len(late):,}")
print(f"  Mean: {late[rate_col].mean()*1000:.2f} mm/yr")
print(f"  Median: {late[rate_col].median()*1000:.2f} mm/yr")
print(f"  Std: {late[rate_col].std()*1000:.2f} mm/yr")

# =============================================================================
# TEMPORAL ACCELERATION ANALYSIS
# =============================================================================

mean_change = (late[rate_col].mean() - early[rate_col].mean()) * 1000
percent_change = (mean_change / (early[rate_col].mean() * 1000)) * 100 if early[rate_col].mean() != 0 else 0

t_stat, t_pval = stats.ttest_ind(late[rate_col].dropna(), early[rate_col].dropna())
u_stat, u_pval = stats.mannwhitneyu(late[rate_col].dropna(), early[rate_col].dropna(), alternative='two-sided')

pooled_std = np.sqrt((early[rate_col].std()**2 + late[rate_col].std()**2) / 2)
cohens_d = mean_change / (pooled_std * 1000) if pooled_std > 0 else 0

print(f"\n" + "="*80)
print(f"TEMPORAL ACCELERATION/DECELERATION ANALYSIS")
print(f"="*80)

print(f"\nMean subsidence rate change:")
print(f"  Early: {early[rate_col].mean()*1000:.2f} mm/yr")
print(f"  Late: {late[rate_col].mean()*1000:.2f} mm/yr")
print(f"  Change: {mean_change:.2f} mm/yr ({percent_change:+.1f}%)")

if mean_change > 0:
    direction = "DECELERATING (improving/stabilizing)"
elif mean_change < 0:
    direction = "ACCELERATING (worsening)"
else:
    direction = "STABLE"

print(f"  Direction: {direction}")

print(f"\nStatistical significance:")
print(f"  T-test p-value: {t_pval:.6f} {'✓ SIGNIFICANT' if t_pval < 0.05 else '✗ not significant'}")
print(f"  Mann-Whitney p-value: {u_pval:.6f} {'✓ SIGNIFICANT' if u_pval < 0.05 else '✗ not significant'}")
print(f"  Cohen's d: {cohens_d:.3f} ", end='')
if abs(cohens_d) < 0.2:
    print("(negligible)")
elif abs(cohens_d) < 0.5:
    print("(small)")
elif abs(cohens_d) < 0.8:
    print("(medium)")
else:
    print("(large)")

# =============================================================================
# SAVE STRATIFIED DATA
# =============================================================================

print(f"\nSaving stratified datasets...")

early.to_csv(OUTPUT_DIR / 'icesat2_early_period.csv', index=False)
late.to_csv(OUTPUT_DIR / 'icesat2_late_period.csv', index=False)

print(f"  ✓ icesat2_early_period.csv")
print(f"  ✓ icesat2_late_period.csv")

# Save acceleration metrics
acceleration_metrics = pd.DataFrame({
    'metric': ['early_mean_m_per_yr', 'late_mean_m_per_yr', 'change_m_per_yr', 'change_mm_per_yr', 
               'percent_change', 'direction', 'n_early', 'n_late', 't_statistic', 't_pvalue', 
               'u_statistic', 'u_pvalue', 'cohens_d'],
    'value': [early[rate_col].mean(), late[rate_col].mean(), 
              (late[rate_col].mean() - early[rate_col].mean()), mean_change,
              percent_change, direction, len(early), len(late), 
              t_stat, t_pval, u_stat, u_pval, cohens_d]
})

acceleration_metrics.to_csv(OUTPUT_DIR / 'temporal_acceleration_metrics.csv', index=False)
print(f"  ✓ temporal_acceleration_metrics.csv")

# Save text report
with open(OUTPUT_DIR / 'temporal_period_statistics.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("TEMPORAL PERIOD STRATIFICATION & ACCELERATION ANALYSIS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Split method: {SPLIT_METHOD}\n")
    f.write(f"Split date: {split_date.strftime('%Y-%m-%d')}\n\n")
    f.write(f"EARLY PERIOD: {early[date_col].min().strftime('%Y-%m-%d')} to {early[date_col].max().strftime('%Y-%m-%d')}\n")
    f.write(f"  Points: {len(early):,}\n")
    f.write(f"  Mean: {early[rate_col].mean()*1000:.2f} mm/yr\n")
    f.write(f"  Median: {early[rate_col].median()*1000:.2f} mm/yr\n")
    f.write(f"  Std: {early[rate_col].std()*1000:.2f} mm/yr\n\n")
    f.write(f"LATE PERIOD: {late[date_col].min().strftime('%Y-%m-%d')} to {late[date_col].max().strftime('%Y-%m-%d')}\n")
    f.write(f"  Points: {len(late):,}\n")
    f.write(f"  Mean: {late[rate_col].mean()*1000:.2f} mm/yr\n")
    f.write(f"  Median: {late[rate_col].median()*1000:.2f} mm/yr\n")
    f.write(f"  Std: {late[rate_col].std()*1000:.2f} mm/yr\n\n")
    f.write(f"CHANGE: {mean_change:.2f} mm/yr ({percent_change:+.1f}%)\n")
    f.write(f"DIRECTION: {direction}\n")
    f.write(f"T-test p: {t_pval:.6f} {'SIGNIFICANT' if t_pval < 0.05 else 'not significant'}\n")
    f.write(f"Cohen's d: {cohens_d:.3f}\n")

print(f"  ✓ temporal_period_statistics.txt")

# =============================================================================
# GENERATE COMPARISON FIGURES
# =============================================================================

print(f"\nGenerating comparison figures...")

# Figure 1: Spatial maps
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

vmin, vmax = np.percentile(data_filtered[rate_col]*1000, [5, 95])

scatter1 = axes[0].scatter(early[lon_col], early[lat_col], c=early[rate_col]*1000,
                           cmap='RdBu_r', s=3, vmin=vmin, vmax=vmax, alpha=0.6)
axes[0].set_title(f'EARLY PERIOD\n{early[date_col].min().strftime("%Y-%m-%d")} to {early[date_col].max().strftime("%Y-%m-%d")}\n' +
                  f'n={len(early):,}, mean={early[rate_col].mean()*1000:.2f} mm/yr',
                  fontsize=13, fontweight='bold')
axes[0].set_xlabel('Longitude', fontsize=11)
axes[0].set_ylabel('Latitude', fontsize=11)
axes[0].grid(True, alpha=0.3, linestyle='--')

scatter2 = axes[1].scatter(late[lon_col], late[lat_col], c=late[rate_col]*1000,
                           cmap='RdBu_r', s=3, vmin=vmin, vmax=vmax, alpha=0.6)
axes[1].set_title(f'LATE PERIOD\n{late[date_col].min().strftime("%Y-%m-%d")} to {late[date_col].max().strftime("%Y-%m-%d")}\n' +
                  f'n={len(late):,}, mean={late[rate_col].mean()*1000:.2f} mm/yr',
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel('Longitude', fontsize=11)
axes[1].set_ylabel('Latitude', fontsize=11)
axes[1].grid(True, alpha=0.3, linestyle='--')

cbar = fig.colorbar(scatter2, ax=axes, orientation='horizontal', pad=0.05, aspect=50)
cbar.set_label('Subsidence Rate (mm/year)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'temporal_comparison_maps.png', dpi=DPI, bbox_inches='tight')
plt.close()
print(f"  ✓ temporal_comparison_maps.png")

# Figure 2: Distribution comparison
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

axes[0].hist(early[rate_col].dropna()*1000, bins=60, alpha=0.6, color='steelblue',
            label=f'Early (n={len(early):,})', edgecolor='black', density=True)
axes[0].hist(late[rate_col].dropna()*1000, bins=60, alpha=0.6, color='coral',
            label=f'Late (n={len(late):,})', edgecolor='black', density=True)
axes[0].axvline(early[rate_col].mean()*1000, color='steelblue', linestyle='--', linewidth=2.5, label='Early mean')
axes[0].axvline(late[rate_col].mean()*1000, color='coral', linestyle='--', linewidth=2.5, label='Late mean')
axes[0].set_title('Subsidence Rate Distribution Comparison', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Rate (mm/year)', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

data_for_box = [early[rate_col].dropna()*1000, late[rate_col].dropna()*1000]
bp = axes[1].boxplot(data_for_box, labels=['Early', 'Late'], patch_artist=True,
                     widths=0.6, showmeans=True, meanline=True)
bp['boxes'][0].set_facecolor('steelblue')
bp['boxes'][1].set_facecolor('coral')

for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color='black', linewidth=1.5)

axes[1].set_title('Boxplot Comparison (Temporal Acceleration)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Rate (mm/year)', fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].axhline(0, color='red', linestyle=':', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'temporal_comparison_distributions.png', dpi=DPI, bbox_inches='tight')
plt.close()
print(f"  ✓ temporal_comparison_distributions.png")

# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n" + "="*80)
print(f"SCRIPT 05B COMPLETE")
print(f"="*80)
print(f"\nKEY FINDING:")
if t_pval < 0.05:
    print(f"  Subsidence {direction.split()[0].lower()}: {mean_change:.2f} mm/yr")
    print(f"  p-value: {t_pval:.4f} (STATISTICALLY SIGNIFICANT)")
else:
    print(f"  Subsidence appears STABLE - no significant change (p={t_pval:.4f})")

print(f"\nOutputs saved to: {OUTPUT_DIR}")
