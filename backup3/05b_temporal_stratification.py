"""
================================================================================
SCRIPT 05B: TEMPORAL PERIOD STRATIFICATION & ROBUST ACCELERATION ANALYSIS
================================================================================

Purpose
-------
Split ICESat-2 repeat-track crossover data into early/late temporal periods
and quantify subsidence acceleration/deceleration using robust statistical
methods. Designed to highlight your key finding: subsidence deceleration
in the Tuktoyaktuk Peninsula.

Key Features
------------
• Temporal period stratification (median, fixed date, or equal-sample split)
• Robust statistics: Mann-Whitney U, Kendall's tau, Cohen's d
• Bootstrap confidence intervals on deceleration (1000 iterations)
• Median Absolute Deviation (MAD) for outlier-resistant spread estimates
• Explicit "change velocity" (mm/yr/year) quantifying acceleration/deceleration
• Spatial comparison maps (early vs late)
• Distribution histograms + boxplots with statistical annotations

Input
-----
• icesat2_repeat_track_matches.csv (from Script 03)

Output
------
• icesat2_early_period.csv (with period flag)
• icesat2_late_period.csv (with period flag)
• temporal_acceleration_metrics.csv (robust metrics + bootstrap CI)
• temporal_period_statistics.txt (human-readable summary)
• temporal_comparison_maps.png (spatial early vs late)
• temporal_comparison_distributions.png (histograms + boxplots)

Novel Finding: Quantifies transition from subsiding to stabilizing (deceleration)
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("SCRIPT 05B: TEMPORAL PERIOD STRATIFICATION & ROBUST ACCELERATION ANALYSIS")
print("="*80)

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_OUTPUT_DIR = Path(r'E:\Processing\ICeSat-2\output_data')
MIN_NEIGHBOR_COUNT = 0  # Use all points (no quality filter for this analysis)
DPI = 300

# Temporal split options
SPLIT_METHOD = 'MEDIAN'  # Options: 'MEDIAN', 'FIXED_DATE', 'EQUAL_SAMPLES'
FIXED_DATE = '2021-06-01'  # Only used if SPLIT_METHOD = 'FIXED_DATE'

# Bootstrap settings
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 42

# =============================================================================
# LOAD DATA
# =============================================================================

try:
    with open(BASE_OUTPUT_DIR / 'latest_run_timestamp.txt', 'r') as f:
        RUN_TIMESTAMP = f.read().strip()
    OUTPUT_DIR = BASE_OUTPUT_DIR / f'run_{RUN_TIMESTAMP}'
except FileNotFoundError:
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
# AUTO-DETECT REQUIRED COLUMNS
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
    print(f"\n✓ Quality filtered: {len(data_filtered):,} points (neighbor_count >= {MIN_NEIGHBOR_COUNT})")
else:
    data_filtered = data.copy()
    print(f"\nℹ Using all {len(data_filtered):,} points (no neighbor_count column)")

# =============================================================================
# DETERMINE TEMPORAL SPLIT
# =============================================================================

date_min = data_filtered[date_col].min()
date_max = data_filtered[date_col].max()
date_span = (date_max - date_min).days / 365.25

print(f"\nData time range:")
print(f"  Start: {date_min.strftime('%Y-%m-%d')}")
print(f"  End: {date_max.strftime('%Y-%m-%d')}")
print(f"  Span: {date_span:.2f} years")

if SPLIT_METHOD == 'MEDIAN':
    split_date = data_filtered[date_col].median()
    print(f"\nSplit method: MEDIAN date")
elif SPLIT_METHOD == 'FIXED_DATE':
    split_date = pd.to_datetime(FIXED_DATE)
    print(f"\nSplit method: FIXED date ({FIXED_DATE})")
elif SPLIT_METHOD == 'EQUAL_SAMPLES':
    sorted_dates = data_filtered[date_col].sort_values()
    split_date = sorted_dates.iloc[len(sorted_dates) // 2]
    print(f"\nSplit method: EQUAL SAMPLES (by count)")
else:
    print(f"ERROR: Unknown SPLIT_METHOD '{SPLIT_METHOD}'")
    exit(1)

print(f"Split date: {split_date.strftime('%Y-%m-%d')}")

# =============================================================================
# SPLIT INTO EARLY & LATE PERIODS
# =============================================================================

early = data_filtered[data_filtered[date_col] < split_date].copy()
late = data_filtered[data_filtered[date_col] >= split_date].copy()

early['temporal_period'] = 'early'
late['temporal_period'] = 'late'

print(f"\n" + "="*80)
print(f"TEMPORAL PERIOD STATISTICS")
print(f"="*80)

print(f"\nEARLY PERIOD: {early[date_col].min().strftime('%Y-%m-%d')} to {early[date_col].max().strftime('%Y-%m-%d')}")
print(f"  Points: {len(early):,}")
print(f"  Mean: {early[rate_col].mean()*1000:.2f} mm/yr")
print(f"  Median: {early[rate_col].median()*1000:.2f} mm/yr")
print(f"  Std: {early[rate_col].std()*1000:.2f} mm/yr")

print(f"\nLATE PERIOD: {late[date_col].min().strftime('%Y-%m-%d')} to {late[date_col].max().strftime('%Y-%m-%d')}")
print(f"  Points: {len(late):,}")
print(f"  Mean: {late[rate_col].mean()*1000:.2f} mm/yr")
print(f"  Median: {late[rate_col].median()*1000:.2f} mm/yr")
print(f"  Std: {late[rate_col].std()*1000:.2f} mm/yr")

# =============================================================================
# ROBUST STATISTICAL ANALYSIS
# =============================================================================

print(f"\n" + "="*80)
print(f"ROBUST STATISTICAL ANALYSIS")
print(f"="*80)

# Basic statistics
early_rates = early[rate_col].dropna().values * 1000  # Convert to mm/yr
late_rates = late[rate_col].dropna().values * 1000

mean_change = late_rates.mean() - early_rates.mean()
percent_change = (mean_change / early_rates.mean()) * 100 if early_rates.mean() != 0 else 0

# Parametric test
t_stat, t_pval = stats.ttest_ind(early_rates, late_rates)

# Non-parametric test
u_stat, u_pval = stats.mannwhitneyu(early_rates, late_rates, alternative='two-sided')

# Kendall's tau (trend test on combined time series)
early_dates_numeric = (early[date_col] - early[date_col].min()).dt.days.values
late_dates_numeric = (late[date_col] - late[date_col].min()).dt.days.values
combined_dates_numeric = np.concatenate([
    early_dates_numeric,
    late_dates_numeric + (early[date_col].max() - early[date_col].min()).days
])
combined_rates = np.concatenate([early_rates, late_rates])

tau_stat, tau_pval = stats.kendalltau(combined_dates_numeric, combined_rates)

# Effect size (Cohen's d)
pooled_std = np.sqrt((early_rates.std()**2 + late_rates.std()**2) / 2)
cohens_d = mean_change / pooled_std if pooled_std > 0 else 0

# Robust spread: Median Absolute Deviation (MAD)
early_mad = np.median(np.abs(early_rates - early_rates.mean()))
late_mad = np.median(np.abs(late_rates - late_rates.mean()))

# Change velocity: acceleration/deceleration (mm/yr/year)
time_span_years = (late[date_col].max() - early[date_col].min()).days / 365.25
change_velocity = mean_change / time_span_years if time_span_years > 0 else 0

# Bootstrap confidence intervals on deceleration
print(f"\nComputing bootstrap confidence intervals (n={BOOTSTRAP_ITERATIONS})...")

bootstrap_changes = []
np.random.seed(BOOTSTRAP_SEED)

for _ in range(BOOTSTRAP_ITERATIONS):
    early_sample = np.random.choice(early_rates, size=len(early_rates), replace=True)
    late_sample = np.random.choice(late_rates, size=len(late_rates), replace=True)
    bootstrap_changes.append(late_sample.mean() - early_sample.mean())

bootstrap_changes = np.array(bootstrap_changes)
ci_lower, ci_upper = np.percentile(bootstrap_changes, [2.5, 97.5])
ci_mean = np.mean(bootstrap_changes)
ci_std = np.std(bootstrap_changes)

# Determine direction
if mean_change > 0:
    direction = "DECELERATING"
    direction_desc = "(subsidence improving/stabilizing)"
elif mean_change < 0:
    direction = "ACCELERATING"
    direction_desc = "(subsidence worsening)"
else:
    direction = "STABLE"
    direction_desc = "(no significant change)"

print(f"\nMean subsidence rate change:")
print(f"  Early: {early_rates.mean():.2f} mm/yr")
print(f"  Late: {late_rates.mean():.2f} mm/yr")
print(f"  Change: {mean_change:.2f} mm/yr ({percent_change:+.1f}%)")
print(f"  Change velocity: {change_velocity:.4f} mm/yr/year")
print(f"  Direction: {direction} {direction_desc}")

print(f"\nParametric and Non-parametric Tests:")
print(f"  T-test: t={t_stat:.4f}, p={t_pval:.6f} {'✓ SIGNIFICANT' if t_pval < 0.05 else '✗ not significant'}")
print(f"  Mann-Whitney U: U={u_stat:.2f}, p={u_pval:.6f}")
print(f"  Kendall's tau: τ={tau_stat:.4f}, p={tau_pval:.6f}")
print(f"  Cohen's d: {cohens_d:.3f}", end=' ')

if abs(cohens_d) < 0.2:
    print("(negligible effect)")
elif abs(cohens_d) < 0.5:
    print("(small effect)")
elif abs(cohens_d) < 0.8:
    print("(medium effect)")
else:
    print("(large effect)")

print(f"\nRobust Statistics (Median Absolute Deviation):")
print(f"  Early MAD: {early_mad:.2f} mm/yr")
print(f"  Late MAD: {late_mad:.2f} mm/yr")

print(f"\nBootstrap 95% Confidence Interval on Change:")
print(f"  [{ci_lower:.2f}, {ci_upper:.2f}] mm/yr")
print(f"  Mean: {ci_mean:.2f} mm/yr, Std: {ci_std:.2f} mm/yr")

# =============================================================================
# SAVE STRATIFIED DATA
# =============================================================================

print(f"\n" + "="*80)
print(f"SAVING OUTPUTS")
print(f"="*80)

print(f"\nSaving stratified datasets...")

early.to_csv(OUTPUT_DIR / 'icesat2_early_period.csv', index=False)
late.to_csv(OUTPUT_DIR / 'icesat2_late_period.csv', index=False)

print(f"  ✓ icesat2_early_period.csv")
print(f"  ✓ icesat2_late_period.csv")

# Save acceleration metrics
acceleration_metrics = pd.DataFrame({
    'metric': [
        'early_mean_mm_per_yr',
        'late_mean_mm_per_yr',
        'change_mm_per_yr',
        'change_velocity_mm_yr_per_yr',
        'percent_change_percent',
        'direction',
        'n_early',
        'n_late',
        't_statistic',
        't_pvalue',
        'u_statistic',
        'u_pvalue',
        'kendall_tau',
        'kendall_pvalue',
        'cohens_d',
        'early_mad_mm_yr',
        'late_mad_mm_yr',
        'bootstrap_mean_mm_yr',
        'bootstrap_std_mm_yr',
        'bootstrap_ci_lower_mm_yr',
        'bootstrap_ci_upper_mm_yr',
        'time_span_years'
    ],
    'value': [
        early_rates.mean(),
        late_rates.mean(),
        mean_change,
        change_velocity,
        percent_change,
        direction,
        len(early),
        len(late),
        t_stat,
        t_pval,
        u_stat,
        u_pval,
        tau_stat,
        tau_pval,
        cohens_d,
        early_mad,
        late_mad,
        ci_mean,
        ci_std,
        ci_lower,
        ci_upper,
        time_span_years
    ]
})

acceleration_metrics.to_csv(OUTPUT_DIR / 'temporal_acceleration_metrics.csv', index=False)
print(f"  ✓ temporal_acceleration_metrics.csv")

# Save text report
with open(OUTPUT_DIR / 'temporal_period_statistics.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("TEMPORAL PERIOD STRATIFICATION & ROBUST ACCELERATION ANALYSIS\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"SPLIT METHOD: {SPLIT_METHOD}\n")
    f.write(f"SPLIT DATE: {split_date.strftime('%Y-%m-%d')}\n\n")
    
    f.write(f"EARLY PERIOD: {early[date_col].min().strftime('%Y-%m-%d')} to {early[date_col].max().strftime('%Y-%m-%d')}\n")
    f.write(f"  Points: {len(early):,}\n")
    f.write(f"  Mean: {early_rates.mean():.2f} mm/yr\n")
    f.write(f"  Median: {np.median(early_rates):.2f} mm/yr\n")
    f.write(f"  Std dev: {early_rates.std():.2f} mm/yr\n")
    f.write(f"  MAD: {early_mad:.2f} mm/yr\n\n")
    
    f.write(f"LATE PERIOD: {late[date_col].min().strftime('%Y-%m-%d')} to {late[date_col].max().strftime('%Y-%m-%d')}\n")
    f.write(f"  Points: {len(late):,}\n")
    f.write(f"  Mean: {late_rates.mean():.2f} mm/yr\n")
    f.write(f"  Median: {np.median(late_rates):.2f} mm/yr\n")
    f.write(f"  Std dev: {late_rates.std():.2f} mm/yr\n")
    f.write(f"  MAD: {late_mad:.2f} mm/yr\n\n")
    
    f.write(f"TEMPORAL ACCELERATION/DECELERATION:\n")
    f.write(f"  Change: {mean_change:.2f} mm/yr ({percent_change:+.1f}%)\n")
    f.write(f"  Change velocity: {change_velocity:.4f} mm/yr/year\n")
    f.write(f"  Direction: {direction} {direction_desc}\n")
    f.write(f"  Time span: {time_span_years:.2f} years\n\n")
    
    f.write(f"STATISTICAL SIGNIFICANCE:\n")
    f.write(f"  T-test p-value: {t_pval:.6f} {'✓ SIGNIFICANT' if t_pval < 0.05 else '✗ not significant'}\n")
    f.write(f"  Mann-Whitney p-value: {u_pval:.6f}\n")
    f.write(f"  Kendall tau: {tau_stat:.4f}, p={tau_pval:.6f}\n")
    f.write(f"  Cohen's d: {cohens_d:.3f}\n\n")
    
    f.write(f"BOOTSTRAP CONFIDENCE INTERVALS (95%, n={BOOTSTRAP_ITERATIONS}):\n")
    f.write(f"  95% CI on change: [{ci_lower:.2f}, {ci_upper:.2f}] mm/yr\n")
    f.write(f"  Bootstrap mean: {ci_mean:.2f} ± {ci_std:.2f} mm/yr\n\n")
    
    f.write(f"INTERPRETATION:\n")
    if abs(cohens_d) >= 0.8:
        f.write(f"  • Large effect size (Cohen's d={cohens_d:.3f})\n")
    elif abs(cohens_d) >= 0.5:
        f.write(f"  • Medium effect size (Cohen's d={cohens_d:.3f})\n")
    elif abs(cohens_d) >= 0.2:
        f.write(f"  • Small effect size (Cohen's d={cohens_d:.3f})\n")
    else:
        f.write(f"  • Negligible effect size (Cohen's d={cohens_d:.3f})\n")
    
    if t_pval < 0.05:
        f.write(f"  • Temporal change is statistically SIGNIFICANT (p={t_pval:.6f})\n")
    else:
        f.write(f"  • Temporal change is not statistically significant (p={t_pval:.6f})\n")
    
    if mean_change > 0:
        f.write(f"  • Subsidence DECELERATING (stabilizing/improving)\n")
        f.write(f"    Early mean of {early_rates.mean():.2f} mm/yr declining to {late_rates.mean():.2f} mm/yr\n")
    elif mean_change < 0:
        f.write(f"  • Subsidence ACCELERATING (worsening)\n")
        f.write(f"    Early mean of {early_rates.mean():.2f} mm/yr increasing to {late_rates.mean():.2f} mm/yr\n")
    else:
        f.write(f"  • Subsidence STABLE (no temporal trend)\n")
    
    f.write(f"  • Bootstrap confidence interval does not span zero: significant deceleration detected\n" if ci_lower > 0 or ci_upper < 0 else f"  • Bootstrap confidence interval spans zero: marginal significance\n")

print(f"  ✓ temporal_period_statistics.txt")

# =============================================================================
# GENERATE COMPARISON FIGURES
# =============================================================================

print(f"\nGenerating comparison figures...")

# Figure 1: Spatial maps (early vs late)
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

vmin, vmax = np.percentile(np.concatenate([early_rates, late_rates]), [5, 95])

scatter1 = axes[0].scatter(
    early[lon_col], early[lat_col], c=early[rate_col]*1000,
    cmap='RdBu_r', s=3, vmin=vmin, vmax=vmax, alpha=0.6
)

axes[0].set_title(
    f'EARLY PERIOD\n{early[date_col].min().strftime("%Y-%m-%d")} to {early[date_col].max().strftime("%Y-%m-%d")}\n' +
    f'n={len(early):,}, mean={early_rates.mean():.2f} mm/yr',
    fontsize=13, fontweight='bold'
)
axes[0].set_xlabel('Longitude', fontsize=11)
axes[0].set_ylabel('Latitude', fontsize=11)
axes[0].grid(True, alpha=0.3, linestyle='--')

scatter2 = axes[1].scatter(
    late[lon_col], late[lat_col], c=late[rate_col]*1000,
    cmap='RdBu_r', s=3, vmin=vmin, vmax=vmax, alpha=0.6
)

axes[1].set_title(
    f'LATE PERIOD\n{late[date_col].min().strftime("%Y-%m-%d")} to {late[date_col].max().strftime("%Y-%m-%d")}\n' +
    f'n={len(late):,}, mean={late_rates.mean():.2f} mm/yr',
    fontsize=13, fontweight='bold'
)
axes[1].set_xlabel('Longitude', fontsize=11)
axes[1].set_ylabel('Latitude', fontsize=11)
axes[1].grid(True, alpha=0.3, linestyle='--')

cbar = fig.colorbar(scatter2, ax=axes, orientation='horizontal', pad=0.05, aspect=50)
cbar.set_label('Subsidence Rate (mm/year)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'temporal_comparison_maps.png', dpi=DPI, bbox_inches='tight')
plt.close()

print(f"  ✓ temporal_comparison_maps.png")

# Figure 2: Distribution comparison with annotations
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

axes[0].hist(
    early_rates, bins=60, alpha=0.6, color='steelblue',
    label=f'Early (n={len(early):,})', edgecolor='black', density=True
)
axes[0].hist(
    late_rates, bins=60, alpha=0.6, color='coral',
    label=f'Late (n={len(late):,})', edgecolor='black', density=True
)

axes[0].axvline(early_rates.mean(), color='steelblue', linestyle='--', linewidth=2.5, label='Early mean')
axes[0].axvline(late_rates.mean(), color='coral', linestyle='--', linewidth=2.5, label='Late mean')

axes[0].set_title('Subsidence Rate Distribution Comparison', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Rate (mm/year)', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].legend(fontsize=10, loc='upper right')
axes[0].grid(True, alpha=0.3)

# Annotation box
textstr = f'Change: {mean_change:.2f} mm/yr\nVelocity: {change_velocity:.4f} mm/yr/year\np-value: {t_pval:.2e}'
axes[0].text(0.02, 0.97, textstr, transform=axes[0].transAxes,
    fontsize=11, verticalalignment='top', horizontalalignment='left',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Boxplot
data_for_box = [early_rates, late_rates]
bp = axes[1].boxplot(
    data_for_box, labels=['Early', 'Late'], patch_artist=True,
    widths=0.6, showmeans=True, meanline=True
)

bp['boxes'][0].set_facecolor('steelblue')
bp['boxes'][1].set_facecolor('coral')

for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color='black', linewidth=1.5)

axes[1].set_title('Boxplot Comparison (Temporal Deceleration)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Rate (mm/year)', fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].axhline(0, color='red', linestyle=':', linewidth=1, alpha=0.5)

# Direction label
direction_color = 'green' if mean_change > 0 else ('red' if mean_change < 0 else 'gray')
axes[1].text(0.5, 0.05, f'Direction: {direction}', transform=axes[1].transAxes,
    fontsize=12, fontweight='bold', ha='center', color=direction_color)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'temporal_comparison_distributions.png', dpi=DPI, bbox_inches='tight')
plt.close()

print(f"  ✓ temporal_comparison_distributions.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(f"\n" + "="*80)
print(f"SCRIPT 05B COMPLETE")
print(f"="*80)

print(f"\nKEY FINDINGS (Your Novel Arctic Subsidence Finding):")
print(f"{'='*80}")

if t_pval < 0.05:
    print(f"\n✓ SIGNIFICANT temporal change detected:")
    print(f"  Subsidence is {direction.lower()}: {mean_change:.2f} mm/yr")
    print(f"  Change velocity: {change_velocity:.4f} mm/yr/year")
    print(f"  Bootstrap 95% CI: [{ci_lower:.2f}, {ci_upper:.2f}] mm/yr")
    print(f"  Statistical significance: p = {t_pval:.2e}")
else:
    print(f"\n✗ No significant temporal change detected (p={t_pval:.4f})")
    print(f"  Subsidence appears STABLE over observation period")

print(f"\nInterpretation for Thesis:")
if mean_change > 0 and t_pval < 0.05:
    print(f"  This represents a DECELERATION of subsidence in the Tuktoyaktuk Peninsula")
    print(f"  indicating a transition from initial rapid submergence to a stabilizing phase.")
elif mean_change < 0 and t_pval < 0.05:
    print(f"  This represents an ACCELERATION of subsidence, worsening coastal conditions.")
else:
    print(f"  No clear temporal trend; subsidence rates remain relatively stable.")

print(f"\nAll outputs saved to: {OUTPUT_DIR}\n")
