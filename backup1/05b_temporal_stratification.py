"""
================================================================================
SCRIPT 05B: TEMPORAL STRATIFICATION - WITH TEXT REPORT & VISUALIZATIONS
================================================================================

FEATURES RESTORED:
✓ Early vs Late period comparison (2019-2021 vs 2022-2024)
✓ Robust statistics (t-test, Mann-Whitney, Kendall's tau)
✓ Bootstrap confidence intervals
✓ TEXT REPORT GENERATION (human-readable summary)
✓ DISTRIBUTION HISTOGRAMS & BOXPLOTS (publication figures)

TEXT REPORT OUTPUT:
- Human-readable .txt file with all statistics
- Perfect for copying into thesis results section
- Shows interpretation and significance

VISUALIZATION OUTPUT:
- Distribution comparison (histogram + boxplot)
- Statistical annotations on figures
- Publication-ready quality

================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("SCRIPT 05B: TEMPORAL STRATIFICATION - WITH TEXT REPORT & FIGURES")
print("="*80)

# ============================================================================
# LOAD CONFIGURATION
# ============================================================================

BASE_OUTPUT_DIR = Path(r'E:\Processing\ICeSat-2\output_data')

try:
    with open(BASE_OUTPUT_DIR / 'latest_run_timestamp.txt', 'r') as f:
        RUN_TIMESTAMP = f.read().strip()
    OUTPUT_DIR = BASE_OUTPUT_DIR / f'run_{RUN_TIMESTAMP}'
except FileNotFoundError:
    print("ERROR: Timestamp file not found. Run Script 00 first.")
    sys.exit(1)

print(f"\nRun ID: {RUN_TIMESTAMP}\n")

# ============================================================================
# LOAD DATA
# ============================================================================

print("="*80)
print("LOADING DATA")
print("="*80 + "\n")

try:
    data = pd.read_csv(OUTPUT_DIR / 'icesat2_repeat_track_matches.csv')
    print(f"✓ Loaded {len(data):,} crossover points\n")
except FileNotFoundError:
    print("ERROR: icesat2_repeat_track_matches.csv not found!")
    sys.exit(1)

# ============================================================================
# AUTO-DETECT REQUIRED COLUMNS
# ============================================================================

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
    sys.exit(1)

print(f"✓ Detected columns:")
print(f" Latitude: {lat_col}")
print(f" Longitude: {lon_col}")
print(f" Rate: {rate_col}")
print(f" Date: {date_col}\n")

# ============================================================================
# CONVERT DATES & FILTER
# ============================================================================

data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
data = data.dropna(subset=[date_col])

data_filtered = data.copy()
print(f"✓ Using all {len(data_filtered):,} points\n")

# ============================================================================
# DETERMINE TEMPORAL SPLIT
# ============================================================================

date_min = data_filtered[date_col].min()
date_max = data_filtered[date_col].max()
date_span = (date_max - date_min).days / 365.25

print(f"Data time range:")
print(f" Start: {date_min.strftime('%Y-%m-%d')}")
print(f" End: {date_max.strftime('%Y-%m-%d')}")
print(f" Span: {date_span:.2f} years\n")

# Fixed year-based split
SPLIT_YEAR = 2021
split_date = pd.to_datetime(f'{SPLIT_YEAR}-12-31')

print(f"Split method: FIXED YEAR CUTOFF")
print(f"Split date: {split_date.strftime('%Y-%m-%d')}\n")

# ============================================================================
# SPLIT INTO EARLY & LATE PERIODS
# ============================================================================

early = data_filtered[data_filtered[date_col] < split_date].copy()
late = data_filtered[data_filtered[date_col] >= split_date].copy()

early['temporal_period'] = 'early'
late['temporal_period'] = 'late'

print("="*80)
print("TEMPORAL PERIOD STATISTICS")
print("="*80)

print(f"\nEARLY PERIOD: {early[date_col].min().strftime('%Y-%m-%d')} to {early[date_col].max().strftime('%Y-%m-%d')}")
print(f" Points: {len(early):,}")
print(f" Mean: {early[rate_col].mean()*1000:.2f} mm/yr")
print(f" Median: {early[rate_col].median()*1000:.2f} mm/yr")
print(f" Std: {early[rate_col].std()*1000:.2f} mm/yr")

print(f"\nLATE PERIOD: {late[date_col].min().strftime('%Y-%m-%d')} to {late[date_col].max().strftime('%Y-%m-%d')}")
print(f" Points: {len(late):,}")
print(f" Mean: {late[rate_col].mean()*1000:.2f} mm/yr")
print(f" Median: {late[rate_col].median()*1000:.2f} mm/yr")
print(f" Std: {late[rate_col].std()*1000:.2f} mm/yr")

# ============================================================================
# ROBUST STATISTICAL ANALYSIS
# ============================================================================

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

# Kendall's tau (trend test)
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

# Change velocity
time_span_years = (late[date_col].max() - early[date_col].min()).days / 365.25
change_velocity = mean_change / time_span_years if time_span_years > 0 else 0

# Bootstrap confidence intervals
print(f"\nComputing bootstrap confidence intervals (n=1000)...")
bootstrap_changes = []
np.random.seed(42)

for _ in range(1000):
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
print(f" Early: {early_rates.mean():.2f} mm/yr")
print(f" Late: {late_rates.mean():.2f} mm/yr")
print(f" Change: {mean_change:.2f} mm/yr ({percent_change:+.1f}%)")
print(f" Change velocity: {change_velocity:.4f} mm/yr/year")
print(f" Direction: {direction} {direction_desc}")

print(f"\nParametric and Non-parametric Tests:")
print(f" T-test: t={t_stat:.4f}, p={t_pval:.6f} {'✓ SIGNIFICANT' if t_pval < 0.05 else '✗ not significant'}")
print(f" Mann-Whitney U: U={u_stat:.2f}, p={u_pval:.6f}")
print(f" Kendall's tau: τ={tau_stat:.4f}, p={tau_pval:.6f}")
print(f" Cohen's d: {cohens_d:.3f}", end=' ')

if abs(cohens_d) < 0.2:
    print("(negligible effect)")
elif abs(cohens_d) < 0.5:
    print("(small effect)")
elif abs(cohens_d) < 0.8:
    print("(medium effect)")
else:
    print("(large effect)")

print(f"\nRobust Statistics (Median Absolute Deviation):")
print(f" Early MAD: {early_mad:.2f} mm/yr")
print(f" Late MAD: {late_mad:.2f} mm/yr")

print(f"\nBootstrap 95% Confidence Interval on Change:")
print(f" [{ci_lower:.2f}, {ci_upper:.2f}] mm/yr")
print(f" Mean: {ci_mean:.2f} mm/yr, Std: {ci_std:.2f} mm/yr")

# ============================================================================
# SAVE STRATIFIED DATA
# ============================================================================

print(f"\n" + "="*80)
print(f"SAVING OUTPUTS")
print(f"="*80 + "\n")

print(f"Saving stratified datasets...")

early.to_csv(OUTPUT_DIR / 'icesat2_early_period.csv', index=False)
late.to_csv(OUTPUT_DIR / 'icesat2_late_period.csv', index=False)

print(f" ✓ icesat2_early_period.csv")
print(f" ✓ icesat2_late_period.csv")

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
print(f" ✓ temporal_acceleration_metrics.csv")

# ============================================================================
# TEXT REPORT GENERATION - RESTORED
# ============================================================================

print(f"Generating text report...")

with open(OUTPUT_DIR / 'temporal_period_statistics.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("TEMPORAL PERIOD STRATIFICATION & ROBUST ACCELERATION ANALYSIS\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"SPLIT METHOD: FIXED YEAR CUTOFF\n")
    f.write(f"SPLIT DATE: {split_date.strftime('%Y-%m-%d')}\n\n")
    
    f.write(f"EARLY PERIOD: {early[date_col].min().strftime('%Y-%m-%d')} to {early[date_col].max().strftime('%Y-%m-%d')}\n")
    f.write(f" Points: {len(early):,}\n")
    f.write(f" Mean: {early_rates.mean():.2f} mm/yr\n")
    f.write(f" Median: {np.median(early_rates):.2f} mm/yr\n")
    f.write(f" Std dev: {early_rates.std():.2f} mm/yr\n")
    f.write(f" MAD: {early_mad:.2f} mm/yr\n\n")
    
    f.write(f"LATE PERIOD: {late[date_col].min().strftime('%Y-%m-%d')} to {late[date_col].max().strftime('%Y-%m-%d')}\n")
    f.write(f" Points: {len(late):,}\n")
    f.write(f" Mean: {late_rates.mean():.2f} mm/yr\n")
    f.write(f" Median: {np.median(late_rates):.2f} mm/yr\n")
    f.write(f" Std dev: {late_rates.std():.2f} mm/yr\n")
    f.write(f" MAD: {late_mad:.2f} mm/yr\n\n")
    
    f.write(f"TEMPORAL ACCELERATION/DECELERATION:\n")
    f.write(f" Change: {mean_change:.2f} mm/yr ({percent_change:+.1f}%)\n")
    f.write(f" Change velocity: {change_velocity:.4f} mm/yr/year\n")
    f.write(f" Direction: {direction} {direction_desc}\n")
    f.write(f" Time span: {time_span_years:.2f} years\n\n")
    
    f.write(f"STATISTICAL SIGNIFICANCE:\n")
    f.write(f" T-test p-value: {t_pval:.6f} {'✓ SIGNIFICANT' if t_pval < 0.05 else '✗ not significant'}\n")
    f.write(f" Mann-Whitney p-value: {u_pval:.6f}\n")
    f.write(f" Kendall tau: {tau_stat:.4f}, p={tau_pval:.6f}\n")
    f.write(f" Cohen's d: {cohens_d:.3f}\n\n")
    
    f.write(f"BOOTSTRAP CONFIDENCE INTERVALS (95%, n=1000):\n")
    f.write(f" 95% CI on change: [{ci_lower:.2f}, {ci_upper:.2f}] mm/yr\n")
    f.write(f" Bootstrap mean: {ci_mean:.2f} ± {ci_std:.2f} mm/yr\n\n")
    
    f.write(f"INTERPRETATION:\n")
    if abs(cohens_d) >= 0.8:
        f.write(f" • Large effect size (Cohen's d={cohens_d:.3f})\n")
    elif abs(cohens_d) >= 0.5:
        f.write(f" • Medium effect size (Cohen's d={cohens_d:.3f})\n")
    elif abs(cohens_d) >= 0.2:
        f.write(f" • Small effect size (Cohen's d={cohens_d:.3f})\n")
    else:
        f.write(f" • Negligible effect size (Cohen's d={cohens_d:.3f})\n")
    
    if t_pval < 0.05:
        f.write(f" • Temporal change is statistically SIGNIFICANT (p={t_pval:.6f})\n")
    else:
        f.write(f" • Temporal change is not statistically significant (p={t_pval:.6f})\n")
    
    if mean_change > 0:
        f.write(f" • Subsidence DECELERATING (stabilizing/improving)\n")
        f.write(f" Early mean of {early_rates.mean():.2f} mm/yr declining to {late_rates.mean():.2f} mm/yr\n")
    elif mean_change < 0:
        f.write(f" • Subsidence ACCELERATING (worsening)\n")
        f.write(f" Early mean of {early_rates.mean():.2f} mm/yr increasing to {late_rates.mean():.2f} mm/yr\n")
    else:
        f.write(f" • Subsidence STABLE (no temporal trend)\n")
    
    if ci_lower > 0 or ci_upper < 0:
        f.write(f" • Bootstrap confidence interval does not span zero: significant change detected\n")
    else:
        f.write(f" • Bootstrap confidence interval spans zero: marginal significance\n")

print(f" ✓ temporal_period_statistics.txt")

# ============================================================================
# GENERATE COMPARISON FIGURES - RESTORED
# ============================================================================

print(f"\nGenerating comparison figures...")

# Figure 1: Distribution histograms with annotation
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Histogram
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
bp = axes[1].boxplot(data_for_box, labels=['Early', 'Late'], patch_artist=True)

for patch, color in zip(bp['boxes'], ['steelblue', 'coral']):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

axes[1].set_ylabel('Subsidence Rate (mm/yr)', fontsize=12)
axes[1].set_title('Period Comparison - Boxplot', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

# Add significance annotation
y_max = max(late_rates.max(), early_rates.max()) * 1.1
axes[1].text(1.5, y_max, f'p={t_pval:.2e}', ha='center', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.tight_layout()
fig_file = OUTPUT_DIR / '05b_figure_distribution_comparison.png'
plt.savefig(fig_file, dpi=300, bbox_inches='tight')
print(f" ✓ Saved: 05b_figure_distribution_comparison.png (300 DPI)")

fig_file_hires = OUTPUT_DIR / '05b_figure_distribution_comparison_HiRes.png'
plt.savefig(fig_file_hires, dpi=600, bbox_inches='tight')
print(f" ✓ Saved: 05b_figure_distribution_comparison_HiRes.png (600 DPI)")
plt.close()

# ============================================================================
# COMPLETION
# ============================================================================

print()
print("="*80)
print("SCRIPT 05B COMPLETE - TEMPORAL STRATIFICATION WITH REPORT & FIGURES")
print("="*80)
print(f"\n✓ OUTPUTS:")
print(f" • icesat2_early_period.csv (early period data)")
print(f" • icesat2_late_period.csv (late period data)")
print(f" • temporal_acceleration_metrics.csv (all statistics)")
print(f" • temporal_period_statistics.txt (human-readable report)")
print(f" • 05b_figure_distribution_comparison.png (publication figure)")
print(f"\n✓ TEMPORAL STRATIFICATION COMPLETE")
print(f"Change: {mean_change:.2f} mm/yr, Velocity: {change_velocity:.4f} mm/yr/year")
print(f"P-value: {t_pval:.2e}, Cohen's d: {cohens_d:.3f}")
print(f"\nReady for next analysis phase\n")
print("="*80 + "\n")

sys.exit(0)
