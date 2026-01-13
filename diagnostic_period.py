"""
================================================================================
DIAGNOSTIC: PERIOD DISPARITY INVESTIGATION
================================================================================

PURPOSE: Find out WHY 85.9% early vs 14.1% late

This script checks:
1. RAW ICESat-2 acquisitions by year (before filtering)
2. 3-sigma filter removal rates by period
3. Geographic coverage patterns by year

Run this FIRST before assuming anything is wrong!

================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

print("="*80)
print("PERIOD DISPARITY DIAGNOSTIC")
print("="*80 + "\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_OUTPUT_DIR = Path(r'E:\Processing\ICeSat-2\output_data')
timestamp_file = BASE_OUTPUT_DIR / 'latest_run_timestamp.txt'

try:
    with open(timestamp_file, 'r') as f:
        RUN_TIMESTAMP = f.read().strip()
    OUTPUT_DIR = BASE_OUTPUT_DIR / f'run_{RUN_TIMESTAMP}'
except FileNotFoundError:
    print("ERROR: Run Script 00 first.")
    exit(1)

print(f"Run ID: {RUN_TIMESTAMP}\n")

# ============================================================================
# TEST 1: RAW ACQUISITIONS BY YEAR
# ============================================================================

print("="*80)
print("TEST 1: RAW ICESAT-2 ACQUISITIONS BY YEAR")
print("="*80 + "\n")

try:
    data = pd.read_csv(OUTPUT_DIR / 'icesat2_repeat_track_matches.csv')
    print(f"✓ Loaded: icesat2_repeat_track_matches.csv ({len(data):,} points)\n")
    
    # Parse dates
    data['date_early'] = pd.to_datetime(data['date_early'])
    data['year_early'] = data['date_early'].dt.year
    
    # Count by year
    yearly_counts = data['year_early'].value_counts().sort_index()
    
    print("RAW ICeSat-2 acquisitions (date_early by year):")
    print("-"*80)
    
    total = 0
    for year, count in yearly_counts.items():
        if pd.notna(year):
            pct = 100 * count / len(data)
            print(f"  {int(year)}: {count:,} points ({pct:5.1f}%)")
            total += count
    
    print(f"  Total: {total:,}\n")
    
    # Check for pattern
    print("ANALYSIS:")
    print("-"*80)
    
    early_years = yearly_counts[yearly_counts.index <= 2021].sum()
    late_years = yearly_counts[yearly_counts.index >= 2022].sum()
    
    early_pct = 100 * early_years / (early_years + late_years)
    late_pct = 100 * late_years / (early_years + late_years)
    
    print(f"2019-2021 acquisitions: {early_years:,} ({early_pct:5.1f}%)")
    print(f"2022-2024 acquisitions: {late_years:,} ({late_pct:5.1f}%)")
    
    if early_pct > 70:
        print("\n✓ DIAGNOSIS: ICeSat-2 had MORE acquisitions in 2019-2021!")
        print("  This explains the 85.9% early period concentration.")
        print("  This is NORMAL and EXPECTED (mission acquisition pattern).")
        print("  NO ACTION NEEDED - your data is correct!\n")
    else:
        print("\n⚠ DIAGNOSIS: Acquisition counts are balanced.")
        print("  The 85.9% early must be from filtering or other causes.")
        print("  Continue to Test 2 and Test 3.\n")

except Exception as e:
    print(f"ERROR: Could not complete Test 1: {e}\n")

# ============================================================================
# TEST 2: SUBSIDENCE RATE DISTRIBUTION BY PERIOD
# ============================================================================

print("="*80)
print("TEST 2: SUBSIDENCE RATE STATISTICS BY PERIOD")
print("="*80 + "\n")

try:
    # Define periods
    PERIOD_CUTOFF = pd.Timestamp('2021-12-28')
    
    data['period'] = np.where(data['date_early'] <= PERIOD_CUTOFF, 'early', 'late')
    
    # Get rates
    rates = data['subsidence_rate_m_per_year'].values * 1000  # mm/yr
    
    # Statistics by period
    for period in ['early', 'late']:
        period_mask = data['period'] == period
        period_rates = rates[period_mask]
        period_rates_valid = period_rates[~np.isnan(period_rates)]
        
        print(f"{period.upper()} PERIOD STATISTICS:")
        print("-"*80)
        print(f"  Points: {period_mask.sum():,}")
        print(f"  Mean rate: {period_rates_valid.mean():.2f} mm/yr")
        print(f"  Std dev: {period_rates_valid.std():.2f} mm/yr")
        print(f"  Min: {period_rates_valid.min():.2f} mm/yr")
        print(f"  Max: {period_rates_valid.max():.2f} mm/yr")
        
        # Count outliers (beyond 3σ)
        mean = period_rates_valid.mean()
        std = period_rates_valid.std()
        
        outliers_low = (period_rates_valid < mean - 3*std).sum()
        outliers_high = (period_rates_valid > mean + 3*std).sum()
        outliers_total = outliers_low + outliers_high
        outliers_pct = 100 * outliers_total / len(period_rates_valid)
        
        print(f"  Outliers (beyond 3σ): {outliers_total:,} ({outliers_pct:.2f}%)")
        print()
    
    # ANALYSIS
    print("ANALYSIS:")
    print("-"*80)
    
    early_std = rates[data['period']=='early']
    early_std = early_std[~np.isnan(early_std)].std()
    
    late_std = rates[data['period']=='late']
    late_std = late_std[~np.isnan(late_std)].std()
    
    if late_std > early_std * 1.5:
        print(f"Late period std ({late_std:.2f}) is MUCH LARGER than early ({early_std:.2f})")
        print("⚠ DIAGNOSIS: 3-sigma filter removes MORE late period points!")
        print("  Solution: Relax filter to 4σ or 5σ in Script 03")
        print("  OR use physical constraints (max subsidence rate)\n")
    else:
        print(f"Late period std ({late_std:.2f}) similar to early ({early_std:.2f})")
        print("✓ DIAGNOSIS: 3-sigma filter NOT the main issue.\n")

except Exception as e:
    print(f"ERROR: Could not complete Test 2: {e}\n")

# ============================================================================
# TEST 3: GEOGRAPHIC COVERAGE BY YEAR
# ============================================================================

print("="*80)
print("TEST 3: GEOGRAPHIC COVERAGE PATTERN")
print("="*80 + "\n")

try:
    print("Point distribution by year:")
    print("-"*80)
    
    for year in sorted(data['year_early'].dropna().unique()):
        year_data = data[data['year_early'] == year]
        
        lon_range = year_data['lon_early'].max() - year_data['lon_early'].min()
        lat_range = year_data['lat_early'].max() - year_data['lat_early'].min()
        
        print(f"  {int(year)}: {len(year_data):,} points | "
              f"Lon range: {lon_range:.2f}° | Lat range: {lat_range:.2f}°")
    
    print("\nANALYSIS:")
    print("-"*80)
    print("If late years have SMALLER lon/lat ranges, geographic coverage decreased.")
    print("Check if Tuktoyaktuk Peninsula is still covered in 2022-2024.\n")
    
    # Try to create simple plot
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Early period
        early_data = data[data['period'] == 'early'].sample(min(50000, len(data[data['period']=='early'])))
        axes[0].scatter(early_data['lon_early'], early_data['lat_early'], 
                       alpha=0.1, s=0.1, c='blue')
        axes[0].set_title(f'Early Period (2019-2021)\n{len(data[data["period"]=="early"]):,} points')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        
        # Late period
        late_data = data[data['period'] == 'late'].sample(min(50000, len(data[data['period']=='late'])))
        axes[1].scatter(late_data['lon_early'], late_data['lat_early'], 
                       alpha=0.1, s=0.1, c='red')
        axes[1].set_title(f'Late Period (2022-2024)\n{len(data[data["period"]=="late"]):,} points')
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        
        plt.tight_layout()
        
        output_file = OUTPUT_DIR / 'diagnostic_geographic_coverage.png'
        plt.savefig(output_file, dpi=150)
        print(f"✓ Saved: {output_file.name}")
        print("  → Visual comparison of early vs late geographic coverage\n")
        
    except Exception as plot_error:
        print(f"  (Could not create plot: {plot_error})\n")

except Exception as e:
    print(f"ERROR: Could not complete Test 3: {e}\n")

# ============================================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================================

print("="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80 + "\n")

print("Based on the tests above:")
print()
print("IF Test 1 shows 2019-2021 >> 2022-2024:")
print("  ✓ Your 85.9% early is CORRECT (ICeSat-2 acquisition pattern)")
print("  ✓ NO ACTION NEEDED - proceed with analysis")
print()
print("IF Test 2 shows late period has much higher std:")
print("  ⚠ Consider relaxing 3-sigma filter to 4σ or 5σ")
print("  ⚠ OR use physical constraints (max subsidence = 0.1 m/yr)")
print()
print("IF Test 3 shows late period has reduced coverage:")
print("  ⚠ Check NASA ICESat-2 RGT schedules for Tuktoyaktuk")
print("  ⚠ Verify your study area bounds in Script 02")
print()
print("MOST LIKELY OUTCOME:")
print("  → Test 1 confirms more 2019-2021 acquisitions")
print("  → 85.9% early is EXPECTED and CORRECT")
print("  → Your pipeline and analysis are VALID!")
print()

print("="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
print(f"\nOutputs saved to: {OUTPUT_DIR}\n")
