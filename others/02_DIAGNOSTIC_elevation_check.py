"""
================================================================================
DIAGNOSTIC: Elevation Distribution Analysis for Classification
================================================================================

Purpose: Investigate unexpected elevation ranges in LAND/LAKE/OCEAN classes
         to verify classification logic and identify potential issues.

Questions to Answer:
1. How many land photons are below 0m MSL?
2. How many ocean photons are above 2m MSL?
3. Is classification using elevation or only GIS boundaries?
4. Are there spatial patterns in the anomalies?

Output:
- Detailed elevation statistics by class
- Histogram data for plotting
- List of potential issues
- Recommendations for filtering
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_OUTPUT_DIR = Path(r'E:\Processing\ICeSat-2\output_data')

# Load latest run
timestamp_file = BASE_OUTPUT_DIR / 'latest_run_timestamp.txt'
with open(timestamp_file, 'r') as f:
    RUN_TIMESTAMP = f.read().strip()

OUTPUT_DIR = BASE_OUTPUT_DIR / f'run_{RUN_TIMESTAMP}'

# File paths
LAND_FILE = OUTPUT_DIR / 'icesat2_filtered_ground_photons_LAND.csv'
LAKES_FILE = OUTPUT_DIR / 'icesat2_filtered_ground_photons_LAKES.csv'
OCEAN_FILE = OUTPUT_DIR / 'icesat2_filtered_ground_photons_OCEAN.csv'
DIAGNOSTIC_FILE = OUTPUT_DIR / '02_DIAGNOSTIC_elevation_analysis.txt'

print("="*80)
print("DIAGNOSTIC: ELEVATION DISTRIBUTION ANALYSIS")
print("="*80)
print(f"Run ID: {RUN_TIMESTAMP}\n")

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading classified datasets...")

try:
    df_land = pd.read_csv(LAND_FILE)
    print(f"✓ LAND: {len(df_land):,} photons")
except FileNotFoundError:
    print("✗ LAND file not found")
    df_land = pd.DataFrame()

try:
    df_lakes = pd.read_csv(LAKES_FILE)
    print(f"✓ LAKES: {len(df_lakes):,} photons")
except FileNotFoundError:
    print("✗ LAKES file not found")
    df_lakes = pd.DataFrame()

try:
    df_ocean = pd.read_csv(OCEAN_FILE)
    print(f"✓ OCEAN: {len(df_ocean):,} photons")
except FileNotFoundError:
    print("✗ OCEAN file not found")
    df_ocean = pd.DataFrame()

# Identify elevation column
elev_col = None
if len(df_land) > 0:
    for col in ['Height (m MSL)', 'height', 'Height', 'elevation', 'h_li']:
        if col in df_land.columns:
            elev_col = col
            break

if elev_col is None:
    print("\nERROR: Cannot find elevation column!")
    sys.exit(1)

print(f"\nUsing elevation column: '{elev_col}'")

# ============================================================================
# ANALYSIS 1: BASIC STATISTICS
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 1: BASIC ELEVATION STATISTICS")
print("="*80)

def print_stats(df, name):
    if len(df) == 0:
        print(f"\n{name}: NO DATA")
        return
    
    elev = df[elev_col]
    
    print(f"\n{name} ({len(df):,} photons):")
    print(f"  Range:  {elev.min():.2f} to {elev.max():.2f} m")
    print(f"  Mean:   {elev.mean():.2f} m")
    print(f"  Median: {elev.median():.2f} m")
    print(f"  Std:    {elev.std():.2f} m")
    print(f"  Q25:    {elev.quantile(0.25):.2f} m")
    print(f"  Q75:    {elev.quantile(0.75):.2f} m")

print_stats(df_land, "LAND")
print_stats(df_lakes, "LAKES")
print_stats(df_ocean, "OCEAN")

# ============================================================================
# ANALYSIS 2: ANOMALY DETECTION
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 2: ELEVATION ANOMALIES")
print("="*80)

issues = []

# Check LAND below sea level
if len(df_land) > 0:
    land_below_0 = (df_land[elev_col] < 0).sum()
    land_below_minus2 = (df_land[elev_col] < -2).sum()
    pct_below_0 = land_below_0 / len(df_land) * 100
    
    print(f"\n🚨 LAND ANOMALIES:")
    print(f"   Photons below 0m MSL:  {land_below_0:,} ({pct_below_0:.2f}%)")
    print(f"   Photons below -2m MSL: {land_below_minus2:,}")
    
    if land_below_0 > 0:
        issues.append(f"LAND: {land_below_0:,} photons below 0m MSL")
        print(f"   ⚠ ISSUE: Land should not be below sea level!")

# Check LAKES below sea level
if len(df_lakes) > 0:
    lakes_below_0 = (df_lakes[elev_col] < 0).sum()
    lakes_below_minus2 = (df_lakes[elev_col] < -2).sum()
    pct_below_0 = lakes_below_0 / len(df_lakes) * 100
    
    print(f"\n🚨 LAKE ANOMALIES:")
    print(f"   Photons below 0m MSL:  {lakes_below_0:,} ({pct_below_0:.2f}%)")
    print(f"   Photons below -2m MSL: {lakes_below_minus2:,}")
    
    if lakes_below_0 > 0:
        issues.append(f"LAKES: {lakes_below_0:,} photons below 0m MSL")
        print(f"   ⚠ ISSUE: Inland lakes should not be below sea level!")

# Check OCEAN above sea level
if len(df_ocean) > 0:
    ocean_above_0 = (df_ocean[elev_col] > 0).sum()
    ocean_above_2 = (df_ocean[elev_col] > 2).sum()
    ocean_above_5 = (df_ocean[elev_col] > 5).sum()
    pct_above_0 = ocean_above_0 / len(df_ocean) * 100
    pct_above_2 = ocean_above_2 / len(df_ocean) * 100
    
    print(f"\n🚨 OCEAN ANOMALIES:")
    print(f"   Photons above 0m MSL:  {ocean_above_0:,} ({pct_above_0:.2f}%)")
    print(f"   Photons above 2m MSL:  {ocean_above_2:,} ({pct_above_2:.2f}%)")
    print(f"   Photons above 5m MSL:  {ocean_above_5:,}")
    
    if ocean_above_2 > 0:
        issues.append(f"OCEAN: {ocean_above_2:,} photons above 2m MSL")
        print(f"   ⚠ ISSUE: Ocean should not be >2m above sea level!")
        print(f"   Note: Could be tides, waves, or classification error")

# ============================================================================
# ANALYSIS 3: ELEVATION HISTOGRAM BINS
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 3: ELEVATION DISTRIBUTION (HISTOGRAM BINS)")
print("="*80)

def print_histogram(df, name, bins):
    if len(df) == 0:
        return
    
    elev = df[elev_col]
    counts, bin_edges = np.histogram(elev, bins=bins)
    
    print(f"\n{name}:")
    print(f"{'Elevation Range':<20} {'Count':>12} {'Percentage':>12}")
    print("-" * 45)
    
    for i in range(len(counts)):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i+1]
        count = counts[i]
        pct = count / len(df) * 100
        
        if count > 0:
            print(f"{bin_start:>6.1f} to {bin_end:>6.1f} m  {count:>12,}  {pct:>10.2f}%")

# Define bins for analysis
bins_land = [-10, -5, -2, 0, 2, 5, 10, 20, 30, 50, 100]
bins_lakes = [-10, -5, -2, 0, 2, 5, 10, 20, 30, 50, 100]
bins_ocean = [-50, -30, -20, -10, -5, -2, 0, 2, 5, 10, 20, 50]

print_histogram(df_land, "LAND", bins_land)
print_histogram(df_lakes, "LAKES", bins_lakes)
print_histogram(df_ocean, "OCEAN", bins_ocean)

# ============================================================================
# ANALYSIS 4: SPATIAL DISTRIBUTION OF ANOMALIES
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 4: SPATIAL PATTERNS IN ANOMALIES")
print("="*80)

# Check if we have lat/lon columns
lat_col = None
lon_col = None

for col in ['Latitude (deg)', 'latitude', 'Latitude', 'lat']:
    if len(df_land) > 0 and col in df_land.columns:
        lat_col = col
        break

for col in ['Longitude (deg)', 'longitude', 'Longitude', 'lon']:
    if len(df_land) > 0 and col in df_land.columns:
        lon_col = col
        break

if lat_col and lon_col:
    print("\n✓ Spatial analysis possible")
    
    # Analyze land below sea level
    if len(df_land) > 0 and land_below_0 > 0:
        land_anomalies = df_land[df_land[elev_col] < 0]
        print(f"\nLAND BELOW SEA LEVEL - Spatial Distribution:")
        print(f"  Latitude range:  {land_anomalies[lat_col].min():.4f} to {land_anomalies[lat_col].max():.4f}")
        print(f"  Longitude range: {land_anomalies[lon_col].min():.4f} to {land_anomalies[lon_col].max():.4f}")
        print(f"  Mean location:   {land_anomalies[lat_col].mean():.4f}, {land_anomalies[lon_col].mean():.4f}")
    
    # Analyze ocean above sea level
    if len(df_ocean) > 0 and ocean_above_2 > 0:
        ocean_anomalies = df_ocean[df_ocean[elev_col] > 2]
        print(f"\nOCEAN ABOVE 2m MSL - Spatial Distribution:")
        print(f"  Latitude range:  {ocean_anomalies[lat_col].min():.4f} to {ocean_anomalies[lat_col].max():.4f}")
        print(f"  Longitude range: {ocean_anomalies[lon_col].min():.4f} to {ocean_anomalies[lon_col].max():.4f}")
        print(f"  Mean location:   {ocean_anomalies[lat_col].mean():.4f}, {ocean_anomalies[lon_col].mean():.4f}")
        print(f"  Elevation range: {ocean_anomalies[elev_col].min():.2f} to {ocean_anomalies[elev_col].max():.2f} m")
else:
    print("\n⚠ Cannot perform spatial analysis (no lat/lon columns)")

# ============================================================================
# ANALYSIS 5: CLASSIFICATION METHOD VERIFICATION
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS 5: CLASSIFICATION METHOD CHECK")
print("="*80)

print("\nCurrent classification appears to use:")
print("  ✓ GIS polygon boundaries (Statistics Canada + CanVec)")
print("  ✓ Spatial joins (point-in-polygon tests)")
print("  ✗ NO elevation thresholds applied")

print("\nThis explains the anomalies:")
print("  • Polygon boundaries may not perfectly match actual coastline")
print("  • Tidal effects not accounted for")
print("  • No post-classification elevation filtering")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print("\n1. ADD ELEVATION-BASED REFINEMENT:")
print("   After GIS classification, apply these filters:")
print("   • LAND:  Keep only photons with elevation >= -2m")
print("   • LAKES: Keep only photons with elevation >= -2m AND <= 50m")
print("   • OCEAN: Keep only photons with elevation <= 5m")
print("   (Allows for tides ±2m, waves, and GIS uncertainty)")

print("\n2. HYBRID CLASSIFICATION APPROACH:")
print("   Step 1: Use GIS polygons (current method)")
print("   Step 2: Apply elevation thresholds")
print("   Step 3: Reclassify boundary anomalies")

print("\n3. QUALITY FLAGS:")
print("   Add columns to identify:")
print("   • 'land_below_msl': LAND photons < 0m")
print("   • 'ocean_above_msl': OCEAN photons > 2m")
print("   • 'elevation_verified': Passes elevation threshold")

if len(issues) > 0:
    print("\n4. CURRENT ISSUES DETECTED:")
    for issue in issues:
        print(f"   • {issue}")

# ============================================================================
# SAVE DIAGNOSTIC REPORT
# ============================================================================
print("\n" + "="*80)
print("SAVING DIAGNOSTIC REPORT")
print("="*80)

with open(DIAGNOSTIC_FILE, 'w') as f:
    f.write("="*80 + "\n")
    f.write("DIAGNOSTIC: ELEVATION DISTRIBUTION ANALYSIS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Run ID: {RUN_TIMESTAMP}\n\n")
    
    f.write("SUMMARY OF FINDINGS:\n")
    f.write("-"*80 + "\n\n")
    
    if len(df_land) > 0:
        f.write(f"LAND ({len(df_land):,} photons):\n")
        f.write(f"  Elevation range: {df_land[elev_col].min():.2f} to {df_land[elev_col].max():.2f} m\n")
        f.write(f"  Below 0m MSL: {land_below_0:,} ({pct_below_0:.2f}%)\n\n")
    
    if len(df_lakes) > 0:
        f.write(f"LAKES ({len(df_lakes):,} photons):\n")
        f.write(f"  Elevation range: {df_lakes[elev_col].min():.2f} to {df_lakes[elev_col].max():.2f} m\n")
        lakes_pct = lakes_below_0 / len(df_lakes) * 100
        f.write(f"  Below 0m MSL: {lakes_below_0:,} ({lakes_pct:.2f}%)\n\n")
    
    if len(df_ocean) > 0:
        f.write(f"OCEAN ({len(df_ocean):,} photons):\n")
        f.write(f"  Elevation range: {df_ocean[elev_col].min():.2f} to {df_ocean[elev_col].max():.2f} m\n")
        f.write(f"  Above 0m MSL: {ocean_above_0:,} ({pct_above_0:.2f}%)\n")
        f.write(f"  Above 2m MSL: {ocean_above_2:,} ({pct_above_2:.2f}%)\n\n")
    
    f.write("\nRECOMMENDATIONS:\n")
    f.write("-"*80 + "\n")
    f.write("1. Apply elevation-based refinement after GIS classification\n")
    f.write("2. Use hybrid approach: GIS polygons + elevation thresholds\n")
    f.write("3. Add quality flags for anomalous elevations\n")
    f.write("4. Consider tidal corrections for coastal areas\n")

print(f"\n✓ Diagnostic report saved: {DIAGNOSTIC_FILE.name}")

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
print("\nNext step: Run elevation-based refinement script")
print("            (02_REFINEMENT_elevation_filter.py)")
