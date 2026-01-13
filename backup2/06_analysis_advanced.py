"""
================================================================================
SCRIPT 06: ADVANCED ANALYSIS - SPATIAL CLUSTERING & InSAR VALIDATION
================================================================================

Purpose: Comprehensive spatial analysis combining density clustering, 
spatial autocorrelation, and InSAR validation metrics.

This script consolidates spatial analysis from repeat-track crossover 
measurements with InSAR validation data, providing:

- Spatial density clustering and neighbor analysis
- Moran's I spatial autocorrelation for subsidence rates
- Grid-based zonal stratification
- Local uncertainty quantification by density class
- InSAR agreement analysis (residuals, spatial patterns)
- Hotspot identification

Main Tasks:
-----------

1. Load icesat2_repeat_track_matches.csv
   - Auto-detect coordinate and rate columns
   - Convert to local meter coordinates

2. Density Analysis
   - Build cKDTree spatial index
   - Compute neighbor counts (4m buffer radius)
   - Classify points into density classes (0-1, 1-5, 5-20, 20+)
   - Analyze density distribution

3. Spatial Autocorrelation
   - Calculate Moran's I (100m radius)
   - Identify spatial clustering patterns
   - Compute local Moran's I values

4. Grid-based Zonal Analysis
   - Create 500m grid cells
   - Compute zonal statistics (mean rate, std, point count)
   - Quantify local uncertainty by zone

5. InSAR Validation (if data present)
   - Sample InSAR values at crossover locations
   - Compute residuals (ICESat-2 minus InSAR)
   - Spatial analysis of agreement patterns

6. Output Generation
   - Full dataset with all computed metrics
   - Density class statistics
   - Grid zonal statistics
   - Uncertainty quantification
   - Comprehensive summary report

Inputs:
-------
- icesat2_repeat_track_matches.csv (from Script 03)
- icesat2_insar_integrated.csv (optional, from Script 03)

Outputs:
--------
- density_clusters_all.csv (full dataset with metrics)
- density_class_statistics.csv (stratified analysis)
- grid_zonal_statistics.csv (spatial zones)
- uncertainty_by_density.csv (error quantification)
- insar_agreement_analysis.csv (validation metrics, if InSAR data present)
- density_clustering_summary.txt (comprehensive report)

Configuration:
---------------
- BUFFER_RADIUS_M = 4.0 (neighbor search distance)
- MORANS_I_RADIUS_M = 100.0 (spatial autocorrelation radius)
- GRID_CELL_SIZE_M = 500.0 (zonal grid spacing)

Reproducibility Notes:
----------------------
All parameters (buffer radius, grid size, autocorrelation radius) are 
explicitly documented in the output summary. Keep these fixed when 
comparing different runs or time periods.

"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy import stats
import warnings
import sys

warnings.filterwarnings('ignore')

print("="*80)
print("SCRIPT 06: ADVANCED ANALYSIS - SPATIAL CLUSTERING & InSAR VALIDATION")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_OUTPUT_DIR = Path(r'E:\Processing\ICeSat-2\output_data')
BUFFER_RADIUS_M = 4.0              # Neighbor search radius in meters
MORANS_I_RADIUS_M = 100.0          # Spatial autocorrelation radius
GRID_CELL_SIZE_M = 500.0           # Grid cell size for zonal analysis

# ============================================================================
# LOAD DATA
# ============================================================================

try:
    with open(BASE_OUTPUT_DIR / 'latest_run_timestamp.txt', 'r') as f:
        RUN_TIMESTAMP = f.read().strip()
    OUTPUT_DIR = BASE_OUTPUT_DIR / f'run_{RUN_TIMESTAMP}'
except FileNotFoundError:
    print("ERROR: Timestamp file not found. Run Script 00 or 03 first.")
    sys.exit(1)

print(f"\nRun timestamp: {RUN_TIMESTAMP}")
print(f"Output directory: {OUTPUT_DIR}")

# Load main crossover data
try:
    data = pd.read_csv(OUTPUT_DIR / 'icesat2_repeat_track_matches.csv')
    print(f"✓ Loaded {len(data):,} crossover points")
except FileNotFoundError:
    print("ERROR: icesat2_repeat_track_matches.csv not found!")
    sys.exit(1)

# Try to load InSAR-integrated data (if available)
insar_available = False
insar_data = None
try:
    insar_data = pd.read_csv(OUTPUT_DIR / 'icesat2_insar_integrated.csv')
    insar_available = True
    print(f"✓ Loaded InSAR data ({insar_data.shape[1]} columns)")
except FileNotFoundError:
    print("⚠ InSAR data not available, skipping InSAR validation analysis")

# ============================================================================
# AUTO-DETECT COORDINATE COLUMNS
# ============================================================================

print("\nDetecting coordinate system...")

lat_candidates = ['lat_early', 'lat_late', 'lat_avg', 'latitude', 'lat', 'y']
lon_candidates = ['lon_early', 'lon_late', 'lon_avg', 'longitude', 'lon', 'x']

lat_col = next((c for c in lat_candidates if c in data.columns), None)
lon_col = next((c for c in lon_candidates if c in data.columns), None)

if lat_col is None or lon_col is None:
    for col in data.columns:
        if 'lat' in col.lower() and lat_col is None:
            lat_col = col
        if 'lon' in col.lower() and lon_col is None:
            lon_col = col

if lat_col is None or lon_col is None:
    print("ERROR: No coordinate columns found!")
    print(f"Available: {list(data.columns)}")
    sys.exit(1)

print(f"✓ Using: Latitude={lat_col}, Longitude={lon_col}")

# Auto-detect subsidence rate column
rate_col = next((c for c in data.columns if 'subsidence' in c.lower() or 'rate' in c.lower()), None)
if rate_col is None:
    print("ERROR: No subsidence rate column found!")
    sys.exit(1)
print(f"✓ Using rate column: {rate_col}")

# ============================================================================
# CONVERT TO LOCAL METERS
# ============================================================================

print("\nConverting to local coordinate system...")

lat_center = data[lat_col].mean()
lon_center = data[lon_col].mean()

m_per_deg_lat = 111320
m_per_deg_lon = 111320 * np.cos(np.radians(lat_center))

easting = (data[lon_col] - lon_center) * m_per_deg_lon
northing = (data[lat_col] - lat_center) * m_per_deg_lat
coords = np.column_stack([easting, northing])

print(f"  Center: ({lat_center:.4f}N, {lon_center:.4f}E)")
print(f"  Easting range: {coords[:, 0].min():.1f} to {coords[:, 0].max():.1f} m")
print(f"  Northing range: {coords[:, 1].min():.1f} to {coords[:, 1].max():.1f} m")

# ============================================================================
# SECTION 1: DENSITY CLUSTERING
# ============================================================================

print("\n" + "="*80)
print("SECTION 1: DENSITY CLUSTERING")
print("="*80)

print(f"\nBuilding spatial index (cKDTree)...")
tree = cKDTree(coords)

print(f"Querying neighbors within {BUFFER_RADIUS_M}m...")
try:
    neighbor_counts = tree.query_ball_point(coords, r=BUFFER_RADIUS_M, return_length=True)
except TypeError:
    neighbors_list = tree.query_ball_point(coords, r=BUFFER_RADIUS_M)
    neighbor_counts = np.array([len(n) for n in neighbors_list])

neighbor_counts = neighbor_counts - 1  # Exclude self

print(f"\nNeighbor count statistics:")
print(f"  Mean: {np.mean(neighbor_counts):.1f}")
print(f"  Median: {np.median(neighbor_counts):.1f}")
print(f"  Std: {np.std(neighbor_counts):.1f}")
print(f"  Min: {np.min(neighbor_counts)}")
print(f"  Max: {np.max(neighbor_counts)}")
for percentile in [25, 50, 75, 90, 95]:
    print(f"  {percentile}th percentile: {np.percentile(neighbor_counts, percentile):.1f}")

data['neighbor_count'] = neighbor_counts

# ============================================================================
# SECTION 2: SPATIAL AUTOCORRELATION (MORAN'S I)
# ============================================================================

print("\n" + "="*80)
print("SECTION 2: SPATIAL AUTOCORRELATION (MORAN'S I)")
print("="*80)

print(f"\nComputing Moran's I (radius={MORANS_I_RADIUS_M}m)...")

# Build weight matrix
weights_morans = tree.query_ball_point(coords, r=MORANS_I_RADIUS_M)

# Compute Moran's I
rates = data[rate_col].values
rate_mean = np.mean(rates)
rate_centered = rates - rate_mean
rate_variance = np.var(rates)

morans_i_list = []
for i, neighbors in enumerate(weights_morans):
    if len(neighbors) > 1:
        neighbor_vals = rate_centered[neighbors]
        local_morans = rate_centered[i] * np.mean(neighbor_vals)
        morans_i_list.append(local_morans)
    else:
        morans_i_list.append(np.nan)

global_morans_i = np.mean([m for m in morans_i_list if not np.isnan(m)]) / rate_variance if morans_i_list else np.nan

print(f"\nMoran's I Statistics:")
print(f"  Global Moran's I: {global_morans_i:.4f}")

if global_morans_i > 0.3:
    interpretation = "Strong positive spatial autocorrelation (clustered subsidence rates)"
elif global_morans_i > 0:
    interpretation = "Moderate positive spatial autocorrelation"
elif global_morans_i > -0.1:
    interpretation = "Weak spatial autocorrelation (somewhat dispersed)"
else:
    interpretation = "Negative spatial autocorrelation (dispersed pattern)"

print(f"  Interpretation: {interpretation}")

data['local_morans_i'] = morans_i_list

# ============================================================================
# SECTION 3: DENSITY CLASS ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("SECTION 3: DENSITY CLASS STRATIFICATION")
print("="*80)

print(f"\nClassifying points by density...")

bins = [0, 1, 5, 20, np.inf]
labels = ['0-1', '1-5', '5-20', '20+']
data['density_class'] = pd.cut(data['neighbor_count'], bins=bins, labels=labels, include_lowest=True)

# Statistics by density class
density_stats = data.groupby('density_class', observed=True)[rate_col].agg([
    ('count', 'count'),
    ('mean_m_yr', 'mean'),
    ('std_m_yr', 'std'),
    ('median_m_yr', 'median'),
    ('q25_m_yr', lambda x: np.percentile(x, 25) if len(x) > 0 else np.nan),
    ('q75_m_yr', lambda x: np.percentile(x, 75) if len(x) > 0 else np.nan)
]).reset_index()

# Convert to mm/yr
for col in ['mean_m_yr', 'std_m_yr', 'median_m_yr', 'q25_m_yr', 'q75_m_yr']:
    density_stats[col.replace('_m_yr', '_mm_yr')] = density_stats[col] * 1000
    density_stats = density_stats.drop(col, axis=1)

print(f"\nDensity class statistics:")
print(density_stats.to_string(index=False))

density_stats.to_csv(OUTPUT_DIR / 'density_class_statistics.csv', index=False)
print(f"✓ Saved: density_class_statistics.csv")

# Correlation: density vs rate
valid_mask = ~(data['neighbor_count'].isna() | data[rate_col].isna())
if valid_mask.sum() > 2:
    corr_r, corr_p = stats.pearsonr(data.loc[valid_mask, 'neighbor_count'],
                                     data.loc[valid_mask, rate_col])
    print(f"\nDensity vs subsidence rate correlation:")
    print(f"  Pearson r: {corr_r:.4f}")
    print(f"  p-value: {corr_p:.4e}")

# ============================================================================
# SECTION 4: GRID-BASED ZONAL ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("SECTION 4: GRID-BASED ZONAL ANALYSIS")
print("="*80)

print(f"\nPerforming grid analysis ({GRID_CELL_SIZE_M:.0f}m cells)...")

grid_xmin, grid_xmax = coords[:, 0].min(), coords[:, 0].max()
grid_ymin, grid_ymax = coords[:, 1].min(), coords[:, 1].max()

grid_cols = int(np.ceil((grid_xmax - grid_xmin) / GRID_CELL_SIZE_M))
grid_rows = int(np.ceil((grid_ymax - grid_ymin) / GRID_CELL_SIZE_M))

data['grid_x'] = ((coords[:, 0] - grid_xmin) / GRID_CELL_SIZE_M).astype(int)
data['grid_y'] = ((coords[:, 1] - grid_ymin) / GRID_CELL_SIZE_M).astype(int)

grid_stats = data.groupby(['grid_x', 'grid_y'], observed=True).agg({
    rate_col: ['count', 'mean', 'std'],
    'neighbor_count': 'mean'
}).reset_index()

grid_stats.columns = ['grid_x', 'grid_y', 'n_points', 'mean_rate_m_yr', 'std_rate_m_yr', 'mean_density']
grid_stats['mean_rate_mm_yr'] = grid_stats['mean_rate_m_yr'] * 1000
grid_stats['std_rate_mm_yr'] = grid_stats['std_rate_m_yr'] * 1000
grid_stats = grid_stats[['grid_x', 'grid_y', 'n_points', 'mean_rate_mm_yr', 'std_rate_mm_yr', 'mean_density']]

print(f"  Created {len(grid_stats)} grid cells with data")
print(f"  Mean points per cell: {grid_stats['n_points'].mean():.1f}")
print(f"  Mean rate range: {grid_stats['mean_rate_mm_yr'].min():.2f} to {grid_stats['mean_rate_mm_yr'].max():.2f} mm/yr")

grid_stats.to_csv(OUTPUT_DIR / 'grid_zonal_statistics.csv', index=False)
print(f"✓ Saved: grid_zonal_statistics.csv")

# ============================================================================
# SECTION 5: UNCERTAINTY QUANTIFICATION
# ============================================================================

print("\n" + "="*80)
print("SECTION 5: UNCERTAINTY QUANTIFICATION BY DENSITY")
print("="*80)

print(f"\nQuantifying local uncertainty...")

uncertainty_by_density = data.groupby('density_class', observed=True).agg({
    rate_col: lambda x: (x.std() / np.sqrt(len(x)) * 1000) if len(x) > 1 else np.nan
}).reset_index()

uncertainty_by_density.columns = ['density_class', 'se_mm_yr']

print(f"\nStandard error by density class (mm/yr):")
print(uncertainty_by_density.to_string(index=False))

uncertainty_by_density.to_csv(OUTPUT_DIR / 'uncertainty_by_density.csv', index=False)
print(f"✓ Saved: uncertainty_by_density.csv")

# ============================================================================
# SECTION 6: InSAR AGREEMENT ANALYSIS (if available)
# ============================================================================

if insar_available and insar_data is not None:
    print("\n" + "="*80)
    print("SECTION 6: InSAR VALIDATION AGREEMENT ANALYSIS")
    print("="*80)
    
    # Merge with InSAR data
    insar_rate_col = next((c for c in insar_data.columns if 'insar' in c.lower() and 'rate' in c.lower()), None)
    
    if insar_rate_col:
        print(f"\nAnalyzing InSAR agreement...")
        
        # Calculate residuals (ICESat-2 minus InSAR)
        valid_insar = ~insar_data[insar_rate_col].isna()
        
        if valid_insar.sum() > 0:
            # Compute residuals in mm/yr
            residuals_mm_yr = (insar_data.loc[valid_insar, rate_col].values * 1000 - 
                              insar_data.loc[valid_insar, insar_rate_col].values)
            
            print(f"\nResidual Statistics (ICESat-2 minus InSAR):")
            print(f"  Mean residual: {np.mean(residuals_mm_yr):.3f} mm/yr")
            print(f"  Std residual: {np.std(residuals_mm_yr):.3f} mm/yr")
            print(f"  RMSE: {np.sqrt(np.mean(residuals_mm_yr**2)):.3f} mm/yr")
            print(f"  Min: {np.min(residuals_mm_yr):.3f} mm/yr")
            print(f"  Max: {np.max(residuals_mm_yr):.3f} mm/yr")
            print(f"  Points with InSAR data: {valid_insar.sum()}")
            
            # Correlation analysis
            if valid_insar.sum() > 2:
                corr_insar, p_insar = stats.pearsonr(
                    insar_data.loc[valid_insar, rate_col].values * 1000,
                    insar_data.loc[valid_insar, insar_rate_col].values
                )
                print(f"\nInSAR vs ICESat-2 Correlation:")
                print(f"  Pearson r: {corr_insar:.4f}")
                print(f"  p-value: {p_insar:.4e}")
            
            # Save agreement analysis
            agreement_summary = pd.DataFrame({
                'metric': ['n_pairs', 'mean_residual_mm_yr', 'std_residual_mm_yr', 
                          'rmse_mm_yr', 'pearson_r', 'p_value'],
                'value': [valid_insar.sum(), np.mean(residuals_mm_yr), np.std(residuals_mm_yr),
                         np.sqrt(np.mean(residuals_mm_yr**2)), corr_insar, p_insar]
            })
            agreement_summary.to_csv(OUTPUT_DIR / 'insar_agreement_analysis.csv', index=False)
            print(f"✓ Saved: insar_agreement_analysis.csv")
        else:
            print("No InSAR rate data available")
    else:
        print("InSAR rate column not found")
else:
    print("\n" + "="*80)
    print("SECTION 6: InSAR VALIDATION")
    print("="*80)
    print("⚠ InSAR data not available - skipping InSAR agreement analysis")

# ============================================================================
# SAVE MAIN OUTPUT
# ============================================================================

print("\n" + "="*80)
print("SAVING OUTPUTS")
print("="*80)

data.to_csv(OUTPUT_DIR / 'density_clusters_all.csv', index=False)
print(f"✓ density_clusters_all.csv ({len(data):,} points)")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

summary_file = OUTPUT_DIR / 'density_clustering_summary.txt'
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("ADVANCED SPATIAL ANALYSIS SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Run ID: {RUN_TIMESTAMP}\n")
    f.write(f"Total points analyzed: {len(data):,}\n\n")
    
    f.write("CONFIGURATION PARAMETERS:\n")
    f.write(f"  Neighbor search radius: {BUFFER_RADIUS_M} m\n")
    f.write(f"  Moran's I radius: {MORANS_I_RADIUS_M} m\n")
    f.write(f"  Grid cell size: {GRID_CELL_SIZE_M:.0f} m\n\n")
    
    f.write("DENSITY CLUSTERING:\n")
    f.write(f"  Mean neighbors: {np.mean(neighbor_counts):.1f}\n")
    f.write(f"  Median neighbors: {np.median(neighbor_counts):.1f}\n")
    f.write(f"  Std dev: {np.std(neighbor_counts):.1f}\n")
    f.write(f"  Min: {np.min(neighbor_counts)}\n")
    f.write(f"  Max: {np.max(neighbor_counts)}\n\n")
    
    f.write("SPATIAL AUTOCORRELATION:\n")
    f.write(f"  Moran's I: {global_morans_i:.4f}\n")
    f.write(f"  Interpretation: {interpretation}\n\n")
    
    f.write("DENSITY CLASS STATISTICS:\n")
    f.write(density_stats.to_string(index=False))
    f.write(f"\n\n")
    
    f.write("GRID ZONAL ANALYSIS:\n")
    f.write(f"  Grid dimensions: {grid_cols} × {grid_rows} cells\n")
    f.write(f"  Cells with data: {len(grid_stats)}\n")
    f.write(f"  Mean points per cell: {grid_stats['n_points'].mean():.1f}\n\n")
    
    if valid_mask.sum() > 2:
        f.write("DENSITY vs SUBSIDENCE RATE:\n")
        f.write(f"  Pearson r: {corr_r:.4f}\n")
        f.write(f"  p-value: {corr_p:.4e}\n\n")
    
    if insar_available and insar_data is not None:
        f.write("InSAR VALIDATION:\n")
        f.write(f"  Mean residual (IS2-InSAR): {np.mean(residuals_mm_yr):.3f} mm/yr\n")
        f.write(f"  RMSE: {np.sqrt(np.mean(residuals_mm_yr**2)):.3f} mm/yr\n")
        f.write(f"  Points with InSAR: {valid_insar.sum()}\n\n")
    else:
        f.write("InSAR VALIDATION: Not available\n\n")
    
    f.write("="*80 + "\n")
    f.write("END OF REPORT\n")
    f.write("="*80 + "\n")

print(f"✓ density_clustering_summary.txt")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SCRIPT 06 COMPLETE - ADVANCED SPATIAL ANALYSIS")
print("="*80)

print(f"\nOutputs generated:")
print(f"  • density_clusters_all.csv")
print(f"  • density_class_statistics.csv")
print(f"  • grid_zonal_statistics.csv")
print(f"  • uncertainty_by_density.csv")
if insar_available:
    print(f"  • insar_agreement_analysis.csv")
print(f"  • density_clustering_summary.txt")

print(f"\nAnalysis complete. All files saved to: {OUTPUT_DIR}")
sys.exit(0)
