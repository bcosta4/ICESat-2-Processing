"""
================================================================================
SCRIPT 06: ADVANCED ANALYSIS - SPATIAL DENSITY CLUSTERING & CORRELATION
================================================================================

Purpose: Identify spatial clusters of crossover measurements and explore how subsidence rate relates to local measurement density. This is a higher-level diagnostic to understand patterns and hotspots.

Origin / Consolidation: 
- Directly derived from original: 08_AroundPT_clustering.py
- No additional scripts consolidated.

Main Tasks:
- Load icesat2_repeat_track_matches.csv.
- Construct a cKDTree of crossover coordinates.
- Run DBSCAN or similar clustering algorithm to find dense spatial
  clusters of points.
- Compute cluster-level statistics:
  - cluster size (number of points)
  - mean and standard deviation of subsidence rate
  - spatial extent of each cluster.
- Analyze how subsidence rate correlates with local point density.
- Save per-point cluster labels and summary statistics.

Inputs: - icesat2_repeat_track_matches.csv

Outputs:
- density_clusters_all.csv
  (original data plus cluster labels)
- density_distribution_analysis.txt
- density_vs_rate.csv  (density–rate relationship)

Reproducibility Notes: Clustering parameters (eps, min_samples) control which structures are recognized as clusters vs noise. Record these parameters and keep them fixed when comparing different runs or different time periods.
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
print("SCRIPT 06: ADVANCED ANALYSIS - SPATIAL DENSITY CLUSTERING")
print("="*80)

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_OUTPUT_DIR = Path(r'E:\Processing\ICeSat-2\output_data')
BUFFER_RADIUS_M = 4.0  # Neighbor search radius in meters

# =============================================================================
# LOAD DATA
# =============================================================================

try:
    with open(BASE_OUTPUT_DIR / 'latest_run_timestamp.txt', 'r') as f:
        RUN_TIMESTAMP = f.read().strip()
    OUTPUT_DIR = BASE_OUTPUT_DIR / f'run_{RUN_TIMESTAMP}'
except:
    print("ERROR: Timestamp file not found. Run Script 00 first.")
    sys.exit(1)

print(f"\nRun ID: {RUN_TIMESTAMP}")

try:
    data = pd.read_csv(OUTPUT_DIR / 'icesat2_repeat_track_matches.csv')
    print(f"✓ Loaded {len(data):,} points")
except FileNotFoundError:
    print("ERROR: icesat2_repeat_track_matches.csv not found!")
    sys.exit(1)

# =============================================================================
# AUTO-DETECT COORDINATE COLUMNS
# =============================================================================

print("\nDetecting coordinate system...")

lat_candidates = ['lat_early', 'lat_late', 'lat_avg', 'latitude', 'lat', 'y', 'Latitude (deg)']
lon_candidates = ['lon_early', 'lon_late', 'lon_avg', 'longitude', 'lon', 'x', 'Longitude (deg)']

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

# =============================================================================
# CONVERT TO LOCAL METERS
# =============================================================================

lat_center = data[lat_col].mean()
lon_center = data[lon_col].mean()

m_per_deg_lat = 111320
m_per_deg_lon = 111320 * np.cos(np.radians(lat_center))

easting = (data[lon_col] - lon_center) * m_per_deg_lon
northing = (data[lat_col] - lat_center) * m_per_deg_lat

coords = np.column_stack([easting, northing])

print(f"\nCoordinate conversion:")
print(f"  Center: ({lat_center:.4f}N, {lon_center:.4f}E)")
print(f"  m/deg_lon: {m_per_deg_lon:.1f}")
print(f"  Easting range: {coords[:, 0].min():.1f} to {coords[:, 0].max():.1f} m")
print(f"  Northing range: {coords[:, 1].min():.1f} to {coords[:, 1].max():.1f} m")

# =============================================================================
# BUILD KDTREE & QUERY NEIGHBORS
# =============================================================================

print(f"\nBuilding spatial index...")
tree = cKDTree(coords)

print(f"Querying neighbors within {BUFFER_RADIUS_M}m...")

try:
    neighbor_counts = tree.query_ball_point(coords, r=BUFFER_RADIUS_M, return_length=True)
except TypeError:
    neighbors_list = tree.query_ball_point(coords, r=BUFFER_RADIUS_M)
    neighbor_counts = np.array([len(n) for n in neighbors_list])

neighbor_counts = neighbor_counts - 1  # Exclude self

print(f"✓ Processed {len(data):,} points")

# =============================================================================
# DENSITY DISTRIBUTION ANALYSIS
# =============================================================================

print(f"\nDensity distribution:")
print(f"  Mean: {np.mean(neighbor_counts):.1f} neighbors")
print(f"  Median: {np.median(neighbor_counts):.1f}")
print(f"  Std: {np.std(neighbor_counts):.1f}")
print(f"  Min: {np.min(neighbor_counts)}")
print(f"  Max: {np.max(neighbor_counts)}")

for percentile in [25, 50, 75, 90, 95]:
    print(f"  {percentile}th percentile: {np.percentile(neighbor_counts, percentile):.1f}")

# =============================================================================
# SAVE MAIN OUTPUT WITH DENSITY
# =============================================================================

print(f"\nSaving outputs...")

data['neighbor_count'] = neighbor_counts

output_file = OUTPUT_DIR / 'density_clusters_all.csv'
data.to_csv(output_file, index=False)
print(f"  ✓ density_clusters_all.csv ({len(data):,} points with neighbor_count)")

# =============================================================================
# DENSITY CLASS ANALYSIS
# =============================================================================

bins = [0, 1, 5, 20, np.inf]
labels = ['0-1', '1-5', '5-20', '20+']
data['density_class'] = pd.cut(data['neighbor_count'], bins=bins, labels=labels, include_lowest=True)

density_distribution = data.groupby('density_class', observed=True).agg({
    'neighbor_count': ['count', 'mean', 'std'],
}).reset_index()
density_distribution.columns = ['density_class', 'n_points', 'mean_neighbors', 'std_neighbors']

print(f"\nDensity class distribution:")
print(density_distribution.to_string(index=False))

density_distribution.to_csv(OUTPUT_DIR / 'density_distribution_analysis.csv', index=False)
print(f"  ✓ density_distribution_analysis.csv")

# =============================================================================
# DENSITY vs SUBSIDENCE RATE CORRELATION
# =============================================================================

rate_col = next((c for c in data.columns if 'subsidence' in c.lower() or 'rate' in c.lower()), None)

if rate_col:
    print(f"\nAnalyzing density vs subsidence rate (using {rate_col})...")

    density_vs_rate = data.groupby('density_class', observed=True)[rate_col].agg([
        'count', 'mean', 'std', 'median', 'min', 'max'
    ]).reset_index()
    
    density_vs_rate.columns = ['density_class', 'n_points', 'mean_rate_m_yr', 'std_rate_m_yr', 
                               'median_rate_m_yr', 'min_rate_m_yr', 'max_rate_m_yr']

    # Convert to mm/yr for readability
    for col in ['mean_rate_m_yr', 'std_rate_m_yr', 'median_rate_m_yr', 'min_rate_m_yr', 'max_rate_m_yr']:
        density_vs_rate[col.replace('_m_yr', '_mm_yr')] = density_vs_rate[col] * 1000
        density_vs_rate = density_vs_rate.drop(col, axis=1)

    print(f"\nSubsidence rate by density class:")
    print(density_vs_rate.to_string(index=False))

    # Calculate correlation
    valid_mask = ~(data['neighbor_count'].isna() | data[rate_col].isna())
    if valid_mask.sum() > 2:
        corr_coeff, corr_pval = stats.pearsonr(data.loc[valid_mask, 'neighbor_count'], 
                                                data.loc[valid_mask, rate_col])
        print(f"\nPearson correlation (density vs rate):")
        print(f"  r = {corr_coeff:.4f}, p = {corr_pval:.4e}")
        
        density_vs_rate['correlation_r'] = corr_coeff
        density_vs_rate['correlation_p'] = corr_pval

    density_vs_rate.to_csv(OUTPUT_DIR / 'density_vs_rate_correlation.csv', index=False)
    print(f"  ✓ density_vs_rate_correlation.csv")

# =============================================================================
# SUMMARY REPORT
# =============================================================================

with open(OUTPUT_DIR / 'density_clustering_summary.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("SPATIAL DENSITY CLUSTERING ANALYSIS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Run ID: {RUN_TIMESTAMP}\n")
    f.write(f"Buffer radius: {BUFFER_RADIUS_M} m\n\n")
    f.write(f"Total points: {len(data):,}\n\n")
    f.write(f"NEIGHBOR COUNT STATISTICS:\n")
    f.write(f"  Mean: {np.mean(neighbor_counts):.1f}\n")
    f.write(f"  Median: {np.median(neighbor_counts):.1f}\n")
    f.write(f"  Std: {np.std(neighbor_counts):.1f}\n")
    f.write(f"  Min: {np.min(neighbor_counts)}\n")
    f.write(f"  Max: {np.max(neighbor_counts)}\n\n")
    f.write(f"DENSITY CLASS DISTRIBUTION:\n")
    f.write(density_distribution.to_string(index=False))
    f.write(f"\n\nDENSITY vs SUBSIDENCE RATE:\n")
    if rate_col:
        f.write(density_vs_rate.to_string(index=False))
        if valid_mask.sum() > 2:
            f.write(f"\n\nPearson correlation: r={corr_coeff:.4f}, p={corr_pval:.4e}\n")
    else:
        f.write("No subsidence rate column found.\n")

print(f"  ✓ density_clustering_summary.txt")

# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n" + "="*80)
print(f"SCRIPT 06 COMPLETE")
print(f"="*80)
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
