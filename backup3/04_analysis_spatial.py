"""
SCRIPT 03: SPATIAL ANALYSIS (VISUALIZATION + COVERAGE)

Purpose
-------
Provide spatial diagnostics of the crossover dataset via plots and
coverage statistics. This script helps assess where subsidence is
measured and where data gaps remain.

Origin / Consolidation
----------------------
- Consolidates:
  - 04_visualization.py
  - 05_coverage_analysis.py

Main Tasks – Section 1 (Visualization)
--------------------------------------
- Load icesat2_repeat_track_matches.csv.
- Generate summary plots such as:
  - Histograms of elevation change (dH) and subsidence rates.
  - Spatial scatter plots of crossover locations colored by rate.
  - Uncertainty plots where available.
- Save a multi-panel PNG suitable for inclusion in a thesis or paper.

Main Tasks – Section 2 (Coverage Analysis)
------------------------------------------
- Define a regular spatial grid (e.g. ~1 km cells) over the study area.
- Bin crossover points into grid cells.
- Compute per-cell metrics:
  - number of crossovers
  - mean and standard deviation of subsidence rate
  - basic spatial summary for reporting.
- Optionally summarize coverage by reference ground track (RGT).

Inputs
------
- icesat2_repeat_track_matches.csv

Outputs
-------
- icesat2_summary_plots.png
  (multi-panel overview figure)
- spatial_coverage_grid.csv
- coverage_analysis_summary.txt

Reproducibility Notes
---------------------
Grid resolution, map projections and plotting settings affect the
visual appearance but not the underlying data. Keep grid and plotting
parameters fixed across runs when comparing different time periods or
InSAR products.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("SCRIPT 04: SPATIAL ANALYSIS - ENHANCED")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================

BASE_OUTPUT_DIR = Path(r'E:\Processing\ICeSat-2\output_data')

timestamp_file = BASE_OUTPUT_DIR / 'latest_run_timestamp.txt'

try:
    with open(timestamp_file, 'r') as f:
        RUN_TIMESTAMP = f.read().strip()
    OUTPUT_DIR = BASE_OUTPUT_DIR / f'run_{RUN_TIMESTAMP}'
except FileNotFoundError:
    print("ERROR: Timestamp file not found. Run Script 00 first.")
    sys.exit(1)

print(f"\nRun ID: {RUN_TIMESTAMP}")
print(f"Output directory: {OUTPUT_DIR}\n")

# ============================================================================
# LOAD DATA FROM SCRIPT 03
# ============================================================================

print("="*80)
print("LOADING DATA")
print("="*80 + "\n")

try:
    data = pd.read_csv(OUTPUT_DIR / 'icesat2_repeat_track_matches.csv')
    print(f"✓ Loaded: {len(data):,} crossover points")
except FileNotFoundError:
    print("ERROR: icesat2_repeat_track_matches.csv not found")
    sys.exit(1)

# Auto-detect columns
def get_column(df, *possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None

lat_col = get_column(data, 'latitude', 'lat', 'ref_lat', 'lat_avg')
lon_col = get_column(data, 'longitude', 'lon', 'ref_lon', 'lon_avg')
rate_col = get_column(data, 'dh_rate_m_per_year', 'subsidence_rate_m_per_year', 'rate_m_per_year')

if not all([lat_col, lon_col, rate_col]):
    print(f"ERROR: Could not detect required columns")
    print(f"Available: {list(data.columns)}")
    sys.exit(1)

print(f"✓ Columns detected:")
print(f"  Latitude: {lat_col}")
print(f"  Longitude: {lon_col}")
print(f"  Rate: {rate_col}\n")

# ============================================================================
# SPATIAL ANALYSIS - NEIGHBOR COUNTING
# ============================================================================

print("="*80)
print("SPATIAL ANALYSIS: NEIGHBOR COUNTING")
print("="*80 + "\n")

# Parameters
NEIGHBOR_RADIUS_KM = 4.0
EARTH_RADIUS_KM = 6371.0

# Convert to degrees (approximate)
neighbor_radius_deg = NEIGHBOR_RADIUS_KM / 111.0

print(f"Search radius: {NEIGHBOR_RADIUS_KM} km ({neighbor_radius_deg:.4f}°)\n")

# Build coordinate arrays
coords = np.column_stack([data[lon_col].values, data[lat_col].values])

print(f"Computing pairwise distances...")
# Calculate pairwise distances (in degrees, then to km)
distances = cdist(coords, coords, metric='euclidean')
distances_km = distances * 111.0

# Count neighbors within radius (excluding self)
print(f"Counting neighbors within {NEIGHBOR_RADIUS_KM} km...")
neighbor_count = np.sum(distances_km < NEIGHBOR_RADIUS_KM, axis=1) - 1

data['neighbor_count'] = neighbor_count

print(f"✓ Neighbor count computed")
print(f"  Min: {neighbor_count.min()}")
print(f"  Mean: {neighbor_count.mean():.1f}")
print(f"  Max: {neighbor_count.max()}\n")

# ============================================================================
# DENSITY CLASSIFICATION
# ============================================================================

print("="*80)
print("DENSITY CLASSIFICATION")
print("="*80 + "\n")

# Classify based on neighbor count quartiles
q25, q50, q75 = np.percentile(neighbor_count, [25, 50, 75])

density_class = pd.cut(neighbor_count, 
                       bins=[0, q25, q50, q75, np.inf],
                       labels=['Isolated', 'Sparse', 'Moderate', 'Dense'],
                       include_lowest=True)

data['density_class'] = density_class

print("Density class distribution:")
for cls in ['Isolated', 'Sparse', 'Moderate', 'Dense']:
    count = (density_class == cls).sum()
    pct = 100 * count / len(data)
    print(f"  {cls:12s}: {count:6,} ({pct:5.1f}%)")

print()

# ============================================================================
# LISA HOTSPOT ANALYSIS (Moran's I)
# ============================================================================

print("="*80)
print("LISA HOTSPOT ANALYSIS")
print("="*80 + "\n")

# Get valid data for LISA
valid_mask = ~(data[rate_col].isna() | data[lat_col].isna() | data[lon_col].isna())
valid_indices = np.where(valid_mask)[0]

print(f"Valid points for LISA: {valid_mask.sum():,}")

# Compute spatial weights (inverse distance, within radius)
def compute_weights_matrix(coords, radius_km=4.0):
    """Compute spatial weights matrix (inverse distance, within radius)."""
    distances = cdist(coords, coords, metric='euclidean')
    distances_km = distances * 111.0
    
    # Set to NaN outside radius and on diagonal
    weights = np.zeros_like(distances_km)
    within_radius = (distances_km <= radius_km) & (distances_km > 0)
    weights[within_radius] = 1.0 / (distances_km[within_radius] + 1e-10)
    
    # Row normalize
    row_sums = weights.sum(axis=1)
    row_sums[row_sums == 0] = 1
    weights = weights / row_sums[:, np.newaxis]
    
    return weights

coords_valid = coords[valid_mask]
rates_valid = data.loc[valid_mask, rate_col].values * 1000  # mm/yr

print(f"Computing spatial weights matrix...")
W = compute_weights_matrix(coords_valid, radius_km=NEIGHBOR_RADIUS_KM)

print(f"Computing Moran's I (spatial autocorrelation)...")

# Compute Moran's I
def morans_i(values, weights):
    """Compute Moran's I statistic."""
    n = len(values)
    y = values - values.mean()
    
    W_sum = weights.sum()
    
    numerator = (y @ weights) @ y
    denominator = y @ y
    
    I = (n / W_sum) * (numerator / denominator)
    
    return I

morans_I = morans_i(rates_valid, W)
print(f"✓ Moran's I: {morans_I:.4f}")
print(f"  (Positive = clustering, Negative = dispersion)\n")

# Compute Local Indicators of Spatial Association (LISA)
print(f"Computing Local Moran's I (LISA) for each point...")

local_I = np.zeros(len(rates_valid))
local_pvalues = np.zeros(len(rates_valid))

y = rates_valid - rates_valid.mean()
W_sum = W.sum()

for i in range(len(rates_valid)):
    Ii = (y[i] * (W[i] @ y)) / (np.sum(y**2) / len(y))
    local_I[i] = Ii
    
    # Approximate p-value (simplified)
    local_pvalues[i] = 2 * (1 - stats.norm.cdf(np.abs(Ii)))

# Classify into hotspots
mean_rate = rates_valid.mean()
std_rate = rates_valid.std()

hotspot_class = np.full(len(rates_valid), 'NS', dtype=object)  # Not Significant

# HH: High-High (hotspot)
hh_mask = (rates_valid > mean_rate) & (local_I > 0) & (local_pvalues < 0.05)
hotspot_class[hh_mask] = 'HH'

# LL: Low-Low (cold spot)
ll_mask = (rates_valid < mean_rate) & (local_I > 0) & (local_pvalues < 0.05)
hotspot_class[ll_mask] = 'LL'

# HL: High-Low (outlier)
hl_mask = (rates_valid > mean_rate) & (local_I < 0) & (local_pvalues < 0.05)
hotspot_class[hl_mask] = 'HL'

# LH: Low-High (outlier)
lh_mask = (rates_valid < mean_rate) & (local_I < 0) & (local_pvalues < 0.05)
hotspot_class[lh_mask] = 'LH'

# Add to full dataset
lisa_cluster = np.full(len(data), np.nan, dtype=object)
lisa_cluster[valid_mask] = hotspot_class
data['lisa_cluster'] = lisa_cluster

print(f"✓ LISA classification:")
for cls in ['HH', 'LL', 'HL', 'LH', 'NS']:
    count = (hotspot_class == cls).sum()
    if count > 0:
        pct = 100 * count / len(hotspot_class)
        print(f"  {cls}: {count:6,} ({pct:5.1f}%)")

print()

# ============================================================================
# SAVE DENSITY CLUSTERS
# ============================================================================

print("="*80)
print("SAVING OUTPUTS")
print("="*80 + "\n")

# Save complete dataset with clusters
output_file = OUTPUT_DIR / 'density_clusters_all.csv'
data.to_csv(output_file, index=False)
print(f"✓ Saved: density_clusters_all.csv ({len(data):,} rows)")

# Save summary statistics
summary_data = {
    'Metric': [
        'Total Points',
        'Valid Points (LISA)',
        'Mean Neighbor Count',
        'Mean Subsidence (mm/yr)',
        'Std Subsidence (mm/yr)',
        'Morans I (Global)',
        'Isolated Points',
        'Sparse Points',
        'Moderate Density',
        'Dense Points',
        'HH Hotspots',
        'LL Cold Spots',
        'HL Outliers',
        'LH Outliers',
        'Not Significant'
    ],
    'Value': [
        f"{len(data):,}",
        f"{valid_mask.sum():,}",
        f"{neighbor_count.mean():.1f}",
        f"{rates_valid.mean():.2f}",
        f"{rates_valid.std():.2f}",
        f"{morans_I:.4f}",
        f"{(density_class == 'Isolated').sum()}",
        f"{(density_class == 'Sparse').sum()}",
        f"{(density_class == 'Moderate').sum()}",
        f"{(density_class == 'Dense').sum()}",
        f"{(hotspot_class == 'HH').sum()}",
        f"{(hotspot_class == 'LL').sum()}",
        f"{(hotspot_class == 'HL').sum()}",
        f"{(hotspot_class == 'LH').sum()}",
        f"{(hotspot_class == 'NS').sum()}"
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_file = OUTPUT_DIR / '04_spatial_analysis_summary.csv'
summary_df.to_csv(summary_file, index=False)
print(f"✓ Saved: 04_spatial_analysis_summary.csv")

# ============================================================================
# COMPLETION
# ============================================================================

print("="*80)
print("SCRIPT 04 - COMPLETE")
print("="*80)

print(f"\n✓ OUTPUTS:")
print(f"  • density_clusters_all.csv ({len(data):,} points with spatial metadata)")
print(f"  • 04_spatial_analysis_summary.csv (summary statistics)")

print(f"\n✓ SPATIAL ANALYSIS COMPLETE")
print(f"  • Neighbors: {neighbor_count.min():.0f} - {neighbor_count.max():.0f}")
print(f"  • Density classes: Isolated, Sparse, Moderate, Dense")
print(f"  • LISA hotspots: HH, LL, HL, LH, NS")
print(f"  • Moran's I: {morans_I:.4f}")

print(f"\nReady for Script 05 (Temporal Analysis)")
print("\n" + "="*80 + "\n")

sys.exit(0)
