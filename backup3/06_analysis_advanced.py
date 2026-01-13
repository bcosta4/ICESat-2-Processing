"""
================================================================================
SCRIPT 06: ADVANCED SPATIAL ANALYSIS WITH UNCERTAINTY QUANTIFICATION
================================================================================

INCLUDES:
- Memory-optimized KDTree processing (no MemoryError on 2M+ points)
- LISA hotspot clustering
- Grid zonal statistics
- UNCERTAINTY QUANTIFICATION (4 components, RSS combination)

NEW: Uncertainty calculation integrated directly
  - Measurement uncertainty (quality-dependent)
  - Temporal sampling uncertainty (reduces with √N)
  - Datum/reference frame uncertainty
  - GIA model uncertainty
  
Output includes: rate ± uncertainty, 95% CI, confidence class (High/Medium/Low)

================================================================================
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
import warnings
import sys

warnings.filterwarnings('ignore')

print("="*80)
print("SCRIPT 06: ADVANCED SPATIAL ANALYSIS WITH UNCERTAINTY")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_OUTPUT_DIR = Path(r'E:\Processing\ICeSat-2\output_data')

# Memory optimization
CHUNK_SIZE = 50000
NEIGHBOR_RADIUS_M = 4.0

# LISA settings
LISA_RADIUS_M = 100.0
DENSITY_CLASSES = ['Isolated (0-1)', 'Sparse (1-5)', 'Moderate (5-20)', 'Dense (20+)']

# Grid analysis
GRID_CELL_SIZE_M = 500.0

# UNCERTAINTY PARAMETERS (mm/yr) - tuned for coastal subsidence
UNC_MEASUREMENT_MM_YR = 20.0     # ±20 mm/yr baseline
UNC_DATUM_MM_YR = 10.0           # ±10 mm/yr datum shift
UNC_GIA_MM_YR = 2.0              # ±2 mm/yr GIA trend

print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

# Get run timestamp
try:
    with open(BASE_OUTPUT_DIR / 'latest_run_timestamp.txt', 'r') as f:
        RUN_TIMESTAMP = f.read().strip()
    OUTPUT_DIR = BASE_OUTPUT_DIR / f'run_{RUN_TIMESTAMP}'
except FileNotFoundError:
    print("ERROR: Timestamp file not found. Run Script 00 first.")
    sys.exit(1)

print(f"\nRun ID: {RUN_TIMESTAMP}")
print(f"Output directory: {OUTPUT_DIR}\n")

# Load main data
try:
    data = pd.read_csv(OUTPUT_DIR / 'density_clusters_all.csv')
    print(f"✓ Loaded {len(data):,} crossover points")
except FileNotFoundError:
    print("ERROR: density_clusters_all.csv not found. Run Script 05b first.")
    sys.exit(1)

# Load InSAR data (optional)
insar_data = None
try:
    insar_data = pd.read_csv(OUTPUT_DIR / 'icesat2_insar_integrated.csv')
    print(f"✓ Loaded InSAR data ({len(insar_data):,} points)")
except FileNotFoundError:
    print("⚠ InSAR data not found (optional)")

# Detect columns
print("\nDetecting coordinate system...")
lat_col = None
lon_col = None
rate_col = None

for col in data.columns:
    if 'lat' in col.lower():
        lat_col = col
    if 'lon' in col.lower():
        lon_col = col
    if 'subsidence' in col.lower() or 'rate' in col.lower():
        if 'per_year' in col.lower() or 'mm' in col.lower():
            rate_col = col

if not all([lat_col, lon_col, rate_col]):
    print("ERROR: Could not auto-detect required columns")
    print(f"Available: {list(data.columns)[:10]}")
    sys.exit(1)

print(f"✓ Using: Latitude={lat_col}, Longitude={lon_col}, Rate={rate_col}")

# ============================================================================
# SECTION 1: NEIGHBOR COUNTING & DENSITY CLASSIFICATION
# ============================================================================

print("\n" + "="*80)
print("SECTION 1: NEIGHBOR COUNTING & DENSITY CLASSIFICATION")
print("="*80)

# Extract coordinates
coords = data[[lon_col, lat_col]].values
n_points = len(coords)

print(f"\nBuilding KDTree for {n_points:,} points...")
tree = cKDTree(coords)
print("✓ KDTree built successfully")

# Process in chunks to avoid memory error
print(f"\nProcessing in chunks of {CHUNK_SIZE:,} points...")
neighbor_counts = np.zeros(n_points, dtype=int)

for chunk_start in range(0, n_points, CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, n_points)
    chunk_size = chunk_end - chunk_start
    
    chunk_coords = coords[chunk_start:chunk_end]
    
    try:
        distances, indices = tree.query(chunk_coords, k=min(1000, n_points))
        
        if distances.ndim == 1:
            distances = distances.reshape(1, -1)
            indices = indices.reshape(1, -1)
        
        for i in range(chunk_size):
            neighbors_in_radius = np.sum(distances[i] <= NEIGHBOR_RADIUS_M) - 1
            neighbor_counts[chunk_start + i] = max(0, neighbors_in_radius)
        
        progress = min(chunk_end, n_points)
        print(f"  ✓ Processed {progress:,} / {n_points:,} points")
        
    except Exception as e:
        print(f"  ⚠ Error in chunk {chunk_start}-{chunk_end}: {e}")
        for idx in range(chunk_start, chunk_end):
            try:
                distances = tree.query(coords[idx:idx+1], k=min(500, n_points))[0]
                if distances.ndim == 0:
                    distances = np.array([distances])
                neighbor_counts[idx] = max(0, np.sum(distances <= NEIGHBOR_RADIUS_M) - 1)
            except:
                neighbor_counts[idx] = 0

print(f"✓ Neighbor counting complete")
print(f"  Mean neighbors: {neighbor_counts.mean():.1f}")
print(f"  Max neighbors: {neighbor_counts.max()}")

data['neighbor_count'] = neighbor_counts

# Classify density
print("\nClassifying density classes...")
def classify_density(count):
    if count == 0 or count == 1:
        return 'Isolated (0-1)'
    elif count <= 5:
        return 'Sparse (1-5)'
    elif count <= 20:
        return 'Moderate (5-20)'
    else:
        return 'Dense (20+)'

data['density_class'] = data['neighbor_count'].apply(classify_density)

density_counts = data['density_class'].value_counts()
print("\nDensity class distribution:")
for dc in DENSITY_CLASSES:
    count = density_counts.get(dc, 0)
    pct = 100 * count / len(data)
    print(f"  {dc:20s}: {count:8,} points ({pct:5.1f}%)")

# ============================================================================
# SECTION 2: UNCERTAINTY QUANTIFICATION
# ============================================================================

print("\n" + "="*80)
print("SECTION 2: UNCERTAINTY QUANTIFICATION")
print("="*80)

print("\nCalculating uncertainty components...\n")

# Measurement uncertainty (quality-dependent if available)
if 'quality_ph' in data.columns:
    quality = data['quality_ph'].values
    measurement_unc = np.where(
        quality >= 3,
        UNC_MEASUREMENT_MM_YR / np.sqrt(2),  # Good quality: ±14 mm/yr
        UNC_MEASUREMENT_MM_YR * np.sqrt(1.5) # Poor quality: ±25 mm/yr
    )
else:
    measurement_unc = np.full(n_points, UNC_MEASUREMENT_MM_YR)

# Temporal sampling uncertainty (reduces with √N observations)
if 'num_observations' in data.columns:
    n_obs = np.maximum(data['num_observations'].values, 1)
    temporal_unc = UNC_MEASUREMENT_MM_YR / np.sqrt(n_obs)
else:
    temporal_unc = np.full(n_points, 15.0)

# Datum/reference frame uncertainty
datum_unc = np.full(n_points, UNC_DATUM_MM_YR)

# GIA model uncertainty
gia_unc = np.full(n_points, UNC_GIA_MM_YR)

# Total uncertainty (root sum of squares)
total_unc = np.sqrt(
    measurement_unc**2 + 
    temporal_unc**2 + 
    datum_unc**2 + 
    gia_unc**2
)

# Convert rate to mm/yr if needed
if data[rate_col].max() < 10:
    rate_mm_yr = data[rate_col] * 1000
else:
    rate_mm_yr = data[rate_col]

# Add uncertainty columns
data['subsidence_rate_mm_yr'] = rate_mm_yr
data['uncertainty_mm_yr'] = total_unc
data['ci_lower_mm_yr'] = rate_mm_yr - 1.96 * total_unc
data['ci_upper_mm_yr'] = rate_mm_yr + 1.96 * total_unc

# Confidence classification
unc_low = np.percentile(total_unc, 33)
unc_high = np.percentile(total_unc, 67)

def confidence_class(unc):
    if unc <= unc_low:
        return 'High'
    elif unc <= unc_high:
        return 'Medium'
    else:
        return 'Low'

data['confidence'] = data['uncertainty_mm_yr'].apply(confidence_class)

# Print statistics
print(f"✓ Uncertainty quantification complete\n")
print(f"Uncertainty statistics (mm/yr):")
print(f"  Measurement uncertainty:  {measurement_unc.mean():6.2f} ± {measurement_unc.std():6.2f}")
print(f"  Temporal sampling unc:    {temporal_unc.mean():6.2f} ± {temporal_unc.std():6.2f}")
print(f"  Datum uncertainty:        {datum_unc.mean():6.2f}")
print(f"  GIA uncertainty:          {gia_unc.mean():6.2f}")
print(f"  TOTAL uncertainty:        {total_unc.mean():6.2f} ± {total_unc.std():6.2f}")

conf_counts = pd.Series(data['confidence']).value_counts()
print(f"\nConfidence distribution:")
for conf in ['High', 'Medium', 'Low']:
    count = conf_counts.get(conf, 0)
    pct = 100 * count / len(data)
    print(f"  {conf:10s}: {count:8,} points ({pct:5.1f}%)")

# Uncertainty budget
total_unc_sq = total_unc.mean()**2
print(f"\nUncertainty budget (% contribution):")
print(f"  Measurement:  {100 * (measurement_unc.mean()**2 / total_unc_sq):6.1f}%")
print(f"  Temporal:     {100 * (temporal_unc.mean()**2 / total_unc_sq):6.1f}%")
print(f"  Datum:        {100 * (datum_unc.mean()**2 / total_unc_sq):6.1f}%")
print(f"  GIA:          {100 * (gia_unc.mean()**2 / total_unc_sq):6.1f}%")

# ============================================================================
# SECTION 3: LISA CLUSTERING
# ============================================================================

print("\n" + "="*80)
print("SECTION 3: LISA HOTSPOT CLUSTERING")
print("="*80)

print("\nClassifying LISA clusters...")
data['lisa_cluster'] = 'NS'

# Standardize rates
mean_rate = data[rate_col].mean()
std_rate = data[rate_col].std()
data['z_rate'] = (data[rate_col] - mean_rate) / std_rate

# LISA classification
for idx in range(n_points):
    if idx % 100000 == 0:
        print(f"  Processing {idx:,} / {n_points:,}...")
    
    neighbors = tree.query_ball_point(coords[idx:idx+1], r=LISA_RADIUS_M)[0]
    
    if len(neighbors) < 2:
        data.loc[idx, 'lisa_cluster'] = 'NS'
        continue
    
    z_focal = data.loc[idx, 'z_rate']
    z_neighbors = data.loc[neighbors, 'z_rate'].mean()
    
    if z_focal > 0 and z_neighbors > 0:
        data.loc[idx, 'lisa_cluster'] = 'HH'
    elif z_focal < 0 and z_neighbors < 0:
        data.loc[idx, 'lisa_cluster'] = 'LL'
    elif z_focal > 0 and z_neighbors < 0:
        data.loc[idx, 'lisa_cluster'] = 'HL'
    elif z_focal < 0 and z_neighbors > 0:
        data.loc[idx, 'lisa_cluster'] = 'LH'
    else:
        data.loc[idx, 'lisa_cluster'] = 'NS'

print("✓ LISA classification complete")
lisa_counts = data['lisa_cluster'].value_counts()
print("\nLISA cluster distribution:")
for cluster in ['HH', 'LL', 'HL', 'LH', 'NS']:
    count = lisa_counts.get(cluster, 0)
    pct = 100 * count / len(data)
    label = {
        'HH': 'Hotspot (High-High)',
        'LL': 'Cold Spot (Low-Low)',
        'HL': 'Outlier (High-Low)',
        'LH': 'Outlier (Low-High)',
        'NS': 'Not Significant'
    }
    print(f"  {label[cluster]:30s}: {count:8,} points ({pct:5.1f}%)")

# ============================================================================
# SECTION 4: GRID ZONAL STATISTICS
# ============================================================================

print("\n" + "="*80)
print("SECTION 4: GRID ZONAL STATISTICS WITH UNCERTAINTY")
print("="*80)

print(f"\nAggregating to {GRID_CELL_SIZE_M:.0f}m grid cells...")

# Create grid
lon_min, lon_max = data[lon_col].min(), data[lon_col].max()
lat_min, lat_max = data[lat_col].min(), data[lat_col].max()

lon_range_m = (lon_max - lon_min) * 111000
lat_range_m = (lat_max - lat_min) * 111000

n_lon_cells = int(np.ceil(lon_range_m / GRID_CELL_SIZE_M))
n_lat_cells = int(np.ceil(lat_range_m / GRID_CELL_SIZE_M))

# Assign grid cells
data['grid_lon'] = np.floor((data[lon_col] - lon_min) / ((lon_max - lon_min) / n_lon_cells)).astype(int)
data['grid_lat'] = np.floor((data[lat_col] - lat_min) / ((lat_max - lat_min) / n_lat_cells)).astype(int)

# Compute grid statistics including uncertainty
grid_stats = []

for (grid_lon, grid_lat), group in data.groupby(['grid_lon', 'grid_lat']):
    n_pts = len(group)
    
    # Rate statistics
    mean_rate = group[rate_col].mean()
    std_rate = group[rate_col].std()
    
    # Uncertainty in grid (combines component + spatial)
    mean_unc = group['uncertainty_mm_yr'].mean()
    # Spatial heterogeneity = std of point values minus measurement error
    spatial_unc = np.sqrt(np.maximum(std_rate**2 - (mean_unc/1000)**2, 0))
    total_grid_unc = np.sqrt((mean_unc**2) + (spatial_unc * 1000)**2)
    
    grid_stats.append({
        'grid_lon': grid_lon,
        'grid_lat': grid_lat,
        'lon_center': lon_min + (grid_lon + 0.5) * (lon_max - lon_min) / n_lon_cells,
        'lat_center': lat_min + (grid_lat + 0.5) * (lat_max - lat_min) / n_lat_cells,
        'n_points': n_pts,
        'mean_rate_mm_yr': mean_rate * 1000 if mean_rate < 10 else mean_rate,
        'std_rate_mm_yr': std_rate * 1000 if std_rate < 10 else std_rate,
        'uncertainty_mm_yr': total_grid_unc,
        'relative_unc_pct': 100 * total_grid_unc / np.abs(mean_rate * 1000) if mean_rate != 0 else np.nan,
        'ci_lower_mm_yr': (mean_rate * 1000 if mean_rate < 10 else mean_rate) - 1.96 * total_grid_unc,
        'ci_upper_mm_yr': (mean_rate * 1000 if mean_rate < 10 else mean_rate) + 1.96 * total_grid_unc,
    })

grid_stats_df = pd.DataFrame(grid_stats)

print(f"✓ Created {len(grid_stats_df):,} grid cells")
print(f"  Grid dimensions: {n_lon_cells} × {n_lat_cells} cells")
print(f"\nGrid uncertainty statistics:")
print(f"  Mean uncertainty:       {grid_stats_df['uncertainty_mm_yr'].mean():6.2f} mm/yr")
print(f"  Mean relative unc:      {grid_stats_df['relative_unc_pct'].mean():6.1f}%")
print(f"  High reliability zones: {len(grid_stats_df[grid_stats_df['relative_unc_pct']<30]):,} cells (<30%)")
print(f"  Low reliability zones:  {len(grid_stats_df[grid_stats_df['relative_unc_pct']>100]):,} cells (>100%)")

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

print("\n" + "="*80)
print("SAVING OUTPUTS")
print("="*80)

# Save data with uncertainty
output_file = OUTPUT_DIR / 'density_clusters_all.csv'
data.to_csv(output_file, index=False)
print(f"✓ Saved: {output_file.name}")
print(f"  Columns include: confidence, uncertainty_mm_yr, ci_lower_mm_yr, ci_upper_mm_yr")

# Save grid statistics
grid_file = OUTPUT_DIR / '06_grid_zonal_statistics.csv'
grid_stats_df.to_csv(grid_file, index=False)
print(f"✓ Saved: {grid_file.name}")
print(f"  Columns include: uncertainty_mm_yr, relative_unc_pct, ci_lower/upper_mm_yr")

# Summary report
summary_text = f"""
================================================================================
SCRIPT 06: ADVANCED ANALYSIS WITH UNCERTAINTY - SUMMARY
================================================================================

Generated: {pd.Timestamp.now()}
Run ID: {RUN_TIMESTAMP}

DATASET OVERVIEW:
  Total points: {len(data):,}
  Coordinates: {lat_col}, {lon_col}
  Rate column: {rate_col}

UNCERTAINTY QUANTIFICATION:
  Components: Measurement, Temporal, Datum, GIA (RSS combined)
  Mean total uncertainty: {total_unc.mean():.2f} ± {total_unc.std():.2f} mm/yr
  
  Confidence distribution:
    High:     {len(data[data['confidence']=='High']):8,} ({100*len(data[data['confidence']=='High'])/len(data):5.1f}%)
    Medium:   {len(data[data['confidence']=='Medium']):8,} ({100*len(data[data['confidence']=='Medium'])/len(data):5.1f}%)
    Low:      {len(data[data['confidence']=='Low']):8,} ({100*len(data[data['confidence']=='Low'])/len(data):5.1f}%)

SPATIAL ANALYSIS:
  Density classes:
    Isolated:  {len(data[data['density_class']=='Isolated (0-1)']):8,}
    Sparse:    {len(data[data['density_class']=='Sparse (1-5)']):8,}
    Moderate:  {len(data[data['density_class']=='Moderate (5-20)']):8,}
    Dense:     {len(data[data['density_class']=='Dense (20+)']):8,}
    
  LISA clusters:
    Hotspots (HH):     {len(data[data['lisa_cluster']=='HH']):8,}
    Cold spots (LL):   {len(data[data['lisa_cluster']=='LL']):8,}
    Outliers (HL/LH):  {len(data[(data['lisa_cluster']=='HL') | (data['lisa_cluster']=='LH')]):8,}
    Not significant:   {len(data[data['lisa_cluster']=='NS']):8,}

GRID ANALYSIS:
  Grid cell size: {GRID_CELL_SIZE_M:.0f} m
  Grid dimensions: {n_lon_cells} × {n_lat_cells} cells
  Non-empty cells: {len(grid_stats_df):,}
  
  Grid reliability:
    High reliability (<30% unc):   {len(grid_stats_df[grid_stats_df['relative_unc_pct']<30]):,} cells
    Medium reliability (30-100%):  {len(grid_stats_df[(grid_stats_df['relative_unc_pct']>=30) & (grid_stats_df['relative_unc_pct']<=100)]):,} cells
    Low reliability (>100%):       {len(grid_stats_df[grid_stats_df['relative_unc_pct']>100]):,} cells

UNCERTAINTY BUDGET:
  Measurement:  {100 * (measurement_unc.mean()**2 / total_unc_sq):6.1f}%
  Temporal:     {100 * (temporal_unc.mean()**2 / total_unc_sq):6.1f}%
  Datum:        {100 * (datum_unc.mean()**2 / total_unc_sq):6.1f}%
  GIA:          {100 * (gia_unc.mean()**2 / total_unc_sq):6.1f}%

USAGE NOTES:
  - All subsidence rates should be reported as: value ± uncertainty (95% CI)
  - Use 'confidence' column to filter high-quality data
  - Grid 'relative_unc_pct' shows reliability (use <50% for claims)
  - ci_lower/upper_mm_yr provide 95% confidence intervals

================================================================================
"""

summary_file = OUTPUT_DIR / '06_analysis_summary.txt'
with open(summary_file, 'w') as f:
    f.write(summary_text)
print(f"\n✓ Saved: 06_analysis_summary.txt")

print("\n" + "="*80)
print("SCRIPT 06 COMPLETE - DATA READY FOR SCRIPT 07")
print("="*80)
print("\nKey outputs:")
print("  density_clusters_all.csv now includes:")
print("    - subsidence_rate_mm_yr")
print("    - uncertainty_mm_yr")
print("    - ci_lower/upper_mm_yr (95% CI)")
print("    - confidence (High/Medium/Low)")
print("\n  06_grid_zonal_statistics.csv now includes:")
print("    - uncertainty_mm_yr")
print("    - relative_unc_pct (% of rate magnitude)")
print("    - ci_lower/upper_mm_yr")

sys.exit(0)
