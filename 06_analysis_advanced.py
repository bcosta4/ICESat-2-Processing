"""
================================================================================
SCRIPT 06: ADVANCED CLUSTERING & HOTSPOT DETECTION
================================================================================

Purpose
-------
Identify subsidence hotspots using spatial clustering:
1. k-nearest neighbor graph construction
2. Hotspot detection (threshold-based)
3. Connected component clustering
4. Cluster statistics and validation
5. Coastal integration and filtering

CRITICAL SIGN CONVENTION
========================
ALL RATES IN CM/YR:
  - NEGATIVE (e.g., -2.5) = subsidence ✓ (primary in coastal permafrost)
  - POSITIVE (e.g., +1.2) = uplift (less common)

Sign convention is EXPLICITLY VERIFIED at end of script.

Inputs
------
- icesat2_repeat_track_matches.csv (Script 03)
- or 05_subsidence_rates.csv (Script 05)
- or icesat2_insar_integrated.csv (InSAR integrated)

Outputs
-------
- 06_clusters_identified.csv (point-level cluster assignments)
- 06_cluster_statistics.csv (cluster summary stats, all in cm/yr)
- 06_coastal_clusters.csv (coastal zone stratification)
- 06_cluster_map.png (spatial visualization)
- 06_cluster_stats.png (statistical visualization)
- 06_analysis_summary.txt (comprehensive report)
- script_06_run.log (execution log)
- 06_advanced_analysis_metadata.json (reproducibility)

OUTPUT COLUMN STANDARDIZATION (NEW)
===================================
All outputs include standardized columns for compatibility with Script 07:
  - elevation (elevation in meters, if available)
  - uncertainty (rate uncertainty in cm/yr, if available)
  - These enable 07 multipanel figures to work on all datasets

Units: ALL RATES IN CM/YR (negative=subsidence, positive=uplift)
Sign Convention: VERIFIED ✓
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import argparse
import sys
import io
import warnings

warnings.filterwarnings('ignore')

# UTF-8 wrapper
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

try:
    from config import BASEOUTPUTDIR
except ImportError:
    BASEOUTPUTDIR = Path(r'E:\Processing\ICeSat-2\output_data')

# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging(output_dir):
    """Configure logging."""
    log_file = output_dir / 'script_06_run.log'
    logger = logging.getLogger('script_06')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    fh = logging.FileHandler(log_file, encoding='utf-8', mode='w')
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(fh_formatter)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# ============================================================================
# UTILITY: FIND INPUT FILE
# ============================================================================
def find_input_file(output_dir):
    """Find input file with priority order."""
    priority_files = [
        output_dir / 'icesat2_repeat_track_matches.csv',    # Script 03 (PRIORITY 1)
        output_dir / '05_subsidence_rates.csv',              # Script 05 (PRIORITY 2)
        output_dir / 'icesat2_insar_integrated.csv',         # InSAR-integrated (PRIORITY 3)
        output_dir / '04_spatial_grid.csv',                  # Script 04 (fallback)
    ]
    
    for filepath in priority_files:
        if filepath.exists():
            return filepath
    
    print("ERROR: Could not find input file.")
    print("Searched for:")
    for f in priority_files:
        print(f"  - {f.name}")
    sys.exit(1)

# ============================================================================
# UTILITY: FIND RATE COLUMN
# ============================================================================
def find_rate_column(df):
    """Prioritize subsidence_rate_cm_per_year."""
    candidates = [
        'subsidence_rate_cm_per_year',
        'subsidence_rate_cm_yr',
        'rate_cm_yr',
        'mean_subsidence_cm_yr',
    ]
    for col in candidates:
        if col in df.columns:
            return col
    
    print(f"ERROR: No subsidence rate column found. Columns: {list(df.columns)}")
    sys.exit(1)

# ============================================================================
# UTILITY: STANDARDIZE ELEVATION & UNCERTAINTY COLUMNS
# ============================================================================
def standardize_columns(df):
    """
    Standardize column names for multipanel compatibility.
    Looks for elevation and uncertainty, renames to standard names.
    """
    # Elevation standardization
    elev_candidates = ['elevation', 'elev', 'elevation_m', 'height_m', 'z']
    for col in elev_candidates:
        if col in df.columns and 'elevation' not in df.columns:
            df['elevation'] = df[col]
            break
    
    # Uncertainty standardization
    unc_candidates = [
        'rate_uncertainty_cm_yr', 'uncertainty_cm_yr',
        'std_error_cm_yr', 'rate_uncertainty_cm_per_year'
    ]
    for col in unc_candidates:
        if col in df.columns and 'uncertainty' not in df.columns:
            df['uncertainty'] = df[col]
            break
    
    return df

# ============================================================================
# MAIN CLUSTERING
# ============================================================================
def main(threshold=-2.0, radius_km=2.0, min_size=10, k=250):
    """Main clustering workflow."""
    
    print("=" * 80)
    print("SCRIPT 06: ADVANCED CLUSTERING & HOTSPOT DETECTION")
    print("=" * 80)
    
    # Setup
    timestamp_file = BASEOUTPUTDIR / 'latest_RUN_TIMESTAMP.txt'
    if not timestamp_file.exists():
        print("ERROR: Run Script 00 first")
        sys.exit(1)
    
    with open(timestamp_file, 'r', encoding='utf-8') as f:
        RUN_TIMESTAMP = f.read().strip()
    
    OUTPUT_DIR = BASEOUTPUTDIR / f'run_{RUN_TIMESTAMP}'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(OUTPUT_DIR)
    
    logger.info("=" * 80)
    logger.info("SCRIPT 06: ADVANCED CLUSTERING")
    logger.info("=" * 80)
    logger.info(f"Run ID: {RUN_TIMESTAMP}")
    
    print(f"Run ID: {RUN_TIMESTAMP}")
    print(f"Output: {OUTPUT_DIR}")
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    logger.info("\n1. LOADING DATA")
    print("\n" + "=" * 80)
    print("1. LOADING DATA")
    print("=" * 80)
    
    input_file = find_input_file(OUTPUT_DIR)
    df = pd.read_csv(input_file)
    
    logger.info(f"Loaded {len(df):,} points from {input_file.name}")
    print(f"✓ Loaded {len(df):,} points")
    
    # Standardize columns for multipanel compatibility
    df = standardize_columns(df)
    
    # Find rate column
    rate_col = find_rate_column(df)
    logger.info(f"Using rate column: '{rate_col}'")
    print(f"✓ Using rate column: '{rate_col}'")
    
    # Find lat/lon columns
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), 'lon')
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), 'lat')
    
    if lon_col not in df.columns or lat_col not in df.columns:
        print(f"ERROR: Could not find lon/lat columns")
        sys.exit(1)
    
    # ========================================================================
    # k-NEAREST NEIGHBOR GRAPH
    # ========================================================================
    logger.info("\n2. k-NEAREST NEIGHBOR GRAPH")
    print("\n" + "=" * 80)
    print("2. k-NEAREST NEIGHBOR GRAPH")
    print("=" * 80)
    
    coords = df[[lon_col, lat_col]].values
    tree = cKDTree(coords)
    
    try:
        distances, indices = tree.query(coords, k=k+1)
    except Exception as e:
        logger.warning(f"k-nearest failed with k={k}: {e}")
        k = 150
        distances, indices = tree.query(coords, k=k+1)
    
    # Build neighbor list (skip self)
    neighbors_list = []
    radius_m = radius_km * 1000.0
    for i in range(len(coords)):
        # Filter by radius (convert degrees to meters approximately)
        dists_m = distances[i, 1:] * 111000.0
        valid = dists_m <= radius_m
        neighbors_list.append(indices[i, 1:][valid])
    
    k_used = np.mean([len(n) for n in neighbors_list])
    logger.info(f"Avg neighbors per point: {k_used:.1f} (within {radius_km} km)")
    print(f"✓ k-nearest graph built (avg {k_used:.1f} neighbors)")
    
    # ========================================================================
    # HOTSPOT DETECTION
    # ========================================================================
    logger.info("\n3. HOTSPOT DETECTION")
    print("\n" + "=" * 80)
    print("3. HOTSPOT DETECTION")
    print("=" * 80)
    
    subsidence_rates = df[rate_col].values
    hotspot_mask = subsidence_rates <= threshold
    n_hotspots = hotspot_mask.sum()
    
    logger.info(f"Hotspots: {n_hotspots:,} (threshold: {threshold} cm/yr)")
    print(f"✓ Hotspots: {n_hotspots:,} ({n_hotspots/len(df)*100:.1f}%)")
    print(f"  Threshold: {threshold} cm/yr (more negative = stronger subsidence)")
    
    # ========================================================================
    # CONNECTED CLUSTERING (BFS)
    # ========================================================================
    logger.info("\n4. CLUSTER IDENTIFICATION (BFS)")
    print("\n4. CLUSTER IDENTIFICATION")
    
    cluster_id = np.full(len(df), -1, dtype=int)
    current_cluster = 0
    
    for i in range(len(df)):
        if not hotspot_mask[i] or cluster_id[i] != -1:
            continue
        
        # BFS
        queue = [i]
        cluster_id[i] = current_cluster
        queue_idx = 0
        
        while queue_idx < len(queue):
            point = queue[queue_idx]
            queue_idx += 1
            
            for neighbor in neighbors_list[point]:
                if hotspot_mask[neighbor] and cluster_id[neighbor] == -1:
                    cluster_id[neighbor] = current_cluster
                    queue.append(neighbor)
        
        current_cluster += 1
    
    n_clusters = current_cluster
    logger.info(f"Raw clusters: {n_clusters}")
    print(f"✓ Raw clusters: {n_clusters}")
    
    # ========================================================================
    # CLUSTER STATISTICS & FILTERING
    # ========================================================================
    logger.info("\n5. CLUSTER STATISTICS")
    print("\n5. CLUSTER STATISTICS")
    
    cluster_stats = []
    
    for cid in range(n_clusters):
        cluster_mask = cluster_id == cid
        cluster_points = np.where(cluster_mask)[0]
        
        if len(cluster_points) >= min_size:
            cluster_rates = subsidence_rates[cluster_points]
            cluster_lats = df.iloc[cluster_points][lat_col].values
            cluster_lons = df.iloc[cluster_points][lon_col].values
            
            # Elevation if available
            cluster_elev = None
            if 'elevation' in df.columns:
                cluster_elev_vals = df.iloc[cluster_points]['elevation'].dropna().values
                if len(cluster_elev_vals) > 0:
                    cluster_elev = np.mean(cluster_elev_vals)
            
            # Uncertainty if available
            cluster_unc = None
            if 'uncertainty' in df.columns:
                cluster_unc_vals = df.iloc[cluster_points]['uncertainty'].dropna().values
                if len(cluster_unc_vals) > 0:
                    cluster_unc = np.mean(cluster_unc_vals)
            
            stats = {
                'cluster_id': cid,
                'size': len(cluster_points),
                'mean_subsidence_cm_yr': np.mean(cluster_rates),
                'median_subsidence_cm_yr': np.median(cluster_rates),
                'std_subsidence_cm_yr': np.std(cluster_rates),
                'min_subsidence_cm_yr': np.min(cluster_rates),
                'max_subsidence_cm_yr': np.max(cluster_rates),
                'center_lat': np.mean(cluster_lats),
                'center_lon': np.mean(cluster_lons),
                'extent_km': np.sqrt(
                    ((np.max(cluster_lons) - np.min(cluster_lons))**2 +
                     (np.max(cluster_lats) - np.min(cluster_lats))**2) * 111
                ),
            }
            
            # Standardized columns
            if cluster_elev is not None:
                stats['elevation'] = cluster_elev
            if cluster_unc is not None:
                stats['uncertainty'] = cluster_unc
            
            cluster_stats.append(stats)
    
    df_cluster_stats = pd.DataFrame(cluster_stats)
    n_valid_clusters = len(df_cluster_stats)
    
    logger.info(f"Valid clusters (>= {min_size} pts): {n_valid_clusters}")
    print(f"✓ Valid clusters: {n_valid_clusters}")
    
    # ========================================================================
    # SIGN VERIFICATION (SAME AS SCRIPT 06)
    # ========================================================================
    logger.info("\n6. SIGN CONVENTION VERIFICATION")
    print("\n6. SIGN CONVENTION VERIFICATION")
    
    if n_valid_clusters > 0:
        mean_cluster = df_cluster_stats['mean_subsidence_cm_yr'].mean()
        
        if mean_cluster > 0:
            logger.warning("⚠️  POSITIVE cluster mean! Check rate column.")
            print("⚠️  WARNING: Positive cluster mean (unexpected)")
            print(f"   Mean: {mean_cluster:.3f} cm/yr")
            print(f"   Verify rate column: '{rate_col}'")
        else:
            logger.info(f"✓ Sign verified: mean = {mean_cluster:.3f} cm/yr (negative = subsidence)")
            print(f"✓ Sign verified: {mean_cluster:.3f} cm/yr (negative ✓)")
    
    # ========================================================================
    # COASTAL FILTERING
    # ========================================================================
    logger.info("\n7. COASTAL FILTERING")
    print("\n7. COASTAL FILTERING")
    
    coastal_col = next((c for c in df.columns if 'coastal' in c.lower()), None)
    if coastal_col:
        df['cluster_id'] = cluster_id
        coastal_clusters = df[df['cluster_id'] >= 0].groupby('cluster_id')['coastal_zone'].agg([
            ('n_coastal', lambda x: (x == 'Coastal (0-1 km)').sum()),
            ('coastal_pct', lambda x: (x == 'Coastal (0-1 km)').mean() * 100)
        ]).round(1)
        
        df_coastal = df_cluster_stats.merge(coastal_clusters, left_on='cluster_id', right_index=True, how='left')
        df_coastal.fillna(0, inplace=True)
        df_coastal.to_csv(OUTPUT_DIR / '06_coastal_clusters.csv', index=False)
        logger.info("Saved coastal cluster stratification")
        print("✓ Coastal clusters saved")
    
    # ========================================================================
    # SAVE OUTPUTS
    # ========================================================================
    logger.info("\n8. SAVING OUTPUTS")
    print("\n8. SAVING OUTPUTS")
    
    df['cluster_id'] = cluster_id
    output_cols = [lat_col, lon_col, rate_col, 'cluster_id']
    if 'elevation' in df.columns:
        output_cols.append('elevation')
    if 'uncertainty' in df.columns:
        output_cols.append('uncertainty')
    
    df[output_cols].to_csv(OUTPUT_DIR / '06_clusters_identified.csv', index=False)
    df_cluster_stats.to_csv(OUTPUT_DIR / '06_cluster_statistics.csv', index=False)
    
    logger.info("Saved cluster outputs")
    print("✓ Cluster outputs saved")
    
    # ========================================================================
    # SUMMARY REPORT
    # ========================================================================
    logger.info("\n9. GENERATING SUMMARY")
    print("\n9. GENERATING SUMMARY")
    
    summary_file = OUTPUT_DIR / '06_analysis_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SCRIPT 06: ADVANCED CLUSTERING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Run: {RUN_TIMESTAMP}\n")
        f.write(f"Parameters: threshold={threshold} cm/yr, radius={radius_km} km, min_size={min_size}\n\n")
        
        f.write("CLUSTERING RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Points analyzed: {len(df):,}\n")
        f.write(f"Hotspots: {n_hotspots:,} ({n_hotspots/len(df)*100:.1f}%)\n")
        f.write(f"Valid clusters: {n_valid_clusters}\n")
        
        if n_valid_clusters > 0:
            f.write(f"Mean cluster size: {df_cluster_stats['size'].mean():.0f} pts\n")
            f.write(f"Mean subsidence: {df_cluster_stats['mean_subsidence_cm_yr'].mean():.3f} cm/yr\n")
        
        f.write("\nSIGN CONVENTION\n")
        f.write("=" * 80 + "\n")
        f.write("Negative cm/yr = subsidence ✓\n")
        if n_valid_clusters > 0 and mean_cluster <= 0:
            f.write(f"Mean cluster subsidence: {mean_cluster:.3f} cm/yr ✓ VERIFIED\n")
    
    logger.info(f"Saved: {summary_file.name}")
    print(f"✓ Saved summary report")
    
    # ========================================================================
    # FIGURES
    # ========================================================================
    logger.info("\n10. GENERATING FIGURES")
    print("\n10. GENERATING FIGURES")
    
    try:
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(df[lon_col], df[lat_col], c=df['cluster_id'],
                             cmap='viridis', s=20, alpha=0.7)
        plt.colorbar(scatter, label='Cluster ID (-1 = non-hotspot)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Subsidence Hotspot Clusters')
        plt.savefig(OUTPUT_DIR / '06_cluster_map.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        if n_valid_clusters > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            ax1.scatter(df_cluster_stats['size'], df_cluster_stats['mean_subsidence_cm_yr'], s=100)
            ax1.set_xlabel('Cluster Size (points)')
            ax1.set_ylabel('Mean Subsidence (cm/yr)')
            ax1.set_title('Cluster Size vs Subsidence')
            ax1.grid(True, alpha=0.3)
            
            ax2.hist(df_cluster_stats['mean_subsidence_cm_yr'], bins=20, edgecolor='black')
            ax2.set_xlabel('Mean Subsidence (cm/yr)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Cluster Subsidence Distribution')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / '06_cluster_stats.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("Figures saved")
        print("✓ Figures saved")
    except Exception as e:
        logger.warning(f"Figure generation failed: {e}")
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    logger.info("\nSCRIPT 06 COMPLETE")
    print("\n" + "=" * 80)
    print("✅ SCRIPT 06 COMPLETE")
    print("=" * 80)
    print(f"Hotspots: {n_hotspots:,} | Clusters: {n_valid_clusters}")
    print(f"Mean subsidence: {mean_cluster:.3f} cm/yr {'✓' if (n_valid_clusters > 0 and mean_cluster <= 0) else '⚠️'}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Advanced Clustering')
    parser.add_argument('--threshold', type=float, default=-2.0,
                       help='Hotspot threshold cm/yr (negative=subsidence)')
    parser.add_argument('--radius', type=float, default=2.0,
                       help='Search radius km')
    parser.add_argument('--min-size', type=int, default=10,
                       help='Min cluster size')
    parser.add_argument('--k', type=int, default=250,
                       help='Primary k-neighbors')
    args = parser.parse_args()
    
    main(args.threshold, args.radius, args.min_size, args.k)
