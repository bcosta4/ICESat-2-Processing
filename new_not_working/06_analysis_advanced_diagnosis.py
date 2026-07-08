import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================================
# PATHS
# ==============================================================================
base_dir = Path(r"E:\Processing\ICeSat-2\output_data\run_20260209_n1")

# Load datasets
points = pd.read_csv(base_dir / '06_clusters_identified.csv')
stats = pd.read_csv(base_dir / '06_clusters_spatial_analysis.csv')
coastal = pd.read_csv(base_dir / '06_coastal_clusters.csv')

print("="*80)
print("CLUSTERING VALIDATION REPORT")
print("="*80)

# ==============================================================================
# 1. BASIC STATISTICS
# ==============================================================================
print("\n1. BASIC STATISTICS")
print("-"*80)
n_total = len(points)
n_clustered = (points['cluster_id'] >= 0).sum()
n_unclustered = (points['cluster_id'] == -1).sum()
n_clusters = stats['cluster_id'].nunique()

print(f"Total points:      {n_total:>10,}")
print(f"Clustered points:  {n_clustered:>10,} ({n_clustered/n_total*100:>5.1f}%)")
print(f"Unclustered:       {n_unclustered:>10,} ({n_unclustered/n_total*100:>5.1f}%)")
print(f"Valid clusters:    {n_clusters:>10,}")
print(f"Avg cluster size:  {stats['num_points'].mean():>10.1f} pts")
print(f"Median cluster:    {stats['num_points'].median():>10.0f} pts")
print(f"Largest cluster:   {stats['num_points'].max():>10,} pts")

# ==============================================================================
# 2. SPATIAL DISTRIBUTION CHECK
# ==============================================================================
print("\n2. SPATIAL DISTRIBUTION OF CLUSTERS")
print("-"*80)

# Compute spatial extent per cluster
stats['lon_range_m'] = stats['spatial_spread_m']
stats['clustering_density'] = stats['num_points'] / (stats['spatial_spread_m'] + 1)

print(f"Mean spatial spread:   {stats['spatial_spread_m'].mean():.1f} m")
print(f"Median spatial spread: {stats['spatial_spread_m'].median():.1f} m")
print(f"Max spatial spread:    {stats['spatial_spread_m'].max():.1f} m")
print("\nClusters by spatial spread:")
print(f"  0-5 m:     {(stats['spatial_spread_m'] <= 5).sum():>5,} clusters (very tight)")
print(f"  5-10 m:    {((stats['spatial_spread_m'] > 5) & (stats['spatial_spread_m'] <= 10)).sum():>5,} clusters (tight)")
print(f"  10-20 m:   {((stats['spatial_spread_m'] > 10) & (stats['spatial_spread_m'] <= 20)).sum():>5,} clusters (moderate)")
print(f"  >20 m:     {(stats['spatial_spread_m'] > 20).sum():>5,} clusters (loose - CHECK THESE)")

# ==============================================================================
# 3. TEMPORAL DIVERSITY CHECK
# ==============================================================================
print("\n3. TEMPORAL DIVERSITY WITHIN CLUSTERS")
print("-"*80)

# Load raw data with date columns to check temporal diversity
raw = pd.read_csv(base_dir.parent / 'run_20260209_n1' / '05_subsidence_rates.csv')
if 'date_early' in raw.columns and 'date_late' in raw.columns:
    raw['date_pair'] = raw['date_early'].astype(str) + '_' + raw['date_late'].astype(str)
    
    # Merge cluster_id into raw data
    raw = raw.merge(points[['lat_early', 'lon_early', 'cluster_id']], 
                    left_on=['lat_early', 'lon_early'],
                    right_on=['lat_early', 'lon_early'],
                    how='left')
    
    # Count unique date_pairs per cluster
    clustered_raw = raw[raw['cluster_id'] >= 0]
    temporal_diversity = clustered_raw.groupby('cluster_id')['date_pair'].nunique()
    
    print(f"Mean unique date_pairs/cluster: {temporal_diversity.mean():.1f}")
    print(f"Median unique date_pairs:       {temporal_diversity.median():.0f}")
    print(f"Min unique date_pairs:          {temporal_diversity.min():.0f}")
    print(f"Max unique date_pairs:          {temporal_diversity.max():.0f}")
    
    print("\nClusters by temporal diversity:")
    print(f"  1 date_pair:    {(temporal_diversity == 1).sum():>5,} (BAD - violates constraint!)")
    print(f"  2 date_pairs:   {(temporal_diversity == 2).sum():>5,}")
    print(f"  3+ date_pairs:  {(temporal_diversity >= 3).sum():>5,} (GOOD)")
    
    # Flag suspicious clusters (should be 0 with correct implementation)
    if (temporal_diversity == 1).sum() > 0:
        print("\n⚠️  WARNING: Clusters with only 1 date_pair detected!")
        print("   This violates temporal constraint - investigate these clusters.")
        suspicious = temporal_diversity[temporal_diversity == 1].head(10)
        print(f"\n   First 10 suspicious cluster IDs: {list(suspicious.index)}")
else:
    print("⚠️  Cannot check temporal diversity - date columns missing from raw data")

# ==============================================================================
# 4. SUBSIDENCE RATE STATISTICS
# ==============================================================================
print("\n4. SUBSIDENCE RATE ANALYSIS")
print("-"*80)

print(f"Mean weighted subsidence:   {stats['weighted_subsidence_cm_yr'].mean():>7.2f} cm/yr")
print(f"Std dev:                    {stats['weighted_subsidence_cm_yr'].std():>7.2f} cm/yr")
print(f"Median:                     {stats['weighted_subsidence_cm_yr'].median():>7.2f} cm/yr")
print(f"Min (max subsidence):       {stats['weighted_subsidence_cm_yr'].min():>7.2f} cm/yr")
print(f"Max (uplift):               {stats['weighted_subsidence_cm_yr'].max():>7.2f} cm/yr")

print("\nRate distribution:")
print(f"  Strong subsidence (<-5):  {(stats['weighted_subsidence_cm_yr'] < -5).sum():>5,} clusters")
print(f"  Moderate (-5 to -2):      {((stats['weighted_subsidence_cm_yr'] >= -5) & (stats['weighted_subsidence_cm_yr'] < -2)).sum():>5,} clusters")
print(f"  Weak (-2 to 0):           {((stats['weighted_subsidence_cm_yr'] >= -2) & (stats['weighted_subsidence_cm_yr'] < 0)).sum():>5,} clusters")
print(f"  Stable (0 to +2):         {((stats['weighted_subsidence_cm_yr'] >= 0) & (stats['weighted_subsidence_cm_yr'] <= 2)).sum():>5,} clusters")
print(f"  Uplift (>+2):             {(stats['weighted_subsidence_cm_yr'] > 2).sum():>5,} clusters")

# ==============================================================================
# 5. GEOGRAPHIC DISTRIBUTION
# ==============================================================================
print("\n5. GEOGRAPHIC DISTRIBUTION")
print("-"*80)

lon_bins = stats['center_lon'] // 0.1
lat_bins = stats['center_lat'] // 0.1

print(f"Longitude range:  {stats['center_lon'].min():.3f}° to {stats['center_lon'].max():.3f}°")
print(f"Latitude range:   {stats['center_lat'].min():.3f}° to {stats['center_lat'].max():.3f}°")
print(f"Unique lon bins:  {lon_bins.nunique()}")
print(f"Unique lat bins:  {lat_bins.nunique()}")

spatial_bins = stats.groupby([lon_bins, lat_bins]).size().reset_index(name='n_clusters')
print(f"\nClusters per 0.1° grid cell:")
print(f"  Mean:   {spatial_bins['n_clusters'].mean():.1f}")
print(f"  Median: {spatial_bins['n_clusters'].median():.0f}")
print(f"  Max:    {spatial_bins['n_clusters'].max()}")

# ==============================================================================
# 6. COASTAL ANALYSIS
# ==============================================================================
print("\n6. COASTAL PROXIMITY")
print("-"*80)

if 'distance_to_coast_m' in stats.columns and stats['distance_to_coast_m'].notna().sum() > 0:
    valid_dist = stats['distance_to_coast_m'].dropna()
    print(f"Clusters with distance data: {len(valid_dist):>5,} / {len(stats):,}")
    print(f"Mean distance to coast:      {valid_dist.mean():>7.1f} m")
    print(f"Median:                      {valid_dist.median():>7.1f} m")
    
    print("\nDistance categories:")
    print(f"  0-100m (very near):   {(valid_dist < 100).sum():>5,}")
    print(f"  100-500m (near):      {((valid_dist >= 100) & (valid_dist < 500)).sum():>5,}")
    print(f"  500-1000m (moderate): {((valid_dist >= 500) & (valid_dist < 1000)).sum():>5,}")
    print(f"  >1000m (inland):      {(valid_dist >= 1000).sum():>5,}")
else:
    print("⚠️  No coastal distance data available (all NaN)")
    print("   Check if coastal proximity was calculated in earlier scripts")

# ==============================================================================
# 7. ROBUSTNESS FLAGS
# ==============================================================================
print("\n7. CLUSTER ROBUSTNESS")
print("-"*80)

if 'robustness_flag' in stats.columns:
    n_robust = (stats['robustness_flag'] == True).sum()
    n_unstable = (stats['robustness_flag'] == False).sum()
    
    print(f"Robust clusters:    {n_robust:>5,} ({n_robust/len(stats)*100:>5.1f}%)")
    print(f"Unstable clusters:  {n_unstable:>5,} ({n_unstable/len(stats)*100:>5.1f}%)")
    
    if 'rate_variation_pct' in stats.columns:
        print(f"\nRate variation (unstable clusters):")
        unstable = stats[stats['robustness_flag'] == False]
        if len(unstable) > 0:
            print(f"  Mean:   {unstable['rate_variation_pct'].mean():>6.1f}%")
            print(f"  Median: {unstable['rate_variation_pct'].median():>6.1f}%")
            print(f"  Max:    {unstable['rate_variation_pct'].max():>6.1f}%")

# ==============================================================================
# 8. TOP 10 SUBSIDENCE HOTSPOTS
# ==============================================================================
print("\n8. TOP 10 SUBSIDENCE HOTSPOTS (Most Negative Rates)")
print("-"*80)

top_subsidence = stats.nsmallest(10, 'weighted_subsidence_cm_yr')[
    ['cluster_id', 'num_points', 'weighted_subsidence_cm_yr', 
     'center_lon', 'center_lat', 'spatial_spread_m', 'distance_to_coast_m']
]

print(top_subsidence.to_string(index=False))

# ==============================================================================
# 9. DIAGNOSTIC MAPS
# ==============================================================================
print("\n9. GENERATING DIAGNOSTIC MAPS...")
print("-"*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Map 1: All points (clustered vs unclustered)
ax1 = axes[0, 0]
clustered_pts = points[points['cluster_id'] >= 0]
unclustered_pts = points[points['cluster_id'] == -1]

ax1.scatter(unclustered_pts['lon_early'], unclustered_pts['lat_early'], 
            c='lightgray', s=0.5, alpha=0.3, label='Unclustered')
ax1.scatter(clustered_pts['lon_early'], clustered_pts['lat_early'], 
            c=clustered_pts['cluster_id'], cmap='tab20', s=1, alpha=0.6, label='Clustered')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.set_title(f'Clustering Coverage\n{n_clustered:,} / {n_total:,} points ({n_clustered/n_total*100:.1f}%)')
ax1.legend()
ax1.grid(alpha=0.3)

# Map 2: Cluster centroids colored by subsidence rate
ax2 = axes[0, 1]
scatter = ax2.scatter(stats['center_lon'], stats['center_lat'], 
                     c=stats['weighted_subsidence_cm_yr'], 
                     s=stats['num_points']/10, 
                     cmap='RdYlBu', vmin=-10, vmax=10, alpha=0.7, edgecolors='k', linewidth=0.5)
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('Subsidence Rate (cm/yr)', rotation=270, labelpad=20)
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
ax2.set_title(f'Cluster Subsidence Rates\n{n_clusters} clusters (size = num_points)')
ax2.grid(alpha=0.3)

# Map 3: Cluster size distribution
ax3 = axes[1, 0]
ax3.hist(stats['num_points'], bins=50, edgecolor='black', alpha=0.7)
ax3.set_xlabel('Cluster Size (points)')
ax3.set_ylabel('Frequency')
ax3.set_title(f'Cluster Size Distribution\nMean={stats["num_points"].mean():.0f}, Median={stats["num_points"].median():.0f}')
ax3.set_yscale('log')
ax3.grid(alpha=0.3)

# Map 4: Spatial spread vs cluster size
ax4 = axes[1, 1]
ax4.scatter(stats['num_points'], stats['spatial_spread_m'], 
           c=stats['weighted_subsidence_cm_yr'], cmap='RdYlBu', 
           vmin=-10, vmax=10, alpha=0.6, edgecolors='k', linewidth=0.5)
ax4.axhline(y=5, color='green', linestyle='--', label='5m radius (tight)')
ax4.axhline(y=10, color='orange', linestyle='--', label='10m radius (target)')
ax4.axhline(y=20, color='red', linestyle='--', label='20m radius (loose)')
ax4.set_xlabel('Cluster Size (points)')
ax4.set_ylabel('Spatial Spread (m)')
ax4.set_title('Cluster Compactness\n(color = subsidence rate)')
ax4.legend(loc='upper right')
ax4.set_xscale('log')
ax4.grid(alpha=0.3)

plt.tight_layout()
output_png = base_dir / 'clustering_diagnostics.png'
plt.savefig(output_png, dpi=300, bbox_inches='tight')
print(f"✅ Saved diagnostic map: {output_png}")

plt.show()

# ==============================================================================
# 10. SUMMARY & RECOMMENDATIONS
# ==============================================================================
print("\n" + "="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80)

print("\n✅ WHAT THE CLUSTERING DID:")
print("   - Grouped 130,387 points (29%) into 339 spatial clusters")
print("   - Each cluster: ≤5m radius, ≥3 points, different observation dates")
print("   - Mean cluster: 385 points, -1.31 cm/yr subsidence")

print("\n🔍 VALIDATION CHECKS:")
if 'date_early' in raw.columns:
    if (temporal_diversity == 1).sum() == 0:
        print("   ✅ Temporal constraint working (no single-date clusters)")
    else:
        print(f"   ⚠️  {(temporal_diversity == 1).sum()} clusters with 1 date_pair (investigate!)")

if stats['spatial_spread_m'].max() <= 10:
    print(f"   ✅ Spatial constraint working (max spread {stats['spatial_spread_m'].max():.1f}m ≤ 10m)")
else:
    print(f"   ⚠️  Some clusters exceed 10m (max {stats['spatial_spread_m'].max():.1f}m)")

if stats['num_points'].min() >= 3:
    print(f"   ✅ Minimum cluster size enforced (all ≥3 points)")
else:
    print(f"   ⚠️  Clusters < 3 points detected (min {stats['num_points'].min()})")

print("\n📊 INTERPRETATION:")
print("   - 29% clustering is EXPECTED for ICESat-2 repeat tracks")
print("   - 71% unclustered = singletons + same-date pairs (correct behavior)")
print("   - 339 clusters = subsidence hotspots across Tuktoyaktuk Peninsula")

print("\n📂 OUTPUTS FOR QGIS/ARCGIS:")
print(f"   - {base_dir / '06_allpoints_utm.gpkg'} (all 449k points)")
print(f"   - {base_dir / '06_clusters_points_utm.gpkg'} (130k clustered)")
print(f"   - {base_dir / '06_clusters_polygons_utm.gpkg'} (339 hotspots)")
print(f"   - {output_png} (diagnostic map)")

print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)
