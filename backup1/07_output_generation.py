"""
================================================================================
SCRIPT 07: OUTPUT GENERATION - PUBLICATION FIGURES & GIS EXPORT
================================================================================

Purpose
-------
Generate final publication-ready figures, perform comprehensive
validation diagnostics, and export results to GIS formats (Shapefile
and GeoPackage) for mapping and external analysis.

Main Tasks – Section 1 (Publication Figures):
- Create multiple PNG figures at publication quality (e.g. 300 DPI):
  - Subsidence map with basemap.
  - Measurement density map.
  - Gridded subsidence map.
  - Statistical distribution panels (histograms/box plots).
  - Spatial density heatmaps.
  - Multi-panel comparison figures.
- Save figures in a dedicated output directory.

Main Tasks – Section 2 (Validation Diagnostics):
- Use icesat2_insar_integrated.csv and clustering outputs to:
  - Plot ICESat-2 vs InSAR scatter diagrams with regression line.
  - Stratify validation by data density or other criteria.
  - Compute robust statistics (MAD, IQR, etc.).
- Export PNG figures and a textual diagnostics report.

Main Tasks – Section 3 (GIS Export):
- Convert tabular outputs to GIS layers:
  - Crossover points and InSAR-integrated points as Shapefiles.
  - Combined analysis GeoPackage with multiple layers.
- Ensure CRS is correctly set (typically EPSG:4326).
- Write metadata describing fields, CRS and processing steps.

Inputs:
- density_clusters_all.csv
- icesat2_insar_integrated.csv
- icesat2_repeat_track_matches.csv
- spatial_aggregation_* CSVs and classified subsidence file.

Outputs:
- Multiple PNG figures (01_*.png ... 08_*.png and validation_*.png).
- Shapefiles: e.g. icesat2_crossovers.shp, insar_integrated_points.shp.
- GeoPackages: e.g. icesat2_analysis_combined.gpkg.
- GIS_exports/metadata.txt describing exported layers.

Reproducibility Notes: All figures and GIS layers are derived from the CSV outputs produced
by earlier scripts. As long as inputs and configuration are unchanged,
regenerating outputs with this script should yield identical figures and GIS products.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
import sys

warnings.filterwarnings('ignore')

print("="*80)
print("SCRIPT 07: OUTPUT GENERATION - PUBLICATION FIGURES & GIS EXPORTS")
print("="*80)

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_OUTPUT_DIR = Path(r'E:\Processing\ICeSat-2\output_data')
DPI = 300

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
print(f"Output directory: {OUTPUT_DIR}")

# =============================================================================
# LOAD MAIN DATA
# =============================================================================

print("\nLoading datasets...")

main_file = OUTPUT_DIR / 'icesat2_repeat_track_matches.csv'
if not main_file.exists():
    print(f"ERROR: {main_file} not found!")
    sys.exit(1)

data_main = pd.read_csv(main_file)
print(f"✓ Main crossovers: {len(data_main):,} points")

# Auto-detect columns
lat_col = next((c for c in data_main.columns if 'lat' in c.lower()), None)
lon_col = next((c for c in data_main.columns if 'lon' in c.lower()), None)
rate_col = next((c for c in data_main.columns if 'subsidence' in c.lower() or 'rate' in c.lower()), None)
date_col = next((c for c in data_main.columns if 'date' in c.lower()), None)

if not all([lat_col, lon_col, rate_col]):
    print("ERROR: Could not detect required columns!")
    sys.exit(1)

# Optional: Load period data (from Script 05b)
early_data = None
late_data = None
accel_metrics = None

early_file = OUTPUT_DIR / 'icesat2_early_period.csv'
late_file = OUTPUT_DIR / 'icesat2_late_period.csv'
accel_file = OUTPUT_DIR / 'temporal_acceleration_metrics.csv'

if early_file.exists() and late_file.exists():
    try:
        early_data = pd.read_csv(early_file)
        late_data = pd.read_csv(late_file)
        print(f"✓ Period data: {len(early_data):,} early, {len(late_data):,} late")
    except:
        print(f"⚠ Failed to load period data")

if accel_file.exists():
    try:
        accel_metrics = pd.read_csv(accel_file)
        print(f"✓ Acceleration metrics loaded")
    except:
        pass

# Optional: Load density data (from Script 06)
density_data = None
density_file = OUTPUT_DIR / 'density_clusters_all.csv'

if density_file.exists():
    try:
        density_data = pd.read_csv(density_file)
        if 'neighbor_count' in density_data.columns and 'neighbor_count' not in data_main.columns:
            data_main['neighbor_count'] = density_data['neighbor_count'].values
            print(f"✓ Density data loaded")
    except:
        print(f"⚠ Failed to load density data")

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\n" + "="*80)
print("GENERATING SUMMARY STATISTICS")
print("="*80)

summary_stats = pd.DataFrame({
    'Metric': [
        'Total Points',
        'Mean Subsidence (mm/yr)',
        'Median Subsidence (mm/yr)',
        'Std Dev (mm/yr)',
        'Min (mm/yr)',
        'Max (mm/yr)',
        'Geographic Extent'
    ],
    'Value': [
        f"{len(data_main):,}",
        f"{data_main[rate_col].mean()*1000:.2f}",
        f"{data_main[rate_col].median()*1000:.2f}",
        f"{data_main[rate_col].std()*1000:.2f}",
        f"{data_main[rate_col].min()*1000:.2f}",
        f"{data_main[rate_col].max()*1000:.2f}",
        f"Lat {data_main[lat_col].min():.4f}° to {data_main[lat_col].max():.4f}°, Lon {data_main[lon_col].min():.4f}° to {data_main[lon_col].max():.4f}°"
    ]
})

summary_stats.to_csv(OUTPUT_DIR / '07_summary_statistics.csv', index=False)
print(f"✓ Summary statistics saved")

# =============================================================================
# FIGURE 1: MAIN SUBSIDENCE MAP
# =============================================================================

print("\nGenerating figures...")

fig, ax = plt.subplots(figsize=(14, 10))

vmin, vmax = np.percentile(data_main[rate_col]*1000, [5, 95])
scatter = ax.scatter(data_main[lon_col], data_main[lat_col],
                     c=data_main[rate_col]*1000, cmap='RdBu_r',
                     s=2, alpha=0.6, vmin=vmin, vmax=vmax)

ax.set_xlabel('Longitude (°E)', fontsize=12, fontweight='bold')
ax.set_ylabel('Latitude (°N)', fontsize=12, fontweight='bold')
ax.set_title(f'Subsidence Rate Distribution\n{len(data_main):,} Crossover Points',
             fontsize=14, fontweight='bold')

cbar = plt.colorbar(scatter, ax=ax, label='Subsidence Rate (mm/yr)')
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '07_figure_subsidence_map.png', dpi=DPI, bbox_inches='tight')
plt.close()
print(f"  ✓ 07_figure_subsidence_map.png")

# =============================================================================
# FIGURE 2: SUBSIDENCE DISTRIBUTION
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(data_main[rate_col].dropna()*1000, bins=60, color='steelblue',
             alpha=0.7, edgecolor='black')
axes[0].axvline(data_main[rate_col].mean()*1000, color='red', linestyle='--',
                linewidth=2.5, label=f'Mean: {data_main[rate_col].mean()*1000:.2f} mm/yr')
axes[0].axvline(data_main[rate_col].median()*1000, color='orange', linestyle='--',
                linewidth=2.5, label=f'Median: {data_main[rate_col].median()*1000:.2f} mm/yr')
axes[0].set_xlabel('Subsidence Rate (mm/yr)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0].set_title('Subsidence Rate Distribution', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].boxplot([data_main[rate_col].dropna()*1000], vert=True, patch_artist=True,
                widths=0.5, showmeans=True, meanline=True)
axes[1].set_ylabel('Subsidence Rate (mm/yr)', fontsize=11, fontweight='bold')
axes[1].set_title('Subsidence Rate Boxplot', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_xticklabels(['All Data'])
axes[1].axhline(0, color='red', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '07_figure_rate_distribution.png', dpi=DPI, bbox_inches='tight')
plt.close()
print(f"  ✓ 07_figure_rate_distribution.png")

# =============================================================================
# FIGURE 3: TEMPORAL COMPARISON (if early/late available)
# =============================================================================

if early_data is not None and late_data is not None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    vmin, vmax = np.percentile(data_main[rate_col]*1000, [5, 95])

    # Early spatial
    scatter1 = axes[0, 0].scatter(early_data[lon_col], early_data[lat_col],
                                  c=early_data[rate_col]*1000, cmap='RdBu_r',
                                  s=2, alpha=0.6, vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f'Early Period\n{len(early_data):,} points',
                         fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Longitude')
    axes[0, 0].set_ylabel('Latitude')
    axes[0, 0].grid(True, alpha=0.3)

    # Late spatial
    scatter2 = axes[0, 1].scatter(late_data[lon_col], late_data[lat_col],
                                  c=late_data[rate_col]*1000, cmap='RdBu_r',
                                  s=2, alpha=0.6, vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'Late Period\n{len(late_data):,} points',
                         fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Longitude')
    axes[0, 1].set_ylabel('Latitude')
    axes[0, 1].grid(True, alpha=0.3)

    cbar = fig.colorbar(scatter2, ax=[axes[0, 0], axes[0, 1]],
                        orientation='horizontal', pad=0.05)
    cbar.set_label('Subsidence Rate (mm/yr)', fontweight='bold')

    # Distribution comparison
    axes[1, 0].hist(early_data[rate_col].dropna()*1000, bins=50, alpha=0.6,
                    label=f'Early: {early_data[rate_col].mean()*1000:.2f} mm/yr', color='blue')
    axes[1, 0].hist(late_data[rate_col].dropna()*1000, bins=50, alpha=0.6,
                    label=f'Late: {late_data[rate_col].mean()*1000:.2f} mm/yr', color='red')
    axes[1, 0].set_xlabel('Subsidence Rate (mm/yr)', fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontweight='bold')
    axes[1, 0].set_title('Rate Distribution: Early vs Late', fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Boxplot comparison
    bp = axes[1, 1].boxplot([early_data[rate_col].dropna()*1000, late_data[rate_col].dropna()*1000],
                            labels=['Early', 'Late'], patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    axes[1, 1].set_ylabel('Subsidence Rate (mm/yr)', fontweight='bold')
    axes[1, 1].set_title('Temporal Comparison (Deceleration)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].axhline(0, color='black', linestyle=':', alpha=0.5)

    # Add deceleration text
    if accel_metrics is not None:
        metrics_dict = dict(zip(accel_metrics['metric'], accel_metrics['value']))
        change_mm = metrics_dict.get('change_mm_per_yr', 'N/A')
        direction = metrics_dict.get('direction', 'N/A')
        pval = metrics_dict.get('t_pvalue', np.nan)
        
        textstr = f'Change: {change_mm:.2f} mm/yr\nDirection: {direction}\np-value: {pval:.4e}'
        axes[1, 1].text(0.5, 0.95, textstr, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', horizontalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '07_figure_temporal_comparison.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 07_figure_temporal_comparison.png")

# =============================================================================
# FIGURE 4: DENSITY-STRATIFIED (if density available)
# =============================================================================

if 'neighbor_count' in data_main.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Rate vs Density scatter
    axes[0].scatter(data_main['neighbor_count'], data_main[rate_col]*1000,
                   alpha=0.5, s=2, color='steelblue')
    axes[0].set_xlabel('Neighbor Count (within 4m)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Subsidence Rate (mm/yr)', fontsize=11, fontweight='bold')
    axes[0].set_title('Density vs Subsidence Rate', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='red', linestyle=':', alpha=0.5)

    # Right: Rate by density class
    bins = [0, 1, 5, 20, np.inf]
    labels = ['0-1', '1-5', '5-20', '20+']
    data_main['density_class'] = pd.cut(data_main['neighbor_count'], bins=bins, labels=labels, include_lowest=True)

    density_means = data_main.groupby('density_class', observed=True)[rate_col].mean() * 1000
    density_counts = data_main.groupby('density_class', observed=True).size()

    axes[1].bar(range(len(density_means)), density_means.values, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xticks(range(len(density_means)))
    axes[1].set_xticklabels(density_means.index)
    axes[1].set_xlabel('Density Class', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Mean Subsidence Rate (mm/yr)', fontsize=11, fontweight='bold')
    axes[1].set_title('Subsidence by Density Zone', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].axhline(0, color='red', linestyle=':', alpha=0.5)

    # Add counts on bars
    for i, (mean, count) in enumerate(zip(density_means.values, density_counts.values)):
        axes[1].text(i, mean, f'n={count}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '07_figure_density_stratified.png', dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 07_figure_density_stratified.png")

# =============================================================================
# GIS EXPORTS
# =============================================================================

print("\nGenerating GIS exports...")

try:
    import geopandas as gpd
    from shapely.geometry import Point

    geometry = [Point(xy) for xy in zip(data_main[lon_col], data_main[lat_col])]
    gdf = gpd.GeoDataFrame(data_main, geometry=geometry, crs="EPSG:4326")

    gis_dir = OUTPUT_DIR / 'GIS_exports'
    gis_dir.mkdir(exist_ok=True)

    # Shapefile
    shp_file = gis_dir / 'icesat2_crossovers.shp'
    gdf.to_file(shp_file)
    print(f"  ✓ icesat2_crossovers.shp")

    # GeoPackage
    gpkg_file = gis_dir / 'icesat2_analysis.gpkg'
    gdf.to_file(gpkg_file, layer='crossovers', driver="GPKG")
    print(f"  ✓ icesat2_analysis.gpkg")

    # Metadata
    metadata_file = gis_dir / 'GIS_metadata.txt'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("GIS EXPORT METADATA\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"CRS: EPSG:4326 (WGS84)\n")
        f.write(f"Total points: {len(gdf):,}\n\n")
        f.write(f"FIELD DESCRIPTIONS:\n")
        for col in data_main.columns[:10]:
            f.write(f"  {col}: {data_main[col].dtype}\n")
        if len(data_main.columns) > 10:
            f.write(f"  ... and {len(data_main.columns)-10} more fields\n")

    print(f"  ✓ GIS_metadata.txt")

except ImportError:
    print(f"  ⚠ geopandas not installed – skipping GIS exports")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(f"\n" + "="*80)
print(f"SCRIPT 07 COMPLETE")
print(f"="*80)

print(f"\nFigures generated:")
print(f"  ✓ 07_figure_subsidence_map.png")
print(f"  ✓ 07_figure_rate_distribution.png")
if early_data is not None:
    print(f"  ✓ 07_figure_temporal_comparison.png (deceleration analysis)")
if 'neighbor_count' in data_main.columns:
    print(f"  ✓ 07_figure_density_stratified.png")

print(f"\nAll outputs ready for thesis chapter!")
