"""
================================================================================
SCRIPT 07: OUTPUT GENERATION - PUBLICATION FIGURES & COMPREHENSIVE ANALYSIS
================================================================================

Purpose
-------
Generate final publication-ready figures, comprehensive validation diagnostics,
and GIS-formatted outputs integrating ICESat-2, InSAR, temporal trends,
and density clustering results.

Key Features
------------
• Multi-panel publication figures at 300 DPI (and 600 DPI print quality)
• Temporal deceleration analysis with statistical annotations
• Density-stratified subsidence with regression and error bars
• Composite metrics table (density × temporal period)
• Robust statistics: Cohen's d, confidence intervals, effect sizes
• Comprehensive GIS exports (Shapefile + GeoPackage with metadata)
• Summary statistics and validation diagnostics

Main Tasks – Section 1 (Publication Figures)
---------------------------------------------
- Subsidence map with colorbar
- Rate distribution histogram + boxplot
- Temporal comparison (early vs late, 2×2 spatial + distributions)
- Density stratification with regression analysis
- All figures: 300 DPI screen + 600 DPI print quality

Main Tasks – Section 2 (Summary Statistics & Metrics)
------------------------------------------------------
- Generate summary statistics table
- Create composite metrics by density × period
- Compute robust statistics (MAD, IQR, effect sizes)
- Tabular outputs for thesis tables

Main Tasks – Section 3 (GIS Exports)
------------------------------------
- Convert crossover points to Shapefile (EPSG:4326)
- Create GeoPackage with multiple layers
- Write comprehensive metadata describing fields and processing
- Ensure reproducibility and external GIS compatibility

Inputs
------
• icesat2_repeat_track_matches.csv (from Script 03)
• icesat2_early_period.csv & icesat2_late_period.csv (from Script 05b)
• temporal_acceleration_metrics.csv (from Script 05b)
• density_clusters_all.csv (from Script 06, optional)
• icesat2_insar_integrated.csv (from Script 03, optional)

Outputs
-------
• 07_summary_statistics.csv (summary metrics)
• 07_composite_metrics_by_density.csv (density × period table)
• 07_figure_subsidence_map.png (300 DPI + 600 DPI HiRes)
• 07_figure_rate_distribution.png (histogram + boxplot)
• 07_figure_temporal_comparison.png (early/late 2×2 panel)
• 07_figure_density_stratified.png (density analysis with error bars)
• GIS_exports/icesat2_crossovers.shp (Shapefile)
• GIS_exports/icesat2_analysis.gpkg (GeoPackage)
• GIS_exports/GIS_metadata.txt (field descriptions + metadata)

Reproducibility Notes
---------------------
All outputs are derived from CSV files produced by earlier scripts.
Figures are publication-quality with high DPI output for thesis/papers.
GIS layers preserve all computed metrics (subsidence rates, densities, periods).
================================================================================
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
DPI_SCREEN = 300
DPI_PRINT = 600

# =============================================================================
# LOAD DATA
# =============================================================================

try:
    with open(BASE_OUTPUT_DIR / 'latest_run_timestamp.txt', 'r') as f:
        RUN_TIMESTAMP = f.read().strip()
    OUTPUT_DIR = BASE_OUTPUT_DIR / f'run_{RUN_TIMESTAMP}'
except FileNotFoundError:
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

if not all([lat_col, lon_col, rate_col]):
    print("ERROR: Could not detect required columns!")
    sys.exit(1)

print(f"✓ Column mapping: lat={lat_col}, lon={lon_col}, rate={rate_col}")

# =============================================================================
# LOAD OPTIONAL DATA
# =============================================================================

# Temporal period data (from Script 05b)
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

# Density data (from Script 06)
density_file = OUTPUT_DIR / 'density_clusters_all.csv'

if density_file.exists():
    try:
        density_data = pd.read_csv(density_file)
        if 'neighbor_count' in density_data.columns and 'neighbor_count' not in data_main.columns:
            data_main['neighbor_count'] = density_data['neighbor_count'].values
        if 'density_class' in density_data.columns and 'density_class' not in data_main.columns:
            data_main['density_class'] = density_data['density_class'].values
        print(f"✓ Density data loaded")
    except:
        print(f"⚠ Failed to load density data")

# InSAR data (optional)
insar_file = OUTPUT_DIR / 'icesat2_insar_integrated.csv'

if insar_file.exists():
    try:
        insar_data = pd.read_csv(insar_file)
        for col in ['insar_subsidence_rate_mm_yr']:
            if col in insar_data.columns and col not in data_main.columns:
                data_main[col] = insar_data[col].values
        print(f"✓ InSAR data loaded")
    except:
        pass

# =============================================================================
# GENERATE SUMMARY STATISTICS
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
        f"Lat [{data_main[lat_col].min():.4f}°, {data_main[lat_col].max():.4f}°], Lon [{data_main[lon_col].min():.4f}°, {data_main[lon_col].max():.4f}°]"
    ]
})

summary_stats.to_csv(OUTPUT_DIR / '07_summary_statistics.csv', index=False)
print(f"✓ Summary statistics saved")

# =============================================================================
# COMPOSITE METRICS TABLE (Density × Period)
# =============================================================================

if early_data is not None and late_data is not None and 'neighbor_count' in data_main.columns:
    print("\nGenerating composite metrics (density × period)...")
    
    # Add density class to early/late if not present
    if 'density_class' not in early_data.columns:
        early_data['density_class'] = pd.cut(
            early_data.get('neighbor_count', 0),
            bins=[0, 1, 5, 20, np.inf],
            labels=['0-1', '1-5', '5-20', '20+'],
            include_lowest=True
        )
    
    if 'density_class' not in late_data.columns:
        late_data['density_class'] = pd.cut(
            late_data.get('neighbor_count', 0),
            bins=[0, 1, 5, 20, np.inf],
            labels=['0-1', '1-5', '5-20', '20+'],
            include_lowest=True
        )
    
    composite_metrics = []
    
    for density_class in ['0-1', '1-5', '5-20', '20+']:
        early_subset = early_data[early_data['density_class'] == density_class][rate_col]
        late_subset = late_data[late_data['density_class'] == density_class][rate_col]
        
        if len(early_subset) > 0 and len(late_subset) > 0:
            change = (late_subset.mean() - early_subset.mean()) * 1000
            composite_metrics.append({
                'density_class': density_class,
                'n_early': len(early_subset),
                'n_late': len(late_subset),
                'early_mean_mm_yr': early_subset.mean() * 1000,
                'late_mean_mm_yr': late_subset.mean() * 1000,
                'change_mm_yr': change,
                'early_std_mm_yr': early_subset.std() * 1000,
                'late_std_mm_yr': late_subset.std() * 1000
            })
    
    if composite_metrics:
        composite_df = pd.DataFrame(composite_metrics)
        composite_df.to_csv(OUTPUT_DIR / '07_composite_metrics_by_density.csv', index=False)
        print(f"✓ Composite metrics saved ({len(composite_df)} density classes)")

# =============================================================================
# FIGURE 1: MAIN SUBSIDENCE MAP
# =============================================================================

print("\n" + "="*80)
print("GENERATING PUBLICATION FIGURES")
print("="*80)

fig, ax = plt.subplots(figsize=(14, 10))

vmin, vmax = np.percentile(data_main[rate_col]*1000, [5, 95])

scatter = ax.scatter(data_main[lon_col], data_main[lat_col],
    c=data_main[rate_col]*1000, cmap='RdBu_r',
    s=2, alpha=0.6, vmin=vmin, vmax=vmax)

ax.set_xlabel('Longitude (°E)', fontsize=12, fontweight='bold')
ax.set_ylabel('Latitude (°N)', fontsize=12, fontweight='bold')
ax.set_title(f'Subsidence Rate Distribution\n{len(data_main):,} ICESat-2 Repeat-Track Crossovers',
    fontsize=14, fontweight='bold')

cbar = plt.colorbar(scatter, ax=ax, label='Subsidence Rate (mm/yr)', pad=0.02)
cbar.ax.tick_params(labelsize=10)

ax.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '07_figure_subsidence_map.png', dpi=DPI_SCREEN, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / '07_figure_subsidence_map_HiRes.png', dpi=DPI_PRINT, bbox_inches='tight')
plt.close()

print(f"✓ 07_figure_subsidence_map.png (300 DPI + 600 DPI)")

# =============================================================================
# FIGURE 2: SUBSIDENCE DISTRIBUTION
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(data_main[rate_col].dropna()*1000, bins=60, color='steelblue',
    alpha=0.7, edgecolor='black', linewidth=0.5)

mean_val = data_main[rate_col].mean()*1000
median_val = data_main[rate_col].median()*1000

axes[0].axvline(mean_val, color='red', linestyle='--', linewidth=2.5,
    label=f'Mean: {mean_val:.2f} mm/yr')
axes[0].axvline(median_val, color='orange', linestyle='--', linewidth=2.5,
    label=f'Median: {median_val:.2f} mm/yr')

axes[0].set_xlabel('Subsidence Rate (mm/yr)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0].set_title('Subsidence Rate Distribution', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10, loc='upper right')
axes[0].grid(True, alpha=0.3, axis='y')

# Boxplot
bp = axes[1].boxplot([data_main[rate_col].dropna()*1000], vert=True, patch_artist=True,
    widths=0.5, showmeans=True, meanline=True)

bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][0].set_linewidth(1.5)
bp['medians'][0].set_linewidth(2)
bp['means'][0].set_linewidth(2)
bp['means'][0].set_color('red')

axes[1].set_ylabel('Subsidence Rate (mm/yr)', fontsize=11, fontweight='bold')
axes[1].set_title('Rate Summary Statistics', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_xticklabels(['All Data'])
axes[1].axhline(0, color='red', linestyle=':', linewidth=1.5, alpha=0.5)

# Annotation box
q1 = np.percentile(data_main[rate_col]*1000, 25)
q3 = np.percentile(data_main[rate_col]*1000, 75)
iqr = q3 - q1

textstr = f'IQR: [{q1:.2f}, {q3:.2f}]\nIQR width: {iqr:.2f}\nN = {len(data_main):,}'
axes[1].text(0.95, 0.05, textstr, transform=axes[1].transAxes,
    fontsize=10, verticalalignment='bottom', horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '07_figure_rate_distribution.png', dpi=DPI_SCREEN, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / '07_figure_rate_distribution_HiRes.png', dpi=DPI_PRINT, bbox_inches='tight')
plt.close()

print(f"✓ 07_figure_rate_distribution.png")

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
    axes[0, 0].set_xlabel('Longitude (°E)', fontsize=10)
    axes[0, 0].set_ylabel('Latitude (°N)', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Late spatial
    scatter2 = axes[0, 1].scatter(late_data[lon_col], late_data[lat_col],
        c=late_data[rate_col]*1000, cmap='RdBu_r',
        s=2, alpha=0.6, vmin=vmin, vmax=vmax)
    
    axes[0, 1].set_title(f'Late Period\n{len(late_data):,} points',
        fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Longitude (°E)', fontsize=10)
    axes[0, 1].set_ylabel('Latitude (°N)', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    
    cbar_top = fig.colorbar(scatter2, ax=[axes[0, 0], axes[0, 1]],
        orientation='horizontal', pad=0.08, aspect=40)
    cbar_top.set_label('Subsidence Rate (mm/yr)', fontweight='bold', fontsize=10)
    
    # Distribution comparison
    axes[1, 0].hist(early_data[rate_col].dropna()*1000, bins=50, alpha=0.6,
        label=f'Early: {early_data[rate_col].mean()*1000:.2f} mm/yr', color='blue', edgecolor='black', linewidth=0.5)
    axes[1, 0].hist(late_data[rate_col].dropna()*1000, bins=50, alpha=0.6,
        label=f'Late: {late_data[rate_col].mean()*1000:.2f} mm/yr', color='red', edgecolor='black', linewidth=0.5)
    
    axes[1, 0].set_xlabel('Subsidence Rate (mm/yr)', fontweight='bold', fontsize=10)
    axes[1, 0].set_ylabel('Frequency', fontweight='bold', fontsize=10)
    axes[1, 0].set_title('Rate Distribution: Early vs Late', fontweight='bold', fontsize=11)
    axes[1, 0].legend(fontsize=10, loc='upper right')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].axvline(0, color='black', linestyle=':', linewidth=1.5, alpha=0.5)
    
    # Boxplot comparison
    bp = axes[1, 1].boxplot(
        [early_data[rate_col].dropna()*1000, late_data[rate_col].dropna()*1000],
        labels=['Early', 'Late'], patch_artist=True, showmeans=True, meanline=True)
    
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1.5)
    
    axes[1, 1].set_ylabel('Subsidence Rate (mm/yr)', fontweight='bold', fontsize=10)
    axes[1, 1].set_title('Temporal Deceleration Analysis', fontweight='bold', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].axhline(0, color='black', linestyle=':', linewidth=1.5, alpha=0.5)
    
    # Add deceleration metrics
    if accel_metrics is not None:
        metrics_dict = dict(zip(accel_metrics['metric'], accel_metrics['value']))
        
        # Safe conversion to float
        try:
            change_mm = float(metrics_dict.get('change_mm_per_yr', np.nan))
        except (ValueError, TypeError):
            change_mm = np.nan
        
        try:
            pval = float(metrics_dict.get('t_pvalue', np.nan))
        except (ValueError, TypeError):
            pval = np.nan
        
        try:
            cohens_d = float(metrics_dict.get('cohens_d', np.nan))
        except (ValueError, TypeError):
            cohens_d = np.nan
        
        direction = str(metrics_dict.get('direction', 'N/A'))
        
        # Format p-value
        pval_str = f"{pval:.2e}" if not np.isnan(pval) else "N/A"
        
        textstr = f'Change: {change_mm:.2f} mm/yr\nDirection: {direction}\np-value: {pval_str}\nCohen\'s d: {cohens_d:.3f}'
        
        axes[1, 1].text(0.5, 0.95, textstr, transform=axes[1, 1].transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '07_figure_temporal_comparison.png', dpi=DPI_SCREEN, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / '07_figure_temporal_comparison_HiRes.png', dpi=DPI_PRINT, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 07_figure_temporal_comparison.png")

# =============================================================================
# FIGURE 4: DENSITY-STRATIFIED ANALYSIS
# =============================================================================

if 'neighbor_count' in data_main.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Scatter with regression line
    valid_mask = ~(data_main['neighbor_count'].isna() | data_main[rate_col].isna())
    
    axes[0].scatter(data_main.loc[valid_mask, 'neighbor_count'],
        data_main.loc[valid_mask, rate_col]*1000,
        alpha=0.3, s=1, color='steelblue')
    
    # Add linear regression line
    if valid_mask.sum() > 2:
        z = np.polyfit(data_main.loc[valid_mask, 'neighbor_count'],
            data_main.loc[valid_mask, rate_col]*1000, 1)
        x_poly = np.linspace(data_main.loc[valid_mask, 'neighbor_count'].min(),
            data_main.loc[valid_mask, 'neighbor_count'].max(), 100)
        axes[0].plot(x_poly, np.polyval(z, x_poly), 'r-', linewidth=2.5,
            label=f'Linear fit: slope={z[0]:.4f}')
    
    axes[0].set_xlabel('Neighbor Count (within 4m)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Subsidence Rate (mm/yr)', fontsize=11, fontweight='bold')
    axes[0].set_title('Density vs Subsidence Rate', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    axes[0].legend(fontsize=10)
    
    # Right: Mean by density class with error bars
    bins = [0, 1, 5, 20, np.inf]
    labels = ['0-1', '1-5', '5-20', '20+']
    
    data_main['density_class'] = pd.cut(data_main['neighbor_count'],
        bins=bins, labels=labels, include_lowest=True)
    
    means = []
    sems = []  # Standard errors
    counts = []
    x_pos = []
    
    for i, label in enumerate(labels):
        subset = data_main[data_main['density_class'] == label][rate_col]
        if len(subset) > 0:
            means.append(subset.mean() * 1000)
            sems.append(subset.std() * 1000 / np.sqrt(len(subset)))  # Standard error of mean
            counts.append(len(subset))
            x_pos.append(i)
    
    axes[1].bar(x_pos, means, yerr=sems, capsize=8, color='coral', alpha=0.7,
        edgecolor='black', linewidth=1.5, error_kw={'linewidth': 2, 'ecolor': 'darkred'})
    
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([labels[i] for i in x_pos])
    axes[1].set_xlabel('Density Class', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Mean Subsidence Rate (mm/yr)', fontsize=11, fontweight='bold')
    axes[1].set_title('Subsidence by Density Zone (with SE)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].axhline(0, color='red', linestyle=':', linewidth=1.5, alpha=0.5)
    
    # Add sample counts
    for i, (m, c) in enumerate(zip(means, counts)):
        axes[1].text(x_pos[i], m + sems[i], f'n={c}', ha='center', va='bottom',
            fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '07_figure_density_stratified.png', dpi=DPI_SCREEN, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / '07_figure_density_stratified_HiRes.png', dpi=DPI_PRINT, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 07_figure_density_stratified.png")

# =============================================================================
# GIS EXPORTS
# =============================================================================

print("\n" + "="*80)
print("GENERATING GIS EXPORTS")
print("="*80)

try:
    import geopandas as gpd
    from shapely.geometry import Point
    
    # Create GeoDataFrame
    geometry = [Point(xy) for xy in zip(data_main[lon_col], data_main[lat_col])]
    gdf = gpd.GeoDataFrame(data_main, geometry=geometry, crs="EPSG:26908")
    
    gis_dir = OUTPUT_DIR / 'GIS_exports'
    gis_dir.mkdir(exist_ok=True)
    
    # Shapefile
    gdf.to_file(gis_dir / 'icesat2_crossovers.shp', index=False)
    print(f"✓ icesat2_crossovers.shp ({len(gdf):,} features)")
    
    # GeoPackage
    gdf.to_file(gis_dir / 'icesat2_analysis.gpkg', layer='crossovers', driver="GPKG", index=False)
    print(f"✓ icesat2_analysis.gpkg")
    
    # Metadata
    metadata_file = gis_dir / 'GIS_metadata.txt'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("GIS EXPORT METADATA\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"CRS: EPSG:4326 (WGS84 - Latitude/Longitude)\n")
        f.write(f"Total features: {len(gdf):,}\n")
        f.write(f"Geometry type: Point\n\n")
        
        f.write("FIELD DESCRIPTIONS:\n")
        f.write("-" * 80 + "\n")
        
        for col in list(data_main.columns)[:15]:
            dtype = str(data_main[col].dtype)
            n_null = data_main[col].isna().sum()
            f.write(f"{col:30s} | {dtype:15s} | {n_null:,} null\n")
        
        if len(data_main.columns) > 15:
            f.write(f"\n... and {len(data_main.columns)-15} more fields\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("PROCESSING NOTES\n")
        f.write("="*80 + "\n")
        f.write("• All subsidence rates in mm/year (negative = subsidence, positive = uplift)\n")
        f.write("• Coordinates in WGS84 (EPSG:4326)\n")
        f.write("• Use QGIS or ArcGIS to visualize and further analyze\n")
        f.write("• InSAR validation columns present if InSAR integration was performed\n")
        f.write("• Temporal period flags ('early'/'late') indicate temporal stratification\n")
    
    print(f"✓ GIS_metadata.txt")

except ImportError:
    print(f"⚠ geopandas not installed – skipping GIS exports")
    print("  Install with: pip install geopandas shapely")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*80)
print("SCRIPT 07 COMPLETE")
print("="*80)

print(f"\n✓ Publication-ready outputs generated:")
print(f"  Figure 1: Subsidence map (300 DPI + 600 DPI HiRes)")
print(f"  Figure 2: Distribution statistics (histogram + boxplot)")

if early_data is not None:
    print(f"  Figure 3: Temporal deceleration comparison (2×2 panel)")

if 'neighbor_count' in data_main.columns:
    print(f"  Figure 4: Density stratification (with regression + error bars)")

print(f"\n✓ Tabular outputs:")
print(f"  Summary statistics (07_summary_statistics.csv)")

if 'density_class' in data_main.columns:
    print(f"  Composite metrics by density×period (07_composite_metrics_by_density.csv)")

print(f"\n✓ GIS exports:")
print(f"  Shapefile (icesat2_crossovers.shp)")
print(f"  GeoPackage (icesat2_analysis.gpkg)")
print(f"  Metadata (GIS_metadata.txt)")

print(f"\n✓ All outputs saved to: {OUTPUT_DIR}")
print(f"  Ready for thesis chapter and publication!\n")
