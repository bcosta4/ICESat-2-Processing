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
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("SCRIPT 04: SPATIAL ANALYSIS - VISUALIZATION & COVERAGE")
print("="*80)

BASE_OUTPUT_DIR = Path(r'E:\Processing\ICeSat-2\output_data')

timestamp_file = BASE_OUTPUT_DIR / 'latest_run_timestamp.txt'
with open(timestamp_file, 'r') as f:
    RUN_TIMESTAMP = f.read().strip()

OUTPUT_DIR = BASE_OUTPUT_DIR / f'run_{RUN_TIMESTAMP}'

print(f"Run ID: {RUN_TIMESTAMP}")
print(f"Output directory: {OUTPUT_DIR}")
print("="*80)

# ============================================================================
# SECTION 1: VISUALIZATION (04_visualization.py)
# ============================================================================

print("\n[1/2] CREATING VISUALIZATIONS\n")

repeat_matches_file = OUTPUT_DIR / 'icesat2_repeat_track_matches.csv'

if repeat_matches_file.exists():
    repeat_matches = pd.read_csv(repeat_matches_file)
    print(f"[OK] Loaded {len(repeat_matches):,} repeat matches")
else:
    print(f"ERROR: {repeat_matches_file} not found!")
    exit(1)

def get_column(df, *possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None

print("Creating summary plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'ICeSat-2 Analysis Summary (Run {RUN_TIMESTAMP})', fontsize=14, fontweight='bold')

# Plot 1: Height change distribution
ax = axes[0, 0]
dh_col = get_column(repeat_matches, 'elevation_change_m', 'dh', 'height_change')

if dh_col:
    ax.hist(repeat_matches[dh_col], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel('Elevation Change (m)')
    ax.set_ylabel('Count')
    ax.set_title('Height Change Distribution')
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'Height change column not found', ha='center', va='center')
    ax.set_xticks([])
    ax.set_yticks([])

# Plot 2: Rate distribution
ax = axes[0, 1]
rate_col = get_column(repeat_matches, 'dh_rate_m_per_year', 'rate_m_per_year', 'subsidence_rate')

if rate_col:
    ax.hist(repeat_matches[rate_col], bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax.set_xlabel('Subsidence Rate (m/year)')
    ax.set_ylabel('Count')
    ax.set_title('Subsidence Rate Distribution')
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'Rate column not found', ha='center', va='center')
    ax.set_xticks([])
    ax.set_yticks([])

# Plot 3: Geographic distribution
ax = axes[1, 0]
lat_col = get_column(repeat_matches, 'latitude', 'lat', 'Latitude')
lon_col = get_column(repeat_matches, 'longitude', 'lon', 'Longitude')

if lat_col and lon_col and dh_col:
    scatter = ax.scatter(repeat_matches[lon_col], repeat_matches[lat_col],
                        c=repeat_matches[dh_col], s=5, alpha=0.6, cmap='RdBu_r')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Spatial Distribution (colored by dH)')
    plt.colorbar(scatter, ax=ax, label='dH (m)')
else:
    ax.text(0.5, 0.5, 'Coordinate columns not found', ha='center', va='center')
    ax.set_xticks([])
    ax.set_yticks([])

# Plot 4: Uncertainty analysis
ax = axes[1, 1]
unc_col = get_column(repeat_matches, 'dh_rate_uncertainty_m_per_year', 'uncertainty', 'dh_uncertainty_m')

if unc_col and rate_col:
    ax.scatter(repeat_matches[rate_col], repeat_matches[unc_col], alpha=0.5, s=10)
    ax.set_xlabel('Subsidence Rate (m/year)')
    ax.set_ylabel('Uncertainty (m/year)')
    ax.set_title('Rate vs. Uncertainty')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
else:
    ax.text(0.5, 0.5, 'Uncertainty columns not found', ha='center', va='center')
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
output_fig = OUTPUT_DIR / 'icesat2_summary_plots.png'
plt.savefig(output_fig, dpi=150, bbox_inches='tight')
print(f"[OK] Saved: {output_fig.name}")
plt.close()

# ============================================================================
# SECTION 2: COVERAGE ANALYSIS (05_coverage_analysis.py)
# ============================================================================

print("\n[2/2] COVERAGE ANALYSIS\n")

if len(repeat_matches) == 0:
    print("No data to analyze!")
    exit(1)

print("Computing spatial coverage metrics...")

if lat_col and lon_col:
    bounds = {
        'north': repeat_matches[lat_col].max(),
        'south': repeat_matches[lat_col].min(),
        'east': repeat_matches[lon_col].max(),
        'west': repeat_matches[lon_col].min(),
    }
    
    print(f"\nSpatial Bounds:")
    print(f" Latitude range: {bounds['south']:.4f} to {bounds['north']:.4f}")
    print(f" Longitude range: {bounds['west']:.4f} to {bounds['east']:.4f}")
    print(f" Area extent: {(bounds['east']-bounds['west'])*111:.1f} km x {(bounds['north']-bounds['south'])*111:.1f} km")
    
    print(f"\nGrid-based coverage analysis (0.01 degree cells):")
    
    grid_size = 0.01
    
    repeat_matches['lat_grid'] = (repeat_matches[lat_col] // grid_size * grid_size).round(4)
    repeat_matches['lon_grid'] = (repeat_matches[lon_col] // grid_size * grid_size).round(4)
    
    grid_coverage = repeat_matches.groupby(['lat_grid', 'lon_grid']).size().reset_index(name='count')
    
    print(f" Total grid cells: {len(grid_coverage):,}")
    print(f" Average points per cell: {len(repeat_matches) / len(grid_coverage):.1f}")

print(f"\n" + "="*80)
print(f"ANALYSIS SUMMARY STATISTICS")
print(f"="*80 + "\n")

if dh_col:
    print(f"Elevation Change ({dh_col}):")
    print(f" Mean: {repeat_matches[dh_col].mean():.4f} m")
    print(f" Std: {repeat_matches[dh_col].std():.4f} m")
    print(f" Min: {repeat_matches[dh_col].min():.4f} m")
    print(f" Max: {repeat_matches[dh_col].max():.4f} m")

if rate_col:
    print(f"\nSubsidence Rate ({rate_col}):")
    print(f" Mean: {repeat_matches[rate_col].mean():.6f} m/year")
    print(f" Std: {repeat_matches[rate_col].std():.6f} m/year")
    print(f" Subsiding (>0.01 m/yr): {len(repeat_matches[repeat_matches[rate_col] > 0.01]):,}")
    print(f" Rising (>0.01 m/yr): {len(repeat_matches[repeat_matches[rate_col] < -0.01]):,}")

summary_file = OUTPUT_DIR / 'spatial_coverage_summary.txt'

with open(summary_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("Spatial Coverage Analysis Summary\n")
    f.write("="*80 + "\n")
    f.write(f"Run ID: {RUN_TIMESTAMP}\n")
    f.write(f"Processing date: {datetime.now()}\n\n")
    f.write(f"Total measurement pairs: {len(repeat_matches):,}\n")
    if lat_col and lon_col:
        f.write(f"Spatial bounds:\n")
        f.write(f" Latitude: {bounds['south']:.4f} to {bounds['north']:.4f}\n")
        f.write(f" Longitude: {bounds['west']:.4f} to {bounds['east']:.4f}\n")

print(f"[OK] Summary saved: {summary_file.name}")

print("\n" + "="*80)
print("SCRIPT 04 COMPLETE")
print("="*80 + "\n")
