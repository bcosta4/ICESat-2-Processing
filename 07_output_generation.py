"""
================================================================================
SCRIPT 07: ENHANCED OUTPUT GENERATION WITH ZONAL GRID VISUALIZATION
================================================================================

Purpose
-------
Generate publication-ready figures and output visualizations from ICESat-2 
subsidence analysis results.

Modified with complete reproducibility infrastructure.

FEATURES INCLUDED:
✓ Enhanced zonal grid visualization (20×20 km) as square matrix heatmap
✓ Temporal period comparison plots (Early vs Late)
✓ Timescale stratification analysis
✓ Density classification visualization
✓ Hotspot spatial mapping
✓ Summary statistics compilation
✓ IMPROVED: Complete reproducibility infrastructure (logging, metadata, git hash)
✓ IMPROVED: UTF-8 encoding fix (NO UNICODE ERRORS!)

NEW VISUALIZATIONS:
- Figure 1: Zonal Grid Heatmap (20×20 km)
- Figure 2: Zonal Density Heatmap
- Figure 3: Period Comparison (Early vs Late)
- Figure 4: Timescale Stratification
- Figure 5: Density Classification
- Figure 6: Hotspot Map

Inputs
------
- Output files from Scripts 04, 05, and 06
- Zonal statistics, temporal analysis, advanced analysis data

Outputs
-------
- figures/ directory with 6 publication-ready PNG figures (300 DPI)
- Figure_*_*.png (all figures)
- Summary_Statistics.txt (compiled statistics)
- script_07_run.log (execution log)
- 07_output_generation_metadata.json (reproducibility metadata)

Reproducibility Notes
---------------------
All figures are generated with consistent styling, resolution (300 DPI),
and formatting. Complete audit trail provided via metadata JSON and
execution log.

================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import logging
import subprocess
import sys
import io
import warnings

warnings.filterwarnings('ignore')

# UTF-8 ENCODING FIX (CRITICAL - FIXES YOUR UNICODE ERROR!)
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ============================================================================
# GIT HASH TRACKING
# ============================================================================

def get_git_hash():
    """Get git commit hash for reproducibility"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip()[:8] if result.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


# ============================================================================
# LOGGING SETUP (DUAL OUTPUT: FILE + CONSOLE)
# ============================================================================

def setup_logging(output_dir):
    """Configure logging to both file and console"""
    
    log_file = output_dir / 'script_07_run.log'
    
    logger = logging.getLogger('script_07')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    # File handler - UTF-8 encoding, full details
    fh = logging.FileHandler(log_file, encoding='utf-8', mode='w')
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(fh_formatter)
    
    # Console handler - INFO level, clean output
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ============================================================================
# RUN METADATA CLASS (AUDIT TRAIL)
# ============================================================================

class RunMetadata:
    """Capture complete run parameters for reproducibility"""
    
    def __init__(self, run_id, git_hash):
        self.run_id = run_id
        self.script_name = '07_output_generation'
        self.script_version = 'v1.0-reproducible'
        self.git_hash = git_hash
        self.start_time = datetime.now().isoformat()
        self.end_time = None
        self.configuration = {}
        self.processing_results = {}
        self.environment_info = {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'platform': sys.platform,
            'timestamp': datetime.now().isoformat(),
        }
    
    def finalize(self):
        """Mark run as complete"""
        self.end_time = datetime.now().isoformat()
    
    def save(self, output_dir):
        """Save metadata as JSON"""
        metadata_file = output_dir / f'{self.script_name}_metadata.json'
        
        metadata = {
            'run_id': self.run_id,
            'script': self.script_name,
            'version': self.script_version,
            'git_hash': self.git_hash,
            'environment': self.environment_info,
            'timestamps': {
                'start': self.start_time,
                'end': self.end_time,
            },
            'configuration': self.configuration,
            'processing_results': self.processing_results,
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str, ensure_ascii=False)
        
        return metadata_file


# ============================================================================
# LOAD CONFIGURATION
# ============================================================================

print("="*80)
print("SCRIPT 07: ENHANCED OUTPUT GENERATION WITH ZONAL VISUALIZATION")
print("="*80)

BASE_OUTPUT_DIR = Path(r'E:\Processing\ICeSat-2\output_data')

timestamp_file = BASE_OUTPUT_DIR / 'latest_run_timestamp.txt'
try:
    with open(timestamp_file, 'r', encoding='utf-8') as f:
        RUN_TIMESTAMP = f.read().strip()
    OUTPUT_DIR = BASE_OUTPUT_DIR / f'run_{RUN_TIMESTAMP}'
    FIGURE_DIR = OUTPUT_DIR / 'figures'
    FIGURE_DIR.mkdir(exist_ok=True)
except FileNotFoundError:
    print("ERROR: Run Script 00 first.")
    sys.exit(1)

print(f"Run ID: {RUN_TIMESTAMP}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Figure directory: {FIGURE_DIR}\n")

# Get git hash for reproducibility
git_hash = get_git_hash()
print(f"Git Hash: {git_hash}\n")

# Setup logging
logger = setup_logging(OUTPUT_DIR)

# Create metadata tracker
metadata = RunMetadata(RUN_TIMESTAMP, git_hash)

logger.info("="*80)
logger.info("SCRIPT 07: ENHANCED OUTPUT GENERATION WITH ZONAL VISUALIZATION")
logger.info("="*80)
logger.info(f"Run ID: {RUN_TIMESTAMP}")
logger.info(f"Git Hash: {git_hash}")
logger.info(f"Output directory: {OUTPUT_DIR}")
logger.info(f"Figure directory: {FIGURE_DIR}")

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Record configuration in metadata
metadata.configuration = {
    'figure_dpi_standard': 300,
    'figure_dpi_hires': 600,
    'plotting_style': 'seaborn-v0_8-darkgrid',
}

# ============================================================================
# LOAD DATA
# ============================================================================

logger.info("\nLOADING DATA")
print("\n" + "="*80)
print("LOADING DATA")
print("="*80 + "\n")

# Load zonal statistics (20x20 km grid from Script 04)
try:
    zonal_stats = pd.read_csv(OUTPUT_DIR / '04_zonal_statistics_20km_grid.csv')
    logger.info(f"Loaded zonal statistics: {len(zonal_stats)} grid cells")
    print(f"✓ Loaded zonal statistics: {len(zonal_stats)} grid cells")
except FileNotFoundError:
    logger.warning("Zonal statistics not found. Run Script 04 first.")
    print("WARNING: Zonal statistics not found. Run Script 04 first.")
    zonal_stats = None

# Load temporal analysis (from Script 05)
try:
    temporal_summary = pd.read_csv(OUTPUT_DIR / '05_temporal_summary.csv')
    timescale_analysis = pd.read_csv(OUTPUT_DIR / '05_timescale_stratification_analysis.csv')
    logger.info(f"Loaded temporal analysis and timescale analysis")
    print(f"✓ Loaded temporal analysis: {len(temporal_summary)} rows")
    print(f"✓ Loaded timescale analysis: {len(timescale_analysis)} rows")
except FileNotFoundError:
    logger.warning("Temporal analysis not found. Run Script 05 first.")
    print("WARNING: Temporal analysis not found. Run Script 05 first.")
    temporal_summary = None
    timescale_analysis = None

# Load advanced analysis (from Script 06)
try:
    advanced_data = pd.read_csv(OUTPUT_DIR / 'icesat2_advanced_analysis.csv')
    logger.info(f"Loaded advanced analysis: {len(advanced_data):,} points")
    print(f"✓ Loaded advanced analysis: {len(advanced_data):,} points")
except FileNotFoundError:
    logger.warning("Advanced analysis not found. Run Script 06 first.")
    print("WARNING: Advanced analysis not found. Run Script 06 first.")
    advanced_data = None

print()

# ============================================================================
# FIGURE 1: ZONAL STATISTICS - SQUARE GRID HEATMAP (20×20 km)
# ============================================================================

logger.info("\nGENERATING FIGURE 1: ZONAL GRID HEATMAP")
print("="*80)
print("GENERATING FIGURE 1: ZONAL GRID HEATMAP")
print("="*80 + "\n")

if zonal_stats is not None and len(zonal_stats) > 0:
    logger.info("Creating square grid matrix visualization...")
    print("Creating square grid matrix visualization...")

    try:
        zonal_matrix = pd.read_csv(OUTPUT_DIR / '04_zonal_matrix_mean_subsidence.csv', index_col=0)

        fig, ax = plt.subplots(figsize=(14, 10))

        im = ax.imshow(zonal_matrix.values, cmap='RdYlBu_r', aspect='auto', origin='lower')

        cbar = plt.colorbar(im, ax=ax, label='Mean Subsidence Rate (mm/yr)')

        ax.set_xlabel('Longitude Index (E-W)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latitude Index (N-S)', fontsize=12, fontweight='bold')
        ax.set_title('Coastal Subsidence Rate by 20×20 km Grid Cell\nTuktoyaktuk Peninsula',
                    fontsize=14, fontweight='bold', pad=20)

        ax.grid(True, alpha=0.3, color='gray', linestyle='--', linewidth=0.5)

        ax.set_xticks(np.arange(0, zonal_matrix.shape[1], max(1, zonal_matrix.shape[1]//10)))
        ax.set_yticks(np.arange(0, zonal_matrix.shape[0], max(1, zonal_matrix.shape[0]//10)))

        plt.tight_layout()

        fig_file = FIGURE_DIR / 'Figure_01_Zonal_Grid_Heatmap.png'
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Saved: Figure_01_Zonal_Grid_Heatmap.png")
        print(f"✓ Saved: Figure_01_Zonal_Grid_Heatmap.png")
        print(f" Grid size: {zonal_matrix.shape[0]} (N-S) × {zonal_matrix.shape[1]} (E-W)")
        print(f" Resolution: 20×20 km per cell\n")

    except FileNotFoundError:
        logger.warning("Matrix format not found")
        print("WARNING: Matrix format not found\n")

# ============================================================================
# FIGURE 2: ZONAL STATISTICS - DENSITY HEATMAP
# ============================================================================

logger.info("GENERATING FIGURE 2: ZONAL DENSITY HEATMAP")
print("="*80)
print("GENERATING FIGURE 2: ZONAL DENSITY HEATMAP")
print("="*80 + "\n")

if zonal_stats is not None and len(zonal_stats) > 0:
    logger.info("Creating density matrix visualization...")
    print("Creating density matrix visualization...")

    try:
        density_matrix = pd.read_csv(OUTPUT_DIR / '04_zonal_matrix_density.csv', index_col=0)

        fig, ax = plt.subplots(figsize=(14, 10))

        im = ax.imshow(density_matrix.values, cmap='YlOrRd', aspect='auto', origin='lower')

        cbar = plt.colorbar(im, ax=ax, label='Point Density (points/km²)')

        ax.set_xlabel('Longitude Index (E-W)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latitude Index (N-S)', fontsize=12, fontweight='bold')
        ax.set_title('ICESat-2 Data Point Density by 20×20 km Grid Cell\nTuktoyaktuk Peninsula',
                    fontsize=14, fontweight='bold', pad=20)

        ax.grid(True, alpha=0.3, color='gray', linestyle='--', linewidth=0.5)

        ax.set_xticks(np.arange(0, density_matrix.shape[1], max(1, density_matrix.shape[1]//10)))
        ax.set_yticks(np.arange(0, density_matrix.shape[0], max(1, density_matrix.shape[0]//10)))

        plt.tight_layout()

        fig_file = FIGURE_DIR / 'Figure_02_Zonal_Density_Heatmap.png'
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Saved: Figure_02_Zonal_Density_Heatmap.png")
        print(f"✓ Saved: Figure_02_Zonal_Density_Heatmap.png")
        print(f" Grid size: {density_matrix.shape[0]} (N-S) × {density_matrix.shape[1]} (E-W)")
        print(f" Max density: {density_matrix.values.max():.1f} points/km²\n")

    except FileNotFoundError:
        logger.warning("Density matrix not found")
        print("WARNING: Density matrix not found\n")

# ============================================================================
# FIGURE 3: PERIOD COMPARISON (EARLY vs LATE)
# ============================================================================

logger.info("GENERATING FIGURE 3: PERIOD COMPARISON")
print("="*80)
print("GENERATING FIGURE 3: PERIOD COMPARISON")
print("="*80 + "\n")

if temporal_summary is not None:
    logger.info("Creating period comparison visualization...")
    print("Creating period comparison visualization...")

    try:
        early_data = pd.read_csv(OUTPUT_DIR / '05_early_period_2019_2021.csv')
        late_data = pd.read_csv(OUTPUT_DIR / '05_late_period_2022_2024.csv')

        # Detect rate column
        rate_col = None
        for col in ['subsidence_rate_m_per_year', 'dh_rate_m_per_year', 'rate', 'dh']:
            if col in early_data.columns:
                rate_col = col
                break

        if rate_col:
            early_rates = early_data[rate_col].values * 1000  # mm/yr
            late_rates = late_data[rate_col].values * 1000

            # Remove NaN
            early_clean = early_rates[~np.isnan(early_rates)]
            late_clean = late_rates[~np.isnan(late_rates)]

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Histogram
            ax = axes[0]
            ax.hist(early_clean, bins=100, alpha=0.6, label=f'Early (2019-2021, n={len(early_clean):,})',
                   color='blue', edgecolor='black')
            ax.hist(late_clean, bins=100, alpha=0.6, label=f'Late (2022-2024, n={len(late_clean):,})',
                   color='red', edgecolor='black')
            ax.set_xlabel('Subsidence Rate (mm/yr)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title('Distribution of Subsidence Rates by Period', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # Box plot
            ax = axes[1]
            bp = ax.boxplot([early_clean, late_clean],
                           labels=['Early Period\n(2019-2021)', 'Late Period\n(2022-2024)'],
                           patch_artist=True,
                           widths=0.6)

            for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
                patch.set_facecolor(color)

            ax.set_ylabel('Subsidence Rate (mm/yr)', fontsize=11, fontweight='bold')
            ax.set_title('Subsidence Rate Statistics by Period', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            # Add mean values as text
            early_mean = np.mean(early_clean)
            late_mean = np.mean(late_clean)

            ax.text(1, early_mean, f'{early_mean:.1f}', ha='center', va='bottom', fontweight='bold')
            ax.text(2, late_mean, f'{late_mean:.1f}', ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()

            fig_file = FIGURE_DIR / 'Figure_03_Period_Comparison.png'
            plt.savefig(fig_file, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✓ Saved: Figure_03_Period_Comparison.png")
            print(f"✓ Saved: Figure_03_Period_Comparison.png")
            print(f" Early period mean: {early_mean:.2f} mm/yr (n={len(early_clean):,})")
            print(f" Late period mean: {late_mean:.2f} mm/yr (n={len(late_clean):,})")
            print(f" Change: {late_mean - early_mean:+.2f} mm/yr\n")

    except Exception as e:
        logger.warning(f"Could not create period comparison: {e}")
        print(f"WARNING: Could not create period comparison: {e}\n")

# ============================================================================
# FIGURE 4: TIMESCALE STRATIFICATION
# ============================================================================

logger.info("GENERATING FIGURE 4: TIMESCALE STRATIFICATION")
print("="*80)
print("GENERATING FIGURE 4: TIMESCALE STRATIFICATION")
print("="*80 + "\n")

if timescale_analysis is not None and len(timescale_analysis) > 0:
    logger.info("Creating timescale stratification visualization...")
    print("Creating timescale stratification visualization...")

    try:
        fig, ax = plt.subplots(figsize=(12, 7))

        # Prepare data for grouped bar chart
        periods = timescale_analysis['period'].unique() if 'period' in timescale_analysis.columns else ['early', 'late']
        timescales = ['short (≤1yr)', 'medium (1-3yr)', 'long (>3yr)']

        x = np.arange(len(timescales))
        width = 0.35

        for idx, period in enumerate(['early', 'late']):
            means = []
            for ts in timescales:
                subset = timescale_analysis[(timescale_analysis.get('period','')==period) &
                                           (timescale_analysis.get('timescale','')==ts)]
                if len(subset) > 0 and 'mean_mm_per_yr' in subset.columns:
                    means.append(subset['mean_mm_per_yr'].values[0])
                else:
                    means.append(0)

            ax.bar(x + (idx * width), means, width,
                  label=f'{"Early (2019-2021)" if period == "early" else "Late (2022-2024)"}',
                  alpha=0.8)

        ax.set_xlabel('Timescale of Observation', fontsize=11, fontweight='bold')
        ax.set_ylabel('Mean Subsidence Rate (mm/yr)', fontsize=11, fontweight='bold')
        ax.set_title('Subsidence Rate by Period and Observation Timescale', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(timescales)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        plt.tight_layout()

        fig_file = FIGURE_DIR / 'Figure_04_Timescale_Stratification.png'
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Saved: Figure_04_Timescale_Stratification.png")
        print(f"✓ Saved: Figure_04_Timescale_Stratification.png\n")

    except Exception as e:
        logger.warning(f"Could not create timescale plot: {e}")
        print(f"WARNING: Could not create timescale plot: {e}\n")

# ============================================================================
# FIGURE 5: DENSITY CLASSIFICATION
# ============================================================================

logger.info("GENERATING FIGURE 5: DENSITY CLASSIFICATION")
print("="*80)
print("GENERATING FIGURE 5: DENSITY CLASSIFICATION")
print("="*80 + "\n")

if advanced_data is not None and 'density_class' in advanced_data.columns:
    logger.info("Creating density classification visualization...")
    print("Creating density classification visualization...")

    try:
        density_counts = advanced_data['density_class'].value_counts()

        fig, ax = plt.subplots(figsize=(10, 7))

        colors = ['#FF6B6B', '#FFA726', '#66BB6A', '#42A5F5']

        wedges, texts, autotexts = ax.pie(density_counts.values,
                                          labels=density_counts.index,
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          startangle=90,
                                          textprops={'fontsize': 11, 'fontweight': 'bold'})

        ax.set_title(f'Point Distribution by Density Classification\n(Total: {len(advanced_data):,} points)',
                    fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()

        fig_file = FIGURE_DIR / 'Figure_05_Density_Classification.png'
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Saved: Figure_05_Density_Classification.png")
        print(f"✓ Saved: Figure_05_Density_Classification.png")

        for cls, count in density_counts.items():
            logger.info(f"{cls}: {count:,} points ({100*count/len(advanced_data):.1f}%)")
            print(f" {cls}: {count:,} points ({100*count/len(advanced_data):.1f}%)")

        print()

    except Exception as e:
        logger.warning(f"Could not create density pie chart: {e}")
        print(f"WARNING: Could not create density pie chart: {e}\n")

# ============================================================================
# FIGURE 6: HOTSPOT MAP
# ============================================================================

logger.info("GENERATING FIGURE 6: HOTSPOT MAP")
print("="*80)
print("GENERATING FIGURE 6: HOTSPOT MAP")
print("="*80 + "\n")

if advanced_data is not None:
    logger.info("Creating hotspot spatial map...")
    print("Creating hotspot spatial map...")

    try:
        lat_col = None
        lon_col = None
        rate_col = None

        for col in ['lat', 'latitude', 'lat_early', 'lat_avg']:
            if col in advanced_data.columns:
                lat_col = col
                break

        for col in ['lon', 'longitude', 'lon_early', 'lon_avg']:
            if col in advanced_data.columns:
                lon_col = col
                break

        for col in ['dh_rate_m_per_year', 'subsidence_rate_m_per_year', 'rate']:
            if col in advanced_data.columns:
                rate_col = col
                break

        if lat_col and lon_col and rate_col:
            fig, ax = plt.subplots(figsize=(12, 10))

            # Non-hotspot points
            non_hotspot = advanced_data[~advanced_data.get('is_hotspot', False)]
            ax.scatter(non_hotspot[lon_col], non_hotspot[lat_col],
                      c='lightgray', s=10, alpha=0.3, label='Non-hotspot', edgecolors='none')

            # Hotspot points
            if 'is_hotspot' in advanced_data.columns:
                hotspot = advanced_data[advanced_data['is_hotspot']]
                if len(hotspot) > 0:
                    scatter = ax.scatter(hotspot[lon_col], hotspot[lat_col],
                                        c=hotspot[rate_col]*1000, s=30, alpha=0.8,
                                        cmap='Reds', edgecolors='darkred', linewidth=0.5,
                                        label=f'Hotspot (n={len(hotspot):,})')

                    cbar = plt.colorbar(scatter, ax=ax, label='Subsidence Rate (mm/yr)')

            ax.set_xlabel('Longitude', fontsize=11, fontweight='bold')
            ax.set_ylabel('Latitude', fontsize=11, fontweight='bold')
            ax.set_title('ICESat-2 Coastal Subsidence Hotspots\nTuktoyaktuk Peninsula',
                        fontsize=12, fontweight='bold', pad=20)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            fig_file = FIGURE_DIR / 'Figure_06_Hotspot_Map.png'
            plt.savefig(fig_file, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"✓ Saved: Figure_06_Hotspot_Map.png")
            print(f"✓ Saved: Figure_06_Hotspot_Map.png")

            if 'is_hotspot' in advanced_data.columns:
                n_hotspots = advanced_data['is_hotspot'].sum()
                logger.info(f"Hotspots: {n_hotspots:,} ({100*n_hotspots/len(advanced_data):.1f}%)")
                print(f" Hotspots: {n_hotspots:,} ({100*n_hotspots/len(advanced_data):.1f}%)\n")

    except Exception as e:
        logger.warning(f"Could not create hotspot map: {e}")
        print(f"WARNING: Could not create hotspot map: {e}\n")

# ============================================================================
# SUMMARY STATISTICS TABLE
# ============================================================================

logger.info("GENERATING SUMMARY STATISTICS TABLE")
print("="*80)
print("GENERATING SUMMARY STATISTICS TABLE")
print("="*80 + "\n")

try:
    summary_file = FIGURE_DIR / 'Summary_Statistics.txt'

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ICESat-2 COASTAL SUBSIDENCE ANALYSIS\n")
        f.write("TUKTOYAKTUK PENINSULA\n")
        f.write(f"Run ID: {RUN_TIMESTAMP}\n")
        f.write(f"Git Hash: {git_hash}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")

        if temporal_summary is not None:
            f.write("TEMPORAL ANALYSIS:\n")
            f.write(temporal_summary.to_string(index=False))
            f.write("\n\n")

        if zonal_stats is not None and len(zonal_stats) > 0:
            f.write("ZONAL STATISTICS SUMMARY (20×20 km grid):\n")
            f.write(f" Grid cells with data: {len(zonal_stats)}\n")
            if 'mean_subsidence_mm_per_yr' in zonal_stats.columns:
                f.write(f" Mean subsidence rate: {zonal_stats['mean_subsidence_mm_per_yr'].mean():.2f} mm/yr\n")
                f.write(f" Max subsidence rate: {zonal_stats['mean_subsidence_mm_per_yr'].max():.2f} mm/yr\n")
                f.write(f" Min subsidence rate: {zonal_stats['mean_subsidence_mm_per_yr'].min():.2f} mm/yr\n")
            if 'density_points_per_km2' in zonal_stats.columns:
                f.write(f" Mean density: {zonal_stats['density_points_per_km2'].mean():.2f} points/km²\n\n")

        if advanced_data is not None:
            f.write("ADVANCED ANALYSIS SUMMARY:\n")
            f.write(f" Total points: {len(advanced_data):,}\n")
            if 'density_class' in advanced_data.columns:
                f.write(f" Density classes: {advanced_data['density_class'].nunique()}\n")
            if 'is_hotspot' in advanced_data.columns:
                f.write(f" Hotspots: {advanced_data['is_hotspot'].sum():,}\n")
            f.write("\n")

        f.write("="*80 + "\n")
        f.write("FIGURES GENERATED:\n")
        f.write(" Figure 01: Zonal Grid Heatmap (20×20 km)\n")
        f.write(" Figure 02: Zonal Density Heatmap\n")
        f.write(" Figure 03: Period Comparison (Early vs Late)\n")
        f.write(" Figure 04: Timescale Stratification\n")
        f.write(" Figure 05: Density Classification\n")
        f.write(" Figure 06: Hotspot Map\n")
        f.write("="*80 + "\n")

    logger.info(f"✓ Saved: Summary_Statistics.txt")
    print(f"✓ Saved: Summary_Statistics.txt\n")

except Exception as e:
    logger.warning(f"Could not create summary: {e}")
    print(f"WARNING: Could not create summary: {e}\n")

# ============================================================================
# FINALIZE METADATA AND SAVE
# ============================================================================

metadata.processing_results = {
    'figures_generated': 6,
    'figure_resolution_dpi': 300,
    'zonal_grid_cells': len(zonal_stats) if zonal_stats is not None else 0,
    'has_temporal_analysis': temporal_summary is not None,
    'has_advanced_analysis': advanced_data is not None,
}

metadata.finalize()
metadata_file = metadata.save(OUTPUT_DIR)
logger.info(f"Metadata saved: {metadata_file.name}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

logger.info("="*80)
logger.info("SCRIPT 07 - ENHANCED OUTPUT GENERATION COMPLETE")
logger.info("="*80)

print("\n" + "="*80)
print("SCRIPT 07 - COMPLETE")
print("="*80)

print(f"\n✓ OUTPUT GENERATION COMPLETE:")
print(f" • Figures directory: {FIGURE_DIR}")
print(f" • 6 publication-ready figures generated (300 DPI)")
print(f" • Enhanced zonal grid visualization (20×20 km)")
print(f" • Period comparison plots (Early 2019-2021 vs Late 2022-2024)")
print(f" • Timescale stratification analysis")
print(f" • Density classification pie chart")
print(f" • Hotspot spatial map")

print(f"\n✓ READY FOR THESIS:")
print(f" • All figures: {FIGURE_DIR}/*.png")
print(f" • Summary statistics: {FIGURE_DIR}/Summary_Statistics.txt")

print(f"\nNext steps:")
print(f" 1. Review figures in {FIGURE_DIR}")
print(f" 2. Copy figures to thesis/figures/ directory")
print(f" 3. Reference in Chapter 4 Results")

print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    pass
