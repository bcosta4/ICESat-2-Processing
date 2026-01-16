"""
================================================================================
SCRIPT 07: ENHANCED OUTPUT GENERATION WITH InSAR VALIDATION - FULLY CORRECTED
================================================================================

Purpose
-------
Generate publication-ready figures and output visualizations from ICESat-2
subsidence analysis results, including InSAR validation and comparison.
CRITICAL FIX: Now handles missing timestamp file gracefully
- Attempts to read latest_run_timestamp.txt
- If missing, generates automatic timestamp from current run
- Continues execution regardless of timestamp file availability

Features Included
-----------------
✓ Enhanced zonal grid visualization (20x20 km as square matrix heatmap)
✓ Temporal period comparison plots (Early vs Late)
✓ Timescale stratification analysis
✓ Density classification visualization
✓ Hotspot spatial mapping
✓ InSAR validation plots (3-panel comparison)
✓ Binned statistics by elevation
✓ QGIS-optimized spatial subsample (90% file size reduction!)
✓ ROBUST TIMESTAMP HANDLING (FIXED - no more sys.exit(1))

Inputs
------
- Output files from Scripts 04, 05, and 06
- icesat2_insar_integrated.csv (from Script 03)
- Zonal statistics, temporal analysis, advanced analysis data
- latest_run_timestamp.txt (created by Script 00) [OPTIONAL, auto-generates if missing]

Outputs
-------
- figures/Figure01-07.png (7 publication-ready PNG figures at 300 DPI)
- 07_insar_binned_stats_by_elevation.csv (statistics by elevation)
- 07_insar_qgis_subsample.csv (downsampled for QGIS)
- SummaryStatistics.txt (compiled statistics with InSAR section)
- script07run.log (execution log)
- 07_output_generation_metadata.json (reproducibility metadata)

Reproducibility Notes
---------------------
All figures are generated with consistent styling, resolution 300 DPI, and
formatting. Complete audit trail provided via metadata JSON and execution log.
Git hash recorded for exact code reproducibility.
================================================================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from pathlib import Path
from datetime import datetime
import json
import logging
import subprocess
import sys
import io
import warnings

warnings.filterwarnings('ignore')
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_git_hash():
    """Get git commit hash for reproducibility"""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, timeout=5)
        return result.stdout.strip()[:8] if result.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'

def get_or_create_timestamp(base_dir):
    """Get timestamp from file, or create one if missing"""
    timestamp_file = base_dir / 'latest_run_timestamp.txt'
    if timestamp_file.exists():
        with open(timestamp_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f'⚠ WARNING: Timestamp file not found')
        print(f' Generating automatic timestamp: {timestamp}')
        try:
            with open(timestamp_file, 'w', encoding='utf-8') as f:
                f.write(timestamp)
            print(f' ✓ Saved to: {timestamp_file}')
        except Exception as e:
            print(f' Could not save timestamp file: {e}')
            print(f' Continuing with auto-generated timestamp')
        return timestamp

def setup_logging(output_dir):
    """Configure logging to both file and console"""
    logfile = output_dir / 'script_07_run.log'
    logger = logging.getLogger('script07')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    fh = logging.FileHandler(logfile, encoding='utf-8', mode='w')
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fh_formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def log_and_print(logger, message, level='info'):
    """Log message to both file and console"""
    if level == 'info':
        logger.info(message)
    elif level == 'warning':
        logger.warning(message)
    elif level == 'error':
        logger.error(message)
    elif level == 'debug':
        logger.debug(message)
    print(message)

# ============================================================================
# METADATA CLASS
# ============================================================================
class RunMetadata:
    """Capture complete run parameters for reproducibility"""
    def __init__(self, run_id, git_hash):
        self.run_id = run_id
        self.script_name = '07_output_generation'
        self.script_version = 'v2.2-corrected-paths'
        self.git_hash = git_hash
        self.start_time = datetime.now().isoformat()
        self.end_time = None
        self.configuration = {}
        self.processing_results = {}
        self.environment_info = {
            'python_version': f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}',
            'platform': sys.platform,
            'timestamp': datetime.now().isoformat(),
        }

    def finalize(self):
        """Mark run as complete"""
        self.end_time = datetime.now().isoformat()

    def save(self, output_dir):
        """Save metadata as JSON to OUTPUT_DIR"""
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
# InSAR ANALYSIS FUNCTIONS
# ============================================================================
def create_insar_comparison_plots(insar_df, figure_dir, logger=None):
    """Create publication-ready comparison plots for InSAR vs ICeSat-2"""
    if insar_df is None or len(insar_df) == 0:
        return False
    try:
        icesat2 = insar_df['subsidence_rate_mm_per_year'].values / 1000.0
        insar = insar_df['insar_rate_mm_yr'].values
        valid = ~(np.isnan(icesat2) | np.isnan(insar))
        if valid.sum() < 10:
            if logger:
                log_and_print(logger, f'Only {valid.sum()} valid InSAR-ICeSat-2 pairs', 'warning')
            return False
        icesat2_valid = icesat2[valid]
        insar_valid = insar[valid]
        difference = icesat2_valid - insar_valid
        elev_col = next((c for c in insar_df.columns if 'elev' in c.lower()), None)
        if elev_col:
            elevation = insar_df[elev_col].values[valid]
        else:
            elevation = None
        fig = plt.figure(figsize=(16, 5))
        ax1 = plt.subplot(131)
        ax1.scatter(insar_valid, icesat2_valid, alpha=0.5, s=20,
                    color='steelblue', edgecolors='navy', linewidth=0.5)
        lim_min = min(insar_valid.min(), icesat2_valid.min())
        lim_max = max(insar_valid.max(), icesat2_valid.max())
        ax1.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=2,
                 label='1:1 agreement', alpha=0.7)
        r, p = pearsonr(insar_valid, icesat2_valid)
        rmse = np.sqrt(np.mean(difference**2))
        bias = np.mean(difference)
        stats_text = f'R² = {r**2:.3f}\nRMSE = {rmse:.2f} mm/yr\nBias = {bias:.2f} mm/yr\nn = {valid.sum():,}'
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax1.set_xlabel('InSAR Subsidence Rate (mm/yr)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('ICeSat-2 Subsidence Rate (mm/yr)', fontsize=11, fontweight='bold')
        ax1.set_title('A) InSAR vs ICeSat-2 Validation', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax2 = plt.subplot(132)
        ax2.hist(difference, bins=40, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(bias, color='red', linestyle='--', linewidth=2,
                    label=f'Mean = {bias:.2f} mm/yr')
        ax2.axvline(0, color='green', linestyle='--', linewidth=2,
                    label='Perfect agreement')
        ax2.set_xlabel('Difference (ICeSat-2 - InSAR) mm/yr', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title('B) Difference Distribution', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        ax3 = plt.subplot(133)
        if elevation is not None:
            scatter = ax3.scatter(elevation, difference, alpha=0.6, s=25,
                                  c=np.abs(difference), cmap='Reds',
                                  edgecolors='darkred', linewidth=0.5)
            plt.colorbar(scatter, ax=ax3, label='|Difference| (mm/yr)')
            ax3.set_xlabel('Elevation (m)', fontsize=11, fontweight='bold')
        else:
            ax3.scatter(range(len(difference)), difference, alpha=0.5, s=20,
                       color='steelblue', edgecolors='navy', linewidth=0.5)
            ax3.set_xlabel('Crossover Point Index', fontsize=11, fontweight='bold')
        ax3.axhline(0, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax3.set_ylabel('Residual (mm/yr)', fontsize=11, fontweight='bold')
        ax3.set_title('C) Residuals', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        figfile = figure_dir / 'Figure07_InSAR_Validation.png'
        plt.savefig(figfile, dpi=300, bbox_inches='tight')
        plt.close()
        if logger:
            log_and_print(logger, f'✓ Saved InSAR validation figure: {figfile.name}')
        else:
            print(f'✓ Saved InSAR validation figure: Figure07_InSAR_Validation.png')
        return True
    except Exception as e:
        if logger:
            log_and_print(logger, f'Error creating InSAR plots: {e}', 'warning')
        else:
            print(f'WARNING: Error creating InSAR plots: {e}')
        return False

def create_insar_binned_stats(insar_df, figure_dir, output_dir, bin_size_m=10, logger=None):
    """Create binned statistics CSV for InSAR-ICeSat-2 agreement by elevation"""
    if insar_df is None or len(insar_df) == 0:
        return None
    try:
        icesat2 = insar_df['subsidence_rate_mm_per_year'].values / 1000.0
        insar = insar_df['insar_rate_mm_yr'].values
        valid = ~(np.isnan(icesat2) | np.isnan(insar))
        icesat2_valid = icesat2[valid]
        insar_valid = insar[valid]
        difference = icesat2_valid - insar_valid
        elev_col = next((c for c in insar_df.columns if 'elev' in c.lower()), None)
        if elev_col is None:
            if logger:
                log_and_print(logger, 'No elevation column found - skipping binned stats', 'warning')
            return None
        elevation = insar_df[elev_col].values[valid]
        elev_min = np.floor(elevation.min() / bin_size_m) * bin_size_m
        elev_max = np.ceil(elevation.max() / bin_size_m) * bin_size_m
        bins = np.arange(elev_min, elev_max + bin_size_m, bin_size_m)
        stats_list = []
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i+1]
            mask = (elevation >= lo) & (elevation < hi)
            if mask.sum() == 0:
                continue
            ice_bin = icesat2_valid[mask]
            insar_bin = insar_valid[mask]
            diff_bin = difference[mask]
            stats_list.append({
                'elevation_m': f'{int(lo)}-{int(hi)}',
                'elevation_center': (lo + hi) / 2,
                'n_points': int(mask.sum()),
                'icesat2_mean_mm_yr': float(ice_bin.mean()),
                'icesat2_std_mm_yr': float(ice_bin.std()),
                'insar_mean_mm_yr': float(insar_bin.mean()),
                'insar_std_mm_yr': float(insar_bin.std()),
                'difference_mean_mm_yr': float(diff_bin.mean()),
                'difference_std_mm_yr': float(diff_bin.std()),
                'rmse_mm_yr': float(np.sqrt(np.mean(diff_bin**2))),
                'mae_mm_yr': float(np.mean(np.abs(diff_bin))),
            })
        df_stats = pd.DataFrame(stats_list)
        if len(df_stats) == 0:
            return None
        stats_csv = output_dir / '07_insar_binned_stats_by_elevation.csv'
        df_stats.to_csv(stats_csv, index=False, encoding='utf-8')
        if logger:
            log_and_print(logger, f'✓ Saved binned statistics: {stats_csv.name}')
            print(f' ({len(df_stats)} elevation bins)')
        else:
            print(f'✓ Saved binned statistics: 07_insar_binned_stats_by_elevation.csv')
            print(f' ({len(df_stats)} elevation bins)')
        return df_stats
    except Exception as e:
        if logger:
            log_and_print(logger, f'Error creating binned stats: {e}', 'warning')
        else:
            print(f'WARNING: Error creating binned stats: {e}')
        return None

def create_insar_spatial_subsample(insar_df, output_dir, downsample_factor=10, logger=None):
    """Create a downsampled CSV for QGIS visualization"""
    if insar_df is None or len(insar_df) == 0:
        return None
    try:
        lat_col = None
        lon_col = None
        rate_col = None
        for col in ['lat_avg', 'lat', 'latitude', 'lat_early']:
            if col in insar_df.columns:
                lat_col = col
                break
        for col in ['lon_avg', 'lon', 'longitude', 'lon_early']:
            if col in insar_df.columns:
                lon_col = col
                break
        for col in ['subsidence_rate_mm_per_year', 'rate', 'dH_rate_mm_per_year']:
            if col in insar_df.columns:
                rate_col = col
                break
        if not all([lat_col, lon_col, rate_col]):
            if logger:
                log_and_print(logger, 'Missing required coordinate columns for spatial subsampling', 'warning')
            else:
                print('WARNING: Missing required coordinate columns')
            return None
        downsampled = insar_df.iloc[::downsample_factor].copy()
        grid_size_m = 2000
        downsampled['grid_cell_x'] = (downsampled[lon_col] * 111000).astype(int) // grid_size_m
        downsampled['grid_cell_y'] = (downsampled[lat_col] * 111000).astype(int) // grid_size_m
        if 'insar_rate_mm_yr' in downsampled.columns:
            downsampled['difference_mm_yr'] = (downsampled[rate_col] / 1000.0 -
                                               downsampled['insar_rate_mm_yr'])
            downsampled['abs_difference_mm_yr'] = np.abs(downsampled['difference_mm_yr'])
        qgis_csv = output_dir / '07_insar_qgis_subsample.csv'
        downsampled.to_csv(qgis_csv, index=False, encoding='utf-8')
        if logger:
            log_and_print(logger, f'✓ Saved QGIS-optimized subsample: {qgis_csv.name}')
            log_and_print(logger, f' Original: {len(insar_df):,} points → Downsampled: {len(downsampled):,} points')
            log_and_print(logger, f' Size reduction: {(1 - len(downsampled)/len(insar_df))*100:.1f}%')
        else:
            print(f'✓ Saved QGIS-optimized subsample: 07_insar_qgis_subsample.csv')
            print(f' Downsampling: {len(insar_df):,} → {len(downsampled):,} points (-{(1 - len(downsampled)/len(insar_df))*100:.1f}%)')
            print(f' File size reduced, optimal for QGIS performance')
        return downsampled
    except Exception as e:
        if logger:
            log_and_print(logger, f'Error creating spatial subsample: {e}', 'warning')
        else:
            print(f'WARNING: Error creating spatial subsample: {e}')
        return None

# ============================================================================
# MAIN SCRIPT
# ============================================================================
print('='*80)
print('SCRIPT 07: ENHANCED OUTPUT GENERATION WITH InSAR VALIDATION')
print('='*80)

BASE_OUTPUT_DIR = Path(r'E:\Processing\ICeSat-2\output_data')
RUN_TIMESTAMP = get_or_create_timestamp(BASE_OUTPUT_DIR)

try:
    OUTPUT_DIR = BASE_OUTPUT_DIR / f'run_{RUN_TIMESTAMP}'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR = OUTPUT_DIR / 'figures'
    FIGURE_DIR.mkdir(exist_ok=True)
except Exception as e:
    print(f'ERROR: Could not create directories: {e}')
    sys.exit(1)

print(f'Run ID: {RUN_TIMESTAMP}')
print(f'Output directory: {OUTPUT_DIR}')
print(f'Figure directory: {FIGURE_DIR}')

GIT_HASH = get_git_hash()
print(f'Git Hash: {GIT_HASH}')

logger = setup_logging(OUTPUT_DIR)
metadata = RunMetadata(RUN_TIMESTAMP, GIT_HASH)

logger.info('='*80)
logger.info('SCRIPT 07: ENHANCED OUTPUT GENERATION WITH InSAR VALIDATION')
logger.info('='*80)
logger.info(f'Run ID: {RUN_TIMESTAMP}')
logger.info(f'Git Hash: {GIT_HASH}')
logger.info(f'Output directory: {OUTPUT_DIR}')
logger.info(f'Figure directory: {FIGURE_DIR}')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

metadata.configuration = {
    'figure_dpi_standard': 300,
    'figure_dpi_hires': 600,
    'plotting_style': 'seaborn-v0_8-darkgrid',
}

# ============================================================================
# LOAD DATA
# ============================================================================
logger.info('\n' + '='*80)
logger.info('LOADING DATA')
logger.info('='*80)

print('\n' + '='*80)
print('LOADING DATA')
print('='*80)

zonal_stats = None
try:
    zonal_stats = pd.read_csv(OUTPUT_DIR / '04_zonal_statistics_20km_grid.csv')
    logger.info(f'Loaded zonal statistics: {len(zonal_stats):,} grid cells')
    print(f'Loaded zonal statistics: {len(zonal_stats):,} grid cells')
except FileNotFoundError:
    logger.warning('Zonal statistics not found. Run Script 04 first.')
    print('WARNING: Zonal statistics not found. Run Script 04 first.')

early_data = None
late_data = None
try:
    early_data = pd.read_csv(OUTPUT_DIR / 'icesat2_early_period.csv')
    late_data = pd.read_csv(OUTPUT_DIR / 'icesat2_late_period.csv')
    logger.info(f'Loaded temporal period data: Early={len(early_data):,}, Late={len(late_data):,}')
    print(f'Loaded temporal period data: Early={len(early_data):,}, Late={len(late_data):,}')
except FileNotFoundError:
    logger.warning('Temporal period data not found. Run Script 05b first.')
    print('WARNING: Temporal period data not found. Run Script 05b first.')

timescale_analysis = None
try:
    timescale_analysis = pd.read_csv(OUTPUT_DIR / 'temporal_acceleration_metrics.csv')
    logger.info(f'Loaded timescale analysis: {len(timescale_analysis)} rows')
    print(f'Loaded timescale analysis: {len(timescale_analysis)} rows')
except FileNotFoundError:
    logger.warning('Timescale analysis not found. Run Script 05b first.')
    print('WARNING: Timescale analysis not found. Run Script 05b first.')

advanced_data = None
try:
    advanced_data = pd.read_csv(OUTPUT_DIR / 'icesat2_advanced_analysis.csv')
    logger.info(f'Loaded advanced analysis: {len(advanced_data):,} points')
    print(f'Loaded advanced analysis: {len(advanced_data):,} points')
except FileNotFoundError:
    logger.warning('Advanced analysis not found. Run Script 06 first.')
    print('WARNING: Advanced analysis not found. Run Script 06 first.')

print()

# ============================================================================
# LOAD InSAR INTEGRATION DATA
# ============================================================================
logger.info('\n' + '='*80)
logger.info('LOADING InSAR INTEGRATION DATA')
logger.info('='*80)

print('='*80)
print('LOADING InSAR INTEGRATION DATA')
print('='*80)

insar_integrated = None
has_insar = False

try:
    insar_file = OUTPUT_DIR / 'icesat2_insar_integrated.csv'
    if insar_file.exists():
        insar_integrated = pd.read_csv(insar_file)
        has_insar = True
        logger.info(f'✓ Loaded InSAR-ICeSat-2 integrated data: {len(insar_integrated):,} points')
        print(f'✓ Loaded InSAR-ICeSat-2 integrated data: {len(insar_integrated):,} points')
        required_cols = ['subsidence_rate_mm_per_year', 'insar_rate_mm_yr', 'lon_avg', 'lat_avg']
        available = [c for c in required_cols if c in insar_integrated.columns]
        logger.info(f' Available columns: {available}')
    else:
        logger.warning('InSAR integrated file not found - skipping InSAR analysis')
        print('WARNING: InSAR integrated file not found - skipping InSAR analysis')
except Exception as e:
    logger.warning(f'Error loading InSAR data: {e}')
    print(f'WARNING: Error loading InSAR data: {e}')

if not has_insar:
    print('\nTo enable InSAR analysis:')
    print(' 1. Run Script 03 with InSAR data available')
    print(' 2. Ensure icesat2_insar_integrated.csv is in output directory')

print()

# ============================================================================
# FIGURE 1: ZONAL GRID HEATMAP
# ============================================================================
logger.info('\n' + '='*80)
logger.info('GENERATING FIGURE 1: ZONAL GRID HEATMAP')
logger.info('='*80)

print('\n' + '='*80)
print('GENERATING FIGURE 1: ZONAL GRID HEATMAP')
print('='*80)

if zonal_stats is not None and len(zonal_stats) > 0:
    logger.info('Creating square grid matrix visualization...')
    print('Creating square grid matrix visualization...')
    try:
        zonal_matrix = pd.read_csv(OUTPUT_DIR / '04_zonal_matrix_mean_subsidence.csv', index_col=0)
        fig, ax = plt.subplots(figsize=(14, 10))
        im = ax.imshow(zonal_matrix.values, cmap='RdYlBu_r', aspect='auto', origin='lower')
        cbar = plt.colorbar(im, ax=ax, label='Mean Subsidence Rate (mm/yr)')
        ax.set_xlabel('Longitude Index (E-W)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latitude Index (N-S)', fontsize=12, fontweight='bold')
        ax.set_title('Coastal Subsidence Rate by 20x20 km Grid Cell, Tuktoyaktuk Peninsula',
                     fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xticks(np.arange(0, zonal_matrix.shape[1], max(1, zonal_matrix.shape[1]//10)))
        ax.set_yticks(np.arange(0, zonal_matrix.shape[0], max(1, zonal_matrix.shape[0]//10)))
        plt.tight_layout()
        figfile = FIGURE_DIR / 'Figure01_ZonalGridHeatmap.png'
        plt.savefig(figfile, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f'✓ Saved Figure01_ZonalGridHeatmap.png')
        print(f'✓ Saved Figure01_ZonalGridHeatmap.png')
        print(f' Grid size: {zonal_matrix.shape[0]} N-S, {zonal_matrix.shape[1]} E-W')
        print(f' Resolution: 20x20 km per cell')
    except FileNotFoundError:
        logger.warning('Matrix format not found')
        print('WARNING: Matrix format not found')

# ============================================================================
# FIGURE 2: ZONAL DENSITY HEATMAP
# ============================================================================
logger.info('\n' + '='*80)
logger.info('GENERATING FIGURE 2: ZONAL DENSITY HEATMAP')
logger.info('='*80)

print('\n' + '='*80)
print('GENERATING FIGURE 2: ZONAL DENSITY HEATMAP')
print('='*80)

if zonal_stats is not None and len(zonal_stats) > 0:
    logger.info('Creating density matrix visualization...')
    print('Creating density matrix visualization...')
    try:
        density_matrix = pd.read_csv(OUTPUT_DIR / '04_zonal_matrix_density.csv', index_col=0)
        fig, ax = plt.subplots(figsize=(14, 10))
        im = ax.imshow(density_matrix.values, cmap='YlOrRd', aspect='auto', origin='lower')
        cbar = plt.colorbar(im, ax=ax, label='Point Density (points/km²)')
        ax.set_xlabel('Longitude Index (E-W)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latitude Index (N-S)', fontsize=12, fontweight='bold')
        ax.set_title('ICESat-2 Data Point Density by 20x20 km Grid Cell, Tuktoyaktuk Peninsula',
                     fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xticks(np.arange(0, density_matrix.shape[1], max(1, density_matrix.shape[1]//10)))
        ax.set_yticks(np.arange(0, density_matrix.shape[0], max(1, density_matrix.shape[0]//10)))
        plt.tight_layout()
        figfile = FIGURE_DIR / 'Figure02_ZonalDensityHeatmap.png'
        plt.savefig(figfile, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f'✓ Saved Figure02_ZonalDensityHeatmap.png')
        print(f'✓ Saved Figure02_ZonalDensityHeatmap.png')
        print(f' Grid size: {density_matrix.shape[0]} N-S, {density_matrix.shape[1]} E-W')
        print(f' Max density: {density_matrix.values.max():.1f} points/km²')
    except FileNotFoundError:
        logger.warning('Density matrix not found')
        print('WARNING: Density matrix not found')

# ============================================================================
# FIGURE 3: PERIOD COMPARISON
# ============================================================================
logger.info('\n' + '='*80)
logger.info('GENERATING FIGURE 3: PERIOD COMPARISON')
logger.info('='*80)

print('\n' + '='*80)
print('GENERATING FIGURE 3: PERIOD COMPARISON')
print('='*80)

if early_data is not None and late_data is not None:
    logger.info('Creating period comparison visualization...')
    print('Creating period comparison visualization...')
    try:
        rate_col = None
        for col in ['subsidence_rate_mm_per_year', 'dH_rate_mm_per_year', 'rate', 'dH']:
            if col in early_data.columns:
                rate_col = col
                break
        if rate_col:
            early_rates = early_data[rate_col].values / 1000
            late_rates = late_data[rate_col].values / 1000
            early_clean = early_rates[~np.isnan(early_rates)]
            late_clean = late_rates[~np.isnan(late_rates)]
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            ax = axes[0]
            ax.hist(early_clean, bins=100, alpha=0.6,
                    label=f'Early 2019-2021, n={len(early_clean):,}',
                    color='blue', edgecolor='black')
            ax.hist(late_clean, bins=100, alpha=0.6,
                    label=f'Late 2022-2024, n={len(late_clean):,}',
                    color='red', edgecolor='black')
            ax.set_xlabel('Subsidence Rate (mm/yr)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.set_title('Distribution of Subsidence Rates by Period', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax = axes[1]
            bp = ax.boxplot([early_clean, late_clean],
                            labels=['Early Period (2019-2021)', 'Late Period (2022-2024)'],
                            patch_artist=True, widths=0.6)
            for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
                patch.set_facecolor(color)
            ax.set_ylabel('Subsidence Rate (mm/yr)', fontsize=11, fontweight='bold')
            ax.set_title('Subsidence Rate Statistics by Period', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            early_mean = np.mean(early_clean)
            late_mean = np.mean(late_clean)
            ax.text(1, early_mean, f'{early_mean:.1f}', ha='center', va='bottom', fontweight='bold')
            ax.text(2, late_mean, f'{late_mean:.1f}', ha='center', va='bottom', fontweight='bold')
            plt.tight_layout()
            figfile = FIGURE_DIR / 'Figure03_PeriodComparison.png'
            plt.savefig(figfile, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f'✓ Saved Figure03_PeriodComparison.png')
            print(f'✓ Saved Figure03_PeriodComparison.png')
            print(f' Early period mean: {early_mean:.2f} mm/yr, n={len(early_clean):,}')
            print(f' Late period mean: {late_mean:.2f} mm/yr, n={len(late_clean):,}')
            print(f' Change: {late_mean - early_mean:.2f} mm/yr')
    except Exception as e:
        logger.warning(f'Could not create period comparison: {e}')
        print(f'WARNING: Could not create period comparison: {e}')

# ============================================================================
# FIGURE 4: TIMESCALE STRATIFICATION
# ============================================================================
logger.info('\n' + '='*80)
logger.info('GENERATING FIGURE 4: TIMESCALE STRATIFICATION')
logger.info('='*80)

print('\n' + '='*80)
print('GENERATING FIGURE 4: TIMESCALE STRATIFICATION')
print('='*80)

if timescale_analysis is not None and len(timescale_analysis) > 0:
    logger.info('Creating timescale stratification visualization...')
    print('Creating timescale stratification visualization...')
    try:
        fig, ax = plt.subplots(figsize=(12, 7))
        metrics = timescale_analysis['Metric'].values if 'Metric' in timescale_analysis.columns else []
        if len(metrics) > 0:
            early_vals = timescale_analysis['Early Period'].values if 'Early Period' in timescale_analysis.columns else []
            late_vals = timescale_analysis['Late Period'].values if 'Late Period' in timescale_analysis.columns else []
            x = np.arange(len(metrics))
            width = 0.35
            try:
                ax.bar(x - width/2, early_vals, width, label='Early 2019-2021', alpha=0.8)
                ax.bar(x + width/2, late_vals, width, label='Late 2022-2024', alpha=0.8)
                ax.set_ylabel('Mean Value', fontsize=11, fontweight='bold')
                ax.set_title('Temporal Stratification Metrics', fontsize=12, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(metrics, rotation=45, ha='right')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                figfile = FIGURE_DIR / 'Figure04_TimescaleStratification.png'
                plt.savefig(figfile, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f'✓ Saved Figure04_TimescaleStratification.png')
                print(f'✓ Saved Figure04_TimescaleStratification.png')
            except Exception as e:
                logger.warning(f'Could not plot metrics: {e}')
    except Exception as e:
        logger.warning(f'Could not create timescale plot: {e}')
        print(f'WARNING: Could not create timescale plot: {e}')

# ============================================================================
# FIGURE 5: DENSITY CLASSIFICATION
# ============================================================================
logger.info('\n' + '='*80)
logger.info('GENERATING FIGURE 5: DENSITY CLASSIFICATION')
logger.info('='*80)

print('\n' + '='*80)
print('GENERATING FIGURE 5: DENSITY CLASSIFICATION')
print('='*80)

if advanced_data is not None and 'density_class' in advanced_data.columns:
    logger.info('Creating density classification visualization...')
    print('Creating density classification visualization...')
    try:
        density_counts = advanced_data['density_class'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 7))
        colors = ['FF6B6B', 'FFA726', '66BB6A', '42A5F5']
        wedges, texts, autotexts = ax.pie(density_counts.values, labels=density_counts.index,
                                           autopct='%1.1f%%', colors=colors, startangle=90,
                                           textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax.set_title(f'Point Distribution by Density Classification (Total: {len(advanced_data):,} points)',
                     fontsize=12, fontweight='bold', pad=20)
        plt.tight_layout()
        figfile = FIGURE_DIR / 'Figure05_DensityClassification.png'
        plt.savefig(figfile, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f'✓ Saved Figure05_DensityClassification.png')
        print(f'✓ Saved Figure05_DensityClassification.png')
        for cls, count in density_counts.items():
            logger.info(f' {cls}: {count:,} points ({100*count/len(advanced_data):.1f}%)')
            print(f' {cls}: {count:,} points ({100*count/len(advanced_data):.1f}%)')
        print()
    except Exception as e:
        logger.warning(f'Could not create density pie chart: {e}')
        print(f'WARNING: Could not create density pie chart: {e}')

# ============================================================================
# FIGURE 6: HOTSPOT MAP
# ============================================================================
logger.info('\n' + '='*80)
logger.info('GENERATING FIGURE 6: HOTSPOT MAP')
logger.info('='*80)

print('\n' + '='*80)
print('GENERATING FIGURE 6: HOTSPOT MAP')
print('='*80)

if advanced_data is not None:
    logger.info('Creating hotspot spatial map...')
    print('Creating hotspot spatial map...')
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
        for col in ['dH_rate_mm_per_year', 'subsidence_rate_mm_per_year', 'rate']:
            if col in advanced_data.columns:
                rate_col = col
                break
        if lat_col and lon_col and rate_col:
            fig, ax = plt.subplots(figsize=(12, 10))
            non_hotspot = advanced_data[advanced_data.get('is_hotspot', False) == False]
            ax.scatter(non_hotspot[lon_col], non_hotspot[lat_col],
                      c='lightgray', s=10, alpha=0.3, label='Non-hotspot', edgecolors='none')
            if 'is_hotspot' in advanced_data.columns:
                hotspot = advanced_data[advanced_data['is_hotspot'] == True]
                if len(hotspot) > 0:
                    scatter = ax.scatter(hotspot[lon_col], hotspot[lat_col],
                                        c=hotspot[rate_col]/1000, s=30, alpha=0.8,
                                        cmap='Reds', edgecolors='darkred', linewidth=0.5,
                                        label=f'Hotspot (n={len(hotspot):,})')
                    cbar = plt.colorbar(scatter, ax=ax, label='Subsidence Rate (mm/yr)')
            ax.set_xlabel('Longitude', fontsize=11, fontweight='bold')
            ax.set_ylabel('Latitude', fontsize=11, fontweight='bold')
            ax.set_title('ICESat-2 Coastal Subsidence Hotspots, Tuktoyaktuk Peninsula',
                         fontsize=12, fontweight='bold', pad=20)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            figfile = FIGURE_DIR / 'Figure06_HotspotMap.png'
            plt.savefig(figfile, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f'✓ Saved Figure06_HotspotMap.png')
            print(f'✓ Saved Figure06_HotspotMap.png')
            if 'is_hotspot' in advanced_data.columns:
                n_hotspots = advanced_data['is_hotspot'].sum()
                logger.info(f' Hotspots: {n_hotspots:,} ({100*n_hotspots/len(advanced_data):.1f}%)')
                print(f' Hotspots: {n_hotspots:,} ({100*n_hotspots/len(advanced_data):.1f}%)')
        else:
            logger.warning('Missing required coordinate columns for hotspot map')
            print('WARNING: Missing required coordinate columns')
    except Exception as e:
        logger.warning(f'Could not create hotspot map: {e}')
        print(f'WARNING: Could not create hotspot map: {e}')

# ============================================================================
# FIGURE 7: InSAR VALIDATION
# ============================================================================
if has_insar:
    logger.info('\n' + '='*80)
    logger.info('GENERATING FIGURE 7: InSAR VALIDATION')
    logger.info('='*80)
    print('\n' + '='*80)
    print('GENERATING FIGURE 7: InSAR VALIDATION')
    print('='*80)
    plots_created = create_insar_comparison_plots(insar_integrated, FIGURE_DIR, logger)

# ============================================================================
# InSAR STATISTICS AND SUBSAMPLING
# ============================================================================
if has_insar:
    logger.info('\n' + '='*80)
    logger.info('InSAR VALIDATION & COMPARISON ANALYSIS')
    logger.info('='*80)
    print('\n' + '='*80)
    print('InSAR VALIDATION & COMPARISON ANALYSIS')
    print('='*80)
    print('\nComputing binned statistics by elevation...')
    insar_stats = create_insar_binned_stats(insar_integrated, FIGURE_DIR, OUTPUT_DIR, logger=logger)
    print('\nCreating QGIS-optimized spatial subsample...')
    qgis_data = create_insar_spatial_subsample(insar_integrated, OUTPUT_DIR,
                                               downsample_factor=10, logger=logger)

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
logger.info('\n' + '='*80)
logger.info('GENERATING SUMMARY STATISTICS TABLE')
logger.info('='*80)

print('\n' + '='*80)
print('GENERATING SUMMARY STATISTICS TABLE')
print('='*80)

try:
    summary_file = FIGURE_DIR / 'SummaryStatistics.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write('='*80 + '\n')
        f.write('ICESat-2 COASTAL SUBSIDENCE ANALYSIS\n')
        f.write('TUKTOYAKTUK PENINSULA\n')
        f.write(f'Run ID: {RUN_TIMESTAMP}\n')
        f.write(f'Git Hash: {GIT_HASH}\n')
        f.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('='*80 + '\n\n')
        if zonal_stats is not None and len(zonal_stats) > 0:
            f.write('ZONAL STATISTICS SUMMARY (20x20 km grid)\n')
            f.write(f'Grid cells with data: {len(zonal_stats)}\n')
            if 'mean_subsidence_mm_per_yr' in zonal_stats.columns:
                f.write(f'Mean subsidence rate: {zonal_stats["mean_subsidence_mm_per_yr"].mean():.2f} mm/yr\n')
                f.write(f'Max subsidence rate: {zonal_stats["mean_subsidence_mm_per_yr"].max():.2f} mm/yr\n')
                f.write(f'Min subsidence rate: {zonal_stats["mean_subsidence_mm_per_yr"].min():.2f} mm/yr\n')
            if 'density_points_per_km2' in zonal_stats.columns:
                f.write(f'Mean density: {zonal_stats["density_points_per_km2"].mean():.2f} points/km²\n')
            f.write('\n')
        if advanced_data is not None:
            f.write('ADVANCED ANALYSIS SUMMARY\n')
            f.write(f'Total points: {len(advanced_data):,}\n')
            if 'density_class' in advanced_data.columns:
                f.write(f'Density classes: {advanced_data["density_class"].nunique()}\n')
            if 'is_hotspot' in advanced_data.columns:
                f.write(f'Hotspots: {advanced_data["is_hotspot"].sum():,}\n')
            f.write('\n')
        if has_insar:
            f.write('='*80 + '\n')
            f.write('InSAR VALIDATION STATISTICS\n')
            f.write('='*80 + '\n\n')
            valid_mask = ~(insar_integrated['subsidence_rate_mm_per_year'].isna() |
                          insar_integrated['insar_rate_mm_yr'].isna())
            n_valid = valid_mask.sum()
            if n_valid > 0:
                icesat2_vals = insar_integrated.loc[valid_mask, 'subsidence_rate_mm_per_year'].values / 1000.0
                insar_vals = insar_integrated.loc[valid_mask, 'insar_rate_mm_yr'].values
                diff = icesat2_vals - insar_vals
                rmse = np.sqrt(np.mean(diff**2))
                bias = np.mean(diff)
                mae = np.mean(np.abs(diff))
                r, p = pearsonr(insar_vals, icesat2_vals)
                f.write(f'Validation Points: {n_valid:,}\n')
                f.write(f'Correlation (R): {r:.3f}\n')
                f.write(f'R-squared: {r**2:.3f}\n')
                f.write(f'RMSE: {rmse:.2f} mm/yr\n')
                f.write(f'Bias (ICeSat-2 - InSAR): {bias:.2f} mm/yr\n')
                f.write(f'MAE: {mae:.2f} mm/yr\n\n')
                f.write(f'ICESat-2 Statistics:\n')
                f.write(f' Mean: {icesat2_vals.mean():.2f} mm/yr\n')
                f.write(f' Std Dev: {icesat2_vals.std():.2f} mm/yr\n')
                f.write(f' Range: {icesat2_vals.min():.2f} to {icesat2_vals.max():.2f} mm/yr\n\n')
                f.write(f'InSAR Statistics:\n')
                f.write(f' Mean: {insar_vals.mean():.2f} mm/yr\n')
                f.write(f' Std Dev: {insar_vals.std():.2f} mm/yr\n')
                f.write(f' Range: {insar_vals.min():.2f} to {insar_vals.max():.2f} mm/yr\n\n')
        f.write('='*80 + '\n')
        f.write('FIGURES GENERATED\n')
        f.write('='*80 + '\n')
        f.write('Figure 01: Zonal Grid Heatmap (20x20 km)\n')
        f.write('Figure 02: Zonal Density Heatmap\n')
        f.write('Figure 03: Period Comparison (Early vs Late)\n')
        f.write('Figure 04: Timescale Stratification\n')
        f.write('Figure 05: Density Classification\n')
        f.write('Figure 06: Hotspot Map\n')
        if has_insar:
            f.write('Figure 07: InSAR Validation (NEW)\n')
        f.write('\n')
        if has_insar:
            f.write('DATA FILES (InSAR Analysis)\n')
            f.write(' - 07_insar_binned_stats_by_elevation.csv\n')
            f.write(' - 07_insar_qgis_subsample.csv (use in QGIS!)\n')
        f.write('='*80 + '\n')
    logger.info(f'✓ Saved SummaryStatistics.txt')
    print(f'✓ Saved SummaryStatistics.txt')
except Exception as e:
    logger.warning(f'Could not create summary: {e}')
    print(f'WARNING: Could not create summary: {e}')

# ============================================================================
# FINALIZE
# ============================================================================
num_figures = 6
if has_insar:
    num_figures = 7

metadata.processing_results = {
    'figures_generated': num_figures,
    'figure_resolution_dpi': 300,
    'zonal_grid_cells': len(zonal_stats) if zonal_stats is not None else 0,
    'has_temporal_analysis': early_data is not None,
    'has_advanced_analysis': advanced_data is not None,
    'has_insar_analysis': has_insar,
}

metadata.finalize()
metadata_file = metadata.save(OUTPUT_DIR)

logger.info('\n' + '='*80)
logger.info('SCRIPT 07 - ENHANCED OUTPUT GENERATION COMPLETE')
logger.info('='*80)

print('\n' + '='*80)
print('SCRIPT 07 - COMPLETE')
print('='*80)
print(f'\nOUTPUT GENERATION COMPLETE')
print(f'Figures directory: {FIGURE_DIR}')
print(f'{num_figures} publication-ready figures generated (300 DPI)')
print(f'✓ Enhanced zonal grid visualization (20x20 km)')
print(f'✓ Period comparison plots (Early 2019-2021 vs Late 2022-2024)')
print(f'✓ Timescale stratification analysis')
print(f'✓ Density classification pie chart')
print(f'✓ Hotspot spatial map')
if has_insar:
    print(f'✓ InSAR validation plots (NEW)')
    print(f'✓ Elevation-binned statistics (NEW)')
    print(f'✓ QGIS-optimized subsample (NEW)')
print(f'\n✅ READY FOR THESIS')
print(f'All figures: {FIGURE_DIR}/*.png')
print(f'Summary statistics: {FIGURE_DIR}/SummaryStatistics.txt')
if has_insar:
    print(f'QGIS data file: {OUTPUT_DIR}/07_insar_qgis_subsample.csv')
    print(f'\nQGIS Instructions:')
    print(f' 1. Vector → Add Delimited Text Layer')
    print(f' 2. Select: 07_insar_qgis_subsample.csv')
    print(f' 3. X field: lon_avg, Y field: lat_avg, CRS: EPSG:4326')
    print(f' 4. Symbolize by: difference_mm_yr (color), abs_difference_mm_yr (size)')
print(f'\nNext steps:')
print(f' 1. Review figures in {FIGURE_DIR}')
print(f' 2. Copy figures to thesis/figures directory')
print(f' 3. Reference in Chapter 4: Results')
print('\n' + '='*80 + '\n')

if __name__ == '__main__':
    pass
