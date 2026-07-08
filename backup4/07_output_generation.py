"""
================================================================================
SCRIPT 07: FINAL PUBLICATION FIGURES
================================================================================

Purpose
-------
Generate 9+ publication-ready figures with:
1. Sign verification utility (3 checkpoints)
2. Safe DataFrame boolean checks (is not None)
3. TwoSlopeNorm spatial contours (±5 cm/yr, symmetric)
4. 4-panel enhanced multipanel analysis
5. InSAR validation with binned statistics
6. QGIS-optimized subsamples
7. Figures 01-06 generation (zonal grids, temporal, hotspots, etc.)
8. Violin distribution plot

Units: ALL RATES IN CM/YR (negative=subsidence, positive=uplift)
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
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

# UTF-8 FIX: None-safe encoding check (sys.stdout.encoding can be None on pipes)
_stdout_enc = getattr(sys.stdout, 'encoding', None) or ''
if _stdout_enc.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

_stderr_enc = getattr(sys.stderr, 'encoding', None) or ''
if _stderr_enc.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

try:
    from config import BASEOUTPUTDIR
except ImportError:
    BASEOUTPUTDIR = Path(r'E:\Processing\ICeSat-2\output_data')

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_git_hash():
    """Get git commit hash for reproducibility."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


def setup_logging(output_dir):
    """Configure logging to both file and console."""
    log_file = output_dir / 'script_07_run.log'
    logger = logging.getLogger('script_07')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    fh = logging.FileHandler(log_file, encoding='utf-8', mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ============================================================================
# SIGN VERIFICATION UTILITY
# ============================================================================
def verify_sign_convention(df, rate_col='subsidence_rate_cm_per_year', context_name="Dataset"):
    """
    Report sign convention of subsidence rates.
    Negative = subsidence (expected for permafrost).
    Positive = uplift (less common, but not necessarily an error).
    Returns (mean_rate_is_negative: bool, mean_rate: float, message: str|None)
    """
    if df is None or len(df) == 0:
        print(f"⚠️  No data for {context_name} sign check")
        return False, np.nan, "No data"

    if rate_col not in df.columns:
        print(f"⚠️  No rate column '{rate_col}' in {context_name}")
        return False, np.nan, f"Missing {rate_col}"

    rates = df[rate_col].dropna().values
    if len(rates) == 0:
        print(f"⚠️  No valid rates in {context_name}")
        return False, np.nan, "No valid rates"

    mean_rate = float(np.mean(rates))
    is_negative = mean_rate < 0

    if is_negative:
        print(f"✓  {context_name}: mean = {mean_rate:.3f} cm/yr (negative — subsidence ✓)")
    else:
        print(f"⚠️  {context_name}: mean = {mean_rate:.3f} cm/yr "
              f"(positive — verify sign convention!)")

    return is_negative, mean_rate, None


# ============================================================================
# METADATA TRACKING
# ============================================================================
class RunMetadata:
    """Capture complete run parameters for reproducibility."""

    def __init__(self, run_id, git_hash):
        self.run_id = run_id
        self.script_name = '07_output_generation'
        self.script_version = 'v4.4-corrected'
        self.git_hash = git_hash          # full hash
        self.start_time = datetime.now().isoformat()
        self.end_time = None
        self.configuration = {}
        self.processing_results = {}
        self.environment_info = {
            'python_version': (
                f'{sys.version_info.major}.{sys.version_info.minor}'
                f'.{sys.version_info.micro}'
            ),
            'platform': sys.platform,
            'timestamp': datetime.now().isoformat(),
        }

    def finalize(self):
        self.end_time = datetime.now().isoformat()

    def save(self, output_dir):
        metadata_file = output_dir / '07_output_generation_metadata.json'
        metadata = {
            'run_id': self.run_id,
            'script': self.script_name,
            'version': self.script_version,
            'git_hash': self.git_hash,
            'environment': self.environment_info,
            'timestamps': {'start': self.start_time, 'end': self.end_time},
            'configuration': self.configuration,
            'processing_results': self.processing_results,
        }
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str, ensure_ascii=False)
        return metadata_file


# ============================================================================
# FIGURE GENERATION FUNCTIONS (01–06)
# ============================================================================
def create_figures_01_to_06(zonal_stats, early_data, late_data, timescale_analysis,
                            advanced_data, figure_dir, logger):
    """Generate Figures 01–06 (all rates in cm/yr)."""

    # ── FIGURE 01: ZONAL GRID HEATMAP ──────────────────────────────────────
    if zonal_stats is not None and len(zonal_stats) > 0:
        try:
            logger.info("Generating Figure 01...")
            if 'mean_subsidence_cm_yr' in zonal_stats.columns:
                fig, ax = plt.subplots(figsize=(14, 10))

                if ('lat_center' in zonal_stats.columns and
                        'lon_center' in zonal_stats.columns):
                    zonal_matrix = zonal_stats.pivot_table(
                        values='mean_subsidence_cm_yr',
                        index='lat_center',
                        columns='lon_center',
                        aggfunc='mean'
                    )
                else:
                    zonal_sorted = zonal_stats.sort_values('mean_subsidence_cm_yr')
                    zonal_matrix = zonal_sorted[['mean_subsidence_cm_yr']].T

                im = ax.imshow(
                    zonal_matrix.values, cmap='RdYlBu_r',
                    aspect='auto', origin='lower', vmin=-5, vmax=0
                )
                plt.colorbar(im, ax=ax, label='Mean Subsidence Rate (cm/yr)')
                ax.set_xlabel('Longitude Index (E–W)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Latitude Index (N–S)', fontsize=12, fontweight='bold')
                ax.set_title(
                    'Figure 01: Coastal Subsidence Rate by 20×20 km Grid Cell\n'
                    'Tuktoyaktuk Peninsula',
                    fontsize=14, fontweight='bold', pad=20
                )
                ax.grid(True, alpha=0.3, color='gray', linestyle='--', linewidth=0.5)
                plt.tight_layout()
                fig.savefig(
                    figure_dir / 'Figure01_Zonal_Grid_Heatmap.png',
                    dpi=300, bbox_inches='tight'
                )
                plt.close()
                logger.info("✓ Figure 01 saved")
                print("✓ Figure 01: Zonal Grid Heatmap")
        except Exception as e:
            logger.warning(f"Figure 01 error: {e}")
            print(f"⚠️  Figure 01 error: {e}")

    # ── FIGURE 02: DENSITY HEATMAP ──────────────────────────────────────────
    if zonal_stats is not None and len(zonal_stats) > 0:
        try:
            logger.info("Generating Figure 02...")
            if 'density_points_per_km2' in zonal_stats.columns:
                fig, ax = plt.subplots(figsize=(14, 10))

                if ('lat_center' in zonal_stats.columns and
                        'lon_center' in zonal_stats.columns):
                    density_matrix = zonal_stats.pivot_table(
                        values='density_points_per_km2',
                        index='lat_center',
                        columns='lon_center',
                        aggfunc='mean'
                    )
                else:
                    density_sorted = zonal_stats.sort_values('density_points_per_km2')
                    density_matrix = density_sorted[['density_points_per_km2']].T

                im = ax.imshow(
                    density_matrix.values, cmap='YlOrRd',
                    aspect='auto', origin='lower'
                )
                plt.colorbar(im, ax=ax, label='Point Density (points/km²)')
                ax.set_xlabel('Longitude Index (E–W)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Latitude Index (N–S)', fontsize=12, fontweight='bold')
                ax.set_title(
                    'Figure 02: ICESat-2 Data Point Density by 20×20 km Grid Cell\n'
                    'Tuktoyaktuk Peninsula',
                    fontsize=14, fontweight='bold', pad=20
                )
                ax.grid(True, alpha=0.3, color='gray', linestyle='--', linewidth=0.5)
                plt.tight_layout()
                fig.savefig(
                    figure_dir / 'Figure02_Zonal_Density_Heatmap.png',
                    dpi=300, bbox_inches='tight'
                )
                plt.close()
                logger.info("✓ Figure 02 saved")
                print("✓ Figure 02: Zonal Density Heatmap")
        except Exception as e:
            logger.warning(f"Figure 02 error: {e}")
            print(f"⚠️  Figure 02 error: {e}")

    # ── FIGURE 03: PERIOD COMPARISON ────────────────────────────────────────
    if early_data is not None and late_data is not None:
        try:
            logger.info("Generating Figure 03...")
            rate_col = next(
                (c for c in early_data.columns
                 if 'rate' in c.lower() and 'cm' in c.lower()), None
            )
            if rate_col is None:
                rate_col = next(
                    (c for c in early_data.columns if 'rate' in c.lower()), None
                )

            if rate_col:
                early_rates = early_data[rate_col].values.copy()
                late_rates = late_data[rate_col].values.copy()

                if 'mm' in rate_col.lower():
                    early_rates = early_rates / 10.0
                    late_rates = late_rates / 10.0

                early_clean = early_rates[np.isfinite(early_rates)]
                late_clean = late_rates[np.isfinite(late_rates)]

                fig, axes = plt.subplots(1, 2, figsize=(14, 6))

                axes[0].hist(early_clean, bins=100, alpha=0.6,
                             label=f'Early 2019–2021 (n={len(early_clean)})',
                             color='blue', edgecolor='black')
                axes[0].hist(late_clean, bins=100, alpha=0.6,
                             label=f'Late 2022–2024 (n={len(late_clean)})',
                             color='red', edgecolor='black')
                axes[0].set_xlabel('Subsidence Rate (cm/yr)', fontsize=11, fontweight='bold')
                axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
                axes[0].set_title('Distribution of Subsidence Rates by Period',
                                  fontsize=12, fontweight='bold')
                axes[0].legend(fontsize=10)
                axes[0].grid(True, alpha=0.3)

                bp = axes[1].boxplot(
                    [early_clean, late_clean],
                    labels=['Early Period\n2019–2021', 'Late Period\n2022–2024'],
                    patch_artist=True, widths=0.6
                )
                for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
                    patch.set_facecolor(color)

                axes[1].set_ylabel('Subsidence Rate (cm/yr)', fontsize=11, fontweight='bold')
                axes[1].set_title('Subsidence Rate Statistics by Period',
                                  fontsize=12, fontweight='bold')
                axes[1].grid(True, alpha=0.3, axis='y')

                early_mean = float(np.mean(early_clean))
                late_mean = float(np.mean(late_clean))
                axes[1].text(1, early_mean, f'{early_mean:.2f}',
                             ha='center', va='bottom', fontweight='bold')
                axes[1].text(2, late_mean, f'{late_mean:.2f}',
                             ha='center', va='bottom', fontweight='bold')

                plt.tight_layout()
                fig.savefig(
                    figure_dir / 'Figure03_Period_Comparison.png',
                    dpi=300, bbox_inches='tight'
                )
                plt.close()
                logger.info("✓ Figure 03 saved")
                print("✓ Figure 03: Period Comparison")
        except Exception as e:
            logger.warning(f"Figure 03 error: {e}")
            print(f"⚠️  Figure 03 error: {e}")

    # ── FIGURE 04: TIMESCALE STRATIFICATION ─────────────────────────────────
    if timescale_analysis is not None and len(timescale_analysis) > 0:
        try:
            logger.info("Generating Figure 04...")
            fig, ax = plt.subplots(figsize=(12, 7))

            if 'Metric' in timescale_analysis.columns:
                metrics = timescale_analysis['Metric'].values
                early_vals = (timescale_analysis['Early Period'].values
                              if 'Early Period' in timescale_analysis.columns
                              else np.zeros(len(metrics)))
                late_vals = (timescale_analysis['Late Period'].values
                             if 'Late Period' in timescale_analysis.columns
                             else np.zeros(len(metrics)))

                x = np.arange(len(metrics))
                width = 0.35
                ax.bar(x - width / 2, early_vals, width,
                       label='Early 2019–2021', alpha=0.8)
                ax.bar(x + width / 2, late_vals, width,
                       label='Late 2022–2024', alpha=0.8)
                ax.set_ylabel('Mean Value', fontsize=11, fontweight='bold')
                ax.set_title('Temporal Stratification Metrics',
                             fontsize=12, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(metrics, rotation=45, ha='right')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            fig.savefig(
                figure_dir / 'Figure04_Timescale_Stratification.png',
                dpi=300, bbox_inches='tight'
            )
            plt.close()
            logger.info("✓ Figure 04 saved")
            print("✓ Figure 04: Timescale Stratification")
        except Exception as e:
            logger.warning(f"Figure 04 error: {e}")
            print(f"⚠️  Figure 04 error: {e}")

    # ── FIGURE 05: DENSITY CLASSIFICATION PIE ───────────────────────────────
    if advanced_data is not None and 'density_class' in advanced_data.columns:
        try:
            logger.info("Generating Figure 05...")
            density_counts = advanced_data['density_class'].value_counts()

            fig, ax = plt.subplots(figsize=(10, 7))
            colors = ['#FF6B6B', '#FFA726', '#66BB6A', '#42A5F5']

            ax.pie(
                density_counts.values,
                labels=density_counts.index,
                autopct='%1.1f%%',
                colors=colors[:len(density_counts)],
                startangle=90,
                textprops={'fontsize': 11, 'fontweight': 'bold'}
            )
            ax.set_title(
                f'Point Distribution by Density Classification '
                f'(Total: {len(advanced_data)} points)',
                fontsize=12, fontweight='bold', pad=20
            )
            plt.tight_layout()
            fig.savefig(
                figure_dir / 'Figure05_Density_Classification.png',
                dpi=300, bbox_inches='tight'
            )
            plt.close()
            logger.info("✓ Figure 05 saved")
            print("✓ Figure 05: Density Classification")

            for cls, count in density_counts.items():
                pct = count / len(advanced_data) * 100
                logger.info(f"  {cls}: {count} points ({pct:.1f}%)")
        except Exception as e:
            logger.warning(f"Figure 05 error: {e}")
            print(f"⚠️  Figure 05 error: {e}")

    # ── FIGURE 06: HOTSPOT MAP ───────────────────────────────────────────────
    if advanced_data is not None:
        try:
            logger.info("Generating Figure 06...")
            lat_col = next(
                (c for c in advanced_data.columns if 'lat' in c.lower()), None
            )
            lon_col = next(
                (c for c in advanced_data.columns if 'lon' in c.lower()), None
            )
            rate_col = next(
                (c for c in advanced_data.columns
                 if 'rate' in c.lower() and 'cm' in c.lower()), None
            )

            if lat_col and lon_col and rate_col:
                fig, ax = plt.subplots(figsize=(12, 10))

                # FIX: use explicit column check instead of DataFrame.get()
                if 'cluster_id' in advanced_data.columns:
                    non_hotspot = advanced_data[advanced_data['cluster_id'] == -1]
                    hotspot = advanced_data[advanced_data['cluster_id'] >= 0]
                else:
                    non_hotspot = advanced_data
                    hotspot = advanced_data.iloc[0:0]   # empty with same columns

                ax.scatter(
                    non_hotspot[lon_col], non_hotspot[lat_col],
                    c='lightgray', s=10, alpha=0.3, label='Non-hotspot'
                )

                if len(hotspot) > 0:
                    scatter = ax.scatter(
                        hotspot[lon_col], hotspot[lat_col],
                        c=hotspot[rate_col], s=30, alpha=0.8,
                        cmap='RdYlBu_r', edgecolors='darkred', linewidth=0.5,
                        label=f'Hotspot (n={len(hotspot)})',
                        vmin=-5, vmax=0
                    )
                    plt.colorbar(scatter, ax=ax, label='Subsidence Rate (cm/yr)')

                ax.set_xlabel('Longitude', fontsize=11, fontweight='bold')
                ax.set_ylabel('Latitude', fontsize=11, fontweight='bold')
                ax.set_title(
                    'ICESat-2 Coastal Subsidence Hotspots\nTuktoyaktuk Peninsula',
                    fontsize=12, fontweight='bold', pad=20
                )
                ax.legend(loc='best', fontsize=10)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                fig.savefig(
                    figure_dir / 'Figure06_Hotspot_Map.png',
                    dpi=300, bbox_inches='tight'
                )
                plt.close()
                logger.info("✓ Figure 06 saved")
                print("✓ Figure 06: Hotspot Map")

                n_hotspots = len(hotspot)
                logger.info(
                    f"  Hotspots: {n_hotspots} "
                    f"({n_hotspots / len(advanced_data) * 100:.1f}%)"
                )
        except Exception as e:
            logger.warning(f"Figure 06 error: {e}")
            print(f"⚠️  Figure 06 error: {e}")


# ============================================================================
# VIOLIN PERIOD COMPARISON
# ============================================================================
def create_violin_period_comparison(early_data, late_data, figure_dir, logger):
    """
    Violin + boxplot overlay of subsidence rate distributions for
    early vs late periods.  Rates in cm/yr; negative = subsidence.
    """
    if early_data is None or late_data is None:
        print("⚠️  Violin plot skipped: missing early/late data")
        return

    rate_col = next(
        (c for c in early_data.columns
         if 'rate' in c.lower() and 'cm' in c.lower()), None
    )
    if rate_col is None:
        rate_col = next(
            (c for c in early_data.columns if 'rate' in c.lower()), None
        )
    if rate_col is None:
        print("⚠️  Violin plot skipped: no rate column found")
        return

    early_rates = early_data[rate_col].values.copy()
    late_rates = late_data[rate_col].values.copy()
    if 'mm' in rate_col.lower():
        early_rates = early_rates / 10.0
        late_rates = late_rates / 10.0

    early_clean = early_rates[np.isfinite(early_rates)]
    late_clean = late_rates[np.isfinite(late_rates)]

    if len(early_clean) < 10 or len(late_clean) < 10:
        print("⚠️  Violin plot skipped: not enough valid values")
        return

    df_violin = pd.DataFrame({
        'subsidence_cm_yr': np.concatenate([early_clean, late_clean]),
        'Period': (
            ['Early 2019–2021'] * len(early_clean) +
            ['Late 2022–2024'] * len(late_clean)
        )
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    # FIX: pass ax= explicitly to both seaborn calls so they share the same axes
    sns.violinplot(
        data=df_violin, x='Period', y='subsidence_cm_yr',
        inner='quartile', cut=0, linewidth=1.0,
        palette=['#99d8ff', '#f4a259'],
        ax=ax
    )
    sns.boxplot(
        data=df_violin, x='Period', y='subsidence_cm_yr',
        width=0.15, showcaps=True,
        boxprops={'zorder': 3, 'facecolor': 'white', 'alpha': 0.9},
        showfliers=False,
        whiskerprops={'linewidth': 1.2},
        medianprops={'color': 'black', 'linewidth': 1.5},
        ax=ax                           # FIX: was missing
    )

    ax.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.6)
    ax.set_ylabel('Subsidence rate (cm/yr)', fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.set_title('Violin distribution of subsidence rates by period',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()
    figfile = figure_dir / 'Figure03_Violin_PeriodComparison.png'
    fig.savefig(figfile, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Figure 03 (violin) saved: {figfile.name}")
    print("✓ Figure 03 violin plot generated")


# ============================================================================
# SPATIAL CONTOUR & MULTIPANEL FUNCTIONS
# ============================================================================
def create_spatial_contour_map(spatial_grid, output_dir, logger, contour_type='zonal'):
    """Generate TwoSlopeNorm contour map."""
    if spatial_grid is None or len(spatial_grid) == 0:
        logger.warning(f"Empty spatial_grid for {contour_type} contours")
        return

    if contour_type == 'zonal':
        rate_col = 'mean_subsidence_cm_yr'
        lat_col, lon_col = 'lat_center', 'lon_center'
        title_suffix = '20 km Zonal Grid'
    else:
        rate_col = 'subsidence_rate_cm_per_year'
        lat_col, lon_col = 'latitude', 'longitude'
        title_suffix = 'Point-Based Grid'

    if rate_col not in spatial_grid.columns:
        logger.warning(f"No '{rate_col}' column in {contour_type}")
        return
    if lat_col not in spatial_grid.columns or lon_col not in spatial_grid.columns:
        logger.warning(f"Missing lat/lon columns ({lat_col}, {lon_col}) in {contour_type}")
        return

    try:
        pivot = spatial_grid.pivot_table(
            values=rate_col, index=lat_col, columns=lon_col, aggfunc='mean'
        )

        if pivot.empty or pivot.shape[0] < 2 or pivot.shape[1] < 2:
            logger.warning(f"Pivot insufficient dimensions: {pivot.shape}")
            return

        fig, ax = plt.subplots(figsize=(14, 10))
        # FIX: use keyword arguments (positional vmin/vcenter/vmax deprecated in mpl ≥3.7)
        norm = TwoSlopeNorm(vmin=-5.0, vcenter=0.0, vmax=5.0)

        cs = ax.contourf(
            pivot.columns, pivot.index, pivot.values,
            levels=25, cmap='RdYlBu_r', norm=norm, extend='both'
        )
        cbar = plt.colorbar(cs, ax=ax, shrink=0.85)
        cbar.set_label('Subsidence Rate (cm/yr)', fontsize=12, fontweight='bold')

        ax.set_xlabel('Longitude (°W)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latitude (°N)', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Spatial Subsidence Contours — {title_suffix}\nSymmetric ±5 cm/yr',
            fontsize=14, fontweight='bold'
        )
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        fig_file = output_dir / f'Figure08_Spatial_Contours_{contour_type}.png'
        fig.savefig(fig_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved {fig_file.name}")
        print(f"✓ {contour_type.title()} contour map: pivot shape {pivot.shape}")

    except Exception as e:
        logger.warning(f"Contour error ({contour_type}): {e}")


def create_enhanced_multipanel_figure(spatial_grid, clusters_df, output_dir, logger):
    """Generate 4-panel enhanced multipanel figure."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('Enhanced Subsidence Analysis',
                     fontsize=16, fontweight='bold', y=0.995)

        rate_col = 'subsidence_rate_cm_per_year'

        # PANEL A: Spatial Contour
        axes[0, 0].set_title('A: Spatial Subsidence Contours (TwoSlopeNorm ±5 cm/yr)',
                             fontweight='bold', fontsize=12)
        if spatial_grid is not None and len(spatial_grid) > 10:
            try:
                if 'latitude' in spatial_grid.columns and rate_col in spatial_grid.columns:
                    pivot = spatial_grid.pivot_table(
                        rate_col, 'latitude', 'longitude', 'mean'
                    )
                elif 'lat_center' in spatial_grid.columns:
                    pivot = spatial_grid.pivot_table(
                        'mean_subsidence_cm_yr', 'lat_center', 'lon_center', 'mean'
                    )
                else:
                    pivot = None

                if pivot is not None and not pivot.empty and pivot.shape[0] >= 2:
                    # FIX: keyword arguments for TwoSlopeNorm
                    norm = TwoSlopeNorm(vmin=-5.0, vcenter=0.0, vmax=5.0)
                    cs = axes[0, 0].contourf(
                        pivot.columns, pivot.index, pivot.values,
                        levels=20, cmap='RdYlBu_r', norm=norm, extend='both'
                    )
                    cbar = plt.colorbar(cs, ax=axes[0, 0], shrink=0.9)
                    cbar.set_label('cm/yr')
                    axes[0, 0].set_xlabel('Longitude')
                    axes[0, 0].set_ylabel('Latitude')
                    axes[0, 0].grid(True, alpha=0.3)
            except Exception as e:
                logger.debug(f"Panel A error: {e}")

        # PANEL B: Elevation vs Subsidence
        axes[0, 1].set_title('B: Elevation Dependence', fontweight='bold', fontsize=12)
        if (spatial_grid is not None and
                'elevation' in spatial_grid.columns and
                rate_col in spatial_grid.columns):
            axes[0, 1].scatter(
                spatial_grid['elevation'], spatial_grid[rate_col],
                alpha=0.6, s=40,
                c=spatial_grid[rate_col], cmap='RdYlBu_r',
                edgecolors='black', linewidth=0.5
            )
            axes[0, 1].set_xlabel('Elevation (m)', fontweight='bold')
            axes[0, 1].set_ylabel('Subsidence (cm/yr)', fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].axhline(0, color='gray', linestyle='--', alpha=0.5)
        else:
            axes[0, 1].text(0.5, 0.5, 'No elevation data', ha='center', va='center',
                            transform=axes[0, 1].transAxes, fontsize=11)

        # PANEL C: Uncertainty vs Subsidence
        axes[1, 0].set_title('C: Uncertainty Analysis', fontweight='bold', fontsize=12)
        if spatial_grid is not None and rate_col in spatial_grid.columns:
            uncertainty_col = next(
                (c for c in spatial_grid.columns
                 if 'uncertain' in c.lower() or 'std' in c.lower()), None
            )
            if uncertainty_col:
                axes[1, 0].scatter(
                    spatial_grid[rate_col], spatial_grid[uncertainty_col],
                    alpha=0.6, s=40, c='steelblue',
                    edgecolors='navy', linewidth=0.5
                )
                axes[1, 0].set_xlabel('Subsidence (cm/yr)', fontweight='bold')
                axes[1, 0].set_ylabel('Uncertainty (cm/yr)', fontweight='bold')
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'No uncertainty data',
                                ha='center', va='center',
                                transform=axes[1, 0].transAxes, fontsize=11)
        else:
            axes[1, 0].text(0.5, 0.5, 'No data available', ha='center', va='center',
                            transform=axes[1, 0].transAxes, fontsize=11)

        # PANEL D: Cluster Hotspots
        axes[1, 1].set_title('D: Hotspot Clusters', fontweight='bold', fontsize=12)
        if (clusters_df is not None and len(clusters_df) > 0 and
                'center_lon' in clusters_df.columns and
                'center_lat' in clusters_df.columns and
                'mean_subsidence_cm_yr' in clusters_df.columns):
            scatter = axes[1, 1].scatter(
                clusters_df['center_lon'], clusters_df['center_lat'],
                c=clusters_df['mean_subsidence_cm_yr'],
                s=clusters_df['size'] * 2 if 'size' in clusters_df.columns else 40,
                cmap='RdYlBu_r', alpha=0.7,
                edgecolors='black', linewidth=1,
                vmin=-5, vmax=5
            )
            cbar = plt.colorbar(scatter, ax=axes[1, 1], shrink=0.9)
            cbar.set_label('Mean subsidence (cm/yr)')
            axes[1, 1].set_xlabel('Longitude', fontweight='bold')
            axes[1, 1].set_ylabel('Latitude', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(
                0.5, 0.5, 'No cluster data\nRun Script 06',
                ha='center', va='center',
                transform=axes[1, 1].transAxes, fontsize=11
            )

        plt.tight_layout()
        fig_file = output_dir / 'Figure09_Multipanel_Enhanced.png'
        fig.savefig(fig_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved {fig_file.name}")
        print("✓ 4-panel multipanel generated")

    except Exception as e:
        logger.warning(f"Multipanel error: {e}")

# ============================================================================
# InSAR VALIDATION FUNCTIONS
# ============================================================================
def create_insar_comparison_plots(insar_df, figure_dir, logger):
    """Create publication-ready InSAR vs ICESat-2 comparison plots."""
    if insar_df is None or len(insar_df) == 0:
        return False

    try:
        icesat2 = insar_df['subsidence_rate_cm_per_year'].values
        insar = insar_df['insar_rate_mm_yr'].values / 10.0   # mm/yr → cm/yr

        valid = np.isfinite(icesat2) & np.isfinite(insar)
        if valid.sum() < 10:
            logger.warning(f"Only {valid.sum()} valid InSAR–ICESat-2 pairs")
            return False

        icesat2_valid = icesat2[valid]
        insar_valid = insar[valid]
        difference = icesat2_valid - insar_valid

        elev_col = next((c for c in insar_df.columns if 'elev' in c.lower()), None)
        elevation = insar_df[elev_col].values[valid] if elev_col else None

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        lim_min = min(float(insar_valid.min()), float(icesat2_valid.min()))
        lim_max = max(float(insar_valid.max()), float(icesat2_valid.max()))

        axes[0].scatter(insar_valid, icesat2_valid,
                        alpha=0.6, s=50, c='steelblue',
                        edgecolors='navy', linewidth=0.5)
        axes[0].plot([lim_min, lim_max], [lim_min, lim_max],
                     'r--', linewidth=2, label='1:1 agreement', alpha=0.7)

        rmse = float(np.sqrt(np.mean(difference ** 2)))
        bias = float(np.mean(difference))

        # FIX: guard pearsonr against zero-variance arrays
        if np.std(insar_valid) > 0 and np.std(icesat2_valid) > 0:
            r, p = pearsonr(insar_valid, icesat2_valid)
        else:
            r, p = np.nan, np.nan

        r_str = f'{r:.3f}' if np.isfinite(r) else 'N/A'
        stats_text = (
            f'R = {r_str}\n'
            f'RMSE = {rmse:.2f} cm/yr\n'
            f'Bias = {bias:.2f} cm/yr\n'
            f'N = {int(valid.sum())}'
        )

        axes[0].text(0.05, 0.95, stats_text, transform=axes[0].transAxes,
                     fontsize=11, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[0].set_xlabel('InSAR Subsidence Rate (cm/yr)', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('ICESat-2 Subsidence Rate (cm/yr)', fontsize=11, fontweight='bold')
        axes[0].set_title('A: InSAR vs ICESat-2 Validation', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(difference, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
        axes[1].axvline(bias, color='red', linestyle='--', linewidth=2,
                        label=f'Mean bias = {bias:.2f} cm/yr')
        axes[1].axvline(0, color='green', linestyle='--', linewidth=2,
                        label='Perfect agreement')
        axes[1].set_xlabel('Difference (ICESat-2 − InSAR) (cm/yr)',
                           fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[1].set_title('B: Difference Distribution', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')

        if elevation is not None:
            scatter = axes[2].scatter(
                elevation, difference,
                alpha=0.6, s=40,
                c=np.abs(difference), cmap='Reds',
                edgecolors='darkred', linewidth=0.5
            )
            cbar = plt.colorbar(scatter, ax=axes[2])
            cbar.set_label('|Residual| (cm/yr)')
            axes[2].set_xlabel('Elevation (m)', fontsize=11, fontweight='bold')
        else:
            axes[2].scatter(range(len(difference)), difference,
                            alpha=0.6, s=40, c='steelblue',
                            edgecolors='navy', linewidth=0.5)
            axes[2].set_xlabel('Crossover Index', fontsize=11, fontweight='bold')

        axes[2].axhline(0, color='green', linestyle='--', linewidth=2, alpha=0.7)
        axes[2].set_ylabel('Residual (cm/yr)', fontsize=11, fontweight='bold')
        axes[2].set_title('C: Residuals', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_file = figure_dir / 'Figure07_InSAR_Validation.png'
        fig.savefig(fig_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved {fig_file.name}")
        print(f"✓ InSAR validation: R = {r_str}, RMSE = {rmse:.2f} cm/yr")
        return True

    except Exception as e:
        logger.warning(f"InSAR plot error: {e}")
        return False


def create_insar_binned_stats(insar_df, output_dir, bin_size_m=10, logger=None):
    """
    Create binned statistics CSV for InSAR–ICESat-2 agreement by elevation.
    NOTE: figure_dir parameter removed — outputs go to output_dir only.
    """
    if insar_df is None or len(insar_df) == 0:
        return None

    try:
        icesat2 = insar_df['subsidence_rate_cm_per_year'].values
        insar = insar_df['insar_rate_mm_yr'].values / 10.0
        valid = np.isfinite(icesat2) & np.isfinite(insar)

        icesat2_valid = icesat2[valid]
        insar_valid = insar[valid]
        difference = icesat2_valid - insar_valid

        elev_col = next((c for c in insar_df.columns if 'elev' in c.lower()), None)
        if elev_col is None:
            return None

        elevation = insar_df[elev_col].values[valid]

        elev_min = np.floor(elevation.min() / bin_size_m) * bin_size_m
        elev_max = np.ceil(elevation.max() / bin_size_m) * bin_size_m
        bins = np.arange(elev_min, elev_max + bin_size_m, bin_size_m)

        stats_list = []
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            mask = (elevation >= lo) & (elevation < hi)
            if mask.sum() == 0:
                continue

            icebin = icesat2_valid[mask]
            insarbin = insar_valid[mask]
            diffbin = difference[mask]

            stats_list.append({
                'elevation_m': f'{int(lo)}-{int(hi)}',
                'elevation_center': float((lo + hi) / 2),
                'n_points': int(mask.sum()),
                'icesat2_mean_cm_yr': float(icebin.mean()),
                'icesat2_std_cm_yr': float(icebin.std()),
                'insar_mean_cm_yr': float(insarbin.mean()),
                'insar_std_cm_yr': float(insarbin.std()),
                'difference_mean_cm_yr': float(diffbin.mean()),
                'difference_std_cm_yr': float(diffbin.std()),
                'rmse_cm_yr': float(np.sqrt(np.mean(diffbin ** 2))),
                'mae_cm_yr': float(np.mean(np.abs(diffbin))),
            })

        if not stats_list:
            return None

        df_stats = pd.DataFrame(stats_list)
        stats_csv = output_dir / '07_insar_binned_stats_by_elevation.csv'
        df_stats.to_csv(stats_csv, index=False, encoding='utf-8')
        if logger:
            logger.info(f"Saved {stats_csv.name} ({len(df_stats)} elevation bins)")
        print(f"✓ InSAR binned stats: {len(df_stats)} elevation bins")
        return df_stats

    except Exception as e:
        if logger:
            logger.warning(f"Binned stats error: {e}")
        return None


def create_insar_spatial_subsample(insar_df, output_dir, downsample_factor=10, logger=None):
    """Create QGIS-optimised spatial subsample."""
    if insar_df is None or len(insar_df) == 0:
        return None

    try:
        lat_col = next((c for c in insar_df.columns if 'lat' in c.lower()), None)
        lon_col = next((c for c in insar_df.columns if 'lon' in c.lower()), None)
        rate_col = next(
            (c for c in insar_df.columns
             if 'subsidence_rate_cm_per_year' in c.lower() or
             ('rate' in c.lower() and 'cm' in c.lower())), None
        )

        if not all([lat_col, lon_col, rate_col]):
            return None

        downsampled = insar_df.iloc[::downsample_factor].copy()

        # FIX: apply latitude correction for longitude-to-metre conversion
        mean_lat = float(downsampled[lat_col].mean())
        cos_lat = np.cos(np.radians(mean_lat))
        grid_size_m = 2000

        downsampled['grid_cell_x'] = (
            (downsampled[lon_col] * 111_000 * cos_lat).astype(int) // grid_size_m
        )
        downsampled['grid_cell_y'] = (
            (downsampled[lat_col] * 111_000).astype(int) // grid_size_m
        )

        if 'insar_rate_mm_yr' in downsampled.columns:
            downsampled['difference_cm_yr'] = (
                downsampled[rate_col] - downsampled['insar_rate_mm_yr'] / 10.0
            )
            downsampled['abs_difference_cm_yr'] = np.abs(downsampled['difference_cm_yr'])

        qgis_csv = output_dir / '07_insar_qgis_subsample.csv'
        downsampled.to_csv(qgis_csv, index=False, encoding='utf-8')
        if logger:
            logger.info(f"Saved {qgis_csv.name}")
        reduction = (1 - 1 / downsample_factor) * 100
        print(f"✓ QGIS subsample: {len(downsampled)} points (~{reduction:.0f}% size reduction)")
        return downsampled

    except Exception as e:
        if logger:
            logger.warning(f"QGIS subsample error: {e}")
        return None


def create_summary_statistics(zonal_stats, advanced_data, insar_integrated,
                              figure_dir, output_dir, run_timestamp, git_hash, logger):
    """Generate comprehensive summary statistics table."""
    try:
        summary_file = figure_dir / 'SummaryStatistics.txt'
        git_display = git_hash[:8] if git_hash != 'unknown' else 'unknown'

        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write('=' * 80 + '\n')
            f.write('ICESat-2 COASTAL SUBSIDENCE ANALYSIS - SUMMARY STATISTICS\n')
            f.write('TUKTOYAKTUK PENINSULA\n')
            f.write('=' * 80 + '\n\n')
            f.write(f'Run ID: {run_timestamp}\n')
            f.write(f'Git Hash: {git_display}\n')
            f.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write('=' * 80 + '\n\n')

            if zonal_stats is not None and len(zonal_stats) > 0:
                f.write('ZONAL STATISTICS SUMMARY (20 km grid)\n')
                f.write('-' * 80 + '\n')
                f.write(f'Grid cells with data: {len(zonal_stats)}\n')
                if 'mean_subsidence_cm_yr' in zonal_stats.columns:
                    f.write(f'Mean subsidence rate: '
                            f'{zonal_stats["mean_subsidence_cm_yr"].mean():.2f} cm/yr\n')
                    f.write(f'Max subsidence rate:  '
                            f'{zonal_stats["mean_subsidence_cm_yr"].max():.2f} cm/yr\n')
                    f.write(f'Min subsidence rate:  '
                            f'{zonal_stats["mean_subsidence_cm_yr"].min():.2f} cm/yr\n')
                if 'density_points_per_km2' in zonal_stats.columns:
                    f.write(f'Mean density: '
                            f'{zonal_stats["density_points_per_km2"].mean():.2f} points/km²\n')
                f.write('\n')

            if advanced_data is not None and len(advanced_data) > 0:
                f.write('ADVANCED ANALYSIS SUMMARY\n')
                f.write('-' * 80 + '\n')
                f.write(f'Total points analyzed: {len(advanced_data)}\n')
                if 'density_class' in advanced_data.columns:
                    f.write(f'Density classes: {advanced_data["density_class"].nunique()}\n')
                if 'cluster_id' in advanced_data.columns:
                    n_hotspots = int((advanced_data['cluster_id'] > 0).sum())
                    f.write(
                        f'Hotspot points: {n_hotspots} '
                        f'({n_hotspots / len(advanced_data) * 100:.1f}%)\n'
                    )
                f.write('\n')

            f.write('FIGURES GENERATED\n')
            f.write('=' * 80 + '\n')
            f.write('Figure 01: Zonal Grid Heatmap (20 km)\n')
            f.write('Figure 02: Zonal Density Heatmap\n')
            f.write('Figure 03: Period Comparison (Early 2019–2021 vs Late 2022–2024)\n')
            f.write('Figure 03: Violin Distribution by Period\n')
            f.write('Figure 04: Timescale Stratification\n')
            f.write('Figure 05: Density Classification (Pie Chart)\n')
            f.write('Figure 06: Hotspot Map (Spatial Distribution)\n')
            f.write('Figure 07: InSAR Validation (3-Panel Comparison)\n')
            f.write('Figure 08: Spatial Contour Maps (TwoSlopeNorm ±5 cm/yr)\n')
            f.write('Figure 09: 4-Panel Enhanced Multipanel Analysis\n\n')
            f.write('All figures: 300 DPI PNG format\n')
            f.write('Colormap: RdYlBu_r '
                    '(Red = subsidence, Blue = uplift, Yellow = stable)\n\n')
            f.write('=' * 80 + '\n')
            f.write('UNITS AND CONVENTIONS\n')
            f.write('=' * 80 + '\n')
            f.write('All subsidence rates: cm/yr\n')
            f.write('  Negative values = subsidence (expected for permafrost)\n')
            f.write('  Positive values = uplift (less common)\n')
            f.write('  Zero            = no change\n\n')
            f.write('Color convention:\n')
            f.write('  Red    = subsidence (negative rates)\n')
            f.write('  Blue   = uplift (positive rates)\n')
            f.write('  Yellow = stable (near 0)\n\n')
            f.write('Data files:\n')
            f.write('  07_insar_binned_stats_by_elevation.csv\n')
            f.write('  07_insar_qgis_subsample.csv\n\n')
            f.write('QGIS instructions:\n')
            f.write('  Layer → Add Delimited Text Layer → Select CSV\n')
            f.write('  X field: lon_col, Y field: lat_col\n\n')

        logger.info(f"Saved {summary_file.name}")
        print("✓ Summary statistics generated")

    except Exception as e:
        logger.warning(f"Summary statistics error: {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution function."""

    # ── Timestamp ────────────────────────────────────────────────────────────
    timestamp_file = BASEOUTPUTDIR / 'latest_RUN_TIMESTAMP.txt'
    if not timestamp_file.exists():
        print("ERROR: Run Script 00/08 first to generate timestamp!")
        print(f"Expected: {timestamp_file}")
        sys.exit(1)

    with open(timestamp_file, 'r', encoding='utf-8') as f:
        RUN_TIMESTAMP = f.read().strip()

    OUTPUT_DIR = BASEOUTPUTDIR / f'run_{RUN_TIMESTAMP}'
    FIGURE_DIR = OUTPUT_DIR / 'figures'

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        FIGURE_DIR.mkdir(exist_ok=True)
    except Exception as e:
        print(f"Cannot create directories: {e}")
        sys.exit(1)

    # ── Setup ────────────────────────────────────────────────────────────────
    logger = setup_logging(OUTPUT_DIR)
    git_hash = get_git_hash()            # full hash stored; truncated for display only
    metadata = RunMetadata(RUN_TIMESTAMP, git_hash)

    print("=" * 80)
    print("SCRIPT 07: FINAL PUBLICATION FIGURES (v4.4 - CORRECTED)")
    print("=" * 80)
    print(f"Run ID : {RUN_TIMESTAMP}")
    print(f"Output : {OUTPUT_DIR}")
    print(f"Figures: {FIGURE_DIR}")
    print(f"Git    : {git_hash[:8] if git_hash != 'unknown' else 'unknown'}")

    logger.info("=" * 80)
    logger.info("SCRIPT 07: FINAL PUBLICATION FIGURES (v4.4 - CORRECTED)")
    logger.info("=" * 80)
    logger.info(f"Run ID: {RUN_TIMESTAMP}")
    logger.info(f"Git Hash: {git_hash}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # ── Matplotlib style — version-safe fallback ─────────────────────────────
    for _style in ('seaborn-v0_8-darkgrid', 'seaborn-darkgrid', 'default'):
        try:
            plt.style.use(_style)
            break
        except OSError:
            continue

    sns.set_palette('husl')

    # ── Load Data ────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("LOADING DATA")
    logger.info("=" * 80)
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    zonal_stats = None
    early_data = None
    late_data = None
    timescale_analysis = None
    advanced_data = None
    clusters_stats = None
    insar_integrated = None

    try:
        zonal_stats = pd.read_csv(OUTPUT_DIR / '04_zonal_statistics_20km_grid.csv')
        print(f"✓ Zonal statistics: {len(zonal_stats)} cells")
    except FileNotFoundError:
        print("⚠️  Zonal statistics not found (run Script 04)")
    except Exception as e:
        logger.warning(f"Zonal stats error: {e}")

    try:
        early_data = pd.read_csv(OUTPUT_DIR / 'icesat2_early_period.csv')
        late_data = pd.read_csv(OUTPUT_DIR / 'icesat2_late_period.csv')
        print(f"✓ Temporal data: Early={len(early_data)}, Late={len(late_data)}")
    except FileNotFoundError:
        print("⚠️  Temporal data not found (run Script 05b)")
    except Exception as e:
        logger.warning(f"Temporal data error: {e}")

    try:
        timescale_analysis = pd.read_csv(OUTPUT_DIR / 'temporal_acceleration_metrics.csv')
        print(f"✓ Timescale analysis: {len(timescale_analysis)} metrics")
    except FileNotFoundError:
        print("⚠️  Timescale analysis not found (run Script 05b)")
    except Exception as e:
        logger.warning(f"Timescale analysis error: {e}")

    try:
        advanced_data = pd.read_csv(OUTPUT_DIR / '06_clusters_identified.csv')
        clusters_stats = pd.read_csv(OUTPUT_DIR / '06_cluster_statistics.csv')
        print(f"✓ Advanced analysis: {len(advanced_data)} points, "
              f"{len(clusters_stats)} clusters")
    except FileNotFoundError:
        print("⚠️  Advanced analysis not found (run Script 06)")
    except Exception as e:
        logger.warning(f"Advanced analysis error: {e}")

    try:
        insar_integrated = pd.read_csv(OUTPUT_DIR / 'icesat2_insar_integrated.csv')
        print(f"✓ InSAR integrated: {len(insar_integrated)} points")
    except FileNotFoundError:
        print("⚠️  InSAR data not found (optional)")
    except Exception as e:
        logger.warning(f"InSAR data error: {e}")

    # ── Sign Convention Verification ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SIGN CONVENTION VERIFICATION")
    print("=" * 80)

    if advanced_data is not None and len(advanced_data) > 0:
        verify_sign_convention(advanced_data, 'subsidence_rate_cm_per_year', "Advanced Data")

    if zonal_stats is not None and len(zonal_stats) > 0:
        verify_sign_convention(zonal_stats, 'mean_subsidence_cm_yr', "Zonal Stats")

    if insar_integrated is not None and len(insar_integrated) > 0:
        verify_sign_convention(insar_integrated, 'subsidence_rate_cm_per_year', "InSAR")

    print("=" * 80)

    # Safe summary (FIX #2: all None-guarded)
    print(
        f"Zonal: {len(zonal_stats) if zonal_stats is not None else 0} | "
        f"Temporal: {len(early_data) if early_data is not None else 0}/"
        f"{len(late_data) if late_data is not None else 0}"
    )
    print(
        f"Clusters: {len(clusters_stats) if clusters_stats is not None else 0} | "
        f"InSAR: {len(insar_integrated) if insar_integrated is not None else 0}"
    )

    # ── Figures 01–06 ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING FIGURES 01–06")
    logger.info("=" * 80)
    print("\n" + "=" * 80)
    print("GENERATING FIGURES 01–06")
    print("=" * 80 + "\n")

    create_figures_01_to_06(
        zonal_stats, early_data, late_data,
        timescale_analysis, advanced_data, FIGURE_DIR, logger
    )

    if early_data is not None and late_data is not None:
        create_violin_period_comparison(early_data, late_data, FIGURE_DIR, logger)

    # ── Figures 07–09 ────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING FIGURES 07–09 (InSAR, TwoSlopeNorm, Multipanel)")
    logger.info("=" * 80)
    print("\n" + "=" * 80)
    print("GENERATING FIGURES 07–09")
    print("=" * 80 + "\n")

    if insar_integrated is not None and len(insar_integrated) > 0:
        print("→ InSAR validation plots...")
        create_insar_comparison_plots(insar_integrated, FIGURE_DIR, logger)

        print("→ InSAR binned statistics...")
        # FIX: figure_dir param removed from signature
        create_insar_binned_stats(insar_integrated, OUTPUT_DIR, logger=logger)

        print("→ QGIS-optimised subsample...")
        create_insar_spatial_subsample(insar_integrated, OUTPUT_DIR, logger=logger)
    else:
        print("⚠️  InSAR validation skipped (no data)")

    if zonal_stats is not None:
        print("→ Contour map — zonal (±5 cm/yr symmetric)...")
        create_spatial_contour_map(zonal_stats, FIGURE_DIR, logger, 'zonal')

    if advanced_data is not None:
        print("→ Contour map — point-based...")
        create_spatial_contour_map(advanced_data, FIGURE_DIR, logger, 'points')

    print("→ 4-panel enhanced multipanel...")
    # FIX: prefer advanced_data; fall back to zonal_stats
    spatial_grid_for_multipanel = (
        advanced_data if advanced_data is not None else zonal_stats
    )
    create_enhanced_multipanel_figure(
        spatial_grid_for_multipanel, clusters_stats, FIGURE_DIR, logger
    )

    # ── Summary Statistics ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING SUMMARY STATISTICS")
    logger.info("=" * 80)
    print("\n" + "=" * 80)
    print("GENERATING SUMMARY STATISTICS")
    print("=" * 80)

    create_summary_statistics(
        zonal_stats, advanced_data, insar_integrated,
        FIGURE_DIR, OUTPUT_DIR, RUN_TIMESTAMP, git_hash, logger
    )

    # ── Metadata ─────────────────────────────────────────────────────────────
    n_figures = len(list(FIGURE_DIR.glob('Figure*.png')))

    metadata.configuration = {
        'figures': '9+',
        'dpi': 300,
        'colormap_primary': 'RdYlBu_r',
        'units': 'cm/yr',
        'sign_convention': 'negative=subsidence, positive=uplift',
        'contour_symmetric_range': '±5 cm/yr',
    }
    metadata.processing_results = {
        'figures_generated': n_figures,
        'zonal_cells': len(zonal_stats) if zonal_stats is not None else 0,
        'temporal_periods': 2 if (early_data is not None and late_data is not None) else 0,
        'n_clusters': len(clusters_stats) if clusters_stats is not None else 0,
        'has_insar': insar_integrated is not None,
        'csv_exports': 2 if insar_integrated is not None else 0,
    }

    metadata.finalize()
    metadata_file = metadata.save(OUTPUT_DIR)
    logger.info(f"Saved metadata: {metadata_file.name}")

    # ── Completion ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SCRIPT 07 COMPLETE — FINAL PUBLICATION FIGURES (v4.4)")
    print("=" * 80)
    print(f"\n  ✓ Figures generated  : {n_figures} × 300 DPI PNG")
    print(f"  ✓ Zonal cells        : {len(zonal_stats) if zonal_stats is not None else 0}")
    print(f"  ✓ Hotspot clusters   : {len(clusters_stats) if clusters_stats is not None else 0}")
    print(f"  ✓ InSAR validated    : {insar_integrated is not None}")
    print(f"\n  Location : {FIGURE_DIR}")
    print(f"  Summary  : {FIGURE_DIR / 'SummaryStatistics.txt'}")
    print(f"  Log      : {OUTPUT_DIR / 'script_07_run.log'}")
    print("=" * 80 + "\n")

    logger.info("=" * 80)
    logger.info("SCRIPT 07 EXECUTION COMPLETE (v4.4)")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()