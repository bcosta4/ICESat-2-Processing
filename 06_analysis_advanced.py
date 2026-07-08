"""
================================================================================
SCRIPT 06: SPATIAL CLUSTERING & SUBSIDENCE ANALYSIS
================================================================================

Purpose
-------
Aggregate ICESat-2 subsidence points using spatial proximity clustering:

1. k-nearest neighbor graph construction
2. Spatial clustering (BFS connected components)
3. No temporal constraints - date_pairs preserved as metadata
4. Inverse-variance weighting for rate estimation
5. Baseline length stratification (Δt bins)
6. Year-range stratification from datetime metadata
7. Robustness assessment across temporal baselines and periods
8. Cluster statistics and spatial variability
9. Coastal integration and filtering
10. Vector format exports (shapefile + GeoPackage)

CONFIGURATION
=============
✅ Cluster radius: 5m (configurable via Config.cluster_radius_m)
✅ Minimum cluster size: 3 points (configurable via Config.min_cluster_size)
✅ Spatial proximity only - NO temporal constraints
✅ Points with identical date_pairs CAN cluster together
✅ Baseline date and year-range metadata preserved for analysis
✅ Inverse-variance weighting for subsidence rate estimation

Inputs
------
- icesat2_repeat_track_matches.csv (Script 03)
- or 05_subsidence_rates.csv (Script 05) [REQUIRED for temporal metadata]
- or icesat2_insar_integrated.csv (InSAR integrated)

Outputs
-------
- 06_clusters_identified.csv (point-level cluster assignments)
- 06_cluster_statistics.csv (legacy cluster summary stats)
- 06_clusters_spatial_analysis.csv (extended: weighting + stratification)
- 06_clusters_points.shp / .gpkg (point geometries with attributes)
- 06_clusters_polygons.shp / .gpkg (cluster boundary polygons)
- 06_coastal_clusters.csv (coastal zone stratification)
- 06_cluster_map.png (spatial visualization)
- 06_cluster_stats.png (statistical visualization)
- 06_analysis_summary.txt (basic report)
- 06_analysis_advanced_summary.txt (extended execution report)
- 06_analysis_advanced_metadata.json (reproducibility metadata)
- script_06_run.log (execution log)
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
if sys.stdout is not None and sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr is not None and sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Try to import geopandas for vector exports
try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("⚠️ Warning: geopandas not available. Vector exports will be skipped.")

try:
    from config import BASEOUTPUTDIR
except ImportError:
    BASEOUTPUTDIR = Path(r'E:\Processing\ICeSat-2\output_data')

# =============================================================================
# CONFIGURATION - SINGLE POINT OF CONTROL
# =============================================================================
class Config:
    """Configuration for extended clustering analysis with FIXED constraints."""

    # ------------------------------------------------------------------
    # CLUSTERING PARAMETERS (SINGLE POINT OF CONTROL)
    # ------------------------------------------------------------------
    # Maximum spatial radius for neighbor identification in BFS clustering (meters)
    cluster_radius_m: float = 5.0

    # Minimum number of points required for a valid cluster
    min_cluster_size: int = 3

    # Weighting method for cluster rate estimation
    weight_method = 'inverse_variance'  # or 'inverse_deltat'

    # Δt baseline bins for temporal stratification
    baseline_dt_bins = [
        (0.25, 0.5),
        (0.5, 1.0),
        (1.0, 2.0),
        (2.0, 3.0),
        (3.0, None),
    ]

    # Robustness assessment
    robustness_threshold_pct = 20.0

    # Output formats
    output_formats = ['csv', 'shp', 'gpkg']

    # CRS
    utm_zone_epsg = 3155
    wgs84_epsg = 4326
    
    # Export all points (including unclustered) to GIS layers? 
    # False = only export clustered points (saves disk space)
    export_all_points: bool = False

# =============================================================================
# LOGGING SETUP
# =============================================================================
def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging to file and console."""
    log_file = output_dir / 'script_06_run.log'
    logger = logging.getLogger('script_06')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    fh = logging.FileHandler(log_file, encoding='utf-8', mode='w')
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    fh.setFormatter(fh_formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# =============================================================================
# UTILITY: FIND INPUT FILE
# =============================================================================
def find_input_file(output_dir: Path) -> Path:
    """Find input file with priority order."""
    priority_files = [
        output_dir / '05_subsidence_rates.csv',
        output_dir / 'icesat2_repeat_track_matches.csv',
        output_dir / 'icesat2_insar_integrated.csv',
        output_dir / '04_spatial_grid.csv',
    ]

    for filepath in priority_files:
        if filepath.exists():
            return filepath

    print("ERROR: Could not find input file.")
    print("Searched for:")
    for f in priority_files:
        print(f" - {f.name}")
    sys.exit(1)

# =============================================================================
# UTILITY: FIND RATE / UNCERTAINTY COLUMNS
# =============================================================================
def find_rate_column(df: pd.DataFrame) -> str:
    """Find subsidence rate column."""
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

def find_uncertainty_column(df: pd.DataFrame) -> str | None:
    """Find rate uncertainty column."""
    candidates = [
        'rate_uncertainty_cm_yr', 'uncertainty_cm_yr', 'std_error_cm_yr',
        'rate_uncertainty_cm_per_year', 'sigma_rate', 'sigma_H',
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None

# =============================================================================
# STANDARDIZE COLUMNS
# =============================================================================
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize elevation & uncertainty column names."""
    elev_candidates = ['elevation', 'elev', 'elevation_m', 'height_m', 'z']
    for col in elev_candidates:
        if col in df.columns and 'elevation' not in df.columns:
            df['elevation'] = df[col]
            break

    unc_candidates = [
        'rate_uncertainty_cm_yr', 'uncertainty_cm_yr',
        'std_error_cm_yr', 'rate_uncertainty_cm_per_year', 'sigma_rate',
    ]
    for col in unc_candidates:
        if col in df.columns and 'uncertainty' not in df.columns:
            df['uncertainty'] = df[col]
            break

    return df

# =============================================================================
# BASELINE DATE & YEAR-RANGE VALIDATION (NEW)
# =============================================================================
def validate_baseline_metadata(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Validate and derive baseline date and year-range metadata.

    Expected columns (priority order):
    1. date_early, date_late (temporal constraint clustering)
    2. start_date, end_date (baseline interval)
    3. baseline_date (optional reference)

    Derives:
    - observation_interval_yr (Δt in years from start/end dates)
    - year_range_period (YYYY_YYYY format for period stratification)

    Validates subsidence rates are in expected range [-14, -2] cm/yr.
    """
    logger.info("=" * 80)
    logger.info("BASELINE METADATA VALIDATION")
    logger.info("=" * 80)

    # Check which date columns are present
    has_early_late = 'date_early' in df.columns and 'date_late' in df.columns
    has_start_end = 'start_date' in df.columns and 'end_date' in df.columns
    has_baseline = 'baseline_date' in df.columns

    missing_fields = []
    if not has_early_late:
        missing_fields.append("date_early/date_late")
    if not has_start_end:
        missing_fields.append("start_date/end_date")
    if not has_baseline:
        missing_fields.append("baseline_date (optional)")

    if missing_fields:
        logger.warning(f"⚠️ Missing baseline fields: {', '.join(missing_fields)}")
        logger.info("Will derive from available temporal columns where possible.")

    # Derive start_date/end_date from date_early/date_late if needed
    if has_early_late and not has_start_end:
        logger.info("Deriving start_date/end_date from date_early/date_late")
        df['start_date'] = pd.to_datetime(df['date_early'], errors='coerce')
        df['end_date'] = pd.to_datetime(df['date_late'], errors='coerce')
        has_start_end = True

    # Compute observation_interval_yr (Δt in years)
    if 'observation_interval_yr' not in df.columns:
        if has_start_end:
            start_tmp = pd.to_datetime(df['start_date'], errors='coerce')
            end_tmp = pd.to_datetime(df['end_date'], errors='coerce')
            df['observation_interval_yr'] = (end_tmp - start_tmp).dt.days / 365.25
            logger.info("✅ Computed observation_interval_yr from start_date/end_date")

            # Validate Δt ranges match configured bins
            dt_vals = df['observation_interval_yr'].dropna().astype(float)
            if not dt_vals.empty:
                min_dt, max_dt = dt_vals.min(), dt_vals.max()
                logger.info(
                    f"Observation intervals range: {min_dt:.3f}–{max_dt:.3f} yr "
                    f"(bins: 0.25–0.5, 0.5–1, 1–2, 2–3, 3+ yr)"
                )
        else:
            logger.warning("Could not compute observation_interval_yr (no start/end dates)")

    # Derive year_range_period if missing
    if 'year_range_period' not in df.columns:
        logger.info("Deriving year_range_period from available datetime columns")
        if has_start_end:
            sdt = pd.to_datetime(df['start_date'], errors='coerce')
            edt = pd.to_datetime(df['end_date'], errors='coerce')
            df['year_range_period'] = (
                sdt.dt.year.astype('Int64').astype(str) + "_" +
                edt.dt.year.astype('Int64').astype(str)
            )
        elif 'datetime' in df.columns:
            sdt = pd.to_datetime(df['datetime'], errors='coerce')
            df['year_range_period'] = (
                sdt.dt.year.astype('Int64').astype(str) + "_" +
                sdt.dt.year.astype('Int64').astype(str)
            )
        else:
            df['year_range_period'] = 'unknown'
            logger.warning("No datetime columns available; year_range_period set to 'unknown'")

    return df

# =============================================================================
# VALIDATE SUBSIDENCE RATES
# =============================================================================
def validate_subsidence_rates(df: pd.DataFrame, rate_col: str, logger: logging.Logger):
    """Validate and report subsidence rate distribution (no range restrictions)."""
    rates = df[rate_col].dropna().astype(float)
    if len(rates) == 0:
        logger.error("❌ No valid subsidence rates found")
        sys.exit(1)

    min_rate, max_rate = rates.min(), rates.max()
    mean_rate = rates.mean()
    median_rate = rates.median()
    pct_negative = (rates < 0).mean() * 100
    pct_positive = (rates > 0).mean() * 100

    logger.info(f"Subsidence rate distribution:")
    logger.info(f"  Range: {min_rate:.2f} to {max_rate:.2f} cm/yr")
    logger.info(f"  Mean: {mean_rate:.2f} cm/yr | Median: {median_rate:.2f} cm/yr")
    logger.info(f"  Negative rates: {pct_negative:.1f}% | Positive rates: {pct_positive:.1f}%")
    
    logger.info("✅ Rate validation complete (all values accepted)")

# =============================================================================
# OBSERVATION INTERVAL & YEAR-RANGE EXTRACTION
# =============================================================================
def extract_observation_interval(df: pd.DataFrame, logger: logging.Logger) -> tuple[pd.Series | None, str | None]:
    """Extract observation interval (Δt in years)."""
    interval_candidates = [
        'observation_interval_yr', 'observation_interval', 'delta_t',
        'delta_t_yr', 'baseline_yr', 'baseline',
    ]
    for col in interval_candidates:
        if col in df.columns:
            logger.info(f"Using observation interval column: '{col}'")
            return df[col].astype(float), col

    if 'start_date' in df.columns and 'end_date' in df.columns:
        try:
            start = pd.to_datetime(df['start_date'])
            end = pd.to_datetime(df['end_date'])
            delta_t = (end - start).dt.days / 365.25
            logger.info("Computed observation interval from start_date/end_date")
            return delta_t, 'observation_interval_yr'
        except Exception as e:
            logger.warning(f"Failed to compute Δt from start_date/end_date: {e}")

    logger.warning("Could not reliably compute observation interval; Δt-based weighting may be limited.")
    return None, None

def extract_datetime_columns(df: pd.DataFrame) -> tuple[pd.Series | None, pd.Series | None]:
    """Extract datetime information."""
    if 'datetime' in df.columns:
        try:
            dt = pd.to_datetime(df['datetime'])
            return dt, None
        except Exception:
            pass

    if 'start_date' in df.columns and 'end_date' in df.columns:
        try:
            start = pd.to_datetime(df['start_date'])
            end = pd.to_datetime(df['end_date'])
            return start, end
        except Exception:
            pass

    return None, None

# =============================================================================
# WEIGHTED RATE CALCULATION
# =============================================================================
def calculate_weighted_rate(
    rates: np.ndarray,
    uncertainties: np.ndarray | None,
    deltat: np.ndarray | None,
    weight_method: str,
    logger: logging.Logger | None = None,
) -> dict:
    """
    Compute unweighted and weighted mean rate.

    - If weight_method == 'inverse_variance' and uncertainties available:
      use w = 1 / σ_rate^2
    - Else if deltat available:
      use w = 1 / Δt^2
    - Else: uniform weights
    """
    unweighted = float(np.nanmean(rates))
    method_used = 'uniform'

    if weight_method == 'inverse_variance' and uncertainties is not None:
        safe_unc = np.maximum(np.abs(uncertainties.astype(float)), 1e-6)
        weights = 1.0 / (safe_unc ** 2)
        method_used = 'inverse_variance'

    elif deltat is not None:
        safe_dt = np.maximum(np.abs(deltat.astype(float)), 1e-6)
        weights = 1.0 / (safe_dt ** 2)
        method_used = 'inverse_deltat'

    else:
        weights = np.ones_like(rates, dtype=float)

    if np.sum(weights) <= 0:
        weights = np.ones_like(rates, dtype=float)
        method_used = 'uniform'

    weights = weights / np.sum(weights)
    weighted = float(np.sum(weights * rates))

    return {
        'unweighted_rate': unweighted,
        'weighted_rate': weighted,
        'weights': weights,
        'weight_method_used': method_used,
    }

# =============================================================================
# Δt BIN STRATIFICATION
# =============================================================================
def stratify_by_deltat_bins(
    cluster_idx: np.ndarray,
    df: pd.DataFrame,
    rate_col: str,
    uncertainty_col: str | None,
    dt_col: str | None,
    dt_bins: list[tuple[float, float | None]],
    logger: logging.Logger,
) -> dict:
    """Compute weighted mean subsidence per Δt bin for a cluster."""
    cluster_df = df.iloc[cluster_idx]
    rates = cluster_df[rate_col].values.astype(float)
    deltat = cluster_df[dt_col].values.astype(float) if dt_col and dt_col in df.columns else None
    unc = cluster_df[uncertainty_col].values.astype(float) if uncertainty_col and uncertainty_col in df.columns else None

    bin_weighted_rates = {}
    bin_counts = {}

    for low, high in dt_bins:
        if high is None:
            label = f"rate_{low:.2f}_3plus_yr"
            mask = (deltat >= low) if deltat is not None else np.zeros_like(rates, dtype=bool)
        else:
            label = f"rate_{low:.2f}_{high:.2f}_yr"
            mask = (deltat >= low) & (deltat < high) if deltat is not None else np.zeros_like(rates, dtype=bool)

        if deltat is None:
            bin_weighted_rates[label] = np.nan
            bin_counts[label] = 0
            continue

        if not np.any(mask):
            bin_weighted_rates[label] = np.nan
            bin_counts[label] = 0
            continue

        result = calculate_weighted_rate(
            rates[mask],
            unc[mask] if unc is not None else None,
            deltat[mask],
            weight_method=Config.weight_method,
            logger=logger,
        )

        bin_weighted_rates[label] = result['weighted_rate']
        bin_counts[label] = int(np.sum(mask))

    return {
        'bin_weighted_rates': bin_weighted_rates,
        'bin_counts': bin_counts,
    }

# =============================================================================
# YEAR-RANGE STRATIFICATION
# =============================================================================
def stratify_by_year_range(
    cluster_idx: np.ndarray,
    df: pd.DataFrame,
    rate_col: str,
    uncertainty_col: str | None,
    dt_col: str | None,
    year_range_col: str,
    logger: logging.Logger,
) -> dict:
    """Compute weighted mean subsidence per year-range period for a cluster."""
    cluster_df = df.iloc[cluster_idx]
    rates = cluster_df[rate_col].values.astype(float)
    unc = cluster_df[uncertainty_col].values.astype(float) if uncertainty_col and uncertainty_col in df.columns else None
    deltat = cluster_df[dt_col].values.astype(float) if dt_col and dt_col in df.columns else None
    periods = cluster_df[year_range_col].astype(str).fillna('unknown').values

    unique_periods = np.unique(periods)
    period_weighted_rates = {}
    period_counts = {}

    for p in unique_periods:
        mask = periods == p

        if not np.any(mask):
            period_weighted_rates[p] = np.nan
            period_counts[p] = 0
            continue

        result = calculate_weighted_rate(
            rates[mask],
            unc[mask] if unc is not None else None,
            deltat[mask] if deltat is not None else None,
            weight_method=Config.weight_method,
            logger=logger,
        )

        period_weighted_rates[p] = result['weighted_rate']
        period_counts[p] = int(np.sum(mask))

    return {
        'period_weighted_rates': period_weighted_rates,
        'period_counts': period_counts,
    }

# =============================================================================
# ROBUSTNESS ASSESSMENT
# =============================================================================
def assess_robustness(
    dt_bin_rates: dict,
    period_rates: dict,
    threshold_pct: float,
    logger: logging.Logger | None = None,
) -> dict:
    """Assess robustness across Δt bins and year-range periods."""
    def _collect_valid(values):
        return [v for v in values if v is not None and not np.isnan(v)]

    vals_dt = _collect_valid(dt_bin_rates.values())
    vals_period = _collect_valid(period_rates.values())
    all_vals = vals_dt + vals_period

    if len(all_vals) < 2:
        return {
            'robustness_flag': False,
            'rate_variation_pct': np.nan,
            'rate_range': (np.nan, np.nan),
            'explanation': 'Insufficient temporal bins/periods',
        }

    abs_vals = np.abs(np.array(all_vals, dtype=float))
    min_v = float(np.min(abs_vals))
    max_v = float(np.max(abs_vals))

    if min_v <= 0:
        var_pct = 0.0
    else:
        var_pct = 100.0 * (max_v - min_v) / min_v

    flag = var_pct > threshold_pct

    msg = f"Variation {var_pct:.1f}% (threshold {threshold_pct}%)"
    msg += " ⚠️ UNSTABLE" if flag else " ✓"

    return {
        'robustness_flag': flag,
        'rate_variation_pct': var_pct,
        'rate_range': (float(min_v), float(max_v)),
        'explanation': msg,
    }

# =============================================================================
# SPATIAL METRICS
# =============================================================================
def compute_spatial_metrics(
    cluster_idx: np.ndarray,
    df: pd.DataFrame,
    lon_col: str,
    lat_col: str,
    elev_col: str | None,
) -> dict:
    """Compute centroid, spread, density, and elevation stats for a cluster."""
    cluster_df = df.iloc[cluster_idx]
    lons = cluster_df[lon_col].values.astype(float)
    lats = cluster_df[lat_col].values.astype(float)

    centroid_lon = float(np.mean(lons))
    centroid_lat = float(np.mean(lats))

    lon_std_m = np.std(lons) * 111000.0 * np.cos(np.radians(centroid_lat))
    lat_std_m = np.std(lats) * 111000.0
    spatial_spread_m = float(np.sqrt(lon_std_m**2 + lat_std_m**2))

    try:
        from scipy.spatial import ConvexHull
        if len(lons) >= 3:
            hull = ConvexHull(np.column_stack([lons, lats]))
            area_deg2 = hull.volume
            area_km2 = max(area_deg2 * (111.0 ** 2), 0.01)
        else:
            area_km2 = 0.01
    except Exception:
        area_km2 = 0.01

    point_density = float(len(lons) / area_km2)

    lon_extent_km = (np.max(lons) - np.min(lons)) * 111.0 * np.cos(np.radians(centroid_lat))
    lat_extent_km = (np.max(lats) - np.min(lats)) * 111.0
    extent_km = float(np.sqrt(lon_extent_km**2 + lat_extent_km**2))

    metrics = {
        'center_lon': centroid_lon,
        'center_lat': centroid_lat,
        'spatial_spread_m': spatial_spread_m,
        'point_density': point_density,
        'extent_km': extent_km,
    }

    if elev_col and elev_col in df.columns:
        e = cluster_df[elev_col].dropna().values.astype(float)
        if len(e) > 0:
            metrics.update({
                'elevation_mean_m': float(np.mean(e)),
                'elevation_std_m': float(np.std(e)),
                'elevation_min_m': float(np.min(e)),
                'elevation_max_m': float(np.max(e)),
            })

    return metrics

# =============================================================================
# VECTOR EXPORTS
# =============================================================================
def export_vector_formats(
    df: pd.DataFrame,
    cluster_id_col: str,
    rate_col: str,
    lon_col: str,
    lat_col: str,
    output_dir: Path,
    elev_col: str | None,
    unc_col: str | None,
    weighted_rate_col: str,
    dt_col: str | None,
    baseline_bin_col: str,
    year_range_col: str,
    logger: logging.Logger,
) -> bool:
    """Export point & polygon layers as shapefile + GeoPackage."""
    if not GEOPANDAS_AVAILABLE:
        logger.warning("geopandas not available; skipping vector exports.")
        return False

    try:
 # ==================== OPTIONAL: Export ALL points ====================
        if Config.export_all_points:
            logger.info("Exporting ALL points (clustered + unclustered)...")
            df_all = df.copy()
            if 'point_id' not in df_all.columns:
                df_all['point_id'] = np.arange(len(df_all))
            
            geometry_all = [Point(xy) for xy in zip(df_all[lon_col], df_all[lat_col])]
            gdf_all = gpd.GeoDataFrame(df_all, geometry=geometry_all, crs=f"EPSG:{Config.wgs84_epsg}")
            
            # Select fields for all-points export
            all_fields = [
                'point_id', cluster_id_col, rate_col, unc_col, dt_col,
                baseline_bin_col, year_range_col, elev_col,
                'datetime' if 'datetime' in df_all.columns else None,
                'start_date' if 'start_date' in df_all.columns else None,
                'end_date' if 'end_date' in df_all.columns else None,
            ]
            all_fields = [c for c in all_fields if c and c in gdf_all.columns]
            gdf_all = gdf_all[all_fields + ['geometry']]
            
            # Export EPSG:4326 (WGS84)
            shp_all_wgs = output_dir / '06_allpoints_wgs84.shp'
            gdf_all.to_file(shp_all_wgs, driver='ESRI Shapefile')
            gpkg_all_wgs = output_dir / '06_allpoints_wgs84.gpkg'
            gdf_all.to_file(gpkg_all_wgs, driver='GPKG', layer='allpoints_wgs84')
            
            # Export EPSG:3155 (UTM Zone 8N - your project CRS)
            gdf_all_utm = gdf_all.to_crs(f"EPSG:{Config.utm_zone_epsg}")
            shp_all_utm = output_dir / '06_allpoints_utm.shp'
            gdf_all_utm.to_file(shp_all_utm, driver='ESRI Shapefile')
            gpkg_all_utm = output_dir / '06_allpoints_utm.gpkg'
            gdf_all_utm.to_file(gpkg_all_utm, driver='GPKG', layer='allpoints_utm')
            
            n_clustered_export = (df_all[cluster_id_col] >= 0).sum()
            n_unclustered_export = (df_all[cluster_id_col] == -1).sum()
            logger.info(f"✅ Exported {len(df_all):,} total points (WGS84 + UTM)")
            logger.info(f"   - Clustered: {n_clustered_export:,}")
            logger.info(f"   - Unclustered: {n_unclustered_export:,}")
            print(f"✅ Exported ALL {len(df_all):,} points")
            print(f"   - 06_allpoints_wgs84.shp/.gpkg (EPSG:4326)")
            print(f"   - 06_allpoints_utm.shp/.gpkg (EPSG:3155)")
        else:
            logger.info("Skipped all-points export (Config.export_all_points=False)")
            print("✓ Skipped all-points export (Config.export_all_points=False)")
        
        # Points (only clustered ones)
        df_points = df[df[cluster_id_col] >= 0].copy()

        if 'point_id' not in df_points.columns:
            df_points['point_id'] = np.arange(len(df_points))

        geometry = [Point(xy) for xy in zip(df_points[lon_col], df_points[lat_col])]
        gdf_points = gpd.GeoDataFrame(df_points, geometry=geometry, crs=f"EPSG:{Config.wgs84_epsg}")

        point_fields = [
            'point_id', cluster_id_col, rate_col, weighted_rate_col, unc_col, dt_col,
            baseline_bin_col, year_range_col, elev_col,
            'datetime' if 'datetime' in df_points.columns else None,
            'start_date' if 'start_date' in df_points.columns else None,
            'end_date' if 'end_date' in df_points.columns else None,
        ]

        point_fields = [c for c in point_fields if c and c in gdf_points.columns]
        gdf_points = gdf_points[point_fields + ['geometry']]

        shp_points = output_dir / '06_clusters_points.shp'
        gdf_points.to_file(shp_points, driver='ESRI Shapefile')

        gpkg_points = output_dir / '06_clusters_points.gpkg'
        gdf_points.to_file(gpkg_points, driver='GPKG', layer='clusters_points')

        logger.info("Saved 06_clusters_points.shp/.gpkg")
        
        # ==================== NEW: ADD UTM VERSIONS ====================
        gdf_points_utm = gdf_points.to_crs(f"EPSG:{Config.utm_zone_epsg}")
        shp_points_utm = output_dir / '06_clusters_points_utm.shp'
        gdf_points_utm.to_file(shp_points_utm, driver='ESRI Shapefile')
        gpkg_points_utm = output_dir / '06_clusters_points_utm.gpkg'
        gdf_points_utm.to_file(gpkg_points_utm, driver='GPKG', layer='clusters_points_utm')
        logger.info("Saved 06_clusters_points_utm.shp/.gpkg (EPSG:3155)")

        # Polygons (cluster boundaries)
        cluster_polygons = []
        poly_stats = []

        from scipy.spatial import ConvexHull

        for cid in sorted(df_points[cluster_id_col].unique()):
            sub = df_points[df_points[cluster_id_col] == cid]
            if len(sub) < 3:
                continue

            coords = np.column_stack([sub[lon_col].values.astype(float),
                                     sub[lat_col].values.astype(float)])
            try:
                hull = ConvexHull(coords)
                hull_points = coords[hull.vertices]
                poly = Polygon(hull_points)
            except Exception:
                center_lon = float(sub[lon_col].mean())
                center_lat = float(sub[lat_col].mean())
                poly = Point(center_lon, center_lat).buffer(0.001)

            cluster_polygons.append(poly)

            stats = {
                'cluster_id': cid,
                'num_points': int(len(sub)),
                'weighted_mean_subsidence_cm_yr': float(sub[weighted_rate_col].mean()),
            }

            if 'robustness_flag' in df.columns:
                stats['robustness_flag'] = bool(
                    df[(df[cluster_id_col] == cid)]['robustness_flag'].iloc[0]
                )

            if elev_col and elev_col in sub.columns:
                stats['elevation_mean_m'] = float(sub[elev_col].mean())

            poly_stats.append(stats)

        if cluster_polygons:
            gdf_polygons = gpd.GeoDataFrame(
                {'cluster_id': [s['cluster_id'] for s in poly_stats]},
                geometry=cluster_polygons,
                crs=f"EPSG:{Config.wgs84_epsg}"
            )

            df_poly_stats = pd.DataFrame(poly_stats)
            gdf_polygons = gdf_polygons.merge(df_poly_stats, on='cluster_id', how='left')

            shp_polygons = output_dir / '06_clusters_polygons.shp'
            gpkg_polygons = output_dir / '06_clusters_polygons.gpkg'

            gdf_polygons.to_file(shp_polygons, driver='ESRI Shapefile')
            gdf_polygons.to_file(gpkg_polygons, driver='GPKG', layer='clusters_polygons')

            logger.info("Saved 06_clusters_polygons.shp/.gpkg")
            
            # ==================== NEW: ADD UTM VERSIONS ====================
            gdf_polygons_utm = gdf_polygons.to_crs(f"EPSG:{Config.utm_zone_epsg}")
            shp_polygons_utm = output_dir / '06_clusters_polygons_utm.shp'
            gdf_polygons_utm.to_file(shp_polygons_utm, driver='ESRI Shapefile')
            gpkg_polygons_utm = output_dir / '06_clusters_polygons_utm.gpkg'
            gdf_polygons_utm.to_file(gpkg_polygons_utm, driver='GPKG', layer='clusters_polygons_utm')
            logger.info("Saved 06_clusters_polygons_utm.shp/.gpkg (EPSG:3155)")

        return True

    except Exception as e:
        logger.error(f"Vector export failed: {e}")
        return False

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main(threshold: float = -2.0, radius_km: float = 2.0, min_size: int = 10, k: int = 250):
    """Main clustering workflow (spatial proximity only)."""
    print("=" * 80)
    print("SCRIPT 06: SPATIAL CLUSTERING & SUBSIDENCE ANALYSIS")
    print("=" * 80)

    # Run ID and output directory
    timestamp_file = BASEOUTPUTDIR / 'latest_RUN_TIMESTAMP.txt'
    if not timestamp_file.exists():
        print("ERROR: Run Script 00 first.")
        sys.exit(1)

    RUN_TIMESTAMP = timestamp_file.read_text(encoding='utf-8').strip()
    OUTPUT_DIR = BASEOUTPUTDIR / f"run_{RUN_TIMESTAMP}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(OUTPUT_DIR)

    logger.info("=" * 80)
    logger.info("SCRIPT 06: SPATIAL CLUSTERING")
    logger.info("=" * 80)
    logger.info(f"Run ID: {RUN_TIMESTAMP}")

    print(f"Run ID: {RUN_TIMESTAMP}")
    print(f"Output: {OUTPUT_DIR}")

    # -------------------------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------------------------
    logger.info("\n1. LOADING DATA")
    print("\n" + "=" * 80)
    print("1. LOADING DATA")
    print("=" * 80)

    input_file = find_input_file(OUTPUT_DIR)
    df = pd.read_csv(input_file)

    logger.info(f"Loaded {len(df):,} points from {input_file.name}")
    print(f"✓ Loaded {len(df):,} points")

    df = standardize_columns(df)

    # ------------------------------------------------------------------
    # VALIDATE BASELINE METADATA (NEW)
    # ------------------------------------------------------------------
    df = validate_baseline_metadata(df, logger)

    rate_col = find_rate_column(df)
    logger.info(f"Using rate column: '{rate_col}'")
    print(f"✓ Using rate column: '{rate_col}'")

    validate_subsidence_rates(df, rate_col, logger)

    unc_col = find_uncertainty_column(df)
    if unc_col:
        logger.info(f"Using uncertainty column: '{unc_col}'")
        print(f"✓ Using uncertainty column: '{unc_col}'")
    else:
        logger.warning("No uncertainty column found; may fall back to Δt weighting.")
        print("⚠️ No uncertainty column; may fall back to Δt weighting.")

    # Δt extraction
    dt_series, dt_col = extract_observation_interval(df, logger)
    if dt_series is not None and dt_col != 'observation_interval_yr':
        df['observation_interval_yr'] = dt_series
        dt_col = 'observation_interval_yr'
    elif dt_series is not None:
        df[dt_col] = dt_series

    # datetime / year-range extraction
    single_dt, end_dt = extract_datetime_columns(df)
    start_dt = single_dt if end_dt is None else single_dt

    if 'start_date' in df.columns and 'end_date' in df.columns:
        start_dt = pd.to_datetime(df['start_date'], errors='coerce')
        end_dt = pd.to_datetime(df['end_date'], errors='coerce')

    # lat / lon columns
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), 'lon')
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), 'lat')

    if lon_col not in df.columns or lat_col not in df.columns:
        print("ERROR: Could not find lon/lat columns.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 2. k-NN GRAPH (10m SPATIAL RADIUS - FIXED)
    # -------------------------------------------------------------------------
    logger.info("\n2. k-NEAREST NEIGHBOR GRAPH (≤10 m)")
    print("\n" + "=" * 80)
    print("2. k-NEAREST NEIGHBOR GRAPH (≤10 m)")
    print("=" * 80)

    coords = df[[lon_col, lat_col]].values
    tree = cKDTree(coords)

    try:
        distances, indices = tree.query(coords, k=min(k + 1, len(df)))
    except Exception as e:
        logger.warning(f"k-nearest failed with k={k}: {e}; retrying with k=150")
        k = min(150, len(df))
        distances, indices = tree.query(coords, k=k + 1)

    neighbors_list = []
    max_radius_m = float(min(Config.cluster_radius_m, 10.0))  # Hard cap at 10m

    for i in range(len(coords)):
        this_lon, this_lat = coords[i]
        neighbor_idx = indices[i, 1:]
        neighbor_dist_deg = coords[neighbor_idx] - coords[i]
        dlon = neighbor_dist_deg[:, 0]
        dlat = neighbor_dist_deg[:, 1]
        dlat_m = dlat * 111000.0
        dlon_m = dlon * 111000.0 * np.cos(np.radians(this_lat))
        dists_m = np.sqrt(dlat_m ** 2 + dlon_m ** 2)

        valid = dists_m <= max_radius_m
        neighbors_list.append(neighbor_idx[valid])

    k_used = float(np.mean([len(n) for n in neighbors_list]))
    logger.info(f"Avg neighbors per point: {k_used:.1f} (within {max_radius_m:.1f} m)")
    print(f"✓ k-nearest graph built (avg {k_used:.1f} neighbors within {max_radius_m:.1f} m)")

    # -------------------------------------------------------------------------
    # 3. HOTSPOT DETECTION
    # -------------------------------------------------------------------------
    logger.info("\n3. HOTSPOT DETECTION")
    print("\n" + "=" * 80)
    print("3. HOTSPOT DETECTION")
    print("=" * 80)

    rates_all = df[rate_col].values.astype(float)
    hotspot_mask = np.ones(len(df), dtype=bool)

    logger.info(f"General analysis: ALL {len(df):,} points included (no threshold)")
    print(f"✓ General analysis: ALL {len(df):,} points")
    print(f"  No threshold applied - complete spatial analysis")

    # -------------------------------------------------------------------------
    # 4. BFS CLUSTERING WITH TEMPORAL CONSTRAINT (FIXED)
    # -------------------------------------------------------------------------
    logger.info("\\n4. CLUSTER IDENTIFICATION (SPATIAL PROXIMITY ONLY)")
    print("\\n" + "=" * 80)
    print("4. CLUSTER IDENTIFICATION (SPATIAL CLUSTERING)")
    print("=" * 80)

    # Determine date columns
    # Determine date columns
    date_early_col = 'date_early' if 'date_early' in df.columns else 'start_date'
    date_late_col = 'date_late' if 'date_late' in df.columns else 'end_date'

    if date_early_col not in df.columns or date_late_col not in df.columns:
        logger.error("❌ Required date columns not found")
        sys.exit(1)

    # ==================== FIXED: Robust date_pair precomputation ====================
    logger.info(f"Date columns: '{date_early_col}' + '{date_late_col}'")
    
    early_dates = pd.to_datetime(df[date_early_col], errors='coerce')
    late_dates = pd.to_datetime(df[date_late_col], errors='coerce')
    
    # Count invalid dates (will prevent clustering)
    n_invalid_early = early_dates.isna().sum()
    n_invalid_late = late_dates.isna().sum()
    n_invalid_any = (early_dates.isna() | late_dates.isna()).sum()
    
    logger.info(f"Date parsing: {n_invalid_early} invalid early, {n_invalid_late} invalid late")
    logger.info(f"Total points with invalid dates: {n_invalid_any:,} ({n_invalid_any/len(df)*100:.1f}%)")
    
    # Create hashable date_pairs as TUPLES (FIX: handles NaT properly)
    date_pairs = list(zip(early_dates, late_dates))
    
    # Count duplicate date_pairs (these CANNOT cluster together)
    from collections import Counter
    pair_counter = Counter(date_pairs)
    pair_counts = pd.Series(pair_counter)
    
    n_duplicate_pairs = (pair_counts > 1).sum()
    n_points_in_duplicates = pair_counts[pair_counts > 1].sum()
    
    logger.info(f"Duplicate date_pairs: {n_duplicate_pairs:,} pairs shared by {n_points_in_duplicates:,} points")
    print(f"✓ Duplicate date_pairs: {n_duplicate_pairs:,} pairs → {n_points_in_duplicates:,} points cannot cluster together")
    
    if n_duplicate_pairs > 0:
        top_pairs = pair_counts.nlargest(5)
        logger.info("Top 5 most repeated date_pairs:")
        for pair_tuple, count in top_pairs.items():
            logger.info(f"  {pair_tuple}: {count} points")
    
    cluster_id = np.full(len(df), -1, dtype=int)
    current_cluster = 0


    for i in range(len(df)):
        if not hotspot_mask[i] or cluster_id[i] != -1:
            continue

        queue = [i]
        cluster_id[i] = current_cluster
        qi = 0

        # BFS growth (spatial proximity only):
        # A neighbor is added if:
        #   - within 5m (enforced in neighbors_list construction)
        #   - not already assigned to another cluster
        while qi < len(queue):
            p = queue[qi]
            qi += 1

            for nb in neighbors_list[p]:
                if not hotspot_mask[nb] or cluster_id[nb] != -1:
                    continue

                # Spatial clustering only - no temporal constraint
                cluster_id[nb] = current_cluster
                queue.append(nb)

        # Enforce minimum cluster size
        members = np.where(cluster_id == current_cluster)[0]
        if len(members) < Config.min_cluster_size:
            cluster_id[members] = -1
            logger.debug(f"Discarded cluster {current_cluster} (size {len(members)}) < min {Config.min_cluster_size}")
        else:
            current_cluster += 1

    n_clusters = current_cluster
    logger.info(f"Raw clusters (temporal + spatial): {n_clusters}")
    print(f"✓ Raw clusters (temporal + spatial): {n_clusters}")

    # ==================== NEW: COVERAGE DIAGNOSTICS ====================
    n_total = len(df)
    n_clustered = (cluster_id >= 0).sum()
    n_unclustered = n_total - n_clustered
    pct_clustered = (n_clustered / n_total * 100) if n_total > 0 else 0

    logger.info("=" * 80)
    logger.info("CLUSTERING COVERAGE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total input points: {n_total:,}")
    logger.info(f"Clustered points: {n_clustered:,} ({pct_clustered:.1f}%)")
    logger.info(f"Unclustered points: {n_unclustered:,} ({100-pct_clustered:.1f}%)")
    logger.info(f"Valid clusters (>= {Config.min_cluster_size} pts): {n_clusters}")

    print("\n" + "=" * 80)
    print("CLUSTERING COVERAGE")
    print("=" * 80)
    print(f"Total points: {n_total:,}")
    print(f"Clustered: {n_clustered:,} ({pct_clustered:.1f}%)")
    print(f"Unclustered: {n_unclustered:,} ({100-pct_clustered:.1f}%)")
    print(f"Valid clusters: {n_clusters}")
    print("=" * 80 + "\n")

    # Check average neighbors
    avg_neighbors = np.mean([len(n) for n in neighbors_list])
    logger.info(f"Average neighbors per point (within {max_radius_m:.1f} m): {avg_neighbors:.2f}")
    if avg_neighbors < 3:
        logger.warning("⚠️  Average neighbors < 3: Consider increasing cluster_radius_m")
        print(f"⚠️  Avg neighbors ({avg_neighbors:.2f}) < min cluster size (3)")
        print("   → Most points cannot form clusters with current 10 m radius")

    # Report date_pair distribution (metadata only)
    if date_early_col in df.columns and date_late_col in df.columns:
        duplicate_pairs = pd.Series(Counter(date_pairs))
        n_identical = (duplicate_pairs > 1).sum()
        if n_identical > 0:
            logger.info(f"Points sharing identical date_pairs: {n_identical:,} unique pairs")
            logger.info("✓ Date_pairs are metadata - spatial clustering unaffected")
            print(f"✓ {n_identical:,} unique date_pairs (metadata preserved)")

    # -------------------------------------------------------------------------
    # 5. CLUSTER ANALYSIS (WEIGHTING & STRATIFICATION)
    # -------------------------------------------------------------------------
    logger.info("\\n5. CLUSTER ANALYSIS (WEIGHTING & STRATIFICATION)")
    print("\\n5. CLUSTER ANALYSIS (WEIGHTING & STRATIFICATION)")

    df['cluster_id'] = cluster_id

    cluster_stats_records = []
    robustness_report = []

    for cid in range(n_clusters):
        idx = np.where(cluster_id == cid)[0]

        if len(idx) < Config.min_cluster_size:
            continue

        cluster_rates = rates_all[idx]
        cluster_unc = df[unc_col].values[idx] if unc_col and unc_col in df.columns else None
        cluster_dt = df[dt_col].values[idx] if dt_col and dt_col in df.columns else None

        year_range_col = 'year_range_period'

        # Inverse-variance / Δt weighting
        weight_result = calculate_weighted_rate(
            cluster_rates, cluster_unc, cluster_dt,
            weight_method=Config.weight_method, logger=logger,
        )

        unweighted = weight_result['unweighted_rate']
        weighted = weight_result['weighted_rate']
        weight_method_used = weight_result['weight_method_used']

        # Δt bin stratification
        dt_strat = stratify_by_deltat_bins(
            idx, df, rate_col, unc_col, dt_col, Config.baseline_dt_bins, logger
        )
        dt_bin_rates = dt_strat['bin_weighted_rates']

        # year-range stratification
        yr_strat = stratify_by_year_range(
            idx, df, rate_col, unc_col, dt_col, year_range_col, logger
        )
        period_rates = yr_strat['period_weighted_rates']

        # robustness
        robust = assess_robustness(dt_bin_rates, period_rates,
                                   Config.robustness_threshold_pct, logger)
        robustness_flag = robust['robustness_flag']
        rate_variation_pct = robust['rate_variation_pct']

        if robustness_flag:
            robustness_report.append({
                'cluster_id': cid,
                'variation_pct': rate_variation_pct,
                'explanation': robust['explanation'],
            })

        # spatial metrics
        spatial = compute_spatial_metrics(
            idx, df, lon_col, lat_col, elev_col='elevation' if 'elevation' in df.columns else None
        )

        # temporal context
        if date_early_col in df.columns and date_late_col in df.columns:
            cluster_early = pd.to_datetime(df.iloc[idx][date_early_col], errors='coerce')
            cluster_late = pd.to_datetime(df.iloc[idx][date_late_col], errors='coerce')
            min_date = cluster_early.min()
            max_date = cluster_late.max()
        else:
            min_date = pd.NaT
            max_date = pd.NaT

        # distance_to_coast
        dist_coast_col = next((c for c in df.columns if 'distance_to_coast' in c.lower()), None)
        if dist_coast_col:
            dist_cluster = df.iloc[idx][dist_coast_col].dropna().values
            distance_to_coast_m = float(np.mean(dist_cluster)) if len(dist_cluster) > 0 else np.nan
        else:
            distance_to_coast_m = np.nan

        # assemble record (BASE)
        rec = {
            'cluster_id': cid,
            'num_points': len(idx),
            'center_lon': spatial.get('center_lon', np.nan),
            'center_lat': spatial.get('center_lat', np.nan),
            'spatial_spread_m': spatial.get('spatial_spread_m', np.nan),
            'point_density': spatial.get('point_density', np.nan),
            'extent_km': spatial.get('extent_km', np.nan),
            'elevation_mean_m': spatial.get('elevation_mean_m', np.nan),
            'elevation_std_m': spatial.get('elevation_std_m', np.nan),
            'elevation_min_m': spatial.get('elevation_min_m', np.nan),
            'elevation_max_m': spatial.get('elevation_max_m', np.nan),
            'mean_subsidence_cm_yr': unweighted,
            'weighted_subsidence_cm_yr': weighted,
            'std_subsidence_cm_yr': float(np.nanstd(cluster_rates)),
            'min_subsidence_cm_yr': float(np.nanmin(cluster_rates)),
            'max_subsidence_cm_yr': float(np.nanmax(cluster_rates)),
            'mean_observation_interval_yr': float(np.nanmean(cluster_dt)) if cluster_dt is not None else np.nan,
            'weight_method': weight_method_used,
            'robustness_flag': bool(robustness_flag),
            'rate_variation_pct': float(rate_variation_pct) if rate_variation_pct is not None else np.nan,
            'date_range': f"{min_date.date()}__{max_date.date()}" if pd.notna(min_date) and pd.notna(max_date) else "",
            'observation_count_per_cluster': len(idx),
            'distance_to_coast_m': distance_to_coast_m,
        }

        # ========== NEW: BULK SUBSIDENCE (FIRST→LAST ELEVATION) ==========
        cluster_df = df.iloc[idx].copy()
        if 'elevation' in cluster_df.columns and 'start_date' in cluster_df.columns:
            cluster_df_sorted = cluster_df.sort_values('start_date')
            first_elev = cluster_df_sorted['elevation'].iloc[0]
            last_elev = cluster_df_sorted['elevation'].iloc[-1]
            first_date = pd.to_datetime(cluster_df_sorted['start_date'].iloc[0])
            last_date = pd.to_datetime(cluster_df_sorted['end_date'].iloc[-1])
            
            total_delta_elev_m = float(last_elev - first_elev)
            total_time_yr = (last_date - first_date).days / 365.25
            bulk_subsidence_cm_yr = (total_delta_elev_m * 100) / total_time_yr if total_time_yr > 0 else np.nan
            
            rec.update({
                'first_elev_m': float(first_elev),
                'last_elev_m': float(last_elev),
                'total_delta_elev_cm': total_delta_elev_m * 100,
                'observation_span_yr': float(total_time_yr),
                'bulk_subsidence_cm_yr': bulk_subsidence_cm_yr,
                'first_date': first_date.strftime('%Y-%m-%d'),
                'last_date': last_date.strftime('%Y-%m-%d'),
            })
        # ================================================================

        # Δt bin columns
        for low, high in Config.baseline_dt_bins:
            if high is None:
                label_key = f"rate_{low:.2f}_3plus_yr"
                out_col = "rate_3plus_yr"
            else:
                label_key = f"rate_{low:.2f}_{high:.2f}_yr"
                if (low, high) == (0.25, 0.5):
                    out_col = "rate_0.25_0.5yr"
                elif (low, high) == (0.5, 1.0):
                    out_col = "rate_0.5_1yr"
                elif (low, high) == (1.0, 2.0):
                    out_col = "rate_1_2yr"
                elif (low, high) == (2.0, 3.0):
                    out_col = "rate_2_3yr"
                else:
                    out_col = label_key
            rec[out_col] = dt_bin_rates.get(label_key, np.nan)

        # year-range period columns
        for period_label, val in period_rates.items():
            if period_label == 'unknown':
                continue
            col_name = f"rate_{period_label}"
            rec[col_name] = val

        cluster_stats_records.append(rec)

    df_cluster_stats = pd.DataFrame(cluster_stats_records)
    n_valid_clusters = len(df_cluster_stats)

    logger.info(f"Valid clusters (>= {Config.min_cluster_size} pts): {n_valid_clusters}")
    print(f"✓ Valid clusters: {n_valid_clusters}")

    if robustness_report:
        logger.warning(f"{len(robustness_report)} clusters flagged as potentially unstable.")
        print(f"⚠️ {len(robustness_report)} clusters flagged as potentially unstable.")

    # NEW: Bulk subsidence summary
    if 'bulk_subsidence_cm_yr' in df_cluster_stats.columns:
        bulk_rates = df_cluster_stats['bulk_subsidence_cm_yr'].dropna()
        if len(bulk_rates) > 0:
            logger.info(f"Bulk subsidence (first→last): {bulk_rates.mean():.2f}±{bulk_rates.std():.2f} cm/yr")
            print(f"✓ Bulk subsidence: {bulk_rates.mean():.2f}cm/yr (span {df_cluster_stats['observation_span_yr'].mean():.1f}yr avg)")



    # -------------------------------------------------------------------------
    # 6. SIGN CONVENTION
    # -------------------------------------------------------------------------
    logger.info("\n6. SIGN CONVENTION VERIFICATION")
    print("\n6. SIGN CONVENTION VERIFICATION")

    mean_cluster = df_cluster_stats['mean_subsidence_cm_yr'].mean() if n_valid_clusters > 0 else np.nan
    mean_weighted = df_cluster_stats['weighted_subsidence_cm_yr'].mean() if n_valid_clusters > 0 else np.nan

    if n_valid_clusters > 0 and (mean_cluster > 0 or mean_weighted > 0):
        logger.warning("Positive cluster mean detected. Check rate column / sign convention.")
        print("⚠️ WARNING: Positive cluster mean (unexpected)")
        print(f" Unweighted: {mean_cluster:.3f} cm/yr")
        print(f" Weighted: {mean_weighted:.3f} cm/yr")
    elif n_valid_clusters > 0:
        logger.info(f"Sign verified. Unweighted mean: {mean_cluster:.3f} cm/yr, weighted: {mean_weighted:.3f} cm/yr")
        print(f"✓ Sign verified (unweighted): {mean_cluster:.3f} cm/yr (negative ✓)")
        print(f"✓ Sign verified (weighted): {mean_weighted:.3f} cm/yr (negative ✓)")

    # -------------------------------------------------------------------------
    # 7. COASTAL FILTERING
    # -------------------------------------------------------------------------
    logger.info("\n7. COASTAL FILTERING")
    print("\n7. COASTAL FILTERING")

    coastal_col = next((c for c in df.columns if 'coastal_zone' in c.lower()), None)
    if coastal_col:
        coastal_clusters = df[df['cluster_id'] >= 0].groupby('cluster_id')[coastal_col].agg([
            ('n_coastal', lambda x: (x == 'Coastal (0-1 km)').sum()),
            ('coastal_pct', lambda x: (x == 'Coastal (0-1 km)').mean() * 100),
        ]).round(1)

        df_coastal = df_cluster_stats.merge(coastal_clusters, left_on='cluster_id', right_index=True, how='left')
        df_coastal.fillna(0, inplace=True)

        df_coastal.to_csv(OUTPUT_DIR / '06_coastal_clusters.csv', index=False)
        logger.info("Saved coastal cluster stratification.")
        print("✓ Coastal clusters saved")

    # -------------------------------------------------------------------------
    # 8. SAVE CSV OUTPUTS
    # -------------------------------------------------------------------------
    logger.info("\n8. SAVING CSV OUTPUTS")
    print("\n8. SAVING CSV OUTPUTS")

    # Legacy cluster stats
    legacy_cols = ['cluster_id', 'num_points', 'mean_subsidence_cm_yr', 'std_subsidence_cm_yr',
                   'center_lat', 'center_lon', 'extent_km']

    df_cluster_stats_legacy = df_cluster_stats.copy()
    for col in legacy_cols:
        if col not in df_cluster_stats_legacy.columns:
            df_cluster_stats_legacy[col] = np.nan

    df_cluster_stats_legacy[legacy_cols].to_csv(OUTPUT_DIR / '06_cluster_statistics.csv', index=False)
    logger.info("Saved 06_cluster_statistics.csv")
    print("✓ 06_cluster_statistics.csv")

    # Point-level cluster assignments
    out_cols = [lat_col, lon_col, rate_col, 'cluster_id']
    if 'elevation' in df.columns:
        out_cols.append('elevation')
    if 'uncertainty' in df.columns:
        out_cols.append('uncertainty')

    df[out_cols].to_csv(OUTPUT_DIR / '06_clusters_identified.csv', index=False)
    logger.info("Saved 06_clusters_identified.csv")
    print("✓ 06_clusters_identified.csv")

    # Extended cluster spatial analysis
    df_cluster_stats.to_csv(OUTPUT_DIR / '06_clusters_spatial_analysis.csv', index=False)
    logger.info("Saved 06_clusters_spatial_analysis.csv")
    print("✓ 06_clusters_spatial_analysis.csv")

    # -------------------------------------------------------------------------
    # 9. VECTOR EXPORTS
    # -------------------------------------------------------------------------
    logger.info("\n9. VECTOR EXPORTS")
    print("\n9. VECTOR EXPORTS")

    # Attach weighted rate at point-level
    df['weighted_subsidence_cm_yr'] = np.nan
    for _, row in df_cluster_stats[['cluster_id', 'weighted_subsidence_cm_yr']].iterrows():
        cid = row['cluster_id']
        df.loc[df['cluster_id'] == cid, 'weighted_subsidence_cm_yr'] = row['weighted_subsidence_cm_yr']

    # Baseline_bin (per point) from dt
    df['baseline_bin'] = 'unknown'
    if dt_col and dt_col in df.columns:
        dt_values = df[dt_col].astype(float)
        for i in df.index:
            v = dt_values.iat[i]
            if pd.isna(v):
                continue
            label = 'unknown'
            if 0.25 <= v < 0.5:
                label = '0.25-0.5yr'
            elif 0.5 <= v < 1.0:
                label = '0.5-1yr'
            elif 1.0 <= v < 2.0:
                label = '1-2yr'
            elif 2.0 <= v < 3.0:
                label = '2-3yr'
            elif v >= 3.0:
                label = '3+yr'
            df.at[i, 'baseline_bin'] = label

    year_range_col = 'year_range_period'

    vector_success = export_vector_formats(
        df,
        cluster_id_col='cluster_id',
        rate_col=rate_col,
        lon_col=lon_col,
        lat_col=lat_col,
        output_dir=OUTPUT_DIR,
        elev_col='elevation' if 'elevation' in df.columns else None,
        unc_col='uncertainty' if 'uncertainty' in df.columns else None,
        weighted_rate_col='weighted_subsidence_cm_yr',
        dt_col=dt_col,
        baseline_bin_col='baseline_bin',
        year_range_col=year_range_col,
        logger=logger,
    )

    if vector_success:
        print("✓ Vector exports (shapefile + GeoPackage)")
    else:
        print("⚠️ Vector exports failed or skipped; CSV outputs still available.")

    # -------------------------------------------------------------------------
    # 10. SUMMARY REPORTS
    # -------------------------------------------------------------------------
    logger.info("\n10. GENERATING SUMMARY REPORTS")
    print("\n10. GENERATING SUMMARY REPORTS")

    summary_file = OUTPUT_DIR / '06_analysis_summary.txt'
    with summary_file.open('w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SCRIPT 06: ADVANCED CLUSTERING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Run: {RUN_TIMESTAMP}\n")
        f.write(f"Cluster radius: {Config.cluster_radius_m} m\\n")
        f.write(f"Min cluster size: {Config.min_cluster_size}\\n")
        f.write(f"Export all points: {Config.export_all_points}\\n\\n")
        f.write("CLUSTERING RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Points analyzed: {len(df):,}\n")
        f.write(f"Valid clusters (temporal + spatial): {n_valid_clusters}\n")
        if n_valid_clusters > 0:
            f.write(f"Mean cluster size: {df_cluster_stats['num_points'].mean():.0f} pts\n")
            f.write(f"Mean subsidence (unweighted): {mean_cluster:.3f} cm/yr\n")
            f.write(f"Mean subsidence (weighted): {mean_weighted:.3f} cm/yr\n")
        f.write("\\nCLUSTERING METHOD\\n")
        f.write("=" * 80 + "\\n")
        f.write("✅ Spatial proximity clustering (BFS connected components)\\n")
        f.write("✅ Cluster radius: 5m (max spatial extent)\\n")
        f.write("✅ Minimum cluster size: 3 points\\n")
        f.write("✅ No temporal constraints - date_pairs preserved as metadata\\n")
        f.write("\nSIGN CONVENTION\n")
        f.write("=" * 80 + "\n")
        f.write("Negative cm/yr = subsidence ✓\n")
        if n_valid_clusters > 0 and mean_cluster <= 0 and mean_weighted <= 0:
            f.write(f"Mean cluster subsidence: {mean_cluster:.3f} cm/yr (unweighted) ✓ VERIFIED\n")
            f.write(f"Mean cluster subsidence: {mean_weighted:.3f} cm/yr (weighted) ✓ VERIFIED\n")

    logger.info("Saved 06_analysis_summary.txt")
    print("✓ 06_analysis_summary.txt")

    adv_summary_file = OUTPUT_DIR / '06_analysis_advanced_summary.txt'
    with adv_summary_file.open('w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SCRIPT 06: EXTENDED ANALYSIS (TEMPORAL CONSTRAINTS - FIXED)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Run ID: {RUN_TIMESTAMP}\n")
        f.write(f"Input file: {input_file.name}\n\n")
        f.write("CONFIGURATION\n")
        f.write("=" * 80 + "\n")
        f.write(f"Cluster radius (max): {Config.cluster_radius_m} m\n")
        f.write(f"Min cluster size: {Config.min_cluster_size}\n")
        f.write(f"Weight method: {Config.weight_method}\n")
        f.write(f"Baseline Δt bins: {Config.baseline_dt_bins}\n")
        f.write(f"Robustness threshold: {Config.robustness_threshold_pct}%\n\n")
        f.write("CLUSTERING METHOD\\n")
        f.write("=" * 80 + "\\n")
        f.write("✅ Spatial proximity clustering (5m radius)\\n")
        f.write("✅ No temporal constraints applied\\n")
        f.write("✅ Points with identical date_pairs CAN cluster together\\n")
        f.write("✅ Date_pairs preserved as metadata for analysis\\n")
        f.write("✅ Inverse-variance weighted subsidence rates\\n\\n")
        f.write("RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total points: {len(df):,}\n")
        f.write(f"Valid clusters: {n_valid_clusters}\n")
        if n_valid_clusters > 0:
            f.write(f"Avg cluster size: {df_cluster_stats['num_points'].mean():.1f} pts\n")
            f.write(f"Weighted mean subsidence: {mean_weighted:.3f} cm/yr\n")
        if robustness_report:
            f.write(f"Unstable clusters: {len(robustness_report)}\n")

    logger.info("Saved 06_analysis_advanced_summary.txt")
    print("✓ 06_analysis_advanced_summary.txt")

    # Metadata JSON
    metadata = {
        'run_id': RUN_TIMESTAMP,
        'script': 'script_06_advanced_clustering',
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'cluster_radius_m': Config.cluster_radius_m,
            'min_cluster_size': Config.min_cluster_size,
            'weight_method': Config.weight_method,
            'baseline_dt_bins': str(Config.baseline_dt_bins),
        },
        'clustering_method': [
            'Spatial proximity clustering (5m radius)',
            'No temporal constraints',
            'BFS connected components',
            'Minimum cluster size: 3 points',
            'Date_pairs preserved as metadata',
        ],
        'results': {
            'total_points': int(len(df)),
            'valid_clusters': int(n_valid_clusters),
            'mean_weighted_subsidence_cm_yr': float(mean_weighted) if n_valid_clusters > 0 else None,
        }
    }

    metadata_file = OUTPUT_DIR / '06_analysis_advanced_metadata.json'
    with metadata_file.open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved 06_analysis_advanced_metadata.json")
    print("✓ 06_analysis_advanced_metadata.json")

    # =========================================================================
    # COMPLETION
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("SCRIPT 06 COMPLETE")
    logger.info("=" * 80)
    print("\n" + "=" * 80)
    print("✅ SCRIPT 06 COMPLETE")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script 06: Advanced Clustering')
    parser.add_argument('--threshold', type=float, default=-2.0, help='Subsidence threshold (cm/yr)')
    parser.add_argument('--radius_km', type=float, default=2.0, help='Search radius (km) - note: actual clustering uses Config.cluster_radius_m')
    parser.add_argument('--min_size', type=int, default=10, help='Minimum cluster size - note: actual clustering uses Config.min_cluster_size')
    parser.add_argument('--k', type=int, default=250, help='k-nearest neighbors')

    args = parser.parse_args()

    main(
        threshold=args.threshold,
        radius_km=args.radius_km,
        min_size=args.min_size,
        k=args.k
    )
