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
9. Vector format exports (shapefile + GeoPackage)

CONFIGURATION
=============
✅ Cluster radius: 50m (configurable via Config.cluster_radius_m)
✅ Minimum cluster size: 3 points (configurable via Config.min_cluster_size)
✅ Spatial proximity only - NO temporal constraints
✅ Points with identical date_pairs CAN cluster together
✅ Baseline date and year-range metadata preserved for analysis
✅ Inverse-variance weighting for subsidence rate estimation

Inputs
------
- 05_subsidence_rates.csv (Script 05)             [PRIORITY 1]
- icesat2_repeat_track_matches.csv (Script 03)    [fallback]
- icesat2_insar_integrated.csv                    [fallback]
- 04_spatial_grid.csv                             [fallback]

Outputs
-------
- 06_clusters_identified.csv
- 06_cluster_statistics.csv
- 06_clusters_spatial_analysis.csv
- 06_clusters_points.shp / .gpkg  (WGS84 + UTM)
- 06_clusters_polygons.shp / .gpkg (WGS84 + UTM)
- 06_coastal_clusters.csv
- 06_cluster_map.png
- 06_cluster_stats.png
- 06_analysis_summary.txt
- 06_analysis_advanced_summary.txt
- 06_analysis_advanced_metadata.json
- script_06_run.log

Units: ALL RATES IN CM/YR (negative=subsidence, positive=uplift)
"""

from venv import logger

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter          # FIX #12: single top-level import
from typing import Optional, Tuple, List
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import argparse
import sys
import io
import warnings

warnings.filterwarnings('ignore')

# FIX #1: None-safe UTF-8 wrapper
_stdout_enc = getattr(sys.stdout, 'encoding', None) or ''
if _stdout_enc.lower() != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except AttributeError:
        pass

_stderr_enc = getattr(sys.stderr, 'encoding', None) or ''
if _stderr_enc.lower() != 'utf-8':
    try:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except AttributeError:
        pass

# Optional geopandas
try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("Warning: geopandas not available. Vector exports will be skipped.")

try:
    from config import BASEOUTPUTDIR
except ImportError:
    BASEOUTPUTDIR = Path(r'E:\Processing\ICeSat-2\output_data')

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    """Configuration for extended clustering analysis."""

    cluster_radius_m: float = 50.0
    min_cluster_size: int = 3
    weight_method = 'inverse_variance'

    baseline_dt_bins: List[Tuple] = [
        (0.25, 0.5),
        (0.5,  1.0),
        (1.0,  2.0),
        (2.0,  3.0),
        (3.0,  None),
    ]

    robustness_threshold_pct: float = 20.0
    output_formats = ['csv', 'shp', 'gpkg']

    utm_zone_epsg: int = 3155
    wgs84_epsg: int = 4326
    export_all_points: bool = False

# =============================================================================
# LOGGING
# =============================================================================
def setup_logging(output_dir: Path) -> logging.Logger:
    log_file = output_dir / 'script_06_run.log'
    logger = logging.getLogger('script_06')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    fh = logging.FileHandler(log_file, encoding='utf-8', mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# =============================================================================
# UTILITY: FIND INPUT FILE
# =============================================================================
def find_input_file(output_dir: Path) -> Path:
    priority_files = [
        output_dir / '05_subsidence_rates.csv',
        output_dir / 'icesat2_repeat_track_matches.csv',
        output_dir / 'icesat2_insar_integrated.csv',
        output_dir / '04_spatial_grid.csv',
    ]
    for fp in priority_files:
        if fp.exists():
            return fp
    print("ERROR: Could not find input file.")
    for f in priority_files:
        print(f"  - {f.name}")
    sys.exit(1)

# =============================================================================
# UTILITY: COLUMN FINDERS
# =============================================================================
def find_rate_column(df: pd.DataFrame) -> str:
    candidates = [
        'subsidence_rate_cm_per_year', 'subsidence_rate_cm_yr',
        'rate_cm_yr', 'mean_subsidence_cm_yr',
    ]
    for col in candidates:
        if col in df.columns:
            return col
    print(f"ERROR: No subsidence rate column found. Columns: {list(df.columns)}")
    sys.exit(1)

def find_uncertainty_column(df: pd.DataFrame) -> Optional[str]:
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
    for col in ['elevation', 'elev', 'elevation_m', 'height_m', 'z']:
        if col in df.columns and 'elevation' not in df.columns:
            df['elevation'] = df[col]
            break
    for col in ['rate_uncertainty_cm_yr', 'uncertainty_cm_yr',
                'std_error_cm_yr', 'rate_uncertainty_cm_per_year', 'sigma_rate']:
        if col in df.columns and 'uncertainty' not in df.columns:
            df['uncertainty'] = df[col]
            break
    return df

# =============================================================================
# BASELINE METADATA VALIDATION
# =============================================================================
def validate_baseline_metadata(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Validate and derive baseline date and year-range metadata."""
    logger.info("=" * 80)
    logger.info("BASELINE METADATA VALIDATION")
    logger.info("=" * 80)

    has_early_late = 'date_early' in df.columns and 'date_late' in df.columns
    has_start_end  = 'start_date'  in df.columns and 'end_date'   in df.columns

    missing = []
    if not has_early_late:
        missing.append("date_early/date_late")
    if not has_start_end:
        missing.append("start_date/end_date")
    if 'baseline_date' not in df.columns:
        missing.append("baseline_date (optional)")
    if missing:
        logger.warning(f"Missing baseline fields: {', '.join(missing)}")
        logger.info("Will derive from available temporal columns where possible.")

    # Derive start/end from date_early/date_late if needed
    if has_early_late and not has_start_end:
        logger.info("Deriving start_date/end_date from date_early/date_late")
        df['start_date'] = pd.to_datetime(df['date_early'], errors='coerce')
        df['end_date']   = pd.to_datetime(df['date_late'],  errors='coerce')
        has_start_end = True

    # Compute observation_interval_yr
    if 'observation_interval_yr' not in df.columns and has_start_end:
        s = pd.to_datetime(df['start_date'], errors='coerce')
        e = pd.to_datetime(df['end_date'],   errors='coerce')
        df['observation_interval_yr'] = (e - s).dt.days / 365.25
        logger.info("Computed observation_interval_yr from start_date/end_date")
        dt_vals = df['observation_interval_yr'].dropna()
        if not dt_vals.empty:
            logger.info(f"Observation intervals: {dt_vals.min():.3f}–{dt_vals.max():.3f} yr")

    # Derive year_range_period
    if 'year_range_period' not in df.columns:
        if has_start_end:
            s = pd.to_datetime(df['start_date'], errors='coerce')
            e = pd.to_datetime(df['end_date'],   errors='coerce')
            df['year_range_period'] = (
                s.dt.year.astype('Int64').astype(str) + '_' +
                e.dt.year.astype('Int64').astype(str)
            )
        elif 'datetime' in df.columns:
            s = pd.to_datetime(df['datetime'], errors='coerce')
            df['year_range_period'] = (
                s.dt.year.astype('Int64').astype(str) + '_' +
                s.dt.year.astype('Int64').astype(str)
            )
        else:
            df['year_range_period'] = 'unknown'
            logger.warning("No datetime columns available; year_range_period set to 'unknown'")

    return df

# =============================================================================
# VALIDATE SUBSIDENCE RATES
# =============================================================================
def validate_subsidence_rates(df: pd.DataFrame, rate_col: str,
                               logger: logging.Logger) -> None:
    rates = df[rate_col].dropna().astype(float)
    if len(rates) == 0:
        logger.error("No valid subsidence rates found")
        sys.exit(1)

    logger.info("Subsidence rate distribution:")
    logger.info(f"  Range:  {rates.min():.2f} to {rates.max():.2f} cm/yr")
    logger.info(f"  Mean:   {rates.mean():.2f} cm/yr | Median: {rates.median():.2f} cm/yr")
    logger.info(f"  Negative: {(rates < 0).mean()*100:.1f}% | "
                f"Positive: {(rates > 0).mean()*100:.1f}%")
    logger.info("Rate validation complete (all values accepted)")

# =============================================================================
# OBSERVATION INTERVAL & DATETIME EXTRACTION
# =============================================================================
def extract_observation_interval(
    df: pd.DataFrame, logger: logging.Logger
) -> Tuple[Optional[pd.Series], Optional[str]]:
    candidates = ['observation_interval_yr', 'observation_interval',
                  'delta_t', 'delta_t_yr', 'baseline_yr', 'baseline']
    for col in candidates:
        if col in df.columns:
            logger.info(f"Using observation interval column: '{col}'")
            return df[col].astype(float), col

    if 'start_date' in df.columns and 'end_date' in df.columns:
        try:
            s = pd.to_datetime(df['start_date'])
            e = pd.to_datetime(df['end_date'])
            dt = (e - s).dt.days / 365.25
            logger.info("Computed observation interval from start_date/end_date")
            return dt, 'observation_interval_yr'
        except Exception as e_exc:
            logger.warning(f"Failed to compute Δt from start/end dates: {e_exc}")

    logger.warning("Could not reliably compute observation interval.")
    return None, None

def extract_datetime_columns(
    df: pd.DataFrame,
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    # FIX #4: bare except replaced
    if 'datetime' in df.columns:
        try:
            return pd.to_datetime(df['datetime']), None
        except Exception:
            pass

    if 'start_date' in df.columns and 'end_date' in df.columns:
        try:
            return pd.to_datetime(df['start_date']), pd.to_datetime(df['end_date'])
        except Exception:
            pass

    return None, None

# =============================================================================
# WEIGHTED RATE CALCULATION
# =============================================================================
def calculate_weighted_rate(
    rates: np.ndarray,
    uncertainties: Optional[np.ndarray],
    deltat: Optional[np.ndarray],
    weight_method: str,
    logger: Optional[logging.Logger] = None,
) -> dict:
    unweighted   = float(np.nanmean(rates))
    method_used  = 'uniform'

    if weight_method == 'inverse_variance' and uncertainties is not None:
        safe_unc  = np.maximum(np.abs(uncertainties.astype(float)), 1e-6)
        weights   = 1.0 / (safe_unc ** 2)
        method_used = 'inverse_variance'
    elif deltat is not None:
        safe_dt   = np.maximum(np.abs(deltat.astype(float)), 1e-6)
        weights   = 1.0 / (safe_dt ** 2)
        method_used = 'inverse_deltat'
    else:
        weights = np.ones_like(rates, dtype=float)

    if np.sum(weights) <= 0:
        weights     = np.ones_like(rates, dtype=float)
        method_used = 'uniform'

    weights  = weights / np.sum(weights)
    weighted = float(np.sum(weights * rates))

    return {
        'unweighted_rate':    unweighted,
        'weighted_rate':      weighted,
        'weights':            weights,
        'weight_method_used': method_used,
    }

# =============================================================================
# Δt BIN STRATIFICATION
# =============================================================================
def stratify_by_deltat_bins(
    cluster_idx: np.ndarray,
    df: pd.DataFrame,
    rate_col: str,
    uncertainty_col: Optional[str],
    dt_col: Optional[str],
    dt_bins: list,
    logger: logging.Logger,
) -> dict:
    cluster_df = df.iloc[list(cluster_idx)]   # FIX #13
    rates  = cluster_df[rate_col].values.astype(float)
    deltat = cluster_df[dt_col].values.astype(float) if (dt_col and dt_col in df.columns) else None
    unc    = cluster_df[uncertainty_col].values.astype(float) if (uncertainty_col and uncertainty_col in df.columns) else None

    bin_weighted_rates: dict = {}
    bin_counts: dict         = {}

    for low, high in dt_bins:
        if high is None:
            label = f"rate_{low:.2f}_3plus_yr"
            mask  = (deltat >= low) if deltat is not None else np.zeros(len(rates), dtype=bool)
        else:
            label = f"rate_{low:.2f}_{high:.2f}_yr"
            mask  = ((deltat >= low) & (deltat < high)) if deltat is not None else np.zeros(len(rates), dtype=bool)

        if deltat is None or not np.any(mask):
            bin_weighted_rates[label] = np.nan
            bin_counts[label]         = 0
            continue

        result = calculate_weighted_rate(
            rates[mask],
            unc[mask] if unc is not None else None,
            deltat[mask],
            weight_method=Config.weight_method,
            logger=logger,
        )
        bin_weighted_rates[label] = result['weighted_rate']
        bin_counts[label]         = int(np.sum(mask))

    return {'bin_weighted_rates': bin_weighted_rates, 'bin_counts': bin_counts}

# =============================================================================
# YEAR-RANGE STRATIFICATION
# =============================================================================
def stratify_by_year_range(
    cluster_idx: np.ndarray,
    df: pd.DataFrame,
    rate_col: str,
    uncertainty_col: Optional[str],
    dt_col: Optional[str],
    year_range_col: str,
    logger: logging.Logger,
) -> dict:
    cluster_df = df.iloc[list(cluster_idx)]   # FIX #13
    rates   = cluster_df[rate_col].values.astype(float)
    unc     = cluster_df[uncertainty_col].values.astype(float) if (uncertainty_col and uncertainty_col in df.columns) else None
    deltat  = cluster_df[dt_col].values.astype(float) if (dt_col and dt_col in df.columns) else None
    periods = cluster_df[year_range_col].astype(str).fillna('unknown').values

    period_weighted_rates: dict = {}
    period_counts: dict         = {}

    for p in np.unique(periods):
        mask = periods == p
        if not np.any(mask):
            period_weighted_rates[p] = np.nan
            period_counts[p]         = 0
            continue
        result = calculate_weighted_rate(
            rates[mask],
            unc[mask]    if unc    is not None else None,
            deltat[mask] if deltat is not None else None,
            weight_method=Config.weight_method,
            logger=logger,
        )
        period_weighted_rates[p] = result['weighted_rate']
        period_counts[p]         = int(np.sum(mask))

    return {'period_weighted_rates': period_weighted_rates, 'period_counts': period_counts}

# =============================================================================
# ROBUSTNESS ASSESSMENT
# =============================================================================
def assess_robustness(
    dt_bin_rates: dict,
    period_rates: dict,
    threshold_pct: float,
    logger: Optional[logging.Logger] = None,
) -> dict:
    def _valid(vals):
        return [v for v in vals if v is not None and np.isfinite(float(v))]

    all_vals = _valid(list(dt_bin_rates.values()) + list(period_rates.values()))

    if len(all_vals) < 2:
        return {
            'robustness_flag':    False,
            'rate_variation_pct': np.nan,
            'rate_range':         (np.nan, np.nan),
            'explanation':        'Insufficient temporal bins/periods',
        }

    abs_vals = np.abs(np.array(all_vals, dtype=float))
    min_v = float(np.min(abs_vals))
    max_v = float(np.max(abs_vals))
    var_pct = 100.0 * (max_v - min_v) / min_v if min_v > 0 else 0.0
    flag    = var_pct > threshold_pct
    msg     = f"Variation {var_pct:.1f}% (threshold {threshold_pct}%)"
    msg    += " UNSTABLE" if flag else " OK"

    return {
        'robustness_flag':    flag,
        'rate_variation_pct': var_pct,
        'rate_range':         (min_v, max_v),
        'explanation':        msg,
    }

# =============================================================================
# SPATIAL METRICS
# =============================================================================
def compute_spatial_metrics(
    cluster_idx: np.ndarray,
    df: pd.DataFrame,
    lon_col: str,
    lat_col: str,
    elev_col: Optional[str],
) -> dict:
    cluster_df  = df.iloc[list(cluster_idx)]   # FIX #13
    lons = cluster_df[lon_col].values.astype(float)
    lats = cluster_df[lat_col].values.astype(float)

    centroid_lon = float(np.mean(lons))
    centroid_lat = float(np.mean(lats))
    lon_std_m    = np.std(lons) * 111000.0 * np.cos(np.radians(centroid_lat))
    lat_std_m    = np.std(lats) * 111000.0
    spatial_spread_m = float(np.sqrt(lon_std_m**2 + lat_std_m**2))

    try:
        from scipy.spatial import ConvexHull
        if len(lons) >= 3:
            hull    = ConvexHull(np.column_stack([lons, lats]))
            area_km2 = max(hull.volume * (111.0 ** 2), 0.01)
        else:
            area_km2 = 0.01
    except Exception:
        area_km2 = 0.01

    point_density  = float(len(lons) / area_km2)
    lon_ext_km     = (np.max(lons) - np.min(lons)) * 111.0 * np.cos(np.radians(centroid_lat))
    lat_ext_km     = (np.max(lats) - np.min(lats)) * 111.0
    extent_km      = float(np.sqrt(lon_ext_km**2 + lat_ext_km**2))

    metrics = {
        'center_lon':      centroid_lon,
        'center_lat':      centroid_lat,
        'spatial_spread_m': spatial_spread_m,
        'point_density':   point_density,
        'extent_km':       extent_km,
    }

    if elev_col and elev_col in df.columns:
        e = cluster_df[elev_col].dropna().values.astype(float)
        if len(e) > 0:
            metrics.update({
                'elevation_mean_m': float(np.mean(e)),
                'elevation_std_m':  float(np.std(e)),
                'elevation_min_m':  float(np.min(e)),
                'elevation_max_m':  float(np.max(e)),
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
    elev_col: Optional[str],
    unc_col: Optional[str],
    weighted_rate_col: str,
    dt_col: Optional[str],
    baseline_bin_col: str,
    year_range_col: str,
    logger: logging.Logger,
) -> bool:
    if not GEOPANDAS_AVAILABLE:
        logger.warning("geopandas not available; skipping vector exports.")
        return False

    try:
        # ---- Optional: export ALL points ----
        if Config.export_all_points:
            logger.info("Exporting ALL points (clustered + unclustered)...")
            df_all = df.copy()
            if 'point_id' not in df_all.columns:
                df_all['point_id'] = np.arange(len(df_all))

            gdf_all = gpd.GeoDataFrame(
                df_all,
                geometry=[Point(xy) for xy in zip(df_all[lon_col], df_all[lat_col])],
                crs=f"EPSG:{Config.wgs84_epsg}",
            )
            all_fields = [c for c in [
                'point_id', cluster_id_col, rate_col, unc_col, dt_col,
                baseline_bin_col, year_range_col, elev_col,
                'datetime'   if 'datetime'   in df_all.columns else None,
                'start_date' if 'start_date' in df_all.columns else None,
                'end_date'   if 'end_date'   in df_all.columns else None,
            ] if c and c in gdf_all.columns]
            gdf_all = gdf_all[all_fields + ['geometry']]

            gdf_all.to_file(output_dir / '06_allpoints_wgs84.shp',  driver='ESRI Shapefile')
            gdf_all.to_file(output_dir / '06_allpoints_wgs84.gpkg', driver='GPKG', layer='allpoints_wgs84')
            gdf_all_utm = gdf_all.to_crs(f"EPSG:{Config.utm_zone_epsg}")
            gdf_all_utm.to_file(output_dir / '06_allpoints_utm.shp',  driver='ESRI Shapefile')
            gdf_all_utm.to_file(output_dir / '06_allpoints_utm.gpkg', driver='GPKG', layer='allpoints_utm')

            n_c = (df_all[cluster_id_col] >= 0).sum()
            n_u = (df_all[cluster_id_col] == -1).sum()
            logger.info(f"Exported {len(df_all):,} total points — clustered: {n_c:,}, unclustered: {n_u:,}")
        else:
            logger.info("Skipped all-points export (Config.export_all_points=False)")

        # ---- Clustered points only ----
        df_points = df[df[cluster_id_col] >= 0].copy()
        if 'point_id' not in df_points.columns:
            df_points['point_id'] = np.arange(len(df_points))

        gdf_points = gpd.GeoDataFrame(
            df_points,
            geometry=[Point(xy) for xy in zip(df_points[lon_col], df_points[lat_col])],
            crs=f"EPSG:{Config.wgs84_epsg}",
        )
        point_fields = [c for c in [
            'point_id', cluster_id_col, rate_col, weighted_rate_col, unc_col, dt_col,
            baseline_bin_col, year_range_col, elev_col,
            'datetime'   if 'datetime'   in df_points.columns else None,
            'start_date' if 'start_date' in df_points.columns else None,
            'end_date'   if 'end_date'   in df_points.columns else None,
        ] if c and c in gdf_points.columns]
        gdf_points = gdf_points[point_fields + ['geometry']]

        gdf_points.to_file(output_dir / '06_clusters_points.shp',  driver='ESRI Shapefile')
        gdf_points.to_file(output_dir / '06_clusters_points.gpkg', driver='GPKG', layer='clusters_points')

        gdf_points_utm = gdf_points.to_crs(f"EPSG:{Config.utm_zone_epsg}")
        gdf_points_utm.to_file(output_dir / '06_clusters_points_utm.shp',  driver='ESRI Shapefile')
        gdf_points_utm.to_file(output_dir / '06_clusters_points_utm.gpkg', driver='GPKG', layer='clusters_points_utm')
        logger.info("Saved 06_clusters_points WGS84 + UTM")

        # ---- Cluster boundary polygons ----
        from scipy.spatial import ConvexHull as _ConvexHull

        cluster_polygons = []
        poly_stats       = []

        for cid in sorted(df_points[cluster_id_col].unique()):
            sub = df_points[df_points[cluster_id_col] == cid]
            if len(sub) < 3:
                continue
            coords = np.column_stack([sub[lon_col].values.astype(float),
                                      sub[lat_col].values.astype(float)])
            try:
                hull = _ConvexHull(coords)
                poly = Polygon(coords[hull.vertices])
            except Exception:
                poly = Point(float(sub[lon_col].mean()),
                             float(sub[lat_col].mean())).buffer(0.001)

            cluster_polygons.append(poly)
            s = {
                'cluster_id': cid,
                'num_points': int(len(sub)),
                'weighted_mean_subsidence_cm_yr': float(sub[weighted_rate_col].mean()),
            }
            if 'robustness_flag' in df.columns:
                s['robustness_flag'] = bool(
                    df[df[cluster_id_col] == cid]['robustness_flag'].iloc[0]
                )
            if elev_col and elev_col in sub.columns:
                s['elevation_mean_m'] = float(sub[elev_col].mean())
            poly_stats.append(s)

        if cluster_polygons:
            gdf_poly = gpd.GeoDataFrame(
                pd.DataFrame(poly_stats),
                geometry=cluster_polygons,
                crs=f"EPSG:{Config.wgs84_epsg}",
            )
            gdf_poly.to_file(output_dir / '06_clusters_polygons.shp',  driver='ESRI Shapefile')
            gdf_poly.to_file(output_dir / '06_clusters_polygons.gpkg', driver='GPKG', layer='clusters_polygons')

            gdf_poly_utm = gdf_poly.to_crs(f"EPSG:{Config.utm_zone_epsg}")
            gdf_poly_utm.to_file(output_dir / '06_clusters_polygons_utm.shp',  driver='ESRI Shapefile')
            gdf_poly_utm.to_file(output_dir / '06_clusters_polygons_utm.gpkg', driver='GPKG', layer='clusters_polygons_utm')
            logger.info("Saved 06_clusters_polygons WGS84 + UTM")

        return True

    except Exception as e:
        logger.error(f"Vector export failed: {e}")
        return False

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main(threshold: float = -2.0, radius_km: float = 2.0,
         min_size: int = 10, k: int = 250) -> None:
    """Main clustering workflow (spatial proximity only)."""

    print("=" * 80)
    print("SCRIPT 06: SPATIAL CLUSTERING & SUBSIDENCE ANALYSIS")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Run ID and output directory
    # ------------------------------------------------------------------
    timestamp_file = BASEOUTPUTDIR / 'latest_RUN_TIMESTAMP.txt'
    if not timestamp_file.exists():
        print("ERROR: Run Script 00 first.")
        sys.exit(1)

    RUN_TIMESTAMP = timestamp_file.read_text(encoding='utf-8').strip()
    OUTPUT_DIR    = BASEOUTPUTDIR / f"run_{RUN_TIMESTAMP}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(OUTPUT_DIR)
    logger.info("=" * 80)
    logger.info("SCRIPT 06: SPATIAL CLUSTERING")
    logger.info("=" * 80)
    logger.info(f"Run ID: {RUN_TIMESTAMP}")
    print(f"Run ID: {RUN_TIMESTAMP}")
    print(f"Output: {OUTPUT_DIR}")

    # ------------------------------------------------------------------
    # 1. LOAD DATA
    # ------------------------------------------------------------------
    # FIX #2: corrected escaped newlines throughout main()
    logger.info("\n1. LOADING DATA")
    print("\n" + "=" * 80)
    print("1. LOADING DATA")
    print("=" * 80)

    input_file = find_input_file(OUTPUT_DIR)
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df):,} points from {input_file.name}")
    print(f"Loaded {len(df):,} points")

    df        = standardize_columns(df)
    df        = validate_baseline_metadata(df, logger)
    rate_col  = find_rate_column(df)
    logger.info(f"Using rate column: '{rate_col}'")
    print(f"Rate column: '{rate_col}'")

    validate_subsidence_rates(df, rate_col, logger)

    unc_col = find_uncertainty_column(df)
    if unc_col:
        logger.info(f"Using uncertainty column: '{unc_col}'")
        print(f"Uncertainty column: '{unc_col}'")
    else:
        logger.warning("No uncertainty column found; may fall back to Δt weighting.")
        print("WARNING: No uncertainty column; may fall back to Δt weighting.")

    # Δt extraction
    dt_series, dt_col = extract_observation_interval(df, logger)
    if dt_series is not None:
        if dt_col != 'observation_interval_yr':
            df['observation_interval_yr'] = dt_series
            dt_col = 'observation_interval_yr'
        else:
            df[dt_col] = dt_series

    # Datetime extraction (for metadata only)
    single_dt, end_dt = extract_datetime_columns(df)
    if 'start_date' in df.columns and 'end_date' in df.columns:
        single_dt = pd.to_datetime(df['start_date'], errors='coerce')
        end_dt    = pd.to_datetime(df['end_date'],   errors='coerce')

    # Lat / lon columns
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    if lon_col is None or lat_col is None:
        print("ERROR: Could not find lon/lat columns.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. k-NN GRAPH
    # ------------------------------------------------------------------
    logger.info("\n2. k-NEAREST NEIGHBOR GRAPH")
    print("\n" + "=" * 80)
    print("2. k-NEAREST NEIGHBOR GRAPH")
    print("=" * 80)

    coords       = df[[lon_col, lat_col]].values
    tree         = cKDTree(coords)
    max_radius_m = float(min(Config.cluster_radius_m, 10.0))

    try:
        distances, indices = tree.query(coords, k=min(k + 1, len(df)))
    except Exception as e:
        logger.warning(f"k-nearest failed with k={k}: {e}; retrying with k=150")
        k = min(150, len(df))
        distances, indices = tree.query(coords, k=k + 1)

    neighbors_list = []
    for i in range(len(coords)):
        this_lat   = coords[i, 1]
        nb_idx     = indices[i, 1:]
        delta      = coords[nb_idx] - coords[i]
        dlat_m     = delta[:, 1] * 111000.0
        dlon_m     = delta[:, 0] * 111000.0 * np.cos(np.radians(this_lat))
        dists_m    = np.sqrt(dlat_m**2 + dlon_m**2)
        neighbors_list.append(nb_idx[dists_m <= max_radius_m])

    k_used = float(np.mean([len(n) for n in neighbors_list]))
    logger.info(f"Avg neighbors per point: {k_used:.1f} within {max_radius_m:.1f} m")
    print(f"k-NN graph built (avg {k_used:.1f} neighbors within {max_radius_m:.1f} m)")

    # ------------------------------------------------------------------
    # 3. HOTSPOT DETECTION (no threshold — all points included)
    # ------------------------------------------------------------------
    logger.info("\n3. HOTSPOT DETECTION")
    print("\n" + "=" * 80)
    print("3. HOTSPOT DETECTION")
    print("=" * 80)

    hotspot_mask = np.ones(len(df), dtype=bool)
    logger.info(f"All {len(df):,} points included (no threshold applied)")
    print(f"All {len(df):,} points included")

    # ------------------------------------------------------------------
    # 4. BFS SPATIAL CLUSTERING
    # ------------------------------------------------------------------
    logger.info("\n4. CLUSTER IDENTIFICATION (SPATIAL PROXIMITY ONLY)")
    print("\n" + "=" * 80)
    print("4. CLUSTER IDENTIFICATION (SPATIAL CLUSTERING)")
    print("=" * 80)

    date_early_col = 'date_early' if 'date_early' in df.columns else 'start_date'
    date_late_col  = 'date_late'  if 'date_late'  in df.columns else 'end_date'

    if date_early_col not in df.columns or date_late_col not in df.columns:
        logger.error("Required date columns not found")
        sys.exit(1)

    early_dates = pd.to_datetime(df[date_early_col], errors='coerce')
    late_dates  = pd.to_datetime(df[date_late_col],  errors='coerce')

    n_invalid_early = early_dates.isna().sum()
    n_invalid_late  = late_dates.isna().sum()
    n_invalid_any   = (early_dates.isna() | late_dates.isna()).sum()

    logger.info(f"Date parsing: {n_invalid_early} invalid early, "
                f"{n_invalid_late} invalid late")
    logger.info(f"Points with invalid dates: {n_invalid_any:,} "
                f"({n_invalid_any/len(df)*100:.1f}%)")

    date_pairs   = list(zip(early_dates, late_dates))
    pair_counter = Counter(date_pairs)
    pair_counts  = pd.Series(pair_counter)

    n_dup_pairs  = (pair_counts > 1).sum()
    n_pts_in_dup = pair_counts[pair_counts > 1].sum()
    logger.info(f"Duplicate date_pairs: {n_dup_pairs:,} pairs — "
                f"{n_pts_in_dup:,} points share a pair")
    print(f"Duplicate date_pairs: {n_dup_pairs:,} → {n_pts_in_dup:,} points "
          f"can still cluster (spatial-only)")

    if n_dup_pairs > 0:
        for pair_tuple, count in pair_counts.nlargest(5).items():
            logger.info(f"  {pair_tuple}: {count} points")

    # BFS
    cluster_id      = np.full(len(df), -1, dtype=int)
    current_cluster = 0

    for i in range(len(df)):
        if not hotspot_mask[i] or cluster_id[i] != -1:
            continue

        queue              = [i]
        cluster_id[i]      = current_cluster
        qi                 = 0

        while qi < len(queue):
            p  = queue[qi]
            qi += 1
            for nb in neighbors_list[p]:
                if not hotspot_mask[nb] or cluster_id[nb] != -1:
                    continue
                cluster_id[nb] = current_cluster
                queue.append(nb)

        members = np.where(cluster_id == current_cluster)[0]
        if len(members) < Config.min_cluster_size:
            cluster_id[members] = -1
            logger.debug(f"Discarded cluster {current_cluster} "
                         f"(size {len(members)}) < min {Config.min_cluster_size}")
        else:
            current_cluster += 1

    n_clusters    = current_cluster
    n_total       = len(df)
    n_clustered   = (cluster_id >= 0).sum()
    n_unclustered = n_total - n_clustered
    pct_clustered = (n_clustered / n_total * 100) if n_total > 0 else 0.0

    logger.info("=" * 80)
    logger.info("CLUSTERING COVERAGE SUMMARY")
    logger.info(f"Total input points: {n_total:,}")
    logger.info(f"Clustered:          {n_clustered:,} ({pct_clustered:.1f}%)")
    logger.info(f"Unclustered:        {n_unclustered:,} ({100-pct_clustered:.1f}%)")
    logger.info(f"Valid clusters:     {n_clusters}")

    print("\n" + "=" * 80)
    print("CLUSTERING COVERAGE")
    print(f"Total points:  {n_total:,}")
    print(f"Clustered:     {n_clustered:,} ({pct_clustered:.1f}%)")
    print(f"Unclustered:   {n_unclustered:,} ({100-pct_clustered:.1f}%)")
    print(f"Valid clusters:{n_clusters}")
    print("=" * 80)

    avg_neighbors = np.mean([len(n) for n in neighbors_list])
    if avg_neighbors < 3:
        logger.warning(f"Avg neighbors ({avg_neighbors:.2f}) < 3: "
                       "consider increasing cluster_radius_m")
        print(f"WARNING: Avg neighbors ({avg_neighbors:.2f}) < min cluster size")

    n_identical = (pair_counts > 1).sum()
    if n_identical > 0:
        logger.info(f"{n_identical:,} unique date_pairs — "
                    "date_pairs are metadata; spatial clustering unaffected")

    # ------------------------------------------------------------------
    # 5. CLUSTER ANALYSIS
    # ------------------------------------------------------------------
    logger.info("\n5. CLUSTER ANALYSIS (WEIGHTING & STRATIFICATION)")
    print("\n5. CLUSTER ANALYSIS (WEIGHTING & STRATIFICATION)")

    df['cluster_id'] = cluster_id
    rates_all        = df[rate_col].values.astype(float)

    cluster_stats_records = []
    robustness_report     = []
    year_range_col        = 'year_range_period'

    for cid in range(n_clusters):
        idx = np.where(cluster_id == cid)[0]
        if len(idx) < Config.min_cluster_size:
            continue

        cluster_rates = rates_all[idx]
        cluster_unc   = df[unc_col].values[idx]  if (unc_col and unc_col in df.columns)  else None
        cluster_dt    = df[dt_col].values[idx]   if (dt_col  and dt_col  in df.columns)  else None

        weight_result    = calculate_weighted_rate(
            cluster_rates, cluster_unc, cluster_dt,
            weight_method=Config.weight_method, logger=logger,
        )
        unweighted        = weight_result['unweighted_rate']
        weighted          = weight_result['weighted_rate']
        weight_method_used= weight_result['weight_method_used']

        dt_strat     = stratify_by_deltat_bins(
            idx, df, rate_col, unc_col, dt_col, Config.baseline_dt_bins, logger)
        dt_bin_rates = dt_strat['bin_weighted_rates']

        yr_strat     = stratify_by_year_range(
            idx, df, rate_col, unc_col, dt_col, year_range_col, logger)
        period_rates = yr_strat['period_weighted_rates']

        robust           = assess_robustness(dt_bin_rates, period_rates,
                                             Config.robustness_threshold_pct, logger)
        robustness_flag  = robust['robustness_flag']
        rate_variation_pct = robust['rate_variation_pct']

        if robustness_flag:
            robustness_report.append({
                'cluster_id':  cid,
                'variation_pct': rate_variation_pct,
                'explanation': robust['explanation'],
            })

        spatial = compute_spatial_metrics(
            idx, df, lon_col, lat_col,
            elev_col='elevation' if 'elevation' in df.columns else None,
        )

        # Temporal context
        if date_early_col in df.columns and date_late_col in df.columns:
            cluster_early = pd.to_datetime(df.iloc[list(idx)][date_early_col], errors='coerce')
            cluster_late  = pd.to_datetime(df.iloc[list(idx)][date_late_col],  errors='coerce')
            min_date = cluster_early.min()
            max_date = cluster_late.max()
        else:
            min_date = pd.NaT
            max_date = pd.NaT

        # Distance to coast
        dist_coast_col   = next((c for c in df.columns if 'distance_to_coast' in c.lower()), None)
        distance_to_coast_m = np.nan
        if dist_coast_col:
            d = df.iloc[list(idx)][dist_coast_col].dropna().values
            if len(d) > 0:
                distance_to_coast_m = float(np.mean(d))

        rec = {
            'cluster_id':                    cid,
            'num_points':                    len(idx),
            'center_lon':                    spatial.get('center_lon',       np.nan),
            'center_lat':                    spatial.get('center_lat',       np.nan),
            'spatial_spread_m':              spatial.get('spatial_spread_m', np.nan),
            'point_density':                 spatial.get('point_density',    np.nan),
            'extent_km':                     spatial.get('extent_km',        np.nan),
            'elevation_mean_m':              spatial.get('elevation_mean_m', np.nan),
            'elevation_std_m':               spatial.get('elevation_std_m',  np.nan),
            'elevation_min_m':               spatial.get('elevation_min_m',  np.nan),
            'elevation_max_m':               spatial.get('elevation_max_m',  np.nan),
            'mean_subsidence_cm_yr':         unweighted,
            'weighted_subsidence_cm_yr':     weighted,
            'std_subsidence_cm_yr':          float(np.nanstd(cluster_rates)),
            'min_subsidence_cm_yr':          float(np.nanmin(cluster_rates)),
            'max_subsidence_cm_yr':          float(np.nanmax(cluster_rates)),
            'mean_observation_interval_yr':  float(np.nanmean(cluster_dt)) if cluster_dt is not None else np.nan,
            'weight_method':                 weight_method_used,
            'robustness_flag':               bool(robustness_flag),
            'rate_variation_pct':            float(rate_variation_pct) if np.isfinite(float(rate_variation_pct or 0)) else np.nan,
            'date_range':                    (f"{min_date.date()}__{max_date.date()}"
                                              if pd.notna(min_date) and pd.notna(max_date) else ""),
            'observation_count_per_cluster': len(idx),
            'distance_to_coast_m':           distance_to_coast_m,
        }

        # Bulk subsidence (first → last elevation)
        # FIX #11: guard extended to >= 0.1 yr to avoid meaningless rates
        cluster_df_sub = df.iloc[list(idx)].copy()
        if ('elevation' in cluster_df_sub.columns and
                'start_date' in cluster_df_sub.columns and
                'end_date'   in cluster_df_sub.columns):
            cdf_sorted   = cluster_df_sub.sort_values('start_date')
            first_elev   = cdf_sorted['elevation'].iloc[0]
            last_elev    = cdf_sorted['elevation'].iloc[-1]
            first_date   = pd.to_datetime(cdf_sorted['start_date'].iloc[0])
            last_date    = pd.to_datetime(cdf_sorted['end_date'].iloc[-1])
            total_dt_m   = float((last_elev  - first_elev) * 100)          # m → cm
            total_time_yr = (last_date - first_date).days / 365.25
            bulk_rate     = (total_dt_m / total_time_yr) if total_time_yr >= 0.1 else np.nan
            rec.update({
                'first_elev_m':         float(first_elev),
                'last_elev_m':          float(last_elev),
                'total_delta_elev_cm':  total_dt_m,
                'observation_span_yr':  float(total_time_yr),
                'bulk_subsidence_cm_yr': bulk_rate,
                'first_date':           first_date.strftime('%Y-%m-%d'),
                'last_date':            last_date.strftime('%Y-%m-%d'),
            })

        # Δt bin columns
        _bin_label_map = {
            (0.25, 0.5):  'rate_0.25_0.5yr',
            (0.5,  1.0):  'rate_0.5_1yr',
            (1.0,  2.0):  'rate_1_2yr',
            (2.0,  3.0):  'rate_2_3yr',
            (3.0,  None): 'rate_3plus_yr',
        }
        for low, high in Config.baseline_dt_bins:
            key_in  = f"rate_{low:.2f}_3plus_yr" if high is None else f"rate_{low:.2f}_{high:.2f}_yr"
            key_out = _bin_label_map.get((low, high), key_in)
            rec[key_out] = dt_bin_rates.get(key_in, np.nan)

        # Year-range period columns
        for period_label, val in period_rates.items():
            if period_label == 'unknown':
                continue
            rec[f"rate_{period_label}"] = val

        cluster_stats_records.append(rec)

    df_cluster_stats = pd.DataFrame(cluster_stats_records)
    n_valid_clusters = len(df_cluster_stats)

    logger.info(f"Valid clusters (>= {Config.min_cluster_size} pts): {n_valid_clusters}")
    print(f"Valid clusters: {n_valid_clusters}")

    if robustness_report:
        logger.warning(f"{len(robustness_report)} clusters flagged as potentially unstable.")
        print(f"WARNING: {len(robustness_report)} clusters potentially unstable.")

    if 'bulk_subsidence_cm_yr' in df_cluster_stats.columns:
        bulk_rates = df_cluster_stats['bulk_subsidence_cm_yr'].dropna()
        if len(bulk_rates) > 0:
            logger.info(f"Bulk subsidence (first→last): "
                        f"{bulk_rates.mean():.2f}±{bulk_rates.std():.2f} cm/yr")
            print(f"Bulk subsidence: {bulk_rates.mean():.2f} cm/yr "
                  f"(avg span {df_cluster_stats['observation_span_yr'].mean():.1f} yr)")

    # ------------------------------------------------------------------
    # 6. SIGN CONVENTION VERIFICATION
    # ------------------------------------------------------------------
    logger.info("\n6. SIGN CONVENTION VERIFICATION")
    print("\n6. SIGN CONVENTION VERIFICATION")

    mean_cluster  = df_cluster_stats['mean_subsidence_cm_yr'].mean()     if n_valid_clusters > 0 else np.nan
    mean_weighted = df_cluster_stats['weighted_subsidence_cm_yr'].mean() if n_valid_clusters > 0 else np.nan

    if n_valid_clusters > 0 and (mean_cluster > 0 or mean_weighted > 0):
        logger.warning("Positive cluster mean detected — check sign convention.")
        print(f"WARNING: Positive mean (unexpected) — "
              f"unweighted {mean_cluster:.3f}, weighted {mean_weighted:.3f} cm/yr")
    elif n_valid_clusters > 0:
        logger.info(f"Sign OK. Unweighted: {mean_cluster:.3f} | "
                    f"Weighted: {mean_weighted:.3f} cm/yr")
        print(f"Sign verified — unweighted: {mean_cluster:.3f}, "
              f"weighted: {mean_weighted:.3f} cm/yr")

    # ------------------------------------------------------------------
    # 7. SAVE CSV OUTPUTS
    # ------------------------------------------------------------------
    logger.info("\n7. SAVING CSV OUTPUTS")
    print("\n7. SAVING CSV OUTPUTS")

    legacy_cols = ['cluster_id', 'num_points', 'mean_subsidence_cm_yr',
                   'std_subsidence_cm_yr', 'center_lat', 'center_lon', 'extent_km']
    df_legacy = df_cluster_stats.copy()
    for col in legacy_cols:
        if col not in df_legacy.columns:
            df_legacy[col] = np.nan
    df_legacy[legacy_cols].to_csv(OUTPUT_DIR / '06_cluster_statistics.csv', index=False)
    logger.info("Saved 06_cluster_statistics.csv")

    out_cols = [lat_col, lon_col, rate_col, 'cluster_id']
    if 'elevation'   in df.columns: out_cols.append('elevation')
    if 'uncertainty' in df.columns: out_cols.append('uncertainty')
    df[out_cols].to_csv(OUTPUT_DIR / '06_clusters_identified.csv', index=False)
    logger.info("Saved 06_clusters_identified.csv")

    df_cluster_stats.to_csv(OUTPUT_DIR / '06_clusters_spatial_analysis.csv', index=False)
    logger.info("Saved 06_clusters_spatial_analysis.csv")
    print("CSV outputs saved.")

    # ------------------------------------------------------------------
    # 8. VECTOR EXPORTS
    # ------------------------------------------------------------------
    logger.info("\n8. VECTOR EXPORTS")
    print("\n8. VECTOR EXPORTS")

    if n_valid_clusters > 0:
        weighted_map = df_cluster_stats.set_index('cluster_id')['weighted_subsidence_cm_yr']
        df['weighted_subsidence_cm_yr'] = df['cluster_id'].map(weighted_map)

        if dt_col and dt_col in df.columns:
            dt_vals   = df[dt_col].astype(float)
            bin_edges = [0.25, 0.5, 1.0, 2.0, 3.0, np.inf]
            bin_labels= ['0.25-0.5yr', '0.5-1yr', '1-2yr', '2-3yr', '3+yr']
            df['baseline_bin'] = pd.cut(
                dt_vals, bins=bin_edges, labels=bin_labels, right=False
            ).astype(str)
            df.loc[dt_vals.isna() | (dt_vals < 0.25), 'baseline_bin'] = 'unknown'
        else:
            df['baseline_bin'] = 'unknown'

        vector_success = export_vector_formats(
            df,
            cluster_id_col    = 'cluster_id',
            rate_col          = rate_col,
            lon_col           = lon_col,
            lat_col           = lat_col,
            output_dir        = OUTPUT_DIR,
            elev_col          = 'elevation'  if 'elevation'  in df.columns else None,
            unc_col           = 'uncertainty' if 'uncertainty' in df.columns else None,
            weighted_rate_col = 'weighted_subsidence_cm_yr',
            dt_col            = dt_col,
            baseline_bin_col  = 'baseline_bin',
            year_range_col    = year_range_col,
            logger            = logger,
        )
        if vector_success:
            print("Vector exports complete (shapefile + GeoPackage, WGS84 + UTM)")
        else:
            print("WARNING: Vector exports failed or skipped; CSV outputs available.")
    else:
        logger.warning("No valid clusters found — skipping vector exports.")
        print("WARNING: No valid clusters found — skipping vector exports.")

    # ------------------------------------------------------------------
    # 9. SUMMARY REPORTS
    # ------------------------------------------------------------------
    logger.info("\n9. GENERATING SUMMARY REPORTS")
    print("\n9. GENERATING SUMMARY REPORTS")

    # FIX #3: all \\n / \\\\n corrected to \n
    summary_file = OUTPUT_DIR / '06_analysis_summary.txt'
    with summary_file.open('w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SCRIPT 06: ADVANCED CLUSTERING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Run: {RUN_TIMESTAMP}\n")
        f.write(f"Cluster radius: {Config.cluster_radius_m} m\n")
        f.write(f"Min cluster size: {Config.min_cluster_size}\n")
        f.write(f"Export all points: {Config.export_all_points}\n\n")
        f.write("CLUSTERING RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Points analyzed: {len(df):,}\n")
        f.write(f"Valid clusters: {n_valid_clusters}\n")
        if n_valid_clusters > 0:
            f.write(f"Mean cluster size: {df_cluster_stats['num_points'].mean():.0f} pts\n")
            f.write(f"Mean subsidence (unweighted): {mean_cluster:.3f} cm/yr\n")
            f.write(f"Mean subsidence (weighted):   {mean_weighted:.3f} cm/yr\n")
        f.write("\nCLUSTERING METHOD\n")
        f.write("=" * 80 + "\n")
        f.write("Spatial proximity clustering (BFS connected components)\n")
        f.write(f"Cluster radius: {Config.cluster_radius_m} m\n")
        f.write(f"Minimum cluster size: {Config.min_cluster_size} points\n")
        f.write("No temporal constraints — date_pairs preserved as metadata\n")
        f.write("Inverse-variance weighted subsidence rates\n")
        f.write("\nSIGN CONVENTION\n")
        f.write("=" * 80 + "\n")
        f.write("Negative cm/yr = subsidence\n")
        if n_valid_clusters > 0 and mean_cluster <= 0 and mean_weighted <= 0:
            f.write(f"Mean (unweighted): {mean_cluster:.3f} cm/yr  VERIFIED\n")
            f.write(f"Mean (weighted):   {mean_weighted:.3f} cm/yr  VERIFIED\n")
    logger.info("Saved 06_analysis_summary.txt")

    adv_summary_file = OUTPUT_DIR / '06_analysis_advanced_summary.txt'
    with adv_summary_file.open('w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SCRIPT 06: EXTENDED ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp:  {datetime.now().isoformat()}\n")
        f.write(f"Run ID:     {RUN_TIMESTAMP}\n")
        f.write(f"Input file: {input_file.name}\n\n")
        f.write("CONFIGURATION\n")
        f.write("=" * 80 + "\n")
        f.write(f"Cluster radius (max): {Config.cluster_radius_m} m\n")
        f.write(f"Min cluster size:     {Config.min_cluster_size}\n")
        f.write(f"Weight method:        {Config.weight_method}\n")
        f.write(f"Baseline Δt bins:     {Config.baseline_dt_bins}\n")
        f.write(f"Robustness threshold: {Config.robustness_threshold_pct}%\n\n")
        f.write("CLUSTERING METHOD\n")
        f.write("=" * 80 + "\n")
        f.write("Spatial proximity clustering (5 m radius)\n")
        f.write("No temporal constraints applied\n")
        f.write("Points with identical date_pairs CAN cluster together\n")
        f.write("Date_pairs preserved as metadata for analysis\n")
        f.write("Inverse-variance weighted subsidence rates\n\n")
        f.write("RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total points:   {len(df):,}\n")
        f.write(f"Valid clusters: {n_valid_clusters}\n")
        if n_valid_clusters > 0:
            f.write(f"Avg cluster size:         {df_cluster_stats['num_points'].mean():.1f} pts\n")
            f.write(f"Weighted mean subsidence: {mean_weighted:.3f} cm/yr\n")
        if robustness_report:
            f.write(f"Unstable clusters:        {len(robustness_report)}\n")
    logger.info("Saved 06_analysis_advanced_summary.txt")
    print("Summary reports saved.")

    # Metadata JSON — FIX #10: sign_convention added
    run_metadata = {
        'run_id':    RUN_TIMESTAMP,
        'script':    'script_06_advanced_clustering',
        'timestamp': datetime.now().isoformat(),
        'sign_convention': 'negative=subsidence, positive=uplift',
        'configuration': {
            'cluster_radius_m':    Config.cluster_radius_m,
            'min_cluster_size':    Config.min_cluster_size,
            'weight_method':       Config.weight_method,
            'baseline_dt_bins':    str(Config.baseline_dt_bins),
        },
        'clustering_method': [
            'Spatial proximity clustering (5 m radius)',
            'No temporal constraints',
            'BFS connected components',
            f'Minimum cluster size: {Config.min_cluster_size} points',
            'Date_pairs preserved as metadata',
        ],
        'results': {
            'total_points':                  int(len(df)),
            'valid_clusters':                int(n_valid_clusters),
            'mean_weighted_subsidence_cm_yr': float(mean_weighted) if n_valid_clusters > 0 else None,
        },
    }

    metadata_file = OUTPUT_DIR / '06_analysis_advanced_metadata.json'
    with metadata_file.open('w', encoding='utf-8') as f:
        json.dump(run_metadata, f, indent=2)
    logger.info("Saved 06_analysis_advanced_metadata.json")
    print("Metadata JSON saved.")

    # ------------------------------------------------------------------
    # COMPLETION
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 80)
    logger.info("SCRIPT 06 COMPLETE")
    logger.info("=" * 80)
    print("\n" + "=" * 80)
    print("SCRIPT 06 COMPLETE")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script 06: Advanced Clustering')
    parser.add_argument('--threshold', type=float, default=-2.0,
                        help='Subsidence threshold cm/yr (passed to main, not used as filter)')
    parser.add_argument('--radius_km', type=float, default=2.0,
                        help='Search radius km (note: spatial clustering uses Config.cluster_radius_m)')
    parser.add_argument('--min_size', type=int, default=10,
                        help='Min cluster size (note: spatial clustering uses Config.min_cluster_size)')
    parser.add_argument('--k', type=int, default=250,
                        help='k-nearest neighbours for graph construction')
    args = parser.parse_args()
    main(
        threshold=args.threshold,
        radius_km=args.radius_km,
        min_size=args.min_size,
        k=args.k,
    )