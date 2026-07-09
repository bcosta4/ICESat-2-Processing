
"""
SCRIPT 01: ICESat-2 PHOTON/SEGMENT FILTERING - MULTI-PRODUCT (ATL03/ATL06/ATL08)

Purpose
-------
Load raw ICESat-2 data (ATL03 photons, ATL06 land-ice segments, or ATL08
land/canopy/terrain segments) and apply quality filters to extract
high-confidence ground returns suitable for subsidence analysis over the
Tuktoyaktuk Peninsula.

--product {ATL03,ATL06,ATL08} selects data source and loader.
    - ATL03/ATL08: PhoREAL-exported CSV/LAS files (unchanged)
    - ATL06: CSV export with columns Latitude (deg), Longitude (deg),
      Height (m MSL), height_sigma_m, quality_summary, date, rgt, cycle,
      beam, beam_strength, filename, dh_fit_dx, n_fit_photons, year

IMPORTANT CAVEAT: ATL06 "Height (m MSL)" for ATL06 is WGS84 ellipsoidal
height (h_li) as exported upstream; despite the column name, no geoid
correction has been verified here. Treat absolute values with caution;
relative multi-year change over the same footprint is still valid for
subsidence rate calculations, since any constant geoid offset cancels
out in dH.

STUDY AREA
----------
The study-area extent is defined by the polygon boundary in:
    E:\\Processing\\ICESat-2\\raw_data\\processing_area.gpkg
All points are filtered using a true point-in-polygon spatial test against
this boundary (not a rectangular lat/lon bounding box), so the filtered
footprint matches the intended study area exactly, including any irregular
coastline or inland boundary shape.
"""

import pandas as pd
import numpy as np
import laspy
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import glob
from datetime import datetime, timedelta
import warnings
import sys
import io
import argparse

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

from config import INPUTDIR, BASEOUTPUTDIR, VALIDPRODUCTS, DEFAULTPRODUCT, PRODUCTFILEPATTERNS

# =============================================================================
# ARGUMENTS
# =============================================================================
parser = argparse.ArgumentParser(description='ICESat-2 Multi-Product Filtering')
parser.add_argument('--product', type=str, choices=VALIDPRODUCTS, default=DEFAULTPRODUCT,
                     help=f'ICESat-2 product to process. Choices: {VALIDPRODUCTS}')
args = parser.parse_args()
PRODUCT = args.product

# =============================================================================
# CONFIGURATION
# =============================================================================
STRONG_BEAMS = ['gt1l', 'gt2l', 'gt3l']
WEAK_BEAMS   = ['gt1r', 'gt2r', 'gt3r']
CONFIDENCE_THRESHOLD = 3  # 3=high, 4=highest (ATL03/ATL08 signal_conf)

# Study-area boundary: polygon-based (replaces legacy rectangular bbox)
STUDY_AREA_GPKG = Path(r"E:\Processing\ICESat-2\raw_data\processing_area.gpkg")

HEIGHT_COL = 'Height (m MSL)'
BEAM_STRENGTH = {
    'gt1l': 'strong', 'gt1r': 'weak',
    'gt2l': 'strong', 'gt2r': 'weak',
    'gt3l': 'strong', 'gt3r': 'weak',
}

ATLAS_SDP_EPOCH = datetime(2018, 1, 1, 0, 0, 0)

# =============================================================================
# STUDY AREA POLYGON LOADING (cached at module load)
# =============================================================================
_STUDY_AREA_UNION = None
_STUDY_AREA_DESCRIPTION = "UNAVAILABLE"

def load_study_area_polygon():
    """Load and cache the study-area polygon from processing_area.gpkg.
    Returns (unary_union geometry in EPSG:4326, human-readable description str)."""
    global _STUDY_AREA_UNION, _STUDY_AREA_DESCRIPTION
    if _STUDY_AREA_UNION is not None:
        return _STUDY_AREA_UNION, _STUDY_AREA_DESCRIPTION

    if not STUDY_AREA_GPKG.exists():
        print(f"ERROR: Study area GeoPackage not found: {STUDY_AREA_GPKG}", flush=True)
        print("Cannot proceed without a defined study area boundary.", flush=True)
        sys.exit(1)

    try:
        study_gdf = gpd.read_file(STUDY_AREA_GPKG)
        if study_gdf.crs is None:
            print("WARNING: processing_area.gpkg has no CRS defined; assuming EPSG:4326", flush=True)
            study_gdf = study_gdf.set_crs("EPSG:4326")
        else:
            study_gdf = study_gdf.to_crs("EPSG:4326")
    except Exception as e:
        print(f"ERROR: Failed to load study area GeoPackage: {e}", flush=True)
        sys.exit(1)

    _STUDY_AREA_UNION = study_gdf.geometry.unary_union
    minx, miny, maxx, maxy = study_gdf.total_bounds
    n_features = len(study_gdf)
    area_km2 = study_gdf.to_crs("EPSG:3155").geometry.area.sum() / 1e6

    _STUDY_AREA_DESCRIPTION = (
        f"{STUDY_AREA_GPKG.name} ({n_features} feature(s)), "
        f"extent Lat [{miny:.4f}, {maxy:.4f}] Lon [{minx:.4f}, {maxx:.4f}], "
        f"area ~{area_km2:,.1f} km^2"
    )

    print(f" - Study-area source: {STUDY_AREA_GPKG.name}", flush=True)
    print(f" - Study-area features: {n_features}", flush=True)
    print(f" - Study-area bounds: Lat [{miny:.4f}, {maxy:.4f}] Lon [{minx:.4f}, {maxx:.4f}]", flush=True)
    print(f" - Study-area approx. area: {area_km2:,.1f} km^2", flush=True)

    return _STUDY_AREA_UNION, _STUDY_AREA_DESCRIPTION


print("=" * 80, flush=True)
print("SCRIPT 01: ICESat-2 DATA FILTERING - MULTI-PRODUCT", flush=True)
print("=" * 80, flush=True)
print(f"Product selected: {PRODUCT}", flush=True)
print(f"Input directory: {INPUTDIR}", flush=True)
print(f"Output directory: {BASEOUTPUTDIR}", flush=True)
if PRODUCT in ('ATL03', 'ATL08'):
    print(f" - Strong beams only: {STRONG_BEAMS}", flush=True)
    print(f" - Confidence threshold: {CONFIDENCE_THRESHOLD}", flush=True)
if PRODUCT == 'ATL06':
    print(" - Elevation field: 'Height (m MSL)' (as exported, ATL06 CSV)", flush=True)
    print(" - Quality screen: quality_summary == 0 (good fit)", flush=True)

_study_area_union, _study_area_desc = load_study_area_polygon()
print(f" - Study-area filter: polygon-based ({STUDY_AREA_GPKG.name})", flush=True)
print("=" * 80, flush=True)

timestamp_file = Path(BASEOUTPUTDIR) / 'latest_RUN_TIMESTAMP.txt'
if not timestamp_file.exists():
    print(f" Timestamp file not found: {timestamp_file}")
    print("Script 00 must run first!")
    sys.exit(1)
with open(timestamp_file, 'r', encoding='utf-8') as f:
    RUN_TIMESTAMP = f.read().strip()

OUTPUT_DIR = Path(BASEOUTPUTDIR) / f"run_{RUN_TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Run ID: {RUN_TIMESTAMP}")
print(f"Output folder: {OUTPUT_DIR.name}")
print("=" * 80)


# =============================================================================
# METADATA ATTACHMENT
# =============================================================================
def attach_file_metadata(df: pd.DataFrame, filepath: str) -> pd.DataFrame:
    filename = Path(filepath).stem
    parts = filename.split('_')
    df = df.copy()
    df['filename'] = filename
    try:
        df['date'] = pd.to_datetime(parts[2][:8], format='%Y%m%d')
    except (IndexError, ValueError):
        print(f" WARNING: Cannot parse date from filename {filename}, setting NaT", flush=True)
        df['date'] = pd.NaT
    df['year'] = df['date'].dt.year
    df['rgt'] = parts[3][:4] if len(parts) > 3 else 'UNKNOWN'
    df['cycle'] = parts[3][4:6] if len(parts) > 3 else 'UNKNOWN'
    df['beam'] = parts[-1]
    df['beam_strength'] = df['beam'].map(BEAM_STRENGTH)
    return df


# =============================================================================
# PhoREAL LOADERS (ATL03 / ATL08)
# =============================================================================
def load_phoreal_csv(csvpath: str):
    try:
        df = pd.read_csv(csvpath, encoding='utf-8')
        if HEIGHT_COL not in df.columns:
            for candidate in ['Terrain Best Fit (m)', 'Terrain Median (m)', 'h_te_best_fit', 'h_te_median', 'h_li']:
                if candidate in df.columns:
                    df = df.rename(columns={candidate: HEIGHT_COL})
                    break
        df = attach_file_metadata(df, csvpath)
        return df
    except Exception as e:
        print(f"ERROR loading CSV {csvpath}: {e}", flush=True)
        return None


def load_phoreal_las(laspath: str):
    try:
        las = laspy.read(laspath)
        time_attr = None
        for attr in ['gps_time', 'gpstime', 'Time', 'times']:
            if hasattr(las, attr):
                time_attr = getattr(las, attr)
                break
        if time_attr is None:
            time_attr = np.arange(len(las.x))

        utm_e = np.array(las.X, dtype=float) if hasattr(las, 'X') else np.full(len(las.x), np.nan)
        utm_n = np.array(las.Y, dtype=float) if hasattr(las, 'Y') else np.full(len(las.x), np.nan)
        classification = np.array(las.classification, dtype=int) if hasattr(las, 'classification') else np.ones(len(las.x), dtype=int)
        signal_conf = np.array(las.intensity, dtype=int) if hasattr(las, 'intensity') else np.full(len(las.x), 4, dtype=int)

        df = pd.DataFrame({
            'Time (sec)': time_attr,
            'Latitude (deg)': las.y,
            'Longitude (deg)': las.x,
            'UTM Easting (m)': utm_e,
            'UTM Northing (m)': utm_n,
            HEIGHT_COL: las.z,
            'Classification': classification,
            'Signal Confidence': signal_conf,
        })
        df = attach_file_metadata(df, laspath)
        return df
    except Exception as e:
        print(f"ERROR loading LAS {laspath}: {e}", flush=True)
        return None


def load_best_format(csvpath: str):
    p = Path(csvpath)
    laspath = p.with_suffix('.las')
    df_csv = load_phoreal_csv(str(p)) if p.exists() else None
    df_las = load_phoreal_las(str(laspath)) if laspath.exists() else None

    if df_csv is not None and df_las is not None:
        if len(df_csv) >= len(df_las):
            print(f" - Using CSV ({len(df_csv)} rows)", flush=True)
            return df_csv
        else:
            print(f" - Using LAS ({len(df_las)} rows)", flush=True)
            return df_las
    elif df_csv is not None:
        print(f" - Using CSV ({len(df_csv)} rows)", flush=True)
        return df_csv
    elif df_las is not None:
        print(f" - Using LAS ({len(df_las)} rows)", flush=True)
        return df_las
    else:
        return None


# =============================================================================
# ATL06 LOADERS
# =============================================================================
def load_atl06_h5(h5path: str):
    """Read a native NASA ATL06 HDF5 granule (retained for future use;
    current delivery format on this system is CSV)."""
    try:
        import h5py
    except ImportError:
        print("ERROR: h5py is required for ATL06 HDF5 support. Install with: pip install h5py", flush=True)
        return None
    try:
        import pyproj
    except ImportError:
        print("ERROR: pyproj is required for ATL06 UTM conversion. Install with: pip install pyproj", flush=True)
        return None

    p = Path(h5path)
    filename = p.stem
    frames = []
    try:
        with h5py.File(h5path, 'r') as f:
            parts = filename.split('_')
            try:
                date_str = parts[1][:8]
                file_date = pd.to_datetime(date_str, format='%Y%m%d')
            except (IndexError, ValueError):
                print(" WARNING: Cannot parse date from filename", flush=True)
                file_date = pd.NaT
            rgt = parts[2][:4] if len(parts) > 2 else 'UNKNOWN'
            cycle = parts[2][4:6] if len(parts) > 2 else 'UNKNOWN'

            for beam in STRONG_BEAMS:
                grp_path = f'/{beam}/land_ice_segments'
                if grp_path not in f:
                    continue
                grp = f[grp_path]
                lat = np.array(grp['latitude'])
                lon = np.array(grp['longitude'])
                hli = np.array(grp['h_li'])
                delta_time = np.array(grp['delta_time'])
                quality = np.array(grp['atl06_quality_summary']) if 'atl06_quality_summary' in grp else np.zeros(len(lat), dtype=int)

                valid_mask = np.isfinite(hli) & (np.abs(hli) < 1e6)
                if not valid_mask.any():
                    continue

                lat = lat[valid_mask]; lon = lon[valid_mask]; hli = hli[valid_mask]
                delta_time = delta_time[valid_mask]; quality = quality[valid_mask]

                dt_series = pd.to_datetime(ATLAS_SDP_EPOCH) + pd.to_timedelta(delta_time, unit='s')
                signal_conf = np.where(quality == 0, 4, 0)

                df_beam = pd.DataFrame({
                    'Latitude (deg)': lat,
                    'Longitude (deg)': lon,
                    HEIGHT_COL: hli,
                    'Signal Confidence': signal_conf,
                    'quality_summary': quality,
                    'datetime': dt_series,
                })
                df_beam['date'] = df_beam['datetime'].dt.floor('D')
                df_beam['year'] = df_beam['datetime'].dt.year
                df_beam['beam'] = beam
                df_beam['beam_strength'] = 'strong'
                df_beam['rgt'] = rgt
                df_beam['cycle'] = cycle
                df_beam['filename'] = filename
                frames.append(df_beam)
    except Exception as e:
        print(f"ERROR loading ATL06 HDF5 {h5path}: {e}", flush=True)
        return None

    if not frames:
        return None

    df_all = pd.concat(frames, ignore_index=True)
    mean_lon = df_all['Longitude (deg)'].mean()
    mean_lat = df_all['Latitude (deg)'].mean()
    utm_zone = int((mean_lon + 180) / 6) + 1
    hemisphere = 'N' if mean_lat >= 0 else 'S'
    epsg = 32600 + utm_zone if hemisphere == 'N' else 32700 + utm_zone
    transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    easting, northing = transformer.transform(df_all['Longitude (deg)'].values, df_all['Latitude (deg)'].values)
    df_all['UTM Easting (m)'] = easting
    df_all['UTM Northing (m)'] = northing
    df_all['UTM Zone'] = utm_zone
    df_all['UTM Hemisphere'] = hemisphere
    return df_all


REQUIRED_ATL06_CSV_COLS = ['Latitude (deg)', 'Longitude (deg)', HEIGHT_COL, 'quality_summary']

def load_atl06_csv(csvpath: str):
    """Read an ATL06 CSV export using the confirmed schema:
    Latitude (deg), Longitude (deg), Height (m MSL), height_sigma_m,
    quality_summary, date, rgt, cycle, beam, beam_strength, filename,
    dh_fit_dx, n_fit_photons, year.
    """
    try:
        df = pd.read_csv(csvpath, encoding='utf-8')
    except Exception as e:
        print(f"ERROR loading ATL06 CSV {csvpath}: {e}", flush=True)
        return None

    missing = [c for c in REQUIRED_ATL06_CSV_COLS if c not in df.columns]
    if missing:
        print(f" ERROR: Required columns {missing} not found in {Path(csvpath).name}", flush=True)
        print(f" Available columns: {list(df.columns)}", flush=True)
        return None

    try:
        import pyproj
    except ImportError:
        print("ERROR: pyproj is required for ATL06 UTM conversion. Install with: pip install pyproj", flush=True)
        return None

    df = df.copy()

    # Fill-value / invalid-height screen (ATL06 uses ~3.4e38 for invalid h_li)
    valid_mask = np.isfinite(df[HEIGHT_COL]) & (df[HEIGHT_COL].abs() < 1e6)
    n_before = len(df)
    df = df[valid_mask].copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f" Dropped {n_dropped:,} rows with invalid/fill-value height", flush=True)
    if len(df) == 0:
        return df

    # Derive Signal Confidence from quality_summary (0=good fit -> 4, else -> 0)
    df['Signal Confidence'] = np.where(df['quality_summary'] == 0, 4, 0)

    # Parse date column if not already datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if df['date'].isna().all():
            print(" WARNING: date column present but failed to parse for all rows", flush=True)
    else:
        print(" WARNING: date column not found in ATL06 CSV", flush=True)
        df['date'] = pd.NaT

    if 'year' not in df.columns or df['year'].isna().all():
        df['year'] = df['date'].dt.year

    for col, default in [('rgt', 'UNKNOWN'), ('cycle', 'UNKNOWN'), ('beam', 'unknown'), ('filename', Path(csvpath).stem)]:
        if col not in df.columns:
            df[col] = default

    if 'beam_strength' not in df.columns or df['beam_strength'].isna().all():
        df['beam_strength'] = df['beam'].map(BEAM_STRENGTH).fillna('strong')

    # UTM projection (ATL06 CSV has no native UTM columns)
    mean_lon = df['Longitude (deg)'].mean()
    mean_lat = df['Latitude (deg)'].mean()
    utm_zone = int((mean_lon + 180) / 6) + 1
    hemisphere = 'N' if mean_lat >= 0 else 'S'
    epsg = 32600 + utm_zone if hemisphere == 'N' else 32700 + utm_zone
    transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    easting, northing = transformer.transform(df['Longitude (deg)'].values, df['Latitude (deg)'].values)
    df['UTM Easting (m)'] = easting
    df['UTM Northing (m)'] = northing
    df['UTM Zone'] = utm_zone
    df['UTM Hemisphere'] = hemisphere
    return df


# =============================================================================
# STUDY-AREA POLYGON FILTER  (replaces legacy rectangular lat/lon bbox)
# =============================================================================
def apply_bbox_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter points to those within the true study-area polygon defined in
    processing_area.gpkg. Function name retained as 'apply_bbox_filter' for
    backward compatibility with callers, but performs a point-in-polygon
    spatial test rather than a rectangular bounding-box comparison."""
    lat_col, lon_col = 'Latitude (deg)', 'Longitude (deg)'
    if lat_col not in df.columns or lon_col not in df.columns:
        print(" WARNING: Lat/Lon columns not found - study-area filter skipped", flush=True)
        return df

    study_union, _ = load_study_area_polygon()

    n_before = len(df)
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=[Point(xy) for xy in zip(df[lon_col], df[lat_col])],
        crs="EPSG:4326",
    )
    mask = gdf.geometry.within(study_union)
    df_out = df[mask.values].copy()

    n_removed = n_before - len(df_out)
    if n_removed > 0:
        print(f" Study-area polygon filter removed {n_removed:,} rows outside boundary "
              f"- {len(df_out):,} remaining", flush=True)
    else:
        print(f" Study-area polygon filter: all {len(df_out):,} rows within boundary", flush=True)
    return df_out


# =============================================================================
# GROUND-PHOTON / SEGMENT FILTERING
# =============================================================================
def detect_height_col(df: pd.DataFrame):
    for candidate in [HEIGHT_COL, 'Height (m HAE)', 'height', 'Height', 'elevation', 'h_li']:
        if candidate in df.columns:
            return candidate
    return None


def filter_ground_photons(df: pd.DataFrame, product: str, confidence_threshold: int = 3) -> pd.DataFrame:
    """Filter for high-quality ground returns. Behaviour depends on product:
    ATL03/ATL08:
        1. Study-area polygon (processing_area.gpkg)
        2. Signal confidence >= confidence_threshold
        3a. ATL08 classification if present: keep classes 1 (land), 4 (ground)
        3b. Fallback: lowest-20th-percentile height within 100-m segments
    ATL06:
        1. Study-area polygon (processing_area.gpkg)
        2. quality_summary == 0 screen (equivalent quality gate already
           encoded into Signal Confidence by load_atl06_csv/load_atl06_h5,
           so the standard confidence filter is reused here for consistency)
    """
    if len(df) == 0:
        return df

    df_filtered = df.copy()
    initial_count = len(df_filtered)

    df_filtered = apply_bbox_filter(df_filtered)

    if 'Signal Confidence' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Signal Confidence'] >= confidence_threshold]
        print(f" After confidence/quality filter: {len(df_filtered):,} rows", flush=True)
    else:
        print(" WARNING: Signal Confidence column not found - confidence filter skipped", flush=True)

    if len(df_filtered) == 0:
        print(" WARNING: 0 rows remain after confidence filter", flush=True)
        return df_filtered

    if product == 'ATL06':
        pct = len(df_filtered) / initial_count * 100 if initial_count else 0.0
        print(f" Filtered: {initial_count:,} -> {len(df_filtered):,} rows ({pct:.1f}% retained)", flush=True)
        return df_filtered

    # --- ATL03 / ATL08 classification-based filtering ---
    has_classification = 'Classification' in df_filtered.columns and df_filtered['Classification'].notna().any()
    if has_classification:
        df_filtered = df_filtered[df_filtered['Classification'].isin([1, 4])]
        print(f" After classification filter: {len(df_filtered):,} rows (used ATL08)", flush=True)
    else:
        print(" No ATL08 classification - using height-based filtering", flush=True)
        height_col = detect_height_col(df_filtered)
        if 'Along-Track (m)' not in df_filtered.columns:
            print(" WARNING: Along-Track (m) column not found - height-based filter skipped", flush=True)
        elif height_col is None:
            print(" WARNING: No recognised height column found - height-based filter skipped", flush=True)
        else:
            df_filtered = df_filtered.copy()
            df_filtered['segment'] = (df_filtered['Along-Track (m)'] // 100).astype(int)

            def filter_by_height(group: pd.DataFrame) -> pd.DataFrame:
                if len(group) < 5:
                    return group
                threshold = group[height_col].quantile(0.20)
                return group[group[height_col] <= threshold]

            df_filtered = (
                df_filtered
                .groupby(['rgt', 'cycle', 'beam', 'segment'], observed=True)
                .apply(filter_by_height)
                .reset_index(drop=True)
            )
            if 'segment' in df_filtered.columns:
                df_filtered = df_filtered.drop(columns=['segment'])
            print(f" After height-based filter: {len(df_filtered):,} rows", flush=True)

    pct = len(df_filtered) / initial_count * 100 if initial_count else 0.0
    print(f" Filtered: {initial_count:,} -> {len(df_filtered):,} rows ({pct:.1f}% retained)", flush=True)
    return df_filtered


# =============================================================================
# MAIN
# =============================================================================
def main():
    print(f" for ICESat-2 {PRODUCT} files...", flush=True)

    if PRODUCT == 'ATL06':
        h5_files = sorted(glob.glob(str(Path(INPUTDIR) / '*ATL06*.h5')))
        csv_files_atl06 = sorted(glob.glob(str(Path(INPUTDIR) / '*ATL06*.csv')))

        if len(h5_files) == 0 and len(csv_files_atl06) == 0:
            print("ERROR: No ATL06 .h5 or .csv files found!", flush=True)
            print(f"Check path: {INPUTDIR}", flush=True)
            return

        if len(h5_files) > 0:
            print(f"Found {len(h5_files)} ATL06 HDF5 granules", flush=True)
            loader_fn = load_atl06_h5
            file_list = h5_files
            source_format = 'HDF5'
        else:
            print(f"Found {len(csv_files_atl06)} ATL06 CSV files", flush=True)
            loader_fn = load_atl06_csv
            file_list = csv_files_atl06
            source_format = 'CSV'

        print(f"ATL06 source format detected: {source_format}", flush=True)
        print("=" * 80, flush=True)

        all_data = []
        for i, fpath in enumerate(file_list, 1):
            print(f"[{i}/{len(file_list)}] Processing {Path(fpath).name}", flush=True)
            df = loader_fn(fpath)
            if df is None or len(df) == 0:
                print(" WARNING: Skipping - no data loaded", flush=True)
                continue
            df_filtered = filter_ground_photons(df, product='ATL06', confidence_threshold=CONFIDENCE_THRESHOLD)
            if len(df_filtered) > 0:
                all_data.append(df_filtered)
            else:
                print(" WARNING: No segments passed filtering", flush=True)

        if len(all_data) == 0:
            print("No data survived filtering!", flush=True)
            return

        icesat2_data = pd.concat(all_data, ignore_index=True)
        has_dates = icesat2_data['date'].notna().any() if 'date' in icesat2_data.columns else False

    else:
        # ATL03 / ATL08 - PhoREAL CSV/LAS
        pattern = PRODUCTFILEPATTERNS[PRODUCT]
        csv_files = sorted(glob.glob(str(Path(INPUTDIR) / pattern)))
        if len(csv_files) == 0:
            print(f"ERROR: No {PRODUCT} files found!", flush=True)
            print(f"Check path: {INPUTDIR}", flush=True)
            return

        print(f"Found {len(csv_files)} CSV files", flush=True)
        unique_files = sorted(set('_'.join(Path(f).stem.split('_')[:-1]) for f in csv_files))
        print(f"Processing {len(unique_files)} unique acquisitions", flush=True)
        print(f"FILTERING: Strong beams only {STRONG_BEAMS}", flush=True)
        print("=" * 80, flush=True)

        all_data = []
        file_count = 0
        for base_filename in unique_files:
            for beam in STRONG_BEAMS:
                file_count += 1
                filepath = Path(INPUTDIR) / f"{base_filename}_{beam}.csv"
                if not filepath.exists():
                    continue
                print(f"[{file_count}/{len(unique_files)}] Processing {filepath.name}", flush=True)
                df = load_best_format(str(filepath))
                if df is None or len(df) == 0:
                    print(" WARNING: Skipping - no data loaded", flush=True)
                    continue
                df_filtered = filter_ground_photons(df, product=PRODUCT, confidence_threshold=CONFIDENCE_THRESHOLD)
                if len(df_filtered) > 0:
                    all_data.append(df_filtered)
                else:
                    print(" WARNING: No photons passed filtering", flush=True)

        if len(all_data) == 0:
            print("No data survived filtering!", flush=True)
            return

        print(f"{'=' * 80}", flush=True)
        print(f"Combining {len(all_data)} beam/acquisition datasets...", flush=True)
        icesat2_data = pd.concat(all_data, ignore_index=True)
        has_dates = 'date' in icesat2_data.columns and not icesat2_data['date'].isna().all()

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("=" * 80, flush=True)
    print("FILTERING SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(f"Product: {PRODUCT}", flush=True)
    print(f"Total filtered rows: {len(icesat2_data):,}", flush=True)

    if has_dates:
        valid_dates = icesat2_data['date'].dropna()
        print(f"Date range: {valid_dates.min().date()} to {valid_dates.max().date()}", flush=True)
        print(f"Years covered: {sorted(icesat2_data['year'].dropna().astype(int).unique())}", flush=True)

    if 'rgt' in icesat2_data.columns:
        print(f"Unique RGTs: {icesat2_data['rgt'].nunique()}", flush=True)
    if 'beam' in icesat2_data.columns:
        print(f"Unique beams: {sorted(icesat2_data['beam'].unique())}", flush=True)

    total_rows = len(icesat2_data)
    years_sorted = []
    if has_dates:
        years_sorted = sorted(icesat2_data['year'].dropna().astype(int).unique())
        print("=" * 80, flush=True)
        print("COUNTS BY YEAR", flush=True)
        print("=" * 80, flush=True)
        print(f"{'Year':<8}{'Count':<15}{'Percentage':<12}", flush=True)
        print("-" * 40, flush=True)
        for year in years_sorted:
            count = int((icesat2_data['year'] == year).sum())
            pct = 100.0 * count / total_rows
            print(f"{year:<8}{count:<15,}{pct:<11.1f}", flush=True)
        print("-" * 40, flush=True)
        print(f"{'TOTAL':<8}{total_rows:<15,}{100.0:<11}", flush=True)

    # -------------------------------------------------------------------------
    # OUTPUTS
    # -------------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / 'icesat2_filtered_ground_photons.csv'
    print(f" to {output_file}", flush=True)
    icesat2_data.to_csv(output_file, index=False, encoding='utf-8')
    print(f"OK Saved successfully ({output_file.stat().st_size / (1024*1024):.1f} MB)", flush=True)

    count_file = OUTPUT_DIR / '01_photon_counts_by_year.txt'
    with open(count_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"SCRIPT 01: FILTERED COUNTS BY YEAR - PRODUCT {PRODUCT}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Run ID: {RUN_TIMESTAMP}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if has_dates:
            f.write(f"{'Year':<8}{'Count':<15}{'Percentage':<12}\n")
            f.write("-" * 40 + "\n")
            for year in years_sorted:
                count = int((icesat2_data['year'] == year).sum())
                pct = 100.0 * count / total_rows
                f.write(f"{year:<8}{count:<15,}{pct:<11.1f}\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'TOTAL':<8}{total_rows:<15,}{100.0:<11}\n")
        else:
            f.write("WARNING: Date column unavailable - year breakdown not produced.\n")
        f.write("=" * 80 + "\n")
    print(f"OK Year counts saved to {count_file.name}", flush=True)

    summary_file = OUTPUT_DIR / '01_filtering_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"ICESat-2 Filtering Summary - PRODUCT {PRODUCT}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Processing date: {datetime.now()}\n")
        f.write(f"Run ID: {RUN_TIMESTAMP}\n")
        f.write("CONFIGURATION\n")
        f.write(f" Product: {PRODUCT}\n")
        if PRODUCT in ('ATL03', 'ATL08'):
            f.write(f" Strong beams only: {STRONG_BEAMS}\n")
            f.write(f" Confidence threshold: {CONFIDENCE_THRESHOLD}\n")
        if PRODUCT == 'ATL06':
            f.write(" Elevation field: 'Height (m MSL)' (ATL06 CSV export, as delivered)\n")
            f.write(" Quality screen: quality_summary == 0\n")
        f.write(" Study-area filter: polygon-based spatial test (point-in-polygon)\n")
        f.write(f" Study-area source: {STUDY_AREA_GPKG}\n")
        f.write(f" Study-area description: {_study_area_desc}\n")
        f.write(f"Total rows: {total_rows:,}\n")
        if has_dates:
            f.write(f"Date range: {icesat2_data['date'].dropna().min()} to {icesat2_data['date'].dropna().max()}\n")
            f.write(f"Years: {years_sorted}\n")
    print(f"OK Summary saved to {summary_file}", flush=True)

    print("=" * 80, flush=True)
    print("SCRIPT 01 COMPLETE", flush=True)
    print("=" * 80, flush=True)
    print("Next step: Run Script 02 (Terrain Classification)", flush=True)


if __name__ == '__main__':
    main()
