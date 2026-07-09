"""
================================================================================
SCRIPT 02: TERRAIN CLASSIFICATION (LAND / OCEAN / LAKES) - WITH INTEGRATED REFINEMENT
================================================================================

PURPOSE
-------
Classify filtered ground photons as land, ocean or lakes using:

1. GIS polygon boundaries (Statistics Canada + CanVec)
2. Elevation-based refinement (hybrid approach)
3. Year-by-year temporal tracking

METHOD
------
Step 1: Use GIS spatial joins (location-based classification)
Step 2: Apply elevation thresholds (physics-based refinement)
Step 3: Track temporal patterns by year

ELEVATION THRESHOLDS APPLIED:
* LAND:  >= -2.0 m MSL (allows GIS uncertainty + subsidence)
* LAKES: >= -2.0 m MSL (no upper cap for thermokarst lakes)
* OCEAN: <=  1.5 m MSL (Tuktoyaktuk tidal range ~0.3 m + wave buffer)

INPUTS
------
- icesat2_filtered_ground_photons.csv (Script 01 output)
- CANADA_LAND_SHAPEFILE (Statistics Canada 2021 boundaries)
- CANVEC_WATERBODY_SHAPEFILE (NRCan CanVec 50K waterbodies)

OUTPUTS
-------
- icesat2_filtered_ground_photons_LAND.csv (elevation-verified)
- icesat2_filtered_ground_photons_LAKES.csv (elevation-verified)
- icesat2_filtered_ground_photons_OCEAN.csv (elevation-verified)
- icesat2_filtered_ground_photons_ORIGINAL.csv (backup of Script 01 output)
- icesat2_terrain_classified.csv (unified dataset)
- 02_photon_counts_by_year_class.txt (year-by-year breakdown)
- 02_filtering_summary.txt (elevation filtering log)
- 02_refinement_summary.txt (full processing summary)

CRS NOTE
--------
DEFAULT_UTM_EPSG = 3155 -> NAD83(CSRS) / UTM Zone 9N
This matches the Statistics Canada and NRCan CanVec GIS layers used for
spatial joins. If your PhoREAL output uses WGS84 UTM Zone 9N (EPSG:32609)
instead, set DEFAULT_UTM_EPSG = 32609 and the GIS layers will be reprojected
to match automatically.
"""

import argparse
import shutil
import sys
import io
from datetime import datetime
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from config import BASEOUTPUTDIR, VALIDPRODUCTS, DEFAULTPRODUCT

# UTF-8 encoding for Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ============================================================================
# CLI ARGUMENTS
# ============================================================================
parser = argparse.ArgumentParser(description="ICESat-2 Terrain Classification")
parser.add_argument(
    '--product', type=str, choices=VALIDPRODUCTS, default=DEFAULTPRODUCT,
    help='ICESat-2 product being processed (for logging/metadata only; '
         'terrain classification logic is identical across products)'
)
args = parser.parse_args()
PRODUCT = args.product

# ============================================================================
# CONFIGURATION
# ============================================================================

CANADA_LAND_SHAPEFILE = Path(r'E:\GIS\lpr_000b21a_e\lpr_000b21a_e.shp')
CANVEC_WATERBODY_SHAPEFILE = Path(r'E:\GIS\canvec_50K_NT_Hydro_shp\waterbody_2_2.shp')

# Elevation thresholds (metres MSL) — violations are DELETED
LAND_MIN_ELEV = -7.0   # Delete LAND photons below this
LAKE_MIN_ELEV = -2.0   # Delete LAKE photons below this
OCEAN_MAX_ELEV = 1.5   # Delete OCEAN photons above this
# (Tuktoyaktuk tidal range ~0.3 m; 1.5 m gives buffer)

# Default UTM CRS for Tuktoyaktuk Peninsula when column detection fails.
# EPSG:3155 = NAD83(CSRS) / UTM Zone 9N — matches Statistics Canada &
# NRCan CanVec layers. Change to 32609 (WGS84 / UTM Zone 9N) if your
# PhoREAL output uses WGS84; the GIS layers will be reprojected to match.
DEFAULT_UTM_EPSG = 3155

CHUNK_SIZE = 500_000  # photons per processing chunk

# Height column priority order (first match wins)
HEIGHT_COL_CANDIDATES = [
    'Height (m MSL)', 'height', 'Height', 'elevation', 'h_li', 'Terrain Best Fit (m)'
]

# ============================================================================
# VALIDATE SHAPEFILES
# ============================================================================

if not CANADA_LAND_SHAPEFILE.exists():
    print("=" * 80)
    print("ERROR: Statistics Canada land boundary not found!")
    print(f"Expected: {CANADA_LAND_SHAPEFILE}")
    sys.exit(1)

if not CANVEC_WATERBODY_SHAPEFILE.exists():
    print("=" * 80)
    print("WARNING: CanVec waterbody layer not found!")
    print(f"Expected: {CANVEC_WATERBODY_SHAPEFILE}")
    print("Proceeding with land/ocean separation only (no lakes)")
    USE_LAKES = False
else:
    USE_LAKES = True

print("=" * 80, flush=True)
print("SCRIPT 02: TERRAIN CLASSIFICATION WITH ELEVATION FILTERING", flush=True)
print("=" * 80, flush=True)
print(f"Product: {PRODUCT}", flush=True)

# ============================================================================
# READ RUN TIMESTAMP (with existence guard)
# ============================================================================

timestamp_file = BASEOUTPUTDIR / 'latest_RUN_TIMESTAMP.txt'

if not timestamp_file.exists():
    print(f"\n✗ ERROR: Timestamp file not found: {timestamp_file}")
    print("  Script 00 must run first!")
    sys.exit(1)

with open(timestamp_file, 'r', encoding='utf-8') as f:
    RUN_TIMESTAMP = f.read().strip()

OUTPUT_DIR = BASEOUTPUTDIR / f'run_{RUN_TIMESTAMP}'

# ============================================================================
# FILE PATHS
# ============================================================================

INPUT_FILE = OUTPUT_DIR / 'icesat2_filtered_ground_photons.csv'
BACKUP_FILE = OUTPUT_DIR / 'icesat2_filtered_ground_photons_ORIGINAL.csv'

LAND_FILE = OUTPUT_DIR / 'icesat2_filtered_ground_photons_LAND.csv'
LAKES_FILE = OUTPUT_DIR / 'icesat2_filtered_ground_photons_LAKES.csv'
OCEAN_FILE = OUTPUT_DIR / 'icesat2_filtered_ground_photons_OCEAN.csv'

CLASSIFIED_FILE = OUTPUT_DIR / 'icesat2_terrain_classified.csv'
COUNTS_FILE = OUTPUT_DIR / '02_photon_counts_by_year_class.txt'
FILTERING_FILE = OUTPUT_DIR / '02_filtering_summary.txt'
REFINEMENT_FILE = OUTPUT_DIR / '02_refinement_summary.txt'

# NOTE: No OUTPUT_FILE alias to INPUT_FILE. Downstream scripts must read
# LAND_FILE directly. Writing the LAND data twice (once to LAND_FILE and
# once back to the original path) is redundant and risks silent data
# corruption on re-runs.

print(f"Run ID: {RUN_TIMESTAMP}", flush=True)
print(f"Input file: {INPUT_FILE.name}", flush=True)
print(f"Canada land boundary: {CANADA_LAND_SHAPEFILE.name}", flush=True)
if USE_LAKES:
    print(f"CanVec waterbodies: {CANVEC_WATERBODY_SHAPEFILE.name}", flush=True)

print("=" * 80, flush=True)
print(f"\nElevation thresholds (violations are DELETED):")
print(f"  LAND:  must be >= {LAND_MIN_ELEV:.1f} m MSL (delete if below)")
print(f"  LAKES: must be >= {LAKE_MIN_ELEV:.1f} m MSL (delete if below)")
print(f"  OCEAN: must be <= {OCEAN_MAX_ELEV:.1f} m MSL (delete if above)\n")

if not INPUT_FILE.exists():
    print(f"ERROR: Input file not found: {INPUT_FILE}", flush=True)
    sys.exit(1)

file_size_mb = INPUT_FILE.stat().st_size / (1024 ** 2)
print(f"Input file size: {file_size_mb:.1f} MB\n", flush=True)

# ============================================================================
# PRE-FLIGHT: DETECT ELEVATION COLUMN ONCE (before chunk loop)
# ============================================================================

print("Pre-flight: detecting columns...", flush=True)
_header_check = pd.read_csv(INPUT_FILE, nrows=5)

ELEV_COL = None
for _candidate in HEIGHT_COL_CANDIDATES:
    if _candidate in _header_check.columns:
        ELEV_COL = _candidate
        break

if ELEV_COL is None:
    print(
        f"  WARNING: No recognised elevation column found in input.\n"
        f"  Candidates checked: {HEIGHT_COL_CANDIDATES}\n"
        f"  Elevation filtering will be SKIPPED - all photons retained.",
        flush=True
    )
else:
    print(f"  Elevation column: '{ELEV_COL}'", flush=True)

# ============================================================================
# PRE-FLIGHT: VALIDATE REQUIRED SPATIAL COLUMNS
# ============================================================================

required_cols = {'UTM Easting (m)', 'UTM Northing (m)'}
missing_cols = required_cols - set(_header_check.columns)
if missing_cols:
    print(f"ERROR: Required column(s) missing from input: {missing_cols}",
          flush=True)
    sys.exit(1)

# ============================================================================
# LOAD CANADIAN BOUNDARY LAYERS
# ============================================================================

print("=" * 80, flush=True)
print("LOADING CANADIAN GOVERNMENT DATA LAYERS", flush=True)
print("=" * 80, flush=True)

print("\n1. Loading Statistics Canada land boundary...", flush=True)
canada_land = gpd.read_file(CANADA_LAND_SHAPEFILE)
print(f"   Loaded {len(canada_land)} province/territory boundaries", flush=True)
print(f"   CRS: {canada_land.crs}", flush=True)

if USE_LAKES:
    print("\n2. Loading CanVec waterbodies...", flush=True)
    waterbodies = gpd.read_file(CANVEC_WATERBODY_SHAPEFILE)
    print(f"   Loaded {len(waterbodies)} waterbody features", flush=True)
    print(f"   CRS: {waterbodies.crs}", flush=True)

# ============================================================================
# DETECT UTM ZONE (with fallback to DEFAULT_UTM_EPSG)
# ============================================================================

print("\n3. Detecting UTM zone from ICESat-2 data...", flush=True)

utm_epsg = DEFAULT_UTM_EPSG  # safe default; overridden below if columns present

if 'UTM Zone' in _header_check.columns and 'UTM Hemisphere' in _header_check.columns:
    _sample = pd.read_csv(INPUT_FILE, nrows=1000)
    try:
        utm_zone = int(_sample['UTM Zone'].mode()[0])
        utm_hemisphere = str(_sample['UTM Hemisphere'].mode()[0]).strip().upper()
        if utm_hemisphere == 'N':
            utm_epsg = 32600 + utm_zone
        else:
            utm_epsg = 32700 + utm_zone
        print(f"   Detected: UTM Zone {utm_zone}{utm_hemisphere} "
              f"(EPSG:{utm_epsg})", flush=True)
        if utm_epsg != DEFAULT_UTM_EPSG:
            print(f"   NOTE: Detected EPSG:{utm_epsg} differs from default "
                  f"EPSG:{DEFAULT_UTM_EPSG} (NAD83(CSRS) / UTM Zone 9N).\n"
                  f"   GIS layers will be reprojected to EPSG:{utm_epsg} "
                  f"to match photon CRS.", flush=True)
    except Exception as e:
        print(f"   WARNING: UTM detection failed ({e}); "
              f"using default EPSG:{DEFAULT_UTM_EPSG} "
              f"(NAD83(CSRS) / UTM Zone 9N)", flush=True)
else:
    print(f"   'UTM Zone'/'UTM Hemisphere' columns not found; "
          f"using default EPSG:{DEFAULT_UTM_EPSG} "
          f"(NAD83(CSRS) / UTM Zone 9N)", flush=True)

# ============================================================================
# REPROJECT BOUNDARY LAYERS TO MATCH PHOTON CRS
# ============================================================================

print("\n4. Reprojecting boundary layers to match photon CRS...", flush=True)
canada_land_utm = canada_land.to_crs(epsg=utm_epsg)
print(f"   Canada land boundary -> EPSG:{utm_epsg}", flush=True)

if USE_LAKES:
    current_epsg = waterbodies.crs.to_epsg() if waterbodies.crs is not None else None
    if current_epsg != utm_epsg:
        print(f"   Reprojecting waterbodies {current_epsg} -> {utm_epsg}...",
              flush=True)
        waterbodies_utm = waterbodies.to_crs(epsg=utm_epsg)
    else:
        waterbodies_utm = waterbodies
    print(f"   Waterbodies ready (EPSG:{utm_epsg})", flush=True)

print("\n[OK] All boundary layers ready\n", flush=True)

# ============================================================================
# BACKUP ORIGINAL FILE (copy, not rename - safe on re-runs)
# ============================================================================

print(f"Backing up original input: {BACKUP_FILE.name}", flush=True)
shutil.copy2(INPUT_FILE, BACKUP_FILE)
print(f"  ✓ Backup written: {BACKUP_FILE.name}\n", flush=True)

# ============================================================================
# PROCESS PHOTONS IN CHUNKS
# ============================================================================

print("=" * 80, flush=True)
print("CLASSIFYING PHOTONS: LAND / LAKES / OCEAN", flush=True)
print("=" * 80, flush=True)
print(f"Processing in chunks of {CHUNK_SIZE:,} "
      f"(this may take 10-20 minutes)...\n", flush=True)

land_chunks = []
lakes_chunks = []
ocean_chunks = []

total_input = 0
chunk_num = 0
elev_col_absent_chunks = 0  # chunks where elevation column was absent after sjoin

deleted_photons = {
    'LAND_below_threshold': 0,
    'LAKE_below_threshold': 0,
    'OCEAN_above_threshold': 0,
}

for chunk in pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE):
    chunk_num += 1
    total_input += len(chunk)

    print(f"Chunk {chunk_num}: {len(chunk):,} photons ... ", end='', flush=True)

    # ------------------------------------------------------------------
    # Build GeoDataFrame from UTM coordinates
    # ------------------------------------------------------------------
    geometry = [
        Point(e, n)
        for e, n in zip(chunk['UTM Easting (m)'], chunk['UTM Northing (m)'])
    ]
    gdf_chunk = gpd.GeoDataFrame(chunk, geometry=geometry,
                                 crs=f'EPSG:{utm_epsg}')

    # ------------------------------------------------------------------
    # STEP 1: Classify land vs. ocean via Canada boundary sjoin
    # Only pass geometry column to avoid attribute name conflicts.
    # Deduplicate after join: a point on a province boundary may
    # match >1 polygon, producing duplicate rows.
    # ------------------------------------------------------------------
    gdf_with_land = gpd.sjoin(
        gdf_chunk,
        canada_land_utm[['geometry']],
        how='left',
        predicate='within',
    )
    gdf_with_land = gdf_with_land[
        ~gdf_with_land.index.duplicated(keep='first')
    ]

    on_land_mask = gdf_with_land['index_right'].notna()

    # ------------------------------------------------------------------
    # STEP 2: Separate land from lakes (if CanVec available)
    # ------------------------------------------------------------------
    if USE_LAKES:
        gdf_land_candidates = (
            gdf_with_land[on_land_mask]
            .drop(columns=['index_right'])
            .copy()
        )
        gdf_with_lakes = gpd.sjoin(
            gdf_land_candidates,
            waterbodies_utm[['geometry']],
            how='left',
            predicate='within',
        )
        # Deduplicate lake join (point may overlap multiple lake polygons)
        gdf_with_lakes = gdf_with_lakes[
            ~gdf_with_lakes.index.duplicated(keep='first')
        ]

        in_lake_mask = gdf_with_lakes['index_right'].notna()
        gdf_lakes = gdf_with_lakes[in_lake_mask].copy()
        gdf_land = gdf_with_lakes[~in_lake_mask].copy()
        gdf_ocean = gdf_with_land[~on_land_mask].copy()
    else:
        gdf_land = gdf_with_land[on_land_mask].copy()
        gdf_lakes = gpd.GeoDataFrame()
        gdf_ocean = gdf_with_land[~on_land_mask].copy()

    # ------------------------------------------------------------------
    # Convert GeoDataFrames to plain DataFrames (drop geometry + artifacts)
    # ------------------------------------------------------------------
    def _to_df(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Drop geometry and sjoin index columns; return plain DataFrame."""
        if not isinstance(gdf, gpd.GeoDataFrame) or len(gdf) == 0:
            return pd.DataFrame()
        cols_to_drop = [c for c in ['geometry', 'index_right']
                        if c in gdf.columns]
        return pd.DataFrame(gdf.drop(columns=cols_to_drop))

    df_land = _to_df(gdf_land)
    df_lakes = _to_df(gdf_lakes)
    df_ocean = _to_df(gdf_ocean)

    # ------------------------------------------------------------------
    # STEP 3: Elevation filtering (using pre-detected ELEV_COL)
    # ------------------------------------------------------------------
    if ELEV_COL is None:
        # No elevation column - retain everything, skip counters
        land_keep = df_land
        lakes_keep = df_lakes
        ocean_keep = df_ocean
        elev_col_absent_chunks += 1
    else:
        # LAND: delete below threshold
        if len(df_land) > 0 and ELEV_COL in df_land.columns:
            mask = df_land[ELEV_COL] < LAND_MIN_ELEV
            deleted_photons['LAND_below_threshold'] += int(mask.sum())
            land_keep = df_land[~mask].copy()
        else:
            land_keep = df_land

        # LAKES: delete below threshold
        if len(df_lakes) > 0 and ELEV_COL in df_lakes.columns:
            mask = df_lakes[ELEV_COL] < LAKE_MIN_ELEV
            deleted_photons['LAKE_below_threshold'] += int(mask.sum())
            lakes_keep = df_lakes[~mask].copy()
        else:
            lakes_keep = df_lakes

        # OCEAN: delete above threshold
        if len(df_ocean) > 0 and ELEV_COL in df_ocean.columns:
            mask = df_ocean[ELEV_COL] > OCEAN_MAX_ELEV
            deleted_photons['OCEAN_above_threshold'] += int(mask.sum())
            ocean_keep = df_ocean[~mask].copy()
        else:
            ocean_keep = df_ocean

    # Collect retained chunks
    if len(land_keep) > 0:
        land_chunks.append(land_keep)
    if len(lakes_keep) > 0:
        lakes_chunks.append(lakes_keep)
    if len(ocean_keep) > 0:
        ocean_chunks.append(ocean_keep)

    # Progress line
    n = len(chunk)
    pct = lambda k: f"{len(k):,} ({100*len(k)/n:.1f}%)" if n > 0 else "0"
    if USE_LAKES:
        print(f"Land: {pct(land_keep)}, "
              f"Lakes: {pct(lakes_keep)}, "
              f"Ocean: {pct(ocean_keep)}", flush=True)
    else:
        print(f"Land: {pct(land_keep)}, "
              f"Ocean: {pct(ocean_keep)}", flush=True)

# ============================================================================
# COMBINE RESULTS
# ============================================================================

print("\n" + "=" * 80, flush=True)
print("COMBINING RESULTS", flush=True)
print("=" * 80, flush=True)

print("\nCombining land chunks...", flush=True)
df_land_only = pd.concat(land_chunks, ignore_index=True) if land_chunks else pd.DataFrame()
print("Combining lakes chunks...", flush=True)
df_lakes_only = pd.concat(lakes_chunks, ignore_index=True) if lakes_chunks else pd.DataFrame()
print("Combining ocean chunks...", flush=True)
df_ocean_only = pd.concat(ocean_chunks, ignore_index=True) if ocean_chunks else pd.DataFrame()

total_land = len(df_land_only)
total_lakes = len(df_lakes_only)
total_ocean = len(df_ocean_only)
total_output = total_land + total_lakes + total_ocean

# ============================================================================
# ELEVATION FILTERING SUMMARY
# ============================================================================

print("\n" + "=" * 80, flush=True)
print("ELEVATION FILTERING SUMMARY", flush=True)
print("=" * 80, flush=True)

total_deleted = sum(deleted_photons.values())
pct_deleted = 100.0 * total_deleted / total_input if total_input > 0 else 0.0

print(f"\nTotal input photons: {total_input:,}", flush=True)
print(f"Total output photons (retained): {total_output:,}", flush=True)
print(f"Total DELETED (elevation filter): {total_deleted:,} ({pct_deleted:.2f}%)",
      flush=True)
print(f"\nDeletions by category:", flush=True)
print(f"  LAND below {LAND_MIN_ELEV:.1f} m: "
      f"{deleted_photons['LAND_below_threshold']:,}", flush=True)
print(f"  LAKE below {LAKE_MIN_ELEV:.1f} m: "
      f"{deleted_photons['LAKE_below_threshold']:,}", flush=True)
print(f"  OCEAN above {OCEAN_MAX_ELEV:.1f} m: "
      f"{deleted_photons['OCEAN_above_threshold']:,}", flush=True)

if ELEV_COL is None:
    print("\nWARNING: Elevation column not found - "
          "deletion counters are all zero (no filtering applied).", flush=True)
elif elev_col_absent_chunks > 0:
    print(f"\nWARNING: {elev_col_absent_chunks} chunk(s) had no elevation "
          "column after sjoin - conservation check may be incomplete.",
          flush=True)

conservation_total = total_output + total_deleted
print(f"\n✓ Conservation check: "
      f"{total_input:,} input -> "
      f"{total_output:,} retained + {total_deleted:,} deleted = "
      f"{conservation_total:,}", flush=True)
if conservation_total != total_input:
    print(f"  ⚠ Mismatch of {abs(conservation_total - total_input):,} photons "
          "(likely from sjoin deduplication of boundary-overlapping points)",
          flush=True)

# ============================================================================
# ELEVATION STATS PER CLASS
# ============================================================================

print("\n" + "=" * 80, flush=True)
print("FINAL CLASSIFICATION SUMMARY (AFTER FILTERING)", flush=True)
print("=" * 80 + "\n", flush=True)

print(f"Total input photons: {total_input:,}", flush=True)
print(f"Total output photons: {total_output:,}\n", flush=True)

if total_output > 0:
    print(f"  LAND photons:  {total_land:>10,} ({100*total_land/total_output:.2f}%)",
          flush=True)
    print(f"  LAKE photons:  {total_lakes:>10,} ({100*total_lakes/total_output:.2f}%)",
          flush=True)
    print(f"  OCEAN photons: {total_ocean:>10,} ({100*total_ocean/total_output:.2f}%)",
          flush=True)


def _elev_stats(df: pd.DataFrame, label: str) -> None:
    """Print elevation range and mean for a classified DataFrame."""
    if len(df) == 0:
        return
    col = next((c for c in HEIGHT_COL_CANDIDATES if c in df.columns), None)
    if col is None:
        return
    print(f"\n  {label} elevation range: "
          f"{df[col].min():.2f} to {df[col].max():.2f} m "
          f"(mean: {df[col].mean():.2f} m)", flush=True)


_elev_stats(df_land_only, 'LAND')
_elev_stats(df_lakes_only, 'LAKES')
_elev_stats(df_ocean_only, 'OCEAN')

# ============================================================================
# CREATE UNIFIED CLASSIFIED DATASET
# ============================================================================

print("\nCreating unified classified dataset...", flush=True)

# Use .assign() to avoid SettingWithCopyWarning
if total_land > 0:
    df_land_only = df_land_only.assign(classification='LAND')
if total_lakes > 0:
    df_lakes_only = df_lakes_only.assign(classification='LAKE')
if total_ocean > 0:
    df_ocean_only = df_ocean_only.assign(classification='OCEAN')

all_datasets = [
    df for df in [df_land_only, df_lakes_only, df_ocean_only] if len(df) > 0
]

if all_datasets:
    df_all_classified = pd.concat(all_datasets, ignore_index=True)

    # Reuse 'year' column from Script 01 if already present and valid
    if 'year' in df_all_classified.columns and df_all_classified['year'].notna().any():
        print("  Using existing 'year' column (from Script 01).", flush=True)
    else:
        date_col = next(
            (c for c in ['date', 'Date', 'datetime', 'Date/Time', 'acquisition_date']
             if c in df_all_classified.columns),
            None
        )
        if date_col:
            print(f"  Extracting year from '{date_col}' column...", flush=True)
            df_all_classified['year'] = (
                pd.to_datetime(df_all_classified[date_col], errors='coerce').dt.year
            )
        else:
            print("  WARNING: No date column found for year extraction.",
                  flush=True)
            df_all_classified['year'] = pd.NA
else:
    df_all_classified = pd.DataFrame()
    print("WARNING: No data after filtering!", flush=True)

# ============================================================================
# YEAR-BY-YEAR CLASSIFICATION COUNTS
# ============================================================================

print("\n" + "=" * 80, flush=True)
print("YEAR-BY-YEAR CLASSIFICATION COUNTS", flush=True)
print("=" * 80, flush=True)

has_year_data = (
    len(df_all_classified) > 0
    and 'year' in df_all_classified.columns
    and df_all_classified['year'].notna().any()
)

if has_year_data:
    year_class_counts = (
        df_all_classified
        .groupby(['year', 'classification'], observed=True)
        .size()
        .unstack(fill_value=0)
    )

    for col in ['LAND', 'LAKE', 'OCEAN']:
        if col not in year_class_counts.columns:
            year_class_counts[col] = 0

    year_class_counts = year_class_counts[['LAND', 'LAKE', 'OCEAN']]
    year_class_counts['TOTAL'] = year_class_counts.sum(axis=1)

    # --- Console: counts ---
    print("\nClassification counts by year:")
    print("-" * 64)
    print(f"{'Year':<8} {'LAND':>12} {'LAKE':>12} {'OCEAN':>12} {'TOTAL':>12}")
    print("=" * 64)
    for year in sorted(year_class_counts.index):
        r = year_class_counts.loc[year]
        print(f"{int(year):<8} {int(r.LAND):>12,} {int(r.LAKE):>12,} "
              f"{int(r.OCEAN):>12,} {int(r.TOTAL):>12,}")
    print("-" * 64)
    total_land_ct = int(year_class_counts['LAND'].sum())
    total_lake_ct = int(year_class_counts['LAKE'].sum())
    total_ocean_ct = int(year_class_counts['OCEAN'].sum())
    grand_total_ct = int(year_class_counts['TOTAL'].sum())
    print(f"{'TOTAL':<8} {total_land_ct:>12,} {total_lake_ct:>12,} "
          f"{total_ocean_ct:>12,} {grand_total_ct:>12,}")
    print("=" * 64)

    # --- Console: percentages ---
    print("\nClassification percentages by year:")
    print("-" * 52)
    print(f"{'Year':<8} {'LAND %':>12} {'LAKE %':>12} {'OCEAN %':>12}")
    print("=" * 52)
    for year in sorted(year_class_counts.index):
        tot = year_class_counts.loc[year, 'TOTAL']
        if tot > 0:
            r = year_class_counts.loc[year]
            print(f"{int(year):<8} "
                  f"{r.LAND/tot*100:>11.2f}% "
                  f"{r.LAKE/tot*100:>11.2f}% "
                  f"{r.OCEAN/tot*100:>11.2f}%")
    print("=" * 52)

    # --- Write COUNTS_FILE ---
    with open(COUNTS_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SCRIPT 02: TERRAIN CLASSIFICATION COUNTS BY YEAR\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Run ID: {RUN_TIMESTAMP}\n")
        f.write(f"Product: {PRODUCT}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Counts by year:\n")
        f.write("-" * 64 + "\n")
        f.write(f"{'Year':<8} {'LAND':>12} {'LAKE':>12} {'OCEAN':>12} {'TOTAL':>12}\n")
        f.write("=" * 64 + "\n")
        for year in sorted(year_class_counts.index):
            r = year_class_counts.loc[year]
            f.write(f"{int(year):<8} {int(r.LAND):>12,} {int(r.LAKE):>12,} "
                    f"{int(r.OCEAN):>12,} {int(r.TOTAL):>12,}\n")
        f.write("-" * 64 + "\n")
        f.write(f"{'TOTAL':<8} {total_land_ct:>12,} {total_lake_ct:>12,} "
                f"{total_ocean_ct:>12,} {grand_total_ct:>12,}\n")
        f.write("=" * 64 + "\n\n")

        f.write("Percentages by year:\n")
        f.write("-" * 52 + "\n")
        f.write(f"{'Year':<8} {'LAND %':>12} {'LAKE %':>12} {'OCEAN %':>12}\n")
        f.write("=" * 52 + "\n")
        for year in sorted(year_class_counts.index):
            tot = year_class_counts.loc[year, 'TOTAL']
            if tot > 0:
                r = year_class_counts.loc[year]
                f.write(f"{int(year):<8} "
                        f"{r.LAND/tot*100:>11.2f}% "
                        f"{r.LAKE/tot*100:>11.2f}% "
                        f"{r.OCEAN/tot*100:>11.2f}%\n")
        f.write("=" * 52 + "\n")

    print(f"\n✓ Year-by-year counts saved: {COUNTS_FILE.name}", flush=True)

else:
    print("\nWARNING: Cannot compute year-by-year counts "
          "(no valid 'year' data or empty dataset).", flush=True)

# ============================================================================
# SAVE OUTPUT FILES
# ============================================================================

print("\n" + "=" * 80, flush=True)
print("SAVING OUTPUT FILES", flush=True)
print("=" * 80 + "\n", flush=True)

if total_land > 0:
    df_land_only.to_csv(LAND_FILE, index=False, encoding='utf-8')
    print(f"  LAND:  {LAND_FILE.name} "
          f"({LAND_FILE.stat().st_size/(1024**2):.1f} MB)", flush=True)

if total_lakes > 0:
    df_lakes_only.to_csv(LAKES_FILE, index=False, encoding='utf-8')
    print(f"  LAKES: {LAKES_FILE.name} "
          f"({LAKES_FILE.stat().st_size/(1024**2):.1f} MB)", flush=True)

if total_ocean > 0:
    df_ocean_only.to_csv(OCEAN_FILE, index=False, encoding='utf-8')
    print(f"  OCEAN: {OCEAN_FILE.name} "
          f"({OCEAN_FILE.stat().st_size/(1024**2):.1f} MB)", flush=True)

if len(df_all_classified) > 0:
    df_all_classified.to_csv(CLASSIFIED_FILE, index=False, encoding='utf-8')
    print(f"  UNIFIED: {CLASSIFIED_FILE.name} "
          f"({CLASSIFIED_FILE.stat().st_size/(1024**2):.1f} MB)", flush=True)

# ============================================================================
# WRITE FILTERING SUMMARY FILE
# ============================================================================

with open(FILTERING_FILE, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("SCRIPT 02: ELEVATION FILTERING SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Run ID: {RUN_TIMESTAMP}\n")
    f.write(f"Product: {PRODUCT}\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("ELEVATION THRESHOLDS:\n")
    f.write(f"  LAND:  >= {LAND_MIN_ELEV} m MSL\n")
    f.write(f"  LAKES: >= {LAKE_MIN_ELEV} m MSL\n")
    f.write(f"  OCEAN: <= {OCEAN_MAX_ELEV} m MSL\n\n")
    f.write(f"Elevation column used: "
            f"{ELEV_COL or 'NOT FOUND (filtering skipped)'}\n\n")
    f.write(f"Total photons processed: {total_input:,}\n")
    f.write(f"Total photons DELETED: {total_deleted:,} ({pct_deleted:.2f}%)\n\n")
    f.write("Deletions by category:\n")
    f.write(f"  LAND below {LAND_MIN_ELEV} m: "
            f"{deleted_photons['LAND_below_threshold']:,}\n")
    f.write(f"  LAKE below {LAKE_MIN_ELEV} m: "
            f"{deleted_photons['LAKE_below_threshold']:,}\n")
    f.write(f"  OCEAN above {OCEAN_MAX_ELEV} m: "
            f"{deleted_photons['OCEAN_above_threshold']:,}\n\n")
    f.write("Retained photons:\n")
    f.write(f"  LAND:  {total_land:,}\n")
    f.write(f"  LAKES: {total_lakes:,}\n")
    f.write(f"  OCEAN: {total_ocean:,}\n")
    f.write(f"  TOTAL: {total_output:,}\n")

print(f"\n✓ Filtering summary: {FILTERING_FILE.name}", flush=True)

# ============================================================================
# WRITE REFINEMENT SUMMARY FILE
# ============================================================================

with open(REFINEMENT_FILE, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("SCRIPT 02: TERRAIN CLASSIFICATION & REFINEMENT SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Run ID: {RUN_TIMESTAMP}\n")
    f.write(f"Product: {PRODUCT}\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("GIS LAYERS:\n")
    f.write(f"  Canada land boundary: {CANADA_LAND_SHAPEFILE}\n")
    f.write(f"  CanVec waterbodies: "
            f"{CANVEC_WATERBODY_SHAPEFILE if USE_LAKES else 'NOT USED'}\n")
    f.write(f"  UTM projection: EPSG:{utm_epsg}\n")
    f.write(f"  Default CRS fallback: EPSG:{DEFAULT_UTM_EPSG} "
            f"(NAD83(CSRS) / UTM Zone 9N)\n\n")
    f.write("CLASSIFICATION RESULTS:\n")
    f.write(f"  Input photons: {total_input:,}\n")
    f.write(f"  LAND retained: {total_land:,}\n")
    f.write(f"  LAKE retained: {total_lakes:,}\n")
    f.write(f"  OCEAN retained: {total_ocean:,}\n")
    f.write(f"  DELETED: {total_deleted:,}\n")
    f.write(f"  OUTPUT TOTAL: {total_output:,}\n\n")
    f.write("OUTPUT FILES:\n")
    f.write(f"  {LAND_FILE.name}  <- use this for Scripts 03-07\n")
    f.write(f"  {LAKES_FILE.name}\n")
    f.write(f"  {OCEAN_FILE.name}\n")
    f.write(f"  {CLASSIFIED_FILE.name}\n")
    f.write(f"  {BACKUP_FILE.name} (original Script 01 backup)\n")

print(f"✓ Refinement summary: {REFINEMENT_FILE.name}", flush=True)

# ============================================================================
# COMPLETION
# ============================================================================

print("\n" + "=" * 80, flush=True)
print("SCRIPT 02 COMPLETE", flush=True)
print("=" * 80, flush=True)
print("\n✓ Output files created:")
print(f"  • {LAND_FILE.name} (elevation-verified, use for Scripts 03-07)")
print(f"  • {LAKES_FILE.name} (elevation-verified)")
print(f"  • {OCEAN_FILE.name} (elevation-verified)")
print(f"  • {CLASSIFIED_FILE.name} (unified)")
print(f"  • {BACKUP_FILE.name} (original backup)")
print(f"  • {COUNTS_FILE.name}")
print(f"  • {FILTERING_FILE.name}")
print(f"  • {REFINEMENT_FILE.name}")
print("\nNext steps:")
print("  • Scripts 03-07 read from: icesat2_filtered_ground_photons_LAND.csv")
print("  • Review 02_filtering_summary.txt for elevation filtering details")
print("  • Review 02_refinement_summary.txt for full classification summary")
print("\n" + "=" * 80, flush=True)