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

NEW FEATURES:
- Elevation-based refinement integrated (no separate script needed!)
- Removes physically impossible elevations
- Year-by-year classification tracking
- All-in-one solution
- UTF-8 ENCODING FIX (NO UNICODE ERRORS!)

METHOD
------
Step 1: Use GIS spatial joins (location-based classification)
Step 2: Apply elevation thresholds (physics-based refinement)
Step 3: Track temporal patterns by year

ELEVATION THRESHOLDS APPLIED:
* LAND: >= -2m MSL (allows GIS uncertainty + subsidence)
* LAKES: >= -2m MSL (no maximum - lakes occur naturally at all elevations)
* OCEAN: <= 5m MSL (allows tides ±2m + waves)

INPUTS
------
- icesat2_filtered_ground_photons.csv (Script 01 output)
- CANADA_LAND_SHAPEFILE (Statistics Canada 2021 boundaries)
- CANVEC_WATERBODY_SHAPEFILE (NRCan CanVec 50K waterbodies)

OUTPUTS
-------
- icesat2_filtered_ground_photons_LAND.csv (refined, elevation-verified)
- icesat2_filtered_ground_photons_LAKES.csv (refined, elevation-verified)
- icesat2_filtered_ground_photons_OCEAN.csv (refined, elevation-verified)
- icesat2_terrain_classified.csv (unified dataset with classification column)
- 02_photon_counts_by_year_class.txt (year-by-year breakdown)
- 02_refinement_summary.txt (elevation refinement & filtering summary)

REPRODUCIBILITY NOTE
--------------------
Lakes are not capped at any maximum elevation to preserve reproducibility.
Lakes occur naturally at all elevations in the study area and should be
classified as such without artificial constraints. Only the physically
impossible lower bound (-2m MSL) is applied.
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import sys
import io
from datetime import datetime

from config import BASEOUTPUTDIR

# UTF-8 encoding for Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ============================================================================
# CONFIGURATION
# ============================================================================
# Canadian Government Data Layers
CANADA_LAND_SHAPEFILE = Path(r'E:\GIS\lpr_000b21a_e\lpr_000b21a_e.shp')
CANVEC_WATERBODY_SHAPEFILE = Path(r'E:\GIS\canvec_50K_NT_Hydro_shp\waterbody_2_2.shp')

# Elevation thresholds (meters MSL)
LAND_MIN_ELEV = -2.0  # Delete LAND photons below this
LAKE_MIN_ELEV = -2.0  # Delete LAKE photons below this
OCEAN_MAX_ELEV = 5.0  # Delete OCEAN photons above this

# Verify files exist
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

# ============================================================================
# LOAD LATEST RUN
# ============================================================================
timestamp_file = BASEOUTPUTDIR / 'latest_RUN_TIMESTAMP.txt'
with open(timestamp_file, 'r') as f:
    RUN_TIMESTAMP = f.read().strip()

OUTPUT_DIR = BASEOUTPUTDIR / f'run_{RUN_TIMESTAMP}'

INPUT_FILE = OUTPUT_DIR / 'icesat2_filtered_ground_photons.csv'
BACKUP_FILE = OUTPUT_DIR / 'icesat2_filtered_ground_photons_ORIGINAL.csv'

LAND_FILE = OUTPUT_DIR / 'icesat2_filtered_ground_photons_LAND.csv'
LAKES_FILE = OUTPUT_DIR / 'icesat2_filtered_ground_photons_LAKES.csv'
OCEAN_FILE = OUTPUT_DIR / 'icesat2_filtered_ground_photons_OCEAN.csv'
OUTPUT_FILE = INPUT_FILE  # Will be replaced with LAND version

CLASSIFIED_FILE = OUTPUT_DIR / 'icesat2_terrain_classified.csv'
COUNTS_FILE = OUTPUT_DIR / '02_photon_counts_by_year_class.txt'
FILTERING_FILE = OUTPUT_DIR / '02_filtering_summary.txt'
REFINEMENT_FILE = OUTPUT_DIR / '02_refinement_summary.txt'

print(f"Run ID: {RUN_TIMESTAMP}", flush=True)
print(f"Input file: {INPUT_FILE.name}", flush=True)
print(f"Canada land boundary: {CANADA_LAND_SHAPEFILE.name}", flush=True)
if USE_LAKES:
    print(f"CanVec waterbodies: {CANVEC_WATERBODY_SHAPEFILE.name}", flush=True)

print("=" * 80, flush=True)
print(f"\nElevation thresholds (violations are DELETED):")
print(f" LAND: must be >= {LAND_MIN_ELEV:.1f} m MSL (delete if below)")
print(f" LAKES: must be >= {LAKE_MIN_ELEV:.1f} m MSL (delete if below)")
print(f" OCEAN: must be <= {OCEAN_MAX_ELEV:.1f} m MSL (delete if above)\n")

if not INPUT_FILE.exists():
    print(f"ERROR: Input file not found: {INPUT_FILE}", flush=True)
    sys.exit(1)

file_size_mb = INPUT_FILE.stat().st_size / (1024 ** 2)
print(f"Input file size: {file_size_mb:.1f} MB\n", flush=True)

# ============================================================================
# LOAD CANADIAN BOUNDARY LAYERS
# ============================================================================
print("=" * 80, flush=True)
print("LOADING CANADIAN GOVERNMENT DATA LAYERS", flush=True)
print("=" * 80, flush=True)

print("\n1. Loading Statistics Canada land boundary...", flush=True)
canada_land = gpd.read_file(CANADA_LAND_SHAPEFILE)
print(f" Loaded {len(canada_land)} province/territory boundaries", flush=True)
print(f" CRS: {canada_land.crs}", flush=True)

if USE_LAKES:
    print("\n2. Loading CanVec waterbodies...", flush=True)
    waterbodies = gpd.read_file(CANVEC_WATERBODY_SHAPEFILE)
    print(f" Loaded {len(waterbodies)} waterbody features", flush=True)
    print(f" CRS: {waterbodies.crs}", flush=True)

# Detect UTM zone
print("\n3. Detecting UTM zone from ICeSat-2 data...", flush=True)
first_chunk = pd.read_csv(INPUT_FILE, nrows=1000)
utm_zone = int(first_chunk['UTM Zone'].mode()[0])
utm_hemisphere = first_chunk['UTM Hemisphere'].mode()[0]
if utm_hemisphere == 'N':
    utm_epsg = 32600 + utm_zone
else:
    utm_epsg = 32700 + utm_zone
print(f" Detected: UTM Zone {utm_zone}{utm_hemisphere} (EPSG:{utm_epsg})", flush=True)

# Reproject
print("\n4. Reprojecting boundary layers to UTM...", flush=True)
canada_land_utm = canada_land.to_crs(epsg=utm_epsg)
print(f" Canada land boundary reprojected", flush=True)

if USE_LAKES:
    if waterbodies.crs.to_epsg() != utm_epsg:
        print(f" Reprojecting waterbodies...", flush=True)
        waterbodies_utm = waterbodies.to_crs(epsg=utm_epsg)
    else:
        waterbodies_utm = waterbodies
    print(f" Waterbodies ready", flush=True)

print("\n[OK] All boundary layers ready\n", flush=True)

# ============================================================================
# PROCESS PHOTONS IN CHUNKS - CLASSIFY AND FILTER (DELETION BEHAVIOUR)
# ============================================================================
print("\n" + "=" * 80, flush=True)
print("CLASSIFYING PHOTONS: LAND / LAKES / OCEAN", flush=True)
print("=" * 80, flush=True)
print("Processing in chunks (this may take 10-20 minutes)...\n", flush=True)

chunk_size = 500000  # 500k rows per chunk

land_chunks = []
lakes_chunks = []
ocean_chunks = []

total_input = 0
total_processed = 0
chunk_num = 0

deleted_photons = {
    'LAND_below_threshold': 0,
    'LAKE_below_threshold': 0,
    'OCEAN_above_threshold': 0
}

for chunk in pd.read_csv(INPUT_FILE, chunksize=chunk_size):
    chunk_num += 1
    total_input += len(chunk)

    print(f"Chunk {chunk_num}: Processing {len(chunk):,} photons... ", end='', flush=True)

    # Create GeoDataFrame using UTM coordinates
    geometry = [Point(easting, northing) for easting, northing in zip(
        chunk['UTM Easting (m)'],
        chunk['UTM Northing (m)']
    )]
    gdf_chunk = gpd.GeoDataFrame(
        chunk,
        geometry=geometry,
        crs=f'EPSG:{utm_epsg}'
    )

    # STEP 1: Identify points within Canada land boundary
    gdf_with_land = gpd.sjoin(
        gdf_chunk,
        canada_land_utm,
        how='left',
        predicate='within'
    )

    on_land_mask = gdf_with_land['index_right'].notna()

    # STEP 2: If using lakes, separate land from inland water
    if USE_LAKES:
        gdf_land_candidates = gdf_with_land[on_land_mask].copy()
        gdf_with_lakes = gpd.sjoin(
            gdf_land_candidates.drop(columns=['index_right']),
            waterbodies_utm,
            how='left',
            predicate='within'
        )

        in_lake_mask = gdf_with_lakes['index_right'].notna()
        gdf_lakes = gdf_with_lakes[in_lake_mask].copy()
        gdf_land = gdf_with_lakes[~in_lake_mask].copy()
        gdf_ocean = gdf_with_land[~on_land_mask].copy()
    else:
        gdf_land = gdf_with_land[on_land_mask].copy()
        gdf_lakes = pd.DataFrame()
        gdf_ocean = gdf_with_land[~on_land_mask].copy()

    # Clean up join artifacts
    for df in [gdf_land, gdf_lakes, gdf_ocean]:
        if len(df) > 0 and 'index_right' in df.columns:
            df.drop(columns=['index_right'], inplace=True)

    # Remove geometry and convert
    df_land = pd.DataFrame(gdf_land.drop(columns=['geometry'])) if len(gdf_land) > 0 else pd.DataFrame()
    df_lakes = pd.DataFrame(gdf_lakes.drop(columns=['geometry'])) if len(gdf_lakes) > 0 else pd.DataFrame()
    df_ocean = pd.DataFrame(gdf_ocean.drop(columns=['geometry'])) if len(gdf_ocean) > 0 else pd.DataFrame()

    # Apply elevation filters (DELETION, not reclassification)
    # Elevation column detection for this chunk
    elev_col = None
    for col in ['Height (m MSL)', 'height', 'Height', 'elevation', 'h_li']:
        if col in chunk.columns:
            elev_col = col
            break

    if elev_col is None:
        # No elevation column found; keep everything as-is
        land_keep = df_land
        lakes_keep = df_lakes
        ocean_keep = df_ocean

        land_del = 0
        lake_del = 0
        ocean_del = 0
    else:
        # LAND: delete where below threshold
        if len(df_land) > 0:
            land_mask = df_land[elev_col] < LAND_MIN_ELEV
            land_del = int(land_mask.sum())
            land_keep = df_land[~land_mask].copy()
        else:
            land_keep = df_land
            land_del = 0

        # LAKES: delete where below threshold
        if len(df_lakes) > 0:
            lake_mask = df_lakes[elev_col] < LAKE_MIN_ELEV
            lake_del = int(lake_mask.sum())
            lakes_keep = df_lakes[~lake_mask].copy()
        else:
            lakes_keep = df_lakes
            lake_del = 0

        # OCEAN: delete where above threshold
        if len(df_ocean) > 0:
            ocean_mask = df_ocean[elev_col] > OCEAN_MAX_ELEV
            ocean_del = int(ocean_mask.sum())
            ocean_keep = df_ocean[~ocean_mask].copy()
        else:
            ocean_keep = df_ocean
            ocean_del = 0

        # Update global deletion counters
        deleted_photons['LAND_below_threshold'] += land_del
        deleted_photons['LAKE_below_threshold'] += lake_del
        deleted_photons['OCEAN_above_threshold'] += ocean_del

    # Append retained to global collections
    if len(land_keep) > 0:
        land_chunks.append(land_keep)
    if len(lakes_keep) > 0:
        lakes_chunks.append(lakes_keep)
    if len(ocean_keep) > 0:
        ocean_chunks.append(ocean_keep)

    # Progress message in 46 style
    pct_land = len(land_keep) / len(chunk) * 100 if len(chunk) > 0 else 0
    pct_lakes = len(lakes_keep) / len(chunk) * 100 if len(chunk) > 0 else 0
    pct_ocean = len(ocean_keep) / len(chunk) * 100 if len(chunk) > 0 else 0

    if USE_LAKES:
        print(
            f"Land: {len(land_keep):,} ({pct_land:.1f}%), "
            f"Lakes: {len(lakes_keep):,} ({pct_lakes:.1f}%), "
            f"Ocean: {len(ocean_keep):,} ({pct_ocean:.1f}%)",
            flush=True
        )
    else:
        print(
            f"Land: {len(land_keep):,} ({pct_land:.1f}%), "
            f"Ocean: {len(ocean_keep):,} ({pct_ocean:.1f}%)",
            flush=True
        )

    total_processed += len(chunk)

print("\n" + "=" * 80, flush=True)
print("COMBINING RESULTS", flush=True)
print("=" * 80, flush=True)

print("\nCombining land chunks...", flush=True)
df_land_only = pd.concat(land_chunks, ignore_index=True) if land_chunks else pd.DataFrame()
print("Combining lakes chunks...", flush=True)
df_lakes_only = pd.concat(lakes_chunks, ignore_index=True) if (USE_LAKES and lakes_chunks) else pd.DataFrame()
print("Combining ocean chunks...", flush=True)
df_ocean_only = pd.concat(ocean_chunks, ignore_index=True) if ocean_chunks else pd.DataFrame()

# ============================================================================
# ELEVATION FILTERING SUMMARY (DELETION) + BASIC STATS
# ============================================================================
print("\n" + "=" * 80, flush=True)
print("ELEVATION FILTERING SUMMARY", flush=True)
print("=" * 80, flush=True)

total_deleted = sum(deleted_photons.values())
pct_deleted = 100.0 * total_deleted / total_input if total_input > 0 else 0.0

print(f"\nTotal input photons: {total_input:,}", flush=True)
print(f"Total output photons after filtering: {len(df_land_only) + len(df_lakes_only) + len(df_ocean_only):,}", flush=True)
print(f"Total DELETED (failed elevation check): {total_deleted:,} ({pct_deleted:.2f}%)", flush=True)

print(f"\nDeletions by category:", flush=True)
print(f" LAND photons below {LAND_MIN_ELEV:.1f} m: {deleted_photons['LAND_below_threshold']:,}", flush=True)
print(f" LAKE photons below {LAKE_MIN_ELEV:.1f} m: {deleted_photons['LAKE_below_threshold']:,}", flush=True)
print(f" OCEAN photons above {OCEAN_MAX_ELEV:.1f} m: {deleted_photons['OCEAN_above_threshold']:,}", flush=True)

# Save to FILTERING_FILE, preserving 45's style
with open(FILTERING_FILE, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("ELEVATION FILTERING SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Run ID: {RUN_TIMESTAMP}\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("ELEVATION THRESHOLDS:\n")
    f.write(f" LAND: must be >= {LAND_MIN_ELEV}m MSL\n")
    f.write(f" LAKES: must be >= {LAKE_MIN_ELEV}m MSL\n")
    f.write(f" OCEAN: must be <= {OCEAN_MAX_ELEV}m MSL\n\n")
    f.write(f"Total photons processed: {total_input:,}\n")
    f.write(f"Total photons DELETED: {total_deleted:,} ({pct_deleted:.2f}%)\n\n")
    f.write("Deletions by category:\n")
    f.write(f" LAND below {LAND_MIN_ELEV}m: {deleted_photons['LAND_below_threshold']:,}\n")
    f.write(f" LAKE below {LAKE_MIN_ELEV}m: {deleted_photons['LAKE_below_threshold']:,}\n")
    f.write(f" OCEAN above {OCEAN_MAX_ELEV}m: {deleted_photons['OCEAN_above_threshold']:,}\n\n")
    f.write("Retained photons:\n")
    f.write(f" LAND: {len(df_land_only):,}\n")
    f.write(f" LAKES: {len(df_lakes_only):,}\n")
    f.write(f" OCEAN: {len(df_ocean_only):,}\n")
    f.write(f" TOTAL: {len(df_land_only) + len(df_lakes_only) + len(df_ocean_only):,}\n")

print(f"\n✓ Filtering summary saved to {FILTERING_FILE.name}", flush=True)

# ============================================================================
# FINAL CLASSIFICATION SUMMARY (46-style, but with deletion)
# ============================================================================
print("\n" + "=" * 80, flush=True)
print("FINAL CLASSIFICATION SUMMARY (AFTER FILTERING)", flush=True)
print("=" * 80 + "\n", flush=True)

total_land = len(df_land_only)
total_lakes = len(df_lakes_only)
total_ocean = len(df_ocean_only)
total_output = total_land + total_lakes + total_ocean

print(f"Total input photons: {total_input:,}", flush=True)
print(f"Total output photons: {total_output:,}\n", flush=True)

if total_output > 0:
    print(f"LAND photons: {total_land:>10,} ({100 * total_land / total_output:.2f}%)", flush=True)
    print(f"LAKES photons: {total_lakes:>10,} ({100 * total_lakes / total_output:.2f}%)", flush=True)
    print(f"OCEAN photons: {total_ocean:>10,} ({100 * total_ocean / total_output:.2f}%)", flush=True)

# Elevation stats if possible
elev_col_global = None
for col in ['Height (m MSL)', 'height', 'Height', 'elevation', 'h_li']:
    if col in df_land_only.columns or col in df_lakes_only.columns or col in df_ocean_only.columns:
        elev_col_global = col
        break

if elev_col_global is not None:
    if total_land > 0:
        print(f"\nLAND elevation range: {df_land_only[elev_col_global].min():.2f} to {df_land_only[elev_col_global].max():.2f} m", flush=True)
        print(f"LAND mean: {df_land_only[elev_col_global].mean():.2f} m", flush=True)
    if total_lakes > 0:
        print(f"\nLAKES elevation range: {df_lakes_only[elev_col_global].min():.2f} to {df_lakes_only[elev_col_global].max():.2f} m", flush=True)
        print(f"LAKES mean: {df_lakes_only[elev_col_global].mean():.2f} m", flush=True)
    if total_ocean > 0:
        print(f"\nOCEAN elevation range: {df_ocean_only[elev_col_global].min():.2f} to {df_ocean_only[elev_col_global].max():.2f} m", flush=True)
        print(f"OCEAN mean: {df_ocean_only[elev_col_global].mean():.2f} m", flush=True)

print(f"\n✓ Conservation check: {total_input:,} -> {total_output + total_deleted:,} (output + deleted)", flush=True)

# ============================================================================
# CREATE UNIFIED CLASSIFIED DATASET WITH YEAR COLUMN
# ============================================================================
print("\nCreating unified classified dataset...", flush=True)

if total_land > 0:
    df_land_only['classification'] = 'LAND'
if total_lakes > 0:
    df_lakes_only['classification'] = 'LAKE'
if total_ocean > 0:
    df_ocean_only['classification'] = 'OCEAN'

all_datasets = []
if total_land > 0:
    all_datasets.append(df_land_only)
if total_lakes > 0:
    all_datasets.append(df_lakes_only)
if total_ocean > 0:
    all_datasets.append(df_ocean_only)

if all_datasets:
    df_all_classified = pd.concat(all_datasets, ignore_index=True)

    # Extract year from date column
    date_col = None
    for possible_col in ['Date', 'date', 'datetime', 'Date/Time', 'acquisition_date']:
        if possible_col in df_all_classified.columns:
            date_col = possible_col
            break

    if date_col:
        print(f"Extracting year from '{date_col}' column...", flush=True)
        df_all_classified['year'] = pd.to_datetime(df_all_classified[date_col]).dt.year
    else:
        print("WARNING: No date column found for year extraction", flush=True)
        df_all_classified['year'] = None
else:
    df_all_classified = pd.DataFrame()
    print("No data after filtering!", flush=True)

# ============================================================================
# YEAR-BY-YEAR CLASSIFICATION COUNTS (46-style with percentages)
# ============================================================================
print("\n" + "=" * 80, flush=True)
print("COMPUTING YEAR-BY-YEAR CLASSIFICATION COUNTS", flush=True)
print("=" * 80, flush=True)

if len(df_all_classified) > 0 and 'year' in df_all_classified.columns and df_all_classified['year'].notna().any():
    year_class_counts = df_all_classified.groupby(['year', 'classification']).size().unstack(fill_value=0)

    for col in ['LAND', 'LAKE', 'OCEAN']:
        if col not in year_class_counts.columns:
            year_class_counts[col] = 0

    year_class_counts = year_class_counts[['LAND', 'LAKE', 'OCEAN']]
    year_class_counts['TOTAL'] = year_class_counts.sum(axis=1)

    # Console counts
    print("\nClassification counts by year:")
    print("-" * 80)
    print(f"{'Year':<8} {'LAND':>12} {'LAKE':>12} {'OCEAN':>12} {'TOTAL':>12}")
    print("=" * 80)
    for year in sorted(year_class_counts.index):
        land_ct = int(year_class_counts.loc[year, 'LAND'])
        lake_ct = int(year_class_counts.loc[year, 'LAKE'])
        ocean_ct = int(year_class_counts.loc[year, 'OCEAN'])
        total_ct = int(year_class_counts.loc[year, 'TOTAL'])
        print(f"{int(year):<8} {land_ct:>12,} {lake_ct:>12,} {ocean_ct:>12,} {total_ct:>12,}")
    print("-" * 80)
    total_land_counts = int(year_class_counts['LAND'].sum())
    total_lake_counts = int(year_class_counts['LAKE'].sum())
    total_ocean_counts = int(year_class_counts['OCEAN'].sum())
    grand_total_counts = int(year_class_counts['TOTAL'].sum())
    print(f"{'TOTAL':<8} {total_land_counts:>12,} {total_lake_counts:>12,} {total_ocean_counts:>12,} {grand_total_counts:>12,}")
    print("=" * 80)

    # Percentages
    print("\nClassification percentages by year:")
    print("-" * 80)
    print(f"{'Year':<8} {'LAND %':>12} {'LAKE %':>12} {'OCEAN %':>12}")
    print("=" * 80)
    for year in sorted(year_class_counts.index):
        total_year = year_class_counts.loc[year, 'TOTAL']
        if total_year > 0:
            land_pct = year_class_counts.loc[year, 'LAND'] / total_year * 100
            lake_pct = year_class_counts.loc[year, 'LAKE'] / total_year * 100
            ocean_pct = year_class_counts.loc[year, 'OCEAN'] / total_year * 100
            print(f"{int(year):<8} {land_pct:>11.2f}% {lake_pct:>11.2f}% {ocean_pct:>11.2f}%")
    print("=" * 80)

    # Save to COUNTS_FILE (46-style)
    print(f"\nSaving year-by-year counts to: {COUNTS_FILE.name}", flush=True)
    with open(COUNTS_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SCRIPT 02: TERRAIN CLASSIFICATION COUNTS BY YEAR\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Run ID: {RUN_TIMESTAMP}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("Classification counts by year:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Year':<8} {'LAND':>12} {'LAKE':>12} {'OCEAN':>12} {'TOTAL':>12}\n")
        f.write("=" * 80 + "\n")
        for year in sorted(year_class_counts.index):
            land_ct = int(year_class_counts.loc[year, 'LAND'])
            lake_ct = int(year_class_counts.loc[year, 'LAKE'])
            ocean_ct = int(year_class_counts.loc[year, 'OCEAN'])
            total_ct = int(year_class_counts.loc[year, 'TOTAL'])
            f.write(f"{int(year):<8} {land_ct:>12,} {lake_ct:>12,} {ocean_ct:>12,} {total_ct:>12,}\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'TOTAL':<8} {total_land_counts:>12,} {total_lake_counts:>12,} {total_ocean_counts:>12,} {grand_total_counts:>12,}\n")
        f.write("=" * 80 + "\n\n")

        f.write("Classification percentages by year:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Year':<8} {'LAND %':>12} {'LAKE %':>12} {'OCEAN %':>12}\n")
        f.write("=" * 80 + "\n")
        for year in sorted(year_class_counts.index):
            total_year = year_class_counts.loc[year, 'TOTAL']
            if total_year > 0:
                land_pct = year_class_counts.loc[year, 'LAND'] / total_year * 100
                lake_pct = year_class_counts.loc[year, 'LAKE'] / total_year * 100
                ocean_pct = year_class_counts.loc[year, 'OCEAN'] / total_year * 100
                f.write(f"{int(year):<8} {land_pct:>11.2f}% {lake_pct:>11.2f}% {ocean_pct:>11.2f}%\n")
        f.write("=" * 80 + "\n")

    print(f"✓ Year-by-year counts saved to: {COUNTS_FILE.name}", flush=True)
else:
    print("\nWARNING: Cannot compute year-by-year counts (no date column found or no data)", flush=True)

# ============================================================================
# SAVE OUTPUT FILES (46-style, with backup and LAND as default)
# ============================================================================
print("\n" + "=" * 80, flush=True)
print("SAVING OUTPUT FILES", flush=True)
print("=" * 80 + "\n", flush=True)

# Backup original filtered file
print(f"1. Backing up original: {BACKUP_FILE.name}")
if not BACKUP_FILE.exists():
    INPUT_FILE.rename(BACKUP_FILE)
else:
    print(" (already exists, skipping)")

# Save LAND
if total_land > 0:
    df_land_only.to_csv(LAND_FILE, index=False, encoding='utf-8')
    size_mb = LAND_FILE.stat().st_size / (1024 ** 2)
    print(f"2. LAND refined: {LAND_FILE.name} ({size_mb:.1f} MB)")
# Save LAKES
if total_lakes > 0:
    df_lakes_only.to_csv(LAKES_FILE, index=False, encoding='utf-8')
    size_mb = LAKES_FILE.stat().st_size / (1024 ** 2)
    print(f"3. LAKES refined: {LAKES_FILE.name} ({size_mb:.1f} MB)")
# Save OCEAN
if total_ocean > 0:
    df_ocean_only.to_csv(OCEAN_FILE, index=False, encoding='utf-8')
    size_mb = OCEAN_FILE.stat().st_size / (1024 ** 2)
    print(f"4. OCEAN refined: {OCEAN_FILE.name} ({size_mb:.1f} MB)")

# Save unified classified dataset
if len(df_all_classified) > 0:
    df_all_classified.to_csv(CLASSIFIED_FILE, index=False, encoding='utf-8')
    size_mb = CLASSIFIED_FILE.stat().st_size / (1024 ** 2)
    print(f"5. Unified dataset: {CLASSIFIED_FILE.name} ({size_mb:.1f} MB)")

# Set LAND as default input for later scripts
if total_land > 0:
    df_land_only.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"6. Default for pipeline: {OUTPUT_FILE.name}")

print("\n" + "=" * 80, flush=True)
print("SCRIPT 02 COMPLETE", flush=True)
print("=" * 80, flush=True)

print("\n✓ Output files created:")
print(f" • {LAND_FILE.name} (elevation-verified)")
print(f" • {LAKES_FILE.name} (elevation-verified)")
print(f" • {OCEAN_FILE.name} (elevation-verified)")
print(f" • {CLASSIFIED_FILE.name} (unified)")
print(f" • {COUNTS_FILE.name} (year-by-year)")
print(f" • {FILTERING_FILE.name} (filtering summary)")

print("\nNext steps:")
print(" • Scripts 03-07 will process LAND data automatically")
print(" • Review 02_filtering_summary.txt for elevation filtering details")
print(" • Use 02_photon_counts_by_year_class.txt for temporal analysis")
print(" • Use icesat2_terrain_classified.csv for downstream analysis")

print("\n" + "=" * 80, flush=True)
