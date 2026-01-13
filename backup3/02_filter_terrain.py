"""
SCRIPT 02: TERRAIN CLASSIFICATION (LAND / OCEAN / LAKES)

Purpose
-------
Classify filtered ground photons as land, ocean or lakes using national
boundary and hydrology datasets. This separates terrestrial subsidence
signals from marine and lake environments.

Origin / Consolidation
----------------------
- Renamed from: 01b_filter_land_lakes_ocean.py
- No additional scripts consolidated here.

Main Tasks
----------
- Load `icesat2_filtered_ground_photons.csv` (output of Script 01).
- Load Canadian land boundary polygons and CanVec (or equivalent)
  waterbody polygons.
- Perform spatial joins between photon locations and GIS layers.
- Assign each photon to exactly one of:
  - LAND   (on land polygon, not in lake polygons)
  - OCEAN  (outside land polygon)
  - LAKES  (inside waterbody polygon)
- Perform a conservation check so that LAND + OCEAN + LAKES counts
  equal the input photon count.

Inputs
------
- icesat2_filtered_ground_photons.csv
- CANADA_LAND_SHAPEFILE  (landmass polygons)
- CANVEC_WATERBODY_SHAPEFILE  (lake/waterbody polygons)

Outputs
-------
- icesat2_filtered_ground_photons_LAND.csv
- icesat2_filtered_ground_photons_OCEAN.csv
- icesat2_filtered_ground_photons_LAKES.csv

Reproducibility Notes
---------------------
Classification depends on the specific GIS datasets used (CanVec
version, projection, shoreline detail). Record dataset names, versions
and sources in the thesis so that the land/water split can be
reproduced.
"""


import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import sys
import io

# UTF-8 encoding for Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_OUTPUT_DIR = Path(r'E:\Processing\ICESat-2\output_data')

# Canadian Government Data Layers
CANADA_LAND_SHAPEFILE = Path(r'E:\GIS\lpr_000b21a_e\lpr_000b21a_e.shp')
CANVEC_WATERBODY_SHAPEFILE = Path(r'E:\GIS\canvec_50K_NT_Hydro_shp\waterbody_2_2.shp')

# Verify files exist
if not CANADA_LAND_SHAPEFILE.exists():
    print("="*80)
    print("ERROR: Statistics Canada land boundary not found!")
    print(f"Expected: {CANADA_LAND_SHAPEFILE}")
    sys.exit(1)

if not CANVEC_WATERBODY_SHAPEFILE.exists():
    print("="*80)
    print("WARNING: CanVec waterbody layer not found!")
    print(f"Expected: {CANVEC_WATERBODY_SHAPEFILE}")
    print("Proceeding with land/ocean separation only (no lake filtering)")
    USE_LAKES = False
else:
    USE_LAKES = True

print("="*80, flush=True)
print("SCRIPT 02: LAND/WATER/LAKES SEPARATION", flush=True)
print("="*80, flush=True)

# ============================================================================
# LOAD LATEST RUN
# ============================================================================

timestamp_file = BASE_OUTPUT_DIR / 'latest_run_timestamp.txt'
with open(timestamp_file, 'r') as f:
    RUN_TIMESTAMP = f.read().strip()

OUTPUT_DIR = BASE_OUTPUT_DIR / f'run_{RUN_TIMESTAMP}'
INPUT_FILE = OUTPUT_DIR / 'icesat2_filtered_ground_photons.csv'
BACKUP_FILE = OUTPUT_DIR / 'icesat2_filtered_ground_photons_ORIGINAL.csv'
LAND_FILE = OUTPUT_DIR / 'icesat2_filtered_ground_photons_LAND.csv'
LAKES_FILE = OUTPUT_DIR / 'icesat2_filtered_ground_photons_LAKES.csv'
OCEAN_FILE = OUTPUT_DIR / 'icesat2_filtered_ground_photons_OCEAN.csv'
OUTPUT_FILE = INPUT_FILE  # Will be replaced with LAND version

print(f"Run ID: {RUN_TIMESTAMP}", flush=True)
print(f"Input file: {INPUT_FILE.name}", flush=True)
print(f"Canada land boundary: {CANADA_LAND_SHAPEFILE.name}", flush=True)
if USE_LAKES:
    print(f"CanVec waterbodies: {CANVEC_WATERBODY_SHAPEFILE.name}", flush=True)
print("="*80, flush=True)

# Check if input file exists
if not INPUT_FILE.exists():
    print(f"ERROR: Input file not found: {INPUT_FILE}", flush=True)
    sys.exit(1)

# Get file size
file_size_mb = INPUT_FILE.stat().st_size / (1024**2)
print(f"\nInput file size: {file_size_mb:.1f} MB", flush=True)

# ============================================================================
# LOAD CANADIAN BOUNDARY LAYERS
# ============================================================================

print("\n" + "="*80, flush=True)
print("LOADING CANADIAN GOVERNMENT DATA LAYERS", flush=True)
print("="*80, flush=True)

print("\n1. Loading Statistics Canada land boundary...", flush=True)
canada_land = gpd.read_file(CANADA_LAND_SHAPEFILE)
print(f"   Loaded {len(canada_land)} province/territory boundaries", flush=True)
print(f"   CRS: {canada_land.crs}", flush=True)

if USE_LAKES:
    print("\n2. Loading CanVec waterbodies...", flush=True)
    waterbodies = gpd.read_file(CANVEC_WATERBODY_SHAPEFILE)
    print(f"   Loaded {len(waterbodies)} waterbody features", flush=True)
    print(f"   CRS: {waterbodies.crs}", flush=True)

# Detect UTM zone from ICESat-2 data
print("\n3. Detecting UTM zone from ICESat-2 data...", flush=True)
first_chunk = pd.read_csv(INPUT_FILE, nrows=1000)
utm_zone = int(first_chunk['UTM Zone'].mode()[0])
utm_hemisphere = first_chunk['UTM Hemisphere'].mode()[0]

if utm_hemisphere == 'N':
    utm_epsg = 32600 + utm_zone
else:
    utm_epsg = 32700 + utm_zone

print(f"   Detected: UTM Zone {utm_zone}{utm_hemisphere} (EPSG:{utm_epsg})", flush=True)

# Reproject boundary layers to UTM
print("\n4. Reprojecting boundary layers to UTM...", flush=True)
canada_land_utm = canada_land.to_crs(epsg=utm_epsg)
print(f"   Canada land boundary reprojected", flush=True)

if USE_LAKES:
    # Check if waterbodies need reprojection
    if waterbodies.crs.to_epsg() != utm_epsg:
        print(f"   Reprojecting waterbodies...", flush=True)
        waterbodies_utm = waterbodies.to_crs(epsg=utm_epsg)
    else:
        waterbodies_utm = waterbodies
    print(f"   Waterbodies ready", flush=True)

print("\n[OK] All boundary layers ready", flush=True)

# ============================================================================
# PROCESS PHOTONS IN CHUNKS - CLASSIFY AS LAND/LAKES/OCEAN
# ============================================================================

print("\n" + "="*80, flush=True)
print("CLASSIFYING PHOTONS: LAND / LAKES / OCEAN", flush=True)
print("="*80, flush=True)
print("Processing in chunks (this may take 10-20 minutes)...\n", flush=True)

chunk_size = 500000  # 500k rows per chunk
land_chunks = []
lakes_chunks = []
ocean_chunks = []
total_input = 0
total_land = 0
total_lakes = 0
total_ocean = 0
chunk_num = 0

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
    
    # STEP 1: Identify points within Canada land boundary (excludes ocean)
    gdf_with_land = gpd.sjoin(
        gdf_chunk, 
        canada_land_utm, 
        how='left',  # Keep all points
        predicate='within'
    )
    
    # Points on land (not ocean)
    on_land_mask = gdf_with_land['index_right'].notna()
    
    # STEP 2: If using lakes layer, separate land from inland water
    if USE_LAKES:
        # Get points that are on land (not ocean)
        gdf_land_candidates = gdf_with_land[on_land_mask].copy()
        
        # Check which of these are in waterbodies (lakes/rivers)
        gdf_with_lakes = gpd.sjoin(
            gdf_land_candidates.drop(columns=['index_right']),
            waterbodies_utm,
            how='left',
            predicate='within'
        )
        
        in_lake_mask = gdf_with_lakes['index_right'].notna()
        
        # Final classification:
        # - In waterbody = LAKES
        # - On land but not in waterbody = LAND
        # - Not on land = OCEAN
        
        gdf_lakes = gdf_with_lakes[in_lake_mask].copy()
        gdf_land = gdf_with_lakes[~in_lake_mask].copy()
        gdf_ocean = gdf_with_land[~on_land_mask].copy()
        
    else:
        # Without lakes layer: only land vs ocean
        gdf_land = gdf_with_land[on_land_mask].copy()
        gdf_lakes = pd.DataFrame()  # Empty
        gdf_ocean = gdf_with_land[~on_land_mask].copy()
    
    # Clean up join artifacts
    for df in [gdf_land, gdf_lakes, gdf_ocean]:
        if len(df) > 0 and 'index_right' in df.columns:
            df.drop(columns=['index_right'], inplace=True)
    
    # Remove geometry column and convert to DataFrame
    df_land = pd.DataFrame(gdf_land.drop(columns=['geometry'])) if len(gdf_land) > 0 else pd.DataFrame()
    df_lakes = pd.DataFrame(gdf_lakes.drop(columns=['geometry'])) if len(gdf_lakes) > 0 else pd.DataFrame()
    df_ocean = pd.DataFrame(gdf_ocean.drop(columns=['geometry'])) if len(gdf_ocean) > 0 else pd.DataFrame()
    
    # Append to collections
    if len(df_land) > 0:
        land_chunks.append(df_land)
    if len(df_lakes) > 0:
        lakes_chunks.append(df_lakes)
    if len(df_ocean) > 0:
        ocean_chunks.append(df_ocean)
    
    total_land += len(df_land)
    total_lakes += len(df_lakes)
    total_ocean += len(df_ocean)
    
    # Print progress
    pct_land = len(df_land) / len(chunk) * 100 if len(chunk) > 0 else 0
    pct_lakes = len(df_lakes) / len(chunk) * 100 if len(chunk) > 0 else 0
    pct_ocean = len(df_ocean) / len(chunk) * 100 if len(chunk) > 0 else 0
    
    if USE_LAKES:
        print(f"Land: {len(df_land):,} ({pct_land:.1f}%), Lakes: {len(df_lakes):,} ({pct_lakes:.1f}%), Ocean: {len(df_ocean):,} ({pct_ocean:.1f}%)", flush=True)
    else:
        print(f"Land: {len(df_land):,} ({pct_land:.1f}%), Ocean: {len(df_ocean):,} ({pct_ocean:.1f}%)", flush=True)

# ============================================================================
# COMBINE RESULTS
# ============================================================================

print("\n" + "="*80, flush=True)
print("COMBINING RESULTS", flush=True)
print("="*80, flush=True)

print("\nCombining land chunks...", flush=True)
df_land_only = pd.concat(land_chunks, ignore_index=True) if land_chunks else pd.DataFrame()

if USE_LAKES and lakes_chunks:
    print("Combining lakes chunks...", flush=True)
    df_lakes_only = pd.concat(lakes_chunks, ignore_index=True)
else:
    df_lakes_only = pd.DataFrame()

print("Combining ocean chunks...", flush=True)
df_ocean_only = pd.concat(ocean_chunks, ignore_index=True) if ocean_chunks else pd.DataFrame()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80, flush=True)
print("CLASSIFICATION SUMMARY", flush=True)
print("="*80, flush=True)

print(f"\nTotal input photons:       {total_input:,}", flush=True)
print(f"LAND photons:              {total_land:,} ({total_land/total_input*100:.2f}%)", flush=True)
if USE_LAKES:
    print(f"LAKES photons:             {total_lakes:,} ({total_lakes/total_input*100:.2f}%)", flush=True)
print(f"OCEAN photons:             {total_ocean:,} ({total_ocean/total_input*100:.2f}%)", flush=True)

# Verify totals
total_classified = total_land + total_lakes + total_ocean
if total_classified != total_input:
    print(f"\nWARNING: Total classified ({total_classified:,}) != input ({total_input:,})", flush=True)
    print(f"  Difference: {total_input - total_classified:,} photons", flush=True)

# Additional statistics
if total_land > 0:
    print(f"\nLAND dataset statistics:", flush=True)
    print(f"  Elevation range (MSL):   {df_land_only['Height (m MSL)'].min():.2f} to {df_land_only['Height (m MSL)'].max():.2f} m", flush=True)
    print(f"  Mean elevation:          {df_land_only['Height (m MSL)'].mean():.2f} m", flush=True)

if total_lakes > 0:
    print(f"\nLAKES dataset statistics:", flush=True)
    print(f"  Elevation range (MSL):   {df_lakes_only['Height (m MSL)'].min():.2f} to {df_lakes_only['Height (m MSL)'].max():.2f} m", flush=True)
    print(f"  Mean elevation:          {df_lakes_only['Height (m MSL)'].mean():.2f} m", flush=True)

if total_ocean > 0:
    print(f"\nOCEAN dataset statistics:", flush=True)
    print(f"  Elevation range (MSL):   {df_ocean_only['Height (m MSL)'].min():.2f} to {df_ocean_only['Height (m MSL)'].max():.2f} m", flush=True)
    print(f"  Mean elevation:          {df_ocean_only['Height (m MSL)'].mean():.2f} m", flush=True)

# ============================================================================
# SAVE FILES
# ============================================================================

print("\n" + "="*80, flush=True)
print("SAVING OUTPUT FILES", flush=True)
print("="*80, flush=True)

# 1. Backup original
print(f"\n1. Backing up original to:", flush=True)
print(f"   {BACKUP_FILE.name}", flush=True)
if BACKUP_FILE.exists():
    print("   (backup already exists, skipping)", flush=True)
else:
    INPUT_FILE.rename(BACKUP_FILE)

# 2. Save LAND dataset
if total_land > 0:
    print(f"\n2. Saving LAND photons to:", flush=True)
    print(f"   {LAND_FILE.name}", flush=True)
    df_land_only.to_csv(LAND_FILE, index=False)
    land_size_mb = LAND_FILE.stat().st_size / (1024**2)
    print(f"   File size: {land_size_mb:.1f} MB", flush=True)

# 3. Save LAKES dataset
if total_lakes > 0:
    print(f"\n3. Saving LAKES photons to:", flush=True)
    print(f"   {LAKES_FILE.name}", flush=True)
    df_lakes_only.to_csv(LAKES_FILE, index=False)
    lakes_size_mb = LAKES_FILE.stat().st_size / (1024**2)
    print(f"   File size: {lakes_size_mb:.1f} MB", flush=True)

# 4. Save OCEAN dataset
if total_ocean > 0:
    print(f"\n4. Saving OCEAN photons to:", flush=True)
    print(f"   {OCEAN_FILE.name}", flush=True)
    df_ocean_only.to_csv(OCEAN_FILE, index=False)
    ocean_size_mb = OCEAN_FILE.stat().st_size / (1024**2)
    print(f"   File size: {ocean_size_mb:.1f} MB", flush=True)

# 5. Copy LAND version to main filename (for standard pipeline)
if total_land > 0:
    print(f"\n5. Setting LAND dataset as default for pipeline:", flush=True)
    print(f"   {OUTPUT_FILE.name}", flush=True)
    df_land_only.to_csv(OUTPUT_FILE, index=False)
    print(f"   (Pipeline Scripts 02-07 will use LAND-only data)", flush=True)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80, flush=True)
print("SCRIPT 02 COMPLETE", flush=True)
print("="*80, flush=True)

print("\nOutput files created:", flush=True)
print(f"  1. {BACKUP_FILE.name} - Original (all photons)", flush=True)
print(f"  2. {LAND_FILE.name} - True land only ({total_land:,} photons)", flush=True)
if USE_LAKES:
    print(f"  3. {LAKES_FILE.name} - Inland water ({total_lakes:,} photons)", flush=True)
print(f"  4. {OCEAN_FILE.name} - Coastal water ({total_ocean:,} photons)", flush=True)
print(f"  5. {OUTPUT_FILE.name} - LAND (default for pipeline)", flush=True)

print("\nNext steps:", flush=True)
print("  - Scripts 02-07 will process LAND data automatically", flush=True)
print("  - To analyze LAKES separately, run: python 09b_run_water_analysis.py", flush=True)
print("  - To analyze OCEAN separately, manually swap files", flush=True)

print("\n" + "="*80, flush=True)
