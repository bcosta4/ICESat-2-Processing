"""
SCRIPT 03: CORE ANALYSIS (CROSSOVER DETECTION + INSAR INTEGRATION)

Purpose
-------
Detect repeat-track crossover points from land photons, compute
elevation change and subsidence rates, and integrate InSAR-derived
subsidence for independent validation.

Origin / Consolidation
----------------------
- Consolidates:
  - 02_crossover_analysis.py
  - 03_insar_integration.py
- Implemented as two major sections:
  Section 1 = Crossover detection and rate estimation.
  Section 2 = InSAR sampling and validation metrics.

Main Tasks – Section 1 (Crossover Analysis)
-------------------------------------------
- Load land photons from icesat2_filtered_ground_photons_LAND.csv.
- Build a cKDTree of photon coordinates for efficient spatial search.
- Identify crossover photon pairs within a specified horizontal
  distance (e.g. 2.0 m).
- Apply temporal filters according to selected strategy:
  - e.g. 10–365 days, seasonal thaw period, or custom range.
- Compute:
  - elevation change (dH, in metres)
  - time interval in years
  - subsidence rate dH_rate_m_per_year.
- Remove outliers using a 3-sigma criterion on dH or rate.
- Enforce consistent date ordering (older vs newer epoch).
- Save per-pair results to CSV.

Main Tasks – Section 2 (InSAR Integration)
------------------------------------------
- Discover and load InSAR GeoTIFF rasters from INSAR_DIR.
- Reproject crossover locations to raster CRS when needed.
- Sample InSAR subsidence rate at each crossover location.
- Combine ICESat-2 and InSAR measurements in one table.
- Compute validation statistics:
  - Pearson correlation, R-squared, RMSE, bias.
- Write out a textual summary of InSAR validation.

Inputs
------
- icesat2_filtered_ground_photons_LAND.csv
- InSAR GeoTIFF files in INSAR_DIR.

Outputs
-------
- icesat2_repeat_track_matches.csv
  (crossover pairs with dH, dt, and dh_rate_m_per_year)
- icesat2_insar_integrated.csv
  (crossover pairs with InSAR values and differences)
- insar_integration_summary.txt
  (validation metrics and summary statistics)

Reproducibility Notes
---------------------
Temporal filters, distance thresholds and outlier criteria determine
which pairs contribute to subsidence estimates. Document chosen
parameters and InSAR products (sensor, acquisition years, processing
chain) so that the ICESat-2 vs InSAR comparison can be repeated.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import sys
import argparse
import time
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("SCRIPT 03: CORE ANALYSIS - CROSSOVER DETECTION & InSAR INTEGRATION")
print("=" * 80)

# ============================================================================
# PARSE COMMAND-LINE ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(
    description="ICESat-2 Core Crossover Analysis with Temporal Filtering"
)
parser.add_argument(
    '--temporal',
    type=str,
    choices=['1', '2', '3', '4'],
    default='2',
    help='Temporal filtering: 1=June-Oct, 2=June-Sept (default), 3=Early vs Late, 4=Same-month-only'
)
args = parser.parse_args()

TEMPORAL_OPTION = args.temporal

TEMPORAL_OPTIONS = {
    '1': 'June-October (Broad summer window)',
    '2': 'June-September (Standard thaw season)',
    '3': 'Early vs Late Summer (June-July vs Aug-Sept)',
    '4': 'Same-month-only (Strictest)',
}

print(f"\nTemporal option: {TEMPORAL_OPTION} - {TEMPORAL_OPTIONS[TEMPORAL_OPTION]}")

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_OUTPUT_DIR = Path(r'E:\Processing\ICeSat-2\output_data')
INSAR_DIR = Path(r'C:\final_products')

CROSSOVER_DISTANCE_THRESHOLD = 2.0  # meters
TIME_SEPARATION_THRESHOLD = 10  # days minimum

# Get latest run timestamp
timestamp_file = BASE_OUTPUT_DIR / 'latest_run_timestamp.txt'

try:
    with open(timestamp_file, 'r') as f:
        RUN_TIMESTAMP = f.read().strip()
    OUTPUT_DIR = BASE_OUTPUT_DIR / f'run_{RUN_TIMESTAMP}'
except FileNotFoundError:
    print("ERROR: Timestamp file not found. Run Script 00 or 01 first.")
    sys.exit(1)

print(f"Run timestamp: {RUN_TIMESTAMP}")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 80)

# ============================================================================
# SECTION 1: CROSSOVER DETECTION
# ============================================================================

print("\nPART 1/2: CROSSOVER DETECTION")
print("-" * 80)

# Load filtered land photons
INPUT_FILE = OUTPUT_DIR / 'icesat2_filtered_ground_photons_LAND.csv'

if not INPUT_FILE.exists():
    # Try fallback
    INPUT_FILE = OUTPUT_DIR / 'icesat2_filtered_ground_photons.csv'

if not INPUT_FILE.exists():
    print(f"ERROR: No input file found!")
    print(f"  Tried: {OUTPUT_DIR / 'icesat2_filtered_ground_photons_LAND.csv'}")
    print(f"  Tried: {OUTPUT_DIR / 'icesat2_filtered_ground_photons.csv'}")
    sys.exit(1)

print(f"Loading {INPUT_FILE.name}...")
photons = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(photons)} filtered photons")

# Parse datetime
date_cols = [c for c in photons.columns if 'datetime' in c.lower() or 'date' in c.lower()]

if date_cols:
    photons['datetime'] = pd.to_datetime(photons[date_cols[0]])
else:
    if 'year' in photons.columns:
        print("WARNING: Using year column as datetime proxy (no full datetime available)")
        photons['datetime'] = pd.to_datetime(photons['year'], format='%Y')
    else:
        print("ERROR: No datetime column found")
        sys.exit(1)

photons['month'] = photons['datetime'].dt.month
photons['year'] = photons['datetime'].dt.year

# Find coordinate columns
lat_col = next((c for c in photons.columns if 'latitude' in c.lower()), None)
lon_col = next((c for c in photons.columns if 'longitude' in c.lower()), None)
elev_col = next((c for c in photons.columns if 'height' in c.lower() or 'elevation' in c.lower()), None)

if not all([lat_col, lon_col, elev_col]):
    print(f"ERROR: Could not find required columns")
    print(f"  Latitude: {lat_col}")
    print(f"  Longitude: {lon_col}")
    print(f"  Elevation: {elev_col}")
    print(f"  Available columns: {list(photons.columns)}")
    sys.exit(1)

print(f"Column mapping:")
print(f"  Latitude: {lat_col}")
print(f"  Longitude: {lon_col}")
print(f"  Elevation: {elev_col}")

# ============================================================================
# Apply temporal filtering
# ============================================================================

print("\n" + "=" * 80)
print(f"TEMPORAL FILTERING: {TEMPORAL_OPTIONS[TEMPORAL_OPTION]}")
print("=" * 80)

if TEMPORAL_OPTION == '1':
    # June-October
    data = photons[(photons['month'] >= 6) & (photons['month'] <= 10)].copy()
    print("Months allowed: 6, 7, 8, 9, 10 (June-October)")
elif TEMPORAL_OPTION == '2':
    # June-September
    data = photons[(photons['month'] >= 6) & (photons['month'] <= 9)].copy()
    print("Months allowed: 6, 7, 8, 9 (June-September)")
elif TEMPORAL_OPTION == '3':
    # Early vs Late Summer (June-July vs Aug-Sept)
    data = photons[(photons['month'] >= 6) & (photons['month'] <= 9)].copy()
    print("Months allowed: 6, 7, 8, 9 (Early=June-July, Late=Aug-Sept)")
elif TEMPORAL_OPTION == '4':
    # Same-month-only
    data = photons[(photons['month'] >= 6) & (photons['month'] <= 9)].copy()
    print("Months allowed: 6, 7, 8, 9 (Same-month pairs only)")

n_before = len(photons)
n_after = len(data)
print(f"Photons retained: {n_after}/{n_before} ({100*n_after/n_before:.1f}%)")

# ============================================================================
# Crossover detection function
# ============================================================================

def find_crossovers_optimized(df, option_selected, lat_col, lon_col, elev_col):
    """Find crossovers using optimized cKDTree spatial matching."""
    
    start_time = time.time()
    print("\n" + "=" * 80)
    print("SPATIAL-TEMPORAL CROSSOVER MATCHING")
    print("=" * 80)
    
    # Project to meters (UTM approximation)
    mean_lat = df[lat_col].mean()
    cos_lat = np.cos(np.radians(mean_lat))
    
    y = df[lat_col].values * 111000
    x = df[lon_col].values * 111000 * cos_lat
    
    coords = np.column_stack([x, y])
    
    print(f"Building spatial index (cKDTree)...")
    tree = cKDTree(coords)
    
    print(f"Searching for point pairs within {CROSSOVER_DISTANCE_THRESHOLD}m...")
    pairs = tree.query_pairs(r=CROSSOVER_DISTANCE_THRESHOLD, output_type='ndarray')
    
    print(f"Found {len(pairs)} candidate spatial pairs")
    
    if len(pairs) == 0:
        print("No spatial pairs found!")
        return pd.DataFrame()
    
    print(f"Filtering by temporal criteria...")
    
    # Extract indices
    idx1 = pairs[:, 0]
    idx2 = pairs[:, 1]
    
    # Get datetime values
    t1 = df['datetime'].values[idx1]
    t2 = df['datetime'].values[idx2]
    
    # Calculate time delta
    dt_days = (t2 - t1).astype('timedelta64[D]').astype(int)
    
    # Create time filter
    abs_dt = np.abs(dt_days)
    time_mask = (abs_dt >= TIME_SEPARATION_THRESHOLD) & (abs_dt <= 2000)
    
    # Apply same-month filter if option 4
    if option_selected == '4':
        m1 = df['month'].values[idx1]
        m2 = df['month'].values[idx2]
        time_mask = time_mask & (m1 == m2)
    
    valid_idx = np.where(time_mask)[0]
    idx1_valid = idx1[valid_idx]
    idx2_valid = idx2[valid_idx]
    dt_final = dt_days[valid_idx]
    
    print(f"Valid crossovers after temporal filter: {len(valid_idx)}")
    
    if len(valid_idx) == 0:
        return pd.DataFrame()
    
    # Get elevation change
    h1 = df[elev_col].values[idx1_valid]
    h2 = df[elev_col].values[idx2_valid]
    dh = h2 - h1
    
    dt_years = dt_final / 365.25
    rates = dh / dt_years
    
    # Create results
    results = pd.DataFrame({
        'lon_early': df[lon_col].values[idx1_valid],
        'lat_early': df[lat_col].values[idx1_valid],
        'date_early': df['datetime'].values[idx1_valid],
        'elev_early': h1,
        'lon_late': df[lon_col].values[idx2_valid],
        'lat_late': df[lat_col].values[idx2_valid],
        'date_late': df['datetime'].values[idx2_valid],
        'elev_late': h2,
        'dH': dh,
        'dt_days': dt_final,
        'subsidence_rate_m_per_year': rates,
    })
    
    # Ensure chronological ordering
    mask_swap = results['dt_days'] < 0
    if mask_swap.any():
        # Swap early/late if needed
        temp_lon = results.loc[mask_swap, 'lon_early'].copy()
        temp_lat = results.loc[mask_swap, 'lat_early'].copy()
        temp_date = results.loc[mask_swap, 'date_early'].copy()
        temp_elev = results.loc[mask_swap, 'elev_early'].copy()
        
        results.loc[mask_swap, 'lon_early'] = results.loc[mask_swap, 'lon_late']
        results.loc[mask_swap, 'lat_early'] = results.loc[mask_swap, 'lat_late']
        results.loc[mask_swap, 'date_early'] = results.loc[mask_swap, 'date_late']
        results.loc[mask_swap, 'elev_early'] = results.loc[mask_swap, 'elev_late']
        
        results.loc[mask_swap, 'lon_late'] = temp_lon
        results.loc[mask_swap, 'lat_late'] = temp_lat
        results.loc[mask_swap, 'date_late'] = temp_date
        results.loc[mask_swap, 'elev_late'] = temp_elev
        
        results.loc[mask_swap, 'dt_days'] *= -1
        results.loc[mask_swap, 'dH'] *= -1
    
    # Add average coordinates
    results['lon_avg'] = (results['lon_early'] + results['lon_late']) / 2
    results['lat_avg'] = (results['lat_early'] + results['lat_late']) / 2
    results['year_early'] = pd.to_datetime(results['date_early']).dt.year
    results['year_late'] = pd.to_datetime(results['date_late']).dt.year
    
    elapsed = time.time() - start_time
    print(f"Processing time: {elapsed:.1f} seconds")
    
    return results

# Execute crossover detection
crossovers = find_crossovers_optimized(data, TEMPORAL_OPTION, lat_col, lon_col, elev_col)

print(f"\nTotal crossovers found: {len(crossovers)}")

if len(crossovers) > 0:
    print("\n" + "=" * 80)
    print("OUTLIER REMOVAL (3-SIGMA FILTERING)")
    print("=" * 80)
    
    mean_rate = crossovers['subsidence_rate_m_per_year'].mean()
    std_rate = crossovers['subsidence_rate_m_per_year'].std()
    
    print(f"Mean subsidence rate: {mean_rate:.6f} m/year")
    print(f"Std deviation: {std_rate:.6f} m/year")
    print(f"3-sigma bounds: ({mean_rate - 3*std_rate:.6f}, {mean_rate + 3*std_rate:.6f})")
    
    clean = crossovers[
        (crossovers['subsidence_rate_m_per_year'] >= mean_rate - 3*std_rate) &
        (crossovers['subsidence_rate_m_per_year'] <= mean_rate + 3*std_rate)
    ].copy()
    
    removed = len(crossovers) - len(clean)
    print(f"Removed {removed} outliers")
    print(f"Remaining crossovers: {len(clean)}")
    
    # Save results
    out_file_clean = OUTPUT_DIR / 'icesat2_crossover_points_clean.csv'
    out_file_matches = OUTPUT_DIR / 'icesat2_repeat_track_matches.csv'
    
    clean.to_csv(out_file_clean, index=False)
    clean.to_csv(out_file_matches, index=False)
    
    print(f"Saved: {out_file_clean.name}")
    print(f"Saved: {out_file_matches.name}")
    
    print("\n" + "=" * 80)
    print("CROSSOVER STATISTICS")
    print("=" * 80)
    
    print(f"Count: {len(clean)}")
    print(f"Mean subsidence: {clean['subsidence_rate_m_per_year'].mean():.6f} m/yr")
    print(f"Median subsidence: {clean['subsidence_rate_m_per_year'].median():.6f} m/yr")
    print(f"Min: {clean['subsidence_rate_m_per_year'].min():.6f} m/yr")
    print(f"Max: {clean['subsidence_rate_m_per_year'].max():.6f} m/yr")
    print(f"Std dev: {clean['subsidence_rate_m_per_year'].std():.6f} m/yr")

else:
    print("No crossovers found!")
    clean = pd.DataFrame()
    out_file_matches = OUTPUT_DIR / 'icesat2_repeat_track_matches.csv'
    clean.to_csv(out_file_matches, index=False)

# ============================================================================
# SECTION 2: InSAR INTEGRATION (Optional)
# ============================================================================

print("\n" + "=" * 80)
print("PART 2/2: InSAR INTEGRATION & VALIDATION")
print("=" * 80)

icesat2_matches = clean.copy() if len(clean) > 0 else crossovers.copy()

print("Searching for InSAR GeoTIFF files...")

insar_files = list(INSAR_DIR.glob('*average*mm_per_year.tif')) if INSAR_DIR.exists() else []

if not insar_files:
    print("No InSAR GeoTIFF files found. Saving crossover results only.")
    output_file = OUTPUT_DIR / 'icesat2_insar_integrated.csv'
    icesat2_matches.to_csv(output_file, index=False)
    print(f"Saved: {output_file.name}")

else:
    print(f"Found {len(insar_files)} InSAR GeoTIFF file(s)")
    
    # Load InSAR rasters
    insar_rasters = {}
    
    for f in insar_files:
        try:
            year_str = f.stem.split('_')[0]
            year = int(year_str)
            print(f"  - Year {year}: {f.name}")
            
            import rasterio
            insar_rasters[year] = rasterio.open(f)
        except (ValueError, IndexError):
            print(f"  - WARNING: Could not parse year from {f.name}")
            continue
    
    # Sample InSAR values
    if insar_rasters and len(icesat2_matches) > 0:
        print("\nSampling InSAR values...")
        icesat2_matches['insar_subsidence_rate_mm_yr'] = np.nan
        icesat2_matches['insar_year'] = np.nan
        
        for year in sorted(icesat2_matches['year_early'].unique()):
            year = int(year)
            print(f"  Processing year {year}...", end=' ', flush=True)
            
            if year not in insar_rasters:
                print("no InSAR data")
                continue
            
            raster = insar_rasters[year]
            mask = icesat2_matches['year_early'] == year
            subset = icesat2_matches[mask]
            
            if len(subset) == 0:
                print("no crossovers")
                continue
            
            try:
                import pyproj
                
                in_proj = pyproj.CRS("EPSG:4326")
                out_proj = pyproj.CRS(raster.crs)
                transformer = pyproj.Transformer.from_crs(in_proj, out_proj, always_xy=True)
                
                ix, iy = transformer.transform(subset['lon_avg'].values, subset['lat_avg'].values)
                coords = list(zip(ix, iy))
                
                insar_values = [val[0] for val in raster.sample(coords)]
                icesat2_matches.loc[mask, 'insar_subsidence_rate_mm_yr'] = insar_values
                icesat2_matches.loc[mask, 'insar_year'] = year
                
                print(f"{len(subset)} points extracted")
            
            except Exception as e:
                print(f"error: {str(e)[:30]}...")
                continue
        
        # Close rasters
        for r in insar_rasters.values():
            r.close()
        
        # Validation metrics
        print("\n" + "=" * 80)
        print("VALIDATION METRICS")
        print("=" * 80)
        
        valid_pairs = icesat2_matches[
            icesat2_matches['subsidence_rate_m_per_year'].notna() &
            icesat2_matches['insar_subsidence_rate_mm_yr'].notna()
        ].copy()
        
        print(f"Valid pairs for comparison: {len(valid_pairs)}")
        
        if len(valid_pairs) >= 10:
            is2_rate_mm = valid_pairs['subsidence_rate_m_per_year'].values * 1000
            insar_rate_mm = valid_pairs['insar_subsidence_rate_mm_yr'].values
            
            valid_mask = ~(np.isnan(is2_rate_mm) | np.isnan(insar_rate_mm))
            is2_rate_mm = is2_rate_mm[valid_mask]
            insar_rate_mm = insar_rate_mm[valid_mask]
            
            if len(is2_rate_mm) > 1:
                correlation = np.corrcoef(is2_rate_mm, insar_rate_mm)[0, 1]
                r_squared = correlation ** 2
                rmse = np.sqrt(np.mean((is2_rate_mm - insar_rate_mm) ** 2))
                bias = np.mean(is2_rate_mm - insar_rate_mm)
                
                print(f"\n  ICESat-2 Statistics")
                print(f"    Mean: {is2_rate_mm.mean():.3f} mm/yr")
                print(f"    Median: {np.median(is2_rate_mm):.3f} mm/yr")
                print(f"    Std: {is2_rate_mm.std():.3f} mm/yr")
                
                print(f"\n  InSAR Statistics")
                print(f"    Mean: {insar_rate_mm.mean():.3f} mm/yr")
                print(f"    Median: {np.median(insar_rate_mm):.3f} mm/yr")
                print(f"    Std: {insar_rate_mm.std():.3f} mm/yr")
                
                print(f"\n  Validation Metrics")
                print(f"    Pearson R: {correlation:.4f}")
                print(f"    R-squared: {r_squared:.4f}")
                print(f"    RMSE: {rmse:.3f} mm/yr")
                print(f"    Bias (IS2 - InSAR): {bias:.3f} mm/yr")
        else:
            print("Insufficient valid pairs for correlation analysis")
    
    # Save integrated results
    output_file = OUTPUT_DIR / 'icesat2_insar_integrated.csv'
    icesat2_matches.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file.name}")

print("\n" + "=" * 80)
print("SCRIPT 03 COMPLETE")
print("=" * 80)

sys.exit(0)
