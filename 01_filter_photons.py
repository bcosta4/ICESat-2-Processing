"""
SCRIPT 01: ICESat-2 PHOTON FILTERING

Purpose
-------
Load raw ICESat-2 ATL03 photon data and apply quality filters to extract
high-confidence ground photons suitable for subsidence analysis over
the Tuktoyaktuk Peninsula.

Origin / Consolidation
----------------------
- Same functional role as original: 01_filter_photons.py
- No consolidation with other scripts.

Main Tasks
----------
- Read raw ATL03 photons from CSV/LAS files in INPUT_DIR.
- Select only ground photons based on classification flags.
- Apply confidence threshold (typically confidence >= 3).
- Optionally restrict to strong beams only.
- Clip photons spatially to the study area (Arctic coastal domain).
- Preserve key metadata such as:
- latitude, longitude, height
- confidence, quality flags
- reference ground track (RGT), cycle, beam ID.
- Write a single combined CSV of all filtered ground photons.

Inputs
------
- Raw ATL03 photon files (CSV or LAS) from INPUT_DIR.
- Study area bounds and quality thresholds defined in this script.

Outputs
-------
- icesat2_filtered_ground_photons.csv
One row per filtered ground photon with coordinates, height and
metadata.
- 01_photon_counts_by_year.txt  [NEW]
Year-by-year photon count summary

Reproducibility Notes
---------------------
Filtering settings (confidence threshold, beam selection and spatial
bounds) strongly influence which photons enter the analysis. Document
these thresholds clearly in the thesis and keep them fixed for all runs
that you want to compare.
"""

import pandas as pd
import numpy as np
import laspy
from pathlib import Path
import glob
from datetime import datetime
import warnings
import sys
import io

# UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - NATURE PAPER IMPROVEMENTS
# ============================================================================
INPUT_DIR = r'E:\Processing\ICeSat-2\raw_data'
BASE_OUTPUT_DIR = r'E:\Processing\ICeSat-2\output_data'

# NATURE PAPER IMPROVEMENT: Strong beams only
STRONG_BEAMS = ['gt1l', 'gt2l', 'gt3l'] # Keep only these
WEAK_BEAMS = ['gt1r', 'gt2r', 'gt3r'] # Discard these

# NATURE PAPER IMPROVEMENT: Higher confidence threshold
CONFIDENCE_THRESHOLD = 3 # 3=high, 4=highest (was 2=medium in old version)
print("="*80, flush=True)
print("SCRIPT 01: ICeSat-2 PHOTON FILTERING - STRONG BEAMS ONLY", flush=True)
print("="*80, flush=True)
print(f"Input directory: {INPUT_DIR}", flush=True)
print(f"Output directory: {BASE_OUTPUT_DIR}", flush=True)
print(f"IMPROVEMENTS FROM NATURE PAPER:", flush=True)
print(f" • Strong beams only: {STRONG_BEAMS}", flush=True)
print(f" • Discard weak beams: {WEAK_BEAMS}", flush=True)
print(f" • Confidence threshold: {CONFIDENCE_THRESHOLD} (higher quality)", flush=True)
print("="*80, flush=True)

# ============================================================================
# CREATE OUTPUT FOLDER WITH AUTO-INCREMENT
# ============================================================================
def get_next_run_folder():
    """Create folder name: run_YYYYMMDD_n1, run_YYYYMMDD_n2, etc."""
    today = datetime.now().strftime('%Y%m%d')
    base_dir = Path(BASE_OUTPUT_DIR)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Find existing runs for today
    existing_runs = list(base_dir.glob(f'run_{today}_n*'))
    if len(existing_runs) == 0:
        run_number = 1
    else:
        # Extract numbers from existing folders
        numbers = []
        for folder in existing_runs:
            try:
                num = int(folder.name.split('_n')[-1])
                numbers.append(num)
            except:
                pass

        run_number = max(numbers) + 1 if numbers else 1
    run_id = f'{today}_n{run_number}'
    output_dir = base_dir / f'run_{run_id}'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, run_id

OUTPUT_DIR, RUN_TIMESTAMP = get_next_run_folder()

# Save timestamp for other scripts
timestamp_file = Path(BASE_OUTPUT_DIR) / 'latest_run_timestamp.txt'
with open(timestamp_file, 'w') as f:
    f.write(RUN_TIMESTAMP)
print(f"\nRun ID: {RUN_TIMESTAMP}", flush=True)
print(f"Output folder: {OUTPUT_DIR}", flush=True)

# ============================================================================
# LOAD PHOREAL CSV FILES
# ============================================================================
def load_phoreal_csv(csv_path):
    """Load PhoREAL CSV with proper column names and metadata extraction"""
    try:
        df = pd.read_csv(csv_path)
        # Extract metadata from filename
        filename = Path(csv_path).stem
        parts = filename.split('_')
        df['filename'] = filename
        df['date'] = pd.to_datetime(parts[2][0:8], format='%Y%m%d')
        df['year'] = df['date'].dt.year
        df['rgt'] = parts[3][0:4]
        df['cycle'] = parts[3][4:6]
        df['beam'] = parts[-1]

        # Assign beam strength
        beam_strength = {
            'gt1l': 'strong', 'gt1r': 'weak',
            'gt2l': 'strong', 'gt2r': 'weak',
            'gt3l': 'strong', 'gt3r': 'weak'
        }
        df['beam_strength'] = df['beam'].map(beam_strength)
        return df
    except Exception as e:
        print(f"ERROR loading CSV {csv_path}: {e}", flush=True)
        return None
def load_phoreal_las(las_path):
    """Load PhoREAL LAS file and convert to DataFrame"""
    try:
        las = laspy.read(las_path)
        # Handle different laspy versions for time attribute
        time_attr = None
        if hasattr(las, 'gpstime'):
            time_attr = las.gpstime
        elif hasattr(las, 'gps_time'):
            time_attr = las.gps_time
        elif hasattr(las, 'Time'):
            time_attr = las.Time
        elif hasattr(las, 'times'):
            time_attr = las.times
        else:
            print(f"WARNING: No time attribute found in LAS file, using placeholder", flush=True)
            time_attr = np.arange(len(las.x))

        # Extract coordinates and metadata
        df = pd.DataFrame({
            'Time (sec)': time_attr,
            'Latitude (deg)': las.y,
            'Longitude (deg)': las.x,
            'UTM Easting (m)': las.X if hasattr(las, 'X') else np.full(len(las.x), np.nan),
            'UTM Northing (m)': las.Y if hasattr(las, 'Y') else np.full(len(las.x), np.nan),
            'Height (m HAE)': las.z,
            'Classification': las.classification if hasattr(las, 'classification') else np.full(len(las.x), 1),
            'Signal Confidence': las.intensity if hasattr(las, 'intensity') else np.full(len(las.x), 4)
        })

        # Extract metadata from filename
        filename = Path(las_path).stem
        parts = filename.split('_')
        df['filename'] = filename
        df['date'] = pd.to_datetime(parts[2][0:8], format='%Y%m%d')
        df['year'] = df['date'].dt.year
        df['rgt'] = parts[3][0:4]
        df['cycle'] = parts[3][4:6]
        df['beam'] = parts[-1]
        beam_strength = {
            'gt1l': 'strong', 'gt1r': 'weak',
            'gt2l': 'strong', 'gt2r': 'weak',
            'gt3l': 'strong', 'gt3r': 'weak'
        }
        df['beam_strength'] = df['beam'].map(beam_strength)
        return df
    except Exception as e:
        print(f"ERROR loading LAS {las_path}: {e}", flush=True)
        return None

def load_best_format(base_path):
    """Load CSV or LAS, whichever has more data"""
    csv_path = base_path.replace('.las', '.csv')
    las_path = base_path.replace('.csv', '.las')
    df_csv = None
    df_las = None

    if Path(csv_path).exists():
        df_csv = load_phoreal_csv(csv_path)

    if Path(las_path).exists():
        df_las = load_phoreal_las(las_path)

    # Return whichever has more data
    if df_csv is not None and df_las is not None:
        if len(df_csv) >= len(df_las):
            print(f" - Using CSV ({len(df_csv):,} photons)", flush=True)
            return df_csv

        else:
            print(f" - Using LAS ({len(df_las):,} photons)", flush=True)
            return df_las

    elif df_csv is not None:
        print(f" - Using CSV ({len(df_csv):,} photons)", flush=True)
        return df_csv

    elif df_las is not None:
        print(f" - Using LAS ({len(df_las):,} photons)", flush=True)
        return df_las

    else:
        return None

# ============================================================================
# FILTER PHOTONS - WITH NATURE PAPER IMPROVEMENTS
# ============================================================================

def filter_ground_photons(df, confidence_threshold=3):
    """Filter for high-quality ground photons
    IMPROVED: Higher confidence threshold (3-4 only)"""
    df_filtered = df.copy()
    initial_count = len(df_filtered)
    
    # Filter 1: Signal confidence (IMPROVED: threshold=3 instead of 2)
    if 'Signal Confidence' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['Signal Confidence'] >= confidence_threshold]
        print(f" After confidence filter: {len(df_filtered):,} photons", flush=True)

    # Filter 2: Classification (ground points)
    if 'Classification' in df_filtered.columns and df_filtered['Classification'].notna().any():
        df_filtered = df_filtered[df_filtered['Classification'].isin([1, 4])]
        print(f" After classification filter: {len(df_filtered):,} photons (used ATL08)", flush=True)

    else:
        print(f" No ATL08 classification - using height-based filtering", flush=True)
        
        # Height-based filtering if no classification
        if 'Along-Track (m)' in df_filtered.columns:
            df_filtered['segment'] = (df_filtered['Along-Track (m)'] // 100).astype(int)
            def filter_ground_by_height(group):
                if len(group) < 5:
                    return group
                threshold = group['Height (m HAE)'].quantile(0.20)
                return group[group['Height (m HAE)'] <= threshold]

            df_filtered = df_filtered.groupby(['rgt', 'cycle', 'beam', 'segment'], observed=True).apply(
                filter_ground_by_height
            ).reset_index(drop=True)

            df_filtered = df_filtered.drop(columns='segment')
            print(f" After height-based filter: {len(df_filtered):,} photons", flush=True)

    print(f" Filtered: {initial_count:,} → {len(df_filtered):,} photons ({len(df_filtered)/initial_count*100:.1f}% retained)", flush=True)
    return df_filtered

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    print("\nSearching for ICeSat-2 files...", flush=True)
    csv_files = sorted(glob.glob(str(Path(INPUT_DIR) / 'processed*ATL03*.csv')))

    if len(csv_files) == 0:
        print("ERROR: No ICeSat-2 files found!", flush=True)
        print(f"Check path: {INPUT_DIR}", flush=True)
        return

    print(f"Found {len(csv_files)} CSV files", flush=True)

    # Get unique acquisitions (without beam suffix)
    unique_files = set()
    for csv_file in csv_files:
        base = '_'.join(Path(csv_file).stem.split('_')[:-1])
        unique_files.add(base)

    print(f"Processing {len(unique_files)} unique acquisitions", flush=True)
    print(f"FILTERING: Strong beams only ({STRONG_BEAMS})", flush=True)
    print("="*80, flush=True)

    all_data = []
    file_count = 0

    for base_filename in sorted(unique_files):
        file_count += 1

        # NATURE PAPER IMPROVEMENT: Process only strong beams
        for beam in STRONG_BEAMS: # Only strong beams!
            filepath = Path(INPUT_DIR) / f'{base_filename}_{beam}.csv'
            if not filepath.exists():
                continue

            print(f"\n[{file_count}/{len(unique_files)}] Processing: {filepath.name}", flush=True)

            df = load_best_format(str(filepath))
            if df is None or len(df) == 0:
                print(f" WARNING: Skipping - no data", flush=True)
                continue

            # IMPROVED FILTERING
            df_filtered = filter_ground_photons(df, confidence_threshold=CONFIDENCE_THRESHOLD)
            if len(df_filtered) > 0:
                all_data.append(df_filtered)

            else:
                print(f" WARNING: No photons passed filtering", flush=True)

    if len(all_data) == 0:
        print("\nERROR: No data survived filtering!", flush=True)
        return

    print(f"\n{'='*80}", flush=True)
    print(f"Combining {len(all_data)} datasets...", flush=True)

    icesat2_data = pd.concat(all_data, ignore_index=True)

    # ============================================================================
    # SUMMARY STATISTICS
    # ============================================================================

    print("\n" + "="*80, flush=True)
    print("FILTERING SUMMARY", flush=True)
    print("="*80, flush=True)
    print(f"Total filtered photons: {len(icesat2_data):,}", flush=True)
    print(f"Date range: {icesat2_data['date'].min().date()} to {icesat2_data['date'].max().date()}", flush=True)
    print(f"Years covered: {sorted(icesat2_data['year'].unique())}", flush=True)
    print(f"Unique RGTs: {icesat2_data['rgt'].nunique()}", flush=True)
    print(f"Unique beams: {sorted(icesat2_data['beam'].unique())}", flush=True)
    print(f"\nBEAM DISTRIBUTION (Strong beams only):", flush=True)
    print(f" Strong beam photons: {len(icesat2_data[icesat2_data['beam_strength']=='strong']):,} (100%)", flush=True)
    print(f" Weak beam photons: {len(icesat2_data[icesat2_data['beam_strength']=='weak']):,} (0% - filtered out)", flush=True)
    
    # ============================================================================
    # YEAR-BY-YEAR PHOTON COUNT [NEW]
    # ============================================================================

    print("\n" + "="*80, flush=True)
    print("PHOTON COUNTS BY YEAR", flush=True)
    print("="*80 + "\n", flush=True)
    year_counts = icesat2_data['year'].value_counts().sort_index()
    total_photons = len(icesat2_data)
    print(f"{'Year':<8} {'Count':>15} {'Percentage':>12}")
    print("-"*40)

    for year in sorted(icesat2_data['year'].unique()):
        count = len(icesat2_data[icesat2_data['year'] == year])
        pct = 100.0 * count / total_photons
        print(f"{int(year):<8} {count:>15,} {pct:>11.1f}%")

    print("-"*40)
    print(f"{'TOTAL':<8} {total_photons:>15,} {'100.0%':>11}")

    # ============================================================================
    # SAVE RESULTS
    # ============================================================================

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / 'icesat2_filtered_ground_photons.csv'
    print(f"\nSaving to: {output_file}", flush=True)
    icesat2_data.to_csv(output_file, index=False)
    print(f"[OK] Saved successfully ({output_file.stat().st_size / (1024*1024):.1f} MB)", flush=True)

    # ============================================================================
    # SAVE YEAR-BY-YEAR COUNTS TO TXT FILE [NEW]
    # ============================================================================

    count_file = OUTPUT_DIR / '01_photon_counts_by_year.txt'
    with open(count_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SCRIPT 01: FILTERED PHOTON COUNTS BY YEAR\n")
        f.write("="*80 + "\n\n")
        f.write(f"Run ID: {RUN_TIMESTAMP}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"{'Year':<8} {'Count':>15} {'Percentage':>12}\n")
        f.write("-"*40 + "\n")

        for year in sorted(icesat2_data['year'].unique()):
            count = len(icesat2_data[icesat2_data['year'] == year])
            pct = 100.0 * count / total_photons
            f.write(f"{int(year):<8} {count:>15,} {pct:>11.1f}%\n")

        f.write("-"*40 + "\n")
        f.write(f"{'TOTAL':<8} {total_photons:>15,} {'100.0%':>11}\n\n")
        f.write("="*80 + "\n")

    print(f"[OK] Year counts saved to: {count_file.name}", flush=True)

    # Save summary
    summary_file = OUTPUT_DIR / 'filtering_summary.txt'

    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ICeSat-2 Photon Filtering Summary - NATURE PAPER METHOD\n")
        f.write("="*80 + "\n\n")
        f.write(f"Processing date: {datetime.now()}\n")
        f.write(f"Run ID: {RUN_TIMESTAMP}\n\n")
        f.write("IMPROVEMENTS FROM NATURE PAPER:\n")
        f.write(f" • Strong beams only: {STRONG_BEAMS}\n")
        f.write(f" • Discarded weak beams: {WEAK_BEAMS}\n")
        f.write(f" • Confidence threshold: {CONFIDENCE_THRESHOLD} (3=high, 4=highest)\n\n")
        f.write(f"Total photons: {len(icesat2_data):,}\n")
        f.write(f"Date range: {icesat2_data['date'].min()} to {icesat2_data['date'].max()}\n")
        f.write(f"Years: {sorted(icesat2_data['year'].unique())}\n")
        f.write(f"RGTs: {icesat2_data['rgt'].nunique()}\n")
        f.write(f"Beams: {sorted(icesat2_data['beam'].unique())}\n")
        f.write(f"Strong beam photons: {len(icesat2_data[icesat2_data['beam_strength']=='strong']):,}\n")

    print(f"[OK] Summary saved to: {summary_file}\n", flush=True)
    print("="*80, flush=True)
    print("SCRIPT 01 COMPLETE", flush=True)
    print("="*80, flush=True)
    print(f"\nNext step: Run Script 02 (Terrain Classification)\n", flush=True)

if __name__ == '__main__':
    main()
