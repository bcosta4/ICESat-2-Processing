"""
================================================================================
SCRIPT 01: ICeSat-2 PHOTON FILTERING - CORRECTED VERSION (MSL)
================================================================================

Purpose
-------
Load raw ICeSat-2 ATL03 photon data and apply quality filters to extract
high-confidence ground photons suitable for subsidence analysis over
the Tuktoyaktuk Peninsula.

Key Improvements (This Version)
-------------------------------
✓ Fixed: Uses config.py for paths (portable across systems)
✓ Fixed: Reads timestamp from Script 00 output
✓ Fixed: Creates output folder using timestamp from Script 00
✓ Fixed: Strong beams only (NATURE PAPER method)
✓ Fixed: Higher confidence threshold (3 instead of 2)
✓ Fixed: Generates year-by-year photon counts
✓ Fixed: UTF-8 encoding throughout
✓ Fixed: KeyError on 'segment' drop after groupby/apply (pandas compat)
✓ Fixed: Height column detection in height-based fallback filter
✓ Fixed: load_best_format path derivation uses Path stem (not .replace)
✓ Fixed: Zero-photon guard before percentage calculation
✓ Fixed: Unused year_counts variable removed
✓ NEW: Comprehensive filtering summary
✓ NEW: Uses Height (m MSL) for ground filtering (was HAE)
✓ NEW: Study-area bounding box pre-filter (Tuktoyaktuk Peninsula)

Inputs
------
- Raw ATL03 photon files (CSV or LAS) from INPUTDIR
- latest_RUN_TIMESTAMP.txt (created by Script 00)

Outputs
-------
- icesat2_filtered_ground_photons.csv (all filtered photons)
- 01_photon_counts_by_year.txt (year-by-year summary)
- filtering_summary.txt (processing log)
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

# UTF-8 encoding for Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
from config import INPUTDIR, BASEOUTPUTDIR

# NATURE PAPER IMPROVEMENTS: Strong beams only
STRONG_BEAMS = ['gt1l', 'gt2l', 'gt3l']   # Keep only these
WEAK_BEAMS   = ['gt1r', 'gt2r', 'gt3r']   # Discard these

# NATURE PAPER IMPROVEMENT: Higher confidence threshold
CONFIDENCE_THRESHOLD = 3  # 3=high, 4=highest (was 2=medium in old version)

# ATL03 signal_conf values used for ground classification
# 0=noise, 1=buffer, 2=low, 3=medium, 4=high
# PhoREAL maps these to 'Signal Confidence' column

# Study-area bounding box: Tuktoyaktuk Peninsula + buffer (WGS84 degrees)
# Expand slightly beyond peninsula to capture nearshore data
STUDY_AREA_LAT_MIN =  68.5
STUDY_AREA_LAT_MAX =  70.5
STUDY_AREA_LON_MIN = -134.0
STUDY_AREA_LON_MAX = -131.0

# Height datum column name produced by this script
HEIGHT_COL = 'Height (m MSL)'

# Beam strength lookup (used in both loaders)
BEAM_STRENGTH = {
    'gt1l': 'strong', 'gt1r': 'weak',
    'gt2l': 'strong', 'gt2r': 'weak',
    'gt3l': 'strong', 'gt3r': 'weak',
}

print("=" * 80, flush=True)
print("SCRIPT 01: ICeSat-2 PHOTON FILTERING - STRONG BEAMS ONLY (MSL)", flush=True)
print("=" * 80, flush=True)
print(f"Input directory:      {INPUTDIR}", flush=True)
print(f"Output directory:     {BASEOUTPUTDIR}", flush=True)
print(f"IMPROVEMENTS FROM NATURE PAPER:", flush=True)
print(f"  • Strong beams only:      {STRONG_BEAMS}", flush=True)
print(f"  • Discard weak beams:     {WEAK_BEAMS}", flush=True)
print(f"  • Confidence threshold:   {CONFIDENCE_THRESHOLD} (higher quality)", flush=True)
print(f"  • Height datum:           MSL (Mean Sea Level)", flush=True)
print(f"  • Study-area bbox:        "
      f"Lat [{STUDY_AREA_LAT_MIN}, {STUDY_AREA_LAT_MAX}] "
      f"Lon [{STUDY_AREA_LON_MIN}, {STUDY_AREA_LON_MAX}]", flush=True)
print("=" * 80, flush=True)

# ============================================================================
# READ TIMESTAMP FROM SCRIPT 00
# ============================================================================
print("\nReading run timestamp from Script 00...", flush=True)
timestamp_file = Path(BASEOUTPUTDIR) / 'latest_RUN_TIMESTAMP.txt'

if not timestamp_file.exists():
    print(f"\n✗ ERROR: Timestamp file not found: {timestamp_file}")
    print("   Script 00 must run first!")
    sys.exit(1)

with open(timestamp_file, 'r', encoding='utf-8') as f:
    RUN_TIMESTAMP = f.read().strip()

print(f"✓ Run ID: {RUN_TIMESTAMP}")

# Create output folder (should already exist from Script 00)
OUTPUT_DIR = Path(BASEOUTPUTDIR) / f'run_{RUN_TIMESTAMP}'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"✓ Output folder: {OUTPUT_DIR.name}")
print("=" * 80)


# ============================================================================
# HELPER: ATTACH FILENAME METADATA
# ============================================================================
def _attach_file_metadata(df: pd.DataFrame, filepath: str) -> pd.DataFrame:
    """
    Parse acquisition date, RGT, cycle, beam, and beam strength from
    a PhoREAL filename of the form:
        processed_ATL03_YYYYMMDD######_RGTCCbbb_006_02_<beam>.csv

    Adds columns: filename, date, year, rgt, cycle, beam, beam_strength.
    Returns the dataframe with those columns added.
    """
    filename = Path(filepath).stem
    parts = filename.split('_')

    df = df.copy()
    df['filename'] = filename

    # Date is in parts[2], first 8 chars: YYYYMMDD
    try:
        df['date'] = pd.to_datetime(parts[2][0:8], format='%Y%m%d')
    except (IndexError, ValueError):
        print(f"   WARNING: Cannot parse date from filename '{filename}', "
              "setting NaT", flush=True)
        df['date'] = pd.NaT

    df['year']  = df['date'].dt.year
    df['rgt']   = parts[3][0:4]   if len(parts) > 3 else 'UNKNOWN'
    df['cycle'] = parts[3][4:6]   if len(parts) > 3 else 'UNKNOWN'
    df['beam']  = parts[-1]

    df['beam_strength'] = df['beam'].map(BEAM_STRENGTH)

    return df


# ============================================================================
# LOAD PHOREAL CSV FILES
# ============================================================================
def load_phoreal_csv(csv_path: str) -> pd.DataFrame | None:
    """Load PhoREAL CSV with proper column names and metadata extraction."""
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        df = _attach_file_metadata(df, csv_path)
        return df
    except Exception as e:
        print(f"ERROR loading CSV {csv_path}: {e}", flush=True)
        return None


# ============================================================================
# LOAD PHOREAL LAS FILES
# ============================================================================
def load_phoreal_las(las_path: str) -> pd.DataFrame | None:
    """Load PhoREAL LAS file and convert to DataFrame.

    Notes
    -----
    - las.z  → ellipsoidal height as stored by PhoREAL; we label it
               HEIGHT_COL ('Height (m MSL)') because PhoREAL writes MSL
               heights to z when the corrected orthometric option is used.
               If your PhoREAL run wrote HAE, rename accordingly.
    - las.X / las.Y are integer-scaled easting/northing; las.x / las.y
      are the floating-point geographic coordinates (lon, lat).
    - Signal Confidence is stored in the intensity field by PhoREAL.
    """
    try:
        las = laspy.read(las_path)

        # --- GPS time ---
        time_attr = None
        for attr in ('gpstime', 'gps_time', 'Time', 'times'):
            if hasattr(las, attr):
                time_attr = getattr(las, attr)
                break
        if time_attr is None:
            print("   WARNING: No time attribute found in LAS file, "
                  "using placeholder", flush=True)
            time_attr = np.arange(len(las.x))

        # --- UTM coordinates ---
        # PhoREAL stores scaled integer UTM in X/Y; prefer those when present
        utm_e = np.array(las.X, dtype=float) if hasattr(las, 'X') else np.full(len(las.x), np.nan)
        utm_n = np.array(las.Y, dtype=float) if hasattr(las, 'Y') else np.full(len(las.x), np.nan)

        # --- Classification and confidence ---
        classification = (np.array(las.classification, dtype=int)
                          if hasattr(las, 'classification')
                          else np.ones(len(las.x), dtype=int))

        # PhoREAL encodes signal confidence in intensity (uint16 → int)
        signal_conf = (np.array(las.intensity, dtype=int)
                       if hasattr(las, 'intensity')
                       else np.full(len(las.x), 4, dtype=int))

        df = pd.DataFrame({
            'Time (sec)':       time_attr,
            'Latitude (deg)':   las.y,
            'Longitude (deg)':  las.x,
            'UTM Easting (m)':  utm_e,
            'UTM Northing (m)': utm_n,
            HEIGHT_COL:         las.z,   # MSL heights from PhoREAL
            'Classification':   classification,
            'Signal Confidence': signal_conf,
        })

        df = _attach_file_metadata(df, las_path)
        return df

    except Exception as e:
        print(f"ERROR loading LAS {las_path}: {e}", flush=True)
        return None


# ============================================================================
# LOAD BEST FORMAT (CSV preferred; fall back to LAS)
# ============================================================================
def load_best_format(csv_path: str) -> pd.DataFrame | None:
    """
    Given an explicit CSV filepath, attempt to load CSV first; if unavailable
    or the LAS version has more photons, prefer LAS.

    Parameters
    ----------
    csv_path : str
        Full path ending in '.csv' (as discovered by glob).
    """
    p = Path(csv_path)
    las_path = p.with_suffix('.las')

    df_csv = load_phoreal_csv(str(p))          if p.exists()        else None
    df_las = load_phoreal_las(str(las_path))   if las_path.exists() else None

    if df_csv is not None and df_las is not None:
        if len(df_csv) >= len(df_las):
            print(f"   - Using CSV ({len(df_csv):,} photons)", flush=True)
            return df_csv
        else:
            print(f"   - Using LAS ({len(df_las):,} photons)", flush=True)
            return df_las
    elif df_csv is not None:
        print(f"   - Using CSV ({len(df_csv):,} photons)", flush=True)
        return df_csv
    elif df_las is not None:
        print(f"   - Using LAS ({len(df_las):,} photons)", flush=True)
        return df_las
    else:
        return None


# ============================================================================
# STUDY-AREA BOUNDING BOX FILTER
# ============================================================================
def apply_bbox_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove photons outside the Tuktoyaktuk Peninsula study-area bounding box.

    Operates on 'Latitude (deg)' and 'Longitude (deg)' columns.
    If those columns are absent, returns df unchanged with a warning.
    """
    lat_col = 'Latitude (deg)'
    lon_col = 'Longitude (deg)'

    if lat_col not in df.columns or lon_col not in df.columns:
        print("   WARNING: Lat/Lon columns not found – bbox filter skipped",
              flush=True)
        return df

    n_before = len(df)
    mask = (
        (df[lat_col] >= STUDY_AREA_LAT_MIN) &
        (df[lat_col] <= STUDY_AREA_LAT_MAX) &
        (df[lon_col] >= STUDY_AREA_LON_MIN) &
        (df[lon_col] <= STUDY_AREA_LON_MAX)
    )
    df_out = df[mask].copy()
    n_removed = n_before - len(df_out)
    if n_removed > 0:
        print(f"   Bbox filter removed {n_removed:,} photons outside "
              f"study area → {len(df_out):,} remaining", flush=True)
    return df_out


# ============================================================================
# FILTER PHOTONS - WITH NATURE PAPER IMPROVEMENTS
# ============================================================================
def _detect_height_col(df: pd.DataFrame) -> str | None:
    """Return the name of the first recognised height column in df, or None."""
    for candidate in (HEIGHT_COL, 'Height (m HAE)', 'height', 'Height',
                      'elevation', 'h_li'):
        if candidate in df.columns:
            return candidate
    return None


def filter_ground_photons(df: pd.DataFrame,
                          confidence_threshold: int = 3) -> pd.DataFrame:
    """
    Filter for high-quality ground photons.

    Steps
    -----
    1. Study-area bounding box
    2. Signal confidence >= confidence_threshold
    3a. ATL08 classification (if present): keep classes 1 (land) and 4 (ground)
    3b. Fallback: lowest-20th-percentile height within 100-m along-track segments
    """
    if len(df) == 0:
        return df

    df_filtered = df.copy()
    initial_count = len(df_filtered)

    # --- Step 0: bounding-box pre-filter ---
    df_filtered = apply_bbox_filter(df_filtered)

    # --- Step 1: Signal confidence ---
    if 'Signal Confidence' in df_filtered.columns:
        df_filtered = df_filtered[
            df_filtered['Signal Confidence'] >= confidence_threshold
        ]
        print(f"   After confidence filter: {len(df_filtered):,} photons",
              flush=True)
    else:
        print("   WARNING: 'Signal Confidence' column not found – "
              "confidence filter skipped", flush=True)

    # Guard: nothing left after confidence filter
    if len(df_filtered) == 0:
        print("   WARNING: 0 photons remain after confidence filter – "
              "check confidence values in source data", flush=True)
        pct = 0.0
        print(f"   Filtered: {initial_count:,} → 0 photons ({pct:.1f}% retained)",
              flush=True)
        return df_filtered

    # --- Step 2a: ATL08 classification ---
    has_classification = (
        'Classification' in df_filtered.columns
        and df_filtered['Classification'].notna().any()
    )

    if has_classification:
        df_filtered = df_filtered[
            df_filtered['Classification'].isin([1, 4])
        ]
        print(f"   After classification filter: {len(df_filtered):,} photons "
              f"(used ATL08)", flush=True)

    else:
        # --- Step 2b: Height-based fallback ---
        print("   No ATL08 classification – using height-based filtering",
              flush=True)

        height_col = _detect_height_col(df_filtered)

        if 'Along-Track (m)' not in df_filtered.columns:
            print("   WARNING: 'Along-Track (m)' column not found – "
                  "height-based filter skipped", flush=True)

        elif height_col is None:
            print("   WARNING: No recognised height column found – "
                  "height-based filter skipped", flush=True)

        else:
            if height_col != HEIGHT_COL:
                print(f"   NOTE: Using '{height_col}' for height-based filter "
                      f"('{HEIGHT_COL}' not found)", flush=True)

            df_filtered = df_filtered.copy()
            df_filtered['_segment'] = (
                df_filtered['Along-Track (m)'] // 100
            ).astype(int)

            def _filter_by_height(group: pd.DataFrame) -> pd.DataFrame:
                if len(group) < 5:
                    return group
                threshold = group[height_col].quantile(0.20)
                return group[group[height_col] <= threshold]

            df_filtered = (
                df_filtered
                .groupby(['rgt', 'cycle', 'beam', '_segment'], observed=True)
                .apply(_filter_by_height)
                .reset_index(drop=True)
            )

            # FIX: only drop '_segment' if it survived the groupby/apply
            if '_segment' in df_filtered.columns:
                df_filtered = df_filtered.drop(columns='_segment')

            print(f"   After height-based filter: {len(df_filtered):,} photons",
                  flush=True)

    pct = len(df_filtered) / initial_count * 100 if initial_count > 0 else 0.0
    print(f"   Filtered: {initial_count:,} → {len(df_filtered):,} photons "
          f"({pct:.1f}% retained)", flush=True)

    return df_filtered


# ============================================================================
# MAIN PROCESSING
# ============================================================================
def main():
    print("\nSearching for ICeSat-2 files...", flush=True)
    csv_files = sorted(glob.glob(
        str(Path(INPUTDIR) / 'processed*ATL08*.csv')
    ))

    if len(csv_files) == 0:
        print("ERROR: No ICeSat-2 files found!", flush=True)
        print(f"Check path: {INPUTDIR}", flush=True)
        return

    print(f"Found {len(csv_files)} CSV files", flush=True)

    # Derive unique acquisition base names (strip beam suffix)
    unique_files = sorted({
        '_'.join(Path(f).stem.split('_')[:-1]) for f in csv_files
    })

    print(f"Processing {len(unique_files)} unique acquisitions", flush=True)
    print(f"FILTERING: Strong beams only ({STRONG_BEAMS})", flush=True)
    print("=" * 80, flush=True)

    all_data = []
    file_count = 0

    for base_filename in unique_files:
        file_count += 1

        for beam in STRONG_BEAMS:
            filepath = Path(INPUTDIR) / f'{base_filename}_{beam}.csv'

            if not filepath.exists():
                continue

            print(f"\n[{file_count}/{len(unique_files)}] "
                  f"Processing: {filepath.name}", flush=True)

            df = load_best_format(str(filepath))

            if df is None or len(df) == 0:
                print("   WARNING: Skipping – no data loaded", flush=True)
                continue

            df_filtered = filter_ground_photons(
                df, confidence_threshold=CONFIDENCE_THRESHOLD
            )

            if len(df_filtered) > 0:
                all_data.append(df_filtered)
            else:
                print("   WARNING: No photons passed filtering", flush=True)

    if len(all_data) == 0:
        print("\nERROR: No data survived filtering!", flush=True)
        return

    print(f"\n{'='*80}", flush=True)
    print(f"Combining {len(all_data)} beam/acquisition datasets...", flush=True)
    icesat2_data = pd.concat(all_data, ignore_index=True)

    # ========================================================================
    # VALIDATE DATE COLUMN BEFORE USING IT
    # ========================================================================
    if 'date' not in icesat2_data.columns or icesat2_data['date'].isna().all():
        print("WARNING: 'date' column missing or all NaT – "
              "date-based summary will be skipped", flush=True)
        has_dates = False
    else:
        has_dates = True

    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    print("\n" + "=" * 80, flush=True)
    print("FILTERING SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(f"Total filtered photons: {len(icesat2_data):,}", flush=True)

    if has_dates:
        valid_dates = icesat2_data['date'].dropna()
        print(f"Date range:             "
              f"{valid_dates.min().date()} to {valid_dates.max().date()}",
              flush=True)
        print(f"Years covered:          "
              f"{sorted(icesat2_data['year'].dropna().astype(int).unique())}",
              flush=True)

    print(f"Unique RGTs:            {icesat2_data['rgt'].nunique()}", flush=True)
    print(f"Unique beams:           "
          f"{sorted(icesat2_data['beam'].unique())}", flush=True)
    print(f"\nBEAM DISTRIBUTION (Strong beams only):", flush=True)
    n_strong = (icesat2_data['beam_strength'] == 'strong').sum()
    n_weak   = (icesat2_data['beam_strength'] == 'weak').sum()
    print(f"  Strong beam photons: {n_strong:,} (100%)", flush=True)
    print(f"  Weak beam photons:   {n_weak:,} (0% – filtered out)", flush=True)

    # ========================================================================
    # YEAR-BY-YEAR PHOTON COUNT
    # ========================================================================
    total_photons = len(icesat2_data)

    if has_dates:
        print("\n" + "=" * 80, flush=True)
        print("PHOTON COUNTS BY YEAR", flush=True)
        print("=" * 80 + "\n", flush=True)

        print(f"{'Year':<8} {'Count':>15} {'Percentage':>12}")
        print("-" * 40)

        years_sorted = sorted(
            icesat2_data['year'].dropna().astype(int).unique()
        )
        for year in years_sorted:
            count = int((icesat2_data['year'] == year).sum())
            pct   = 100.0 * count / total_photons
            print(f"{year:<8} {count:>15,} {pct:>11.1f}%")

        print("-" * 40)
        print(f"{'TOTAL':<8} {total_photons:>15,} {'100.0%':>11}")

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / 'icesat2_filtered_ground_photons.csv'

    print(f"\nSaving to: {output_file}", flush=True)
    icesat2_data.to_csv(output_file, index=False, encoding='utf-8')
    print(f"[OK] Saved successfully "
          f"({output_file.stat().st_size / (1024*1024):.1f} MB)", flush=True)

    # ========================================================================
    # SAVE YEAR-BY-YEAR COUNTS TO TXT
    # ========================================================================
    count_file = OUTPUT_DIR / '01_photon_counts_by_year.txt'

    with open(count_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("SCRIPT 01: FILTERED PHOTON COUNTS BY YEAR\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Run ID:    {RUN_TIMESTAMP}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if has_dates:
            f.write(f"{'Year':<8} {'Count':>15} {'Percentage':>12}\n")
            f.write("-" * 40 + "\n")
            for year in years_sorted:
                count = int((icesat2_data['year'] == year).sum())
                pct   = 100.0 * count / total_photons
                f.write(f"{year:<8} {count:>15,} {pct:>11.1f}%\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'TOTAL':<8} {total_photons:>15,} {'100.0%':>11}\n\n")
        else:
            f.write("WARNING: Date column unavailable – "
                    "year breakdown not produced.\n\n")

        f.write("=" * 80 + "\n")

    print(f"[OK] Year counts saved to: {count_file.name}", flush=True)

    # ========================================================================
    # SAVE FILTERING SUMMARY
    # ========================================================================
    summary_file = OUTPUT_DIR / 'filtering_summary.txt'

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ICeSat-2 Photon Filtering Summary – NATURE PAPER METHOD (MSL)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Processing date: {datetime.now()}\n")
        f.write(f"Run ID:          {RUN_TIMESTAMP}\n\n")
        f.write("CONFIGURATION:\n")
        f.write(f"  Strong beams only:      {STRONG_BEAMS}\n")
        f.write(f"  Discarded weak beams:   {WEAK_BEAMS}\n")
        f.write(f"  Confidence threshold:   {CONFIDENCE_THRESHOLD} "
                f"(3=high, 4=highest)\n")
        f.write(f"  Height datum:           MSL (Mean Sea Level)\n")
        f.write(f"  Study-area bbox:        "
                f"Lat [{STUDY_AREA_LAT_MIN}, {STUDY_AREA_LAT_MAX}] "
                f"Lon [{STUDY_AREA_LON_MIN}, {STUDY_AREA_LON_MAX}]\n\n")
        f.write(f"Total photons: {total_photons:,}\n")

        if has_dates:
            f.write(f"Date range:    "
                    f"{icesat2_data['date'].dropna().min()} to "
                    f"{icesat2_data['date'].dropna().max()}\n")
            f.write(f"Years:         {years_sorted}\n")

        f.write(f"RGTs:          {icesat2_data['rgt'].nunique()}\n")
        f.write(f"Beams:         {sorted(icesat2_data['beam'].unique())}\n")
        f.write(f"Strong beams:  {n_strong:,}\n")

    print(f"[OK] Summary saved to: {summary_file}\n", flush=True)

    print("=" * 80, flush=True)
    print("SCRIPT 01 COMPLETE", flush=True)
    print("=" * 80, flush=True)
    print(f"\nNext step: Run Script 02 (Terrain Classification)\n", flush=True)


if __name__ == '__main__':
    main()