"""
convert_atl06_to_csv.py
Standalone script to convert ICESat-2 ATL06 (Land Ice Height) HDF5 granules to CSV.

Runs independently of the main permafrost workflow (Scripts 00-08).

>>> EDIT THE PATHS BELOW, THEN RUN:  python convert_atl06_to_csv.py
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import sys
import logging
import time

# =============================================================================
# USER CONFIGURATION — EDIT THESE PATHS AND OPTIONS
# =============================================================================
INPUT_DIR  = r"G:\Data\ICESat-2\ATL06"     # folder containing ATL06 .h5 files
OUTPUT_DIR = r"E:\Processing\ICESat-2\raw_data"    # folder where output CSVs + log will be written

PATTERN          = "*ATL06*.h5"  # glob pattern to match input files
RECURSIVE        = False         # search input_dir subfolders too
QUALITY_FILTER   = 0             # 0 = keep only atl06_quality_summary==0 rows (recommended)
                                  # 1 = keep all rows, no filtering
KEEP_DIAGNOSTICS = True          # keep dh_fit_dx and n_fit_photons columns
MERGE_OUTPUT     = False         # also write one combined ATL06_merged_all.csv
OVERWRITE        = False        # reprocess files even if their CSV already exists
# =============================================================================

BEAMS = ['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r']
ATLAS_EPOCH = datetime(2018, 1, 1, 0, 0, 0)  # ICESat-2 standard data epoch (ATLAS SDP epoch)

def get_beam_strength_map(sc_orient: int) -> dict:
    """Return {beam_name: 'strong'|'weak'} based on spacecraft orientation.
    sc_orient: 0 = backward (R beams strong), 1 = forward (L beams strong)."""
    if sc_orient == 1:
        strong = ['gt1l', 'gt2l', 'gt3l']
    else:
        strong = ['gt1r', 'gt2r', 'gt3r']
    return {b: ('strong' if b in strong else 'weak') for b in BEAMS}

def setup_logging(output_dir: Path) -> logging.Logger:
    logfile = output_dir / 'atl06_to_csv.log'
    logger = logging.getLogger('atl06_convert')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    fh = logging.FileHandler(logfile, encoding='utf-8', mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s',
                                       datefmt='%Y-%m-%d %H:%M:%S'))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def convert_one_file(h5_path: Path, quality_filter: int, keep_diagnostics: bool,
                      logger: logging.Logger) -> pd.DataFrame:
    """Convert a single ATL06 .h5 granule to a DataFrame across all six beams."""
    records = []

    try:
        with h5py.File(h5_path, 'r') as f:
            try:
                rgt = int(f['/orbit_info/rgt'][0])
            except Exception:
                rgt = -1
            try:
                cycle = int(f['/orbit_info/cycle_number'][0])
            except Exception:
                cycle = -1
            try:
                sc_orient = int(f['/orbit_info/sc_orient'][0])
            except Exception:
                sc_orient = -1
            beam_strength_map = get_beam_strength_map(sc_orient)

            for beam in BEAMS:
                grp_path = f'/{beam}/land_ice_segments'
                if grp_path not in f:
                    continue
                grp = f[grp_path]

                try:
                    lat = grp['latitude'][:]
                    lon = grp['longitude'][:]
                    h_li = grp['h_li'][:]
                    h_li_sigma = grp['h_li_sigma'][:] if 'h_li_sigma' in grp else np.full_like(h_li, np.nan)
                    dtime = grp['delta_time'][:]
                    quality = grp['atl06_quality_summary'][:] if 'atl06_quality_summary' in grp else np.zeros_like(h_li, dtype=int)
                except KeyError as e:
                    logger.warning(f"{h5_path.name} [{beam}]: missing dataset {e} - skipping beam")
                    continue

                h_li = np.where(np.abs(h_li) > 1e30, np.nan, h_li)
                h_li_sigma = np.where(np.abs(h_li_sigma) > 1e30, np.nan, h_li_sigma)
                dt_utc = [ATLAS_EPOCH + timedelta(seconds=float(s)) for s in dtime]

                data = {
                    'Latitude (deg)':   lat,
                    'Longitude (deg)':  lon,
                    'Height (m MSL)':   h_li,
                    'height_sigma_m':   h_li_sigma,
                    'quality_summary':  quality.astype(int),
                    'date':             dt_utc,
                    'rgt':              rgt,
                    'cycle':            cycle,
                    'beam':             beam,
                    'beam_strength':    beam_strength_map[beam],
                    'filename':         h5_path.name,
                }

                if keep_diagnostics:
                    dh_fit_dx = grp['fit_statistics/dh_fit_dx'][:] if 'fit_statistics' in grp and 'dh_fit_dx' in grp['fit_statistics'] else np.full_like(h_li, np.nan)
                    n_fit_photons = grp['fit_statistics/n_fit_photons'][:] if 'fit_statistics' in grp and 'n_fit_photons' in grp['fit_statistics'] else np.full_like(h_li, np.nan)
                    data['dh_fit_dx'] = dh_fit_dx
                    data['n_fit_photons'] = n_fit_photons

                df_beam = pd.DataFrame(data)

                if quality_filter == 0:
                    n_before = len(df_beam)
                    df_beam = df_beam[df_beam['quality_summary'] == 0].reset_index(drop=True)
                    logger.info(f"{h5_path.name} [{beam}]: {n_before:,} -> {len(df_beam):,} rows after quality_summary==0 filter")
                else:
                    logger.info(f"{h5_path.name} [{beam}]: {len(df_beam):,} rows loaded, no quality filter applied")

                df_beam = df_beam.dropna(subset=['Height (m MSL)']).reset_index(drop=True)
                records.append(df_beam)

    except Exception as e:
        logger.error(f"Failed to process {h5_path.name}: {e}")
        return pd.DataFrame()

    if not records:
        logger.warning(f"{h5_path.name}: no valid beam data extracted")
        return pd.DataFrame()

    df_all = pd.concat(records, ignore_index=True)
    df_all['year'] = pd.to_datetime(df_all['date']).dt.year
    return df_all

def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    if not input_dir.exists():
        print(f"ERROR: INPUT_DIR does not exist: {input_dir}")
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("=" * 80)
    logger.info("ATL06 -> CSV STANDALONE CONVERSION")
    logger.info("=" * 80)
    logger.info(f"Input dir:  {input_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Pattern:    {PATTERN}  (recursive={RECURSIVE})")
    logger.info(f"Quality filter: {'atl06_quality_summary==0 only' if QUALITY_FILTER == 0 else 'none (all rows)'}")
    logger.info(f"Diagnostics columns (dh_fit_dx, n_fit_photons): {'kept' if KEEP_DIAGNOSTICS else 'DROPPED'}")

    h5_files = sorted(input_dir.rglob(PATTERN) if RECURSIVE else input_dir.glob(PATTERN))
    if not h5_files:
        logger.error(f"No files matching '{PATTERN}' found in {input_dir}")
        sys.exit(1)
    logger.info(f"Found {len(h5_files)} ATL06 files to process")

    t_start = time.time()
    all_dfs = []
    n_ok, n_skip, n_fail = 0, 0, 0

    for i, h5_path in enumerate(h5_files, 1):
        out_csv = output_dir / (h5_path.stem + '.csv')
        if out_csv.exists() and not OVERWRITE:
            logger.info(f"[{i}/{len(h5_files)}] Skipping {h5_path.name} (CSV exists, set OVERWRITE=True to redo)")
            n_skip += 1
            continue

        logger.info(f"[{i}/{len(h5_files)}] Processing {h5_path.name} ...")
        df = convert_one_file(h5_path, quality_filter=QUALITY_FILTER,
                               keep_diagnostics=KEEP_DIAGNOSTICS, logger=logger)
        if df.empty:
            logger.warning(f"[{i}/{len(h5_files)}] No data extracted from {h5_path.name}")
            n_fail += 1
            continue

        df.to_csv(out_csv, index=False, encoding='utf-8')
        logger.info(f"[{i}/{len(h5_files)}] Saved {out_csv.name} ({len(df):,} rows)")
        n_ok += 1
        if MERGE_OUTPUT:
            all_dfs.append(df)

    if MERGE_OUTPUT and all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        merged_csv = output_dir / 'ATL06_merged_all.csv'
        merged.to_csv(merged_csv, index=False, encoding='utf-8')
        logger.info(f"Saved merged file {merged_csv.name} ({len(merged):,} total rows)")

    elapsed = time.time() - t_start
    logger.info("=" * 80)
    logger.info("CONVERSION COMPLETE")
    logger.info(f"Converted: {n_ok}  |  Skipped (existing): {n_skip}  |  Failed/empty: {n_fail}")
    logger.info(f"Elapsed: {elapsed:.1f}s")
    logger.info("=" * 80)

if __name__ == '__main__':
    main()