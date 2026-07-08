# ================================================================================
# SCRIPT 08: WORKFLOW ORCHESTRATION (MASTER EXECUTOR)
# ================================================================================
#
# Purpose
# -------
# Single entry point to run the consolidated ICESat-2 workflow.
# Executes Scripts 00–07 in sequence, manages run state, and allows
# resuming from a failed step.
#
# How to run
# ----------
#   python 08_run_workflow.py --temporal 2
#   python 08_run_workflow.py --temporal 3
#   python 08_run_workflow.py --temporal 4
# ================================================================================

import subprocess
import sys
import time
import io
import warnings
import argparse
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# UTF-8 FIX: None-safe encoding check
_stdout_enc = getattr(sys.stdout, 'encoding', None) or ''
if _stdout_enc.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

_stderr_enc = getattr(sys.stderr, 'encoding', None) or ''
if _stderr_enc.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ============================================================================
# IMPORT CONFIG  (FIX #2: no longer hardcoded)
# ============================================================================
try:
    from config import BASEOUTPUTDIR
except ImportError:
    BASEOUTPUTDIR = Path(r'E:\Processing\ICeSat-2\output_data')

# ============================================================================
# PARSE COMMAND-LINE ARGUMENTS
# ============================================================================
parser = argparse.ArgumentParser(
    description="ICESat-2 Complete Analysis Workflow with Temporal Filtering"
)
parser.add_argument(
    '--temporal',
    type=str,
    choices=['1', '2', '3', '4'],
    default='4',
    help=(
        'Temporal filtering: '
        '1=June-Oct, '
        '2=June-Sept (default), '
        '3=Early vs Late, '
        '4=Same-month-only'
    )
)
args = parser.parse_args()

TEMPORAL_OPTION = args.temporal

TEMPORAL_OPTIONS = {
    '1': 'June-October (Broad summer window)',
    '2': 'June-September (Standard thaw season)',
    '3': 'Early vs Late Summer (June-July vs Aug-Sept)',
    '4': 'Same-month-only (Strictest)',
}

# ============================================================================
# SCRIPT REGISTRY
# Format: (script_num, script_name, description, extra_args, is_required)
# ============================================================================
SCRIPTS = [
    ('00',  '00_setup_validation.py',        'Setup & validation',                          [],                              False),
    ('01',  '01_filter_photons.py',           'Photon filtering (strong beams)',             [],                              True),
    ('02',  '02_filter_terrain.py',           'Terrain classification (land/ocean/lakes)',   [],                              True),
    ('03',  '03_analysis_core.py',            'Crossover analysis & InSAR integration',      ['--temporal', TEMPORAL_OPTION], True),
    ('04',  '04_analysis_spatial.py',         'Spatial analysis',                            [],                              True),
    ('05',  '05_analysis_temporal.py',        'Temporal analysis',                           [],                              True),
    ('05b', '05b_temporal_stratification.py', 'Temporal stratification (seasonal)',          [],                              False),
    ('06',  '06_analysis_advanced.py',        'Advanced analysis',                           [],                              False),
    ('07',  '07_output_generation.py',        'Output generation (figures + GIS)',           [],                              True),
]

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':

    # FIX #3: capture the real pipeline start time before any script runs
    pipeline_start_dt = datetime.now()
    pipeline_start_ts = time.time()

    print("=" * 80)
    print("ICESat-2 COMPLETE ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"\nStart time    : {pipeline_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Temporal opt. : {TEMPORAL_OPTION} — {TEMPORAL_OPTIONS[TEMPORAL_OPTION]}")
    print(f"Total scripts : {len(SCRIPTS)}")
    print("=" * 80)

    # ── Execute scripts ───────────────────────────────────────────────────────
    results = {}   # script_num -> ('SUCCESS'|'FAILED'|'ERROR'|'SKIPPED', elapsed_s)

    for script_num, script_name, description, script_args, is_required in SCRIPTS:
        print(f"\n[{script_num}] {description}")
        print(f"Script: {script_name}")
        if script_args:
            print(f"Arguments: {' '.join(script_args)}")
        print("-" * 80)

        script_start = time.time()
        script_path = Path(script_name)

        if not script_path.exists():
            script_time = time.time() - script_start
            if is_required:
                print(f"✗ [{script_num}] REQUIRED script not found: {script_name}")
                results[script_num] = ('FAILED', script_time)
            else:
                print(f"⊘ [{script_num}] Optional script skipped (not found): {script_name}")
                results[script_num] = ('SKIPPED', script_time)
            continue

        try:
            cmd = [sys.executable, script_name] + script_args

            # FIX #5: removed text=True — it has no effect when capture_output=False
            result = subprocess.run(cmd, capture_output=False, check=False)

            script_time = time.time() - script_start

            if result.returncode == 0:
                print(f"\n✓ [{script_num}] Completed in {script_time:.1f}s")
                results[script_num] = ('SUCCESS', script_time)
            else:
                print(f"\n✗ [{script_num}] FAILED with exit code {result.returncode}")
                results[script_num] = ('FAILED', script_time)

        except Exception as e:
            script_time = time.time() - script_start
            print(f"\n✗ [{script_num}] ERROR: {e}")
            results[script_num] = ('ERROR', script_time)

    # ── Summary table ─────────────────────────────────────────────────────────
    total_time = time.time() - pipeline_start_ts

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"\nTotal time: {total_time / 60:.1f} minutes ({total_time:.0f} seconds)")
    print("-" * 80)
    print(f"{'Script':<8} {'Status':<12} {'Duration (s)':<15} Description")
    print("-" * 80)

    for script_num, script_name, description, _, is_required in SCRIPTS:
        if script_num in results:
            status, duration = results[script_num]
            symbol = '✓' if status == 'SUCCESS' else ('⊘' if status == 'SKIPPED' else '✗')
            req_tag = '' if is_required else ' (opt.)'
            print(f"{script_num:<8} {status:<12} {duration:<15.1f} {symbol} {description}{req_tag}")

    print("=" * 80)

    # ── Statistics ────────────────────────────────────────────────────────────
    success_count = sum(1 for s, _ in results.values() if s == 'SUCCESS')
    failed_count  = sum(1 for s, _ in results.values() if s == 'FAILED')
    error_count   = sum(1 for s, _ in results.values() if s == 'ERROR')
    skipped_count = sum(1 for s, _ in results.values() if s == 'SKIPPED')

    print(f"\nResults Summary:")
    print(f"  ✓ Successful          : {success_count}/{len(SCRIPTS)}")
    print(f"  ✗ Failed              : {failed_count}/{len(SCRIPTS)}")
    print(f"  ⊘ Skipped (optional)  : {skipped_count}/{len(SCRIPTS)}")
    if error_count > 0:
        print(f"  ⚠ Errors             : {error_count}/{len(SCRIPTS)}")

    # FIX #4: simplified required_failed — no redundant inner is_req check
    required_failed = sum(
        1 for script_num, _, _, _, is_req in SCRIPTS
        if is_req
        and script_num in results
        and results[script_num][0] not in ('SUCCESS',)
    )

    if required_failed > 0:
        print(f"\n⚠  WARNING: {required_failed} required script(s) did not complete successfully")
        print("   Workflow may not have full results")
    elif failed_count > 0 or skipped_count > 0:
        print(f"\n⚠  Note: {failed_count + skipped_count} optional script(s) were skipped or failed")
        print("   Core workflow completed successfully")
    else:
        print("\n✓ All scripts completed successfully!")

    # ── Save execution log ────────────────────────────────────────────────────
    timestamp_file = BASEOUTPUTDIR / 'latest_RUN_TIMESTAMP.txt'

    if timestamp_file.exists():
        with open(timestamp_file, 'r', encoding='utf-8') as f:
            run_id = f.read().strip()

        log_dir = BASEOUTPUTDIR / f'run_{run_id}'
        # FIX #6: ensure directory exists before writing log
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f'execution_log_{run_id}.txt'

        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("ICESat-2 Complete Analysis Pipeline — Execution Log\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Run ID         : {run_id}\n")
                # FIX #3: use real start time, not datetime.now() at log-write time
                f.write(f"Start time     : {pipeline_start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"End time       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total duration : {total_time / 60:.1f} minutes\n")
                f.write(f"Temporal option: {TEMPORAL_OPTION} — {TEMPORAL_OPTIONS[TEMPORAL_OPTION]}\n")
                f.write(f"Success        : {success_count}/{len(SCRIPTS)}\n")
                f.write(f"Failed         : {failed_count}/{len(SCRIPTS)}\n")
                f.write(f"Skipped        : {skipped_count}/{len(SCRIPTS)}\n")
                f.write("=" * 80 + "\n\n")

                for script_num, script_name, description, script_args, is_required in SCRIPTS:
                    if script_num in results:
                        status, duration = results[script_num]
                        req_str = "(Required)" if is_required else "(Optional)"
                        f.write(f"[{script_num}] {description} {req_str}\n")
                        f.write(f"  Script   : {script_name}\n")
                        if script_args:
                            f.write(f"  Arguments: {' '.join(script_args)}\n")
                        f.write(f"  Status   : {status}\n")
                        f.write(f"  Duration : {duration:.1f}s\n\n")

                f.write("=" * 80 + "\n")
                f.write("END OF LOG\n")
                f.write("=" * 80 + "\n")

            print(f"\nLog saved: {log_file}")

        except Exception as e:
            print(f"\nWarning: Could not save log file: {e}")

    print("\n" + "=" * 80)
    print("Workflow complete!")
    print("=" * 80 + "\n")

    # FIX #7: sys.exit guarded inside __main__ block (was at bare module level)
    sys.exit(1 if required_failed > 0 else 0)