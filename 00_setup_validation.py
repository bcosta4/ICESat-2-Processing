"""
SCRIPT 00: SETUP & VALIDATION WITH CORRECTED AUTO-INCREMENT TIMESTAMP
Generates daily auto-incrementing run IDs (YYYYMMDD_n1, n2, etc.)
"""

from pathlib import Path
from datetime import datetime
import sys
import io

# UTF-8 FIX: None-safe check (.encoding can be None on Windows pipes/redirects)
_stdout_enc = getattr(sys.stdout, 'encoding', None) or ''
if _stdout_enc.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

_stderr_enc = getattr(sys.stderr, 'encoding', None) or ''
if _stderr_enc.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print("=" * 80)
print("SCRIPT 00: SETUP & VALIDATION")
print("=" * 80)

# ============================================================================
# IMPORT CONFIG
# ============================================================================
try:
    from config import BASEOUTPUTDIR, INPUTDIR, INSARDIR
    print("\n✓ Configuration imported successfully")
    print(f"  BASEOUTPUTDIR: {BASEOUTPUTDIR}")
    print(f"  INPUTDIR:      {INPUTDIR}")
    print(f"  INSARDIR:      {INSARDIR}")
except ImportError as e:
    print(f"\n✗ ERROR: Cannot import config.py")
    print(f"  {e}")
    sys.exit(1)

# ============================================================================
# VALIDATE DIRECTORIES
# ============================================================================
print("\n" + "=" * 80)
print("VALIDATING DIRECTORIES")
print("=" * 80)

BASEOUTPUTDIR.mkdir(parents=True, exist_ok=True)
print(f"✓ Base output directory: {BASEOUTPUTDIR}")

if not INPUTDIR.exists():
    print(f"⚠  WARNING: INPUTDIR does not exist: {INPUTDIR}")
    print(f"   Scripts may fail if no input data available")
else:
    print(f"✓ Input directory: {INPUTDIR}")

if not INSARDIR.exists():
    print(f"⚠  WARNING: INSARDIR does not exist: {INSARDIR}")
    print(f"   InSAR integration will be skipped")
else:
    print(f"✓ InSAR directory: {INSARDIR}")

# ============================================================================
# GENERATE AUTO-INCREMENT RUN ID  (YYYYMMDD_nX FORMAT)
# ============================================================================
def get_next_run_id(base_dir):
    """Generate daily auto-incrementing run ID (YYYYMMDD_n1, n2, etc.).

    Parameters
    ----------
    base_dir : Path | str
        Base output directory.

    Returns
    -------
    str
        Run ID like '20260116_n1'.
    """
    base_dir = Path(base_dir)          # defensive cast; does not shadow param scope
    today = datetime.now().strftime('%Y%m%d')

    existing_runs = list(base_dir.glob(f'run_{today}_n*'))

    if not existing_runs:
        run_number = 1
    else:
        numbers = []
        for folder in existing_runs:
            try:
                num = int(folder.name.split('_n')[-1])
                numbers.append(num)
            except (ValueError, IndexError):
                pass
        run_number = max(numbers) + 1 if numbers else 1

    return f'{today}_n{run_number}'


print("\n" + "=" * 80)
print("GENERATING RUN ID")
print("=" * 80)

run_id = get_next_run_id(BASEOUTPUTDIR)
print(f"\n✓ Generated run ID: {run_id}")

run_OUTPUTDIR = BASEOUTPUTDIR / f'run_{run_id}'
run_OUTPUTDIR.mkdir(parents=True, exist_ok=True)
print(f"✓ Created output directory: {run_OUTPUTDIR}")

# ============================================================================
# SAVE TIMESTAMP FILE FOR DOWNSTREAM SCRIPTS
# ============================================================================
timestamp_file = BASEOUTPUTDIR / 'latest_RUN_TIMESTAMP.txt'

try:
    with open(timestamp_file, 'w', encoding='utf-8') as f:
        f.write(run_id)
    print(f"✓ Saved timestamp file: {timestamp_file.name}")
    print(f"  Content: {run_id}")
except Exception as e:
    print(f"\n✗ CRITICAL ERROR: Could not write timestamp file")
    print(f"  {e}")
    sys.exit(1)

# ============================================================================
# VALIDATION SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)
print(f"\n✓ All checks passed!")
print(f"✓ Ready for next script (01_filter_photons.py)")
print(f"✓ Run ID: {run_id}")
print(f"✓ Output: {run_OUTPUTDIR}")
print("\n" + "=" * 80 + "\n")
