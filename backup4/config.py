"""
Centralized configuration for ICESat-2 processing workflow.
Defines paths, parameters, and environment settings.
All scripts import from this module.
"""

from pathlib import Path
import os
import sys

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
# Output directory (check environment first, then use default)
BASEOUTPUTDIR = Path(
    os.getenv(
        'ICESAT2_OUTPUT',
        'E:/Processing/ICeSat-2/output_data'
    )
)

# Raw data input directory
INPUTDIR = Path(
    os.getenv(
        'ICESAT2_INPUT',
        'E:/Processing/ICESat-2/raw_data'
    )
)

# InSAR validation data (optional - on separate drive, may not exist)
INSARDIR = Path(
    os.getenv(
        'INSAR_DATA',
        'C:/bcosta/InSAR_for_ICESat-2'
    )
)

# CanVec waterbody shapefile for distance-to-ocean calculation (Script 04)
CANVEC_WATERBODY_SHAPEFILE = Path(
    os.getenv(
        'CANVEC_SHAPEFILE',
        r'E:\GIS\canvec_50K_NT_Hydro_shp\waterbody_2_2.shp'
    )
)

# ============================================================================
# EXPORT LIST
# ============================================================================
__all__ = [
    'BASEOUTPUTDIR',
    'INPUTDIR',
    'INSARDIR',
    'CANVEC_WATERBODY_SHAPEFILE',
]

# ============================================================================
# VALIDATION FUNCTION
#      Scripts that need validation should call:
#          import config; config.validate()
#      Script 00 already validates paths explicitly, so this is optional.
# ============================================================================
def validate(create_dirs: bool = True, verbose: bool = True) -> bool:
    """
    Validate configuration paths and optionally create output directories.

    Parameters
    ----------
    create_dirs : bool
        If True, create BASEOUTPUTDIR and INPUTDIR if they do not exist.
        INSARDIR is never auto-created (optional data source).
    verbose : bool
        If True, print status messages to stdout.

    Returns
    -------
    bool
        True if all required paths are accessible, False otherwise.
    """
    # FIX: UTF-8 stdout guard (mirrors all other scripts in the workflow)
    _enc = getattr(sys.stdout, 'encoding', None) or ''
    if _enc.lower() != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    ok = True

    # --- Required directories ---
    if create_dirs:
        BASEOUTPUTDIR.mkdir(parents=True, exist_ok=True)
        INPUTDIR.mkdir(parents=True, exist_ok=True)

    if BASEOUTPUTDIR.exists():
        if verbose:
            print(f"✓ BASEOUTPUTDIR: {BASEOUTPUTDIR}")
    else:
        if verbose:
            print(f"⚠ BASEOUTPUTDIR not found: {BASEOUTPUTDIR}")
        ok = False

    if INPUTDIR.exists():
        if verbose:
            print(f"✓ INPUTDIR:      {INPUTDIR}")
    else:
        if verbose:
            print(f"⚠ INPUTDIR not found: {INPUTDIR}")
            print( "  Scripts will fail if no input data is available")
        ok = False

    # --- Optional InSAR directory (different drive - warn but do not fail) ---
    if INSARDIR.exists():
        if verbose:
            print(f"✓ INSARDIR:      {INSARDIR}")
    else:
        if verbose:
            print(f"⚠ INSARDIR not found: {INSARDIR}")
            print( "  InSAR integration will be skipped (--skip-insar will be set)")
            # FIX: Flag the cross-drive inconsistency for the operator
            if str(INSARDIR.drive).upper() != str(BASEOUTPUTDIR.drive).upper():
                print(f"  Note: INSARDIR is on drive {INSARDIR.drive} while other "
                      f"dirs are on {BASEOUTPUTDIR.drive} - check mount points")

    # --- Optional CanVec shapefile ---
    if CANVEC_WATERBODY_SHAPEFILE.exists():
        if verbose:
            print(f"✓ CanVec SHP:    {CANVEC_WATERBODY_SHAPEFILE}")
    else:
        if verbose:
            print(f"⚠ CanVec shapefile not found: {CANVEC_WATERBODY_SHAPEFILE}")
            print( "  Lake/ocean detection will fall back to elevation-only method")

    return ok

# ============================================================================
# STANDALONE DIAGNOSTIC (python config.py)
# ============================================================================
if __name__ == '__main__':
    print("=" * 80)
    print("ICESat-2 WORKFLOW CONFIGURATION")
    print("=" * 80)
    print(f"BASEOUTPUTDIR            : {BASEOUTPUTDIR}")
    print(f"INPUTDIR                 : {INPUTDIR}")
    print(f"INSARDIR                 : {INSARDIR}")
    print(f"CANVEC_WATERBODY_SHAPEFILE: {CANVEC_WATERBODY_SHAPEFILE}")
    print()
    valid = validate(create_dirs=False, verbose=True)
    print("=" * 80)
    print(f"Configuration valid: {valid}")
    print("=" * 80)
