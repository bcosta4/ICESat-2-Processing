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
BASEOUTPUTDIR = Path(
    os.getenv('ICESAT2_OUTPUT', 'E:/Processing/ICeSat-2/output_data')
)

INPUTDIR = Path(
    os.getenv('ICESAT2_INPUT', 'E:/Processing/ICESat-2/raw_data')
)

INSARDIR = Path(
    os.getenv('INSAR_DATA', 'C:/bcosta/InSAR_for_ICESat-2')
)

CANVEC_WATERBODY_SHAPEFILE = Path(
    os.getenv('CANVEC_SHAPEFILE', r'E:\GIS\canvec_50K_NT_Hydro_shp\waterbody_2_2.shp')
)

# ============================================================================
# PRODUCT SELECTION (NEW)
# ============================================================================
# Valid ICESat-2 products supported by the workflow.
# Passed via --product on 08_run_workflow.py, propagated to Scripts 01-03.
VALIDPRODUCTS = ['ATL03', 'ATL06', 'ATL08']
DEFAULTPRODUCT = 'ATL08'  # matches legacy hardcoded behaviour

# File-glob patterns used by Script 01 when searching INPUTDIR
PRODUCTFILEPATTERNS = {
    'ATL03': 'processed*ATL03*.csv',   # PhoREAL-exported photon CSV/LAS
    'ATL08': 'processed*ATL08*.csv',   # PhoREAL-exported land/veg segment CSV/LAS
    'ATL06': '*ATL06*.h5',             # Standard NASA ATL06 HDF5 granules
}

# ============================================================================
# OUTLIER FILTER DEFAULTS (NEW)
# ============================================================================
# Controls Script 03's Tier-2 progressive sigma filter.
# Tier-1 physical-plausibility filtering is ALWAYS applied (per user decision).
DEFAULT_OUTLIER_FILTER_ENABLED = True
DEFAULT_OUTLIER_SIGMA_TIERS = [5.0, 2.0, 1.5]

# ============================================================================
# EXPORT LIST
# ============================================================================
__all__ = [
    'BASEOUTPUTDIR',
    'INPUTDIR',
    'INSARDIR',
    'CANVEC_WATERBODY_SHAPEFILE',
    'VALID_PRODUCTS',
    'DEFAULT_PRODUCT',
    'PRODUCT_FILE_PATTERNS',
    'DEFAULT_OUTLIER_FILTER_ENABLED',
    'DEFAULT_OUTLIER_SIGMA_TIERS',
]

# ============================================================================
# VALIDATION FUNCTION
# ============================================================================
def validate(create_dirs: bool = True, verbose: bool = True) -> bool:
    """
    Validate configuration paths and optionally create output directories.
    """
    _enc = getattr(sys.stdout, 'encoding', None) or ''
    if _enc.lower() != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    ok = True

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
            print(f"✓ INPUTDIR: {INPUTDIR}")
    else:
        if verbose:
            print(f"⚠ INPUTDIR not found: {INPUTDIR}")
            print("  Scripts will fail if no input data is available")
        ok = False

    if INSARDIR.exists():
        if verbose:
            print(f"✓ INSARDIR: {INSARDIR}")
    else:
        if verbose:
            print(f"⚠ INSARDIR not found: {INSARDIR}")
            print("  InSAR integration will be skipped (--skip-insar will be set)")
        if str(INSARDIR.drive).upper() != str(BASEOUTPUTDIR.drive).upper():
            print(f"  Note: INSARDIR is on drive {INSARDIR.drive} while other "
                  f"dirs are on {BASEOUTPUTDIR.drive} - check mount points")

    if CANVEC_WATERBODY_SHAPEFILE.exists():
        if verbose:
            print(f"✓ CanVec SHP: {CANVEC_WATERBODY_SHAPEFILE}")
    else:
        if verbose:
            print(f"⚠ CanVec shapefile not found: {CANVEC_WATERBODY_SHAPEFILE}")
            print("  Lake/ocean detection will fall back to elevation-only method")

    return ok


if __name__ == '__main__':
    print("=" * 80)
    print("ICESat-2 WORKFLOW CONFIGURATION")
    print("=" * 80)
    print(f"BASEOUTPUTDIR              : {BASEOUTPUTDIR}")
    print(f"INPUTDIR                   : {INPUTDIR}")
    print(f"INSARDIR                   : {INSARDIR}")
    print(f"CANVEC_WATERBODY_SHAPEFILE : {CANVEC_WATERBODY_SHAPEFILE}")
    print(f"VALIDPRODUCTS              : {VALIDPRODUCTS}")
    print(f"DEFAULTPRODUCT             : {DEFAULTPRODUCT}")
    print(f"DEFAULT_OUTLIER_FILTER      : {DEFAULT_OUTLIER_FILTER_ENABLED}")
    print(f"DEFAULT_OUTLIER_SIGMA_TIERS : {DEFAULT_OUTLIER_SIGMA_TIERS}")
    print()
    valid = validate(create_dirs=False, verbose=True)
    print("=" * 80)
    print(f"Configuration valid: {valid}")
    print("=" * 80)