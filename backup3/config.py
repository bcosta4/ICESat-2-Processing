"""
Centralized configuration for ICESat-2 processing workflow.
Defines paths, parameters, and environment settings.
All scripts import from this module.
"""

from pathlib import Path
import os

# Output directory (check environment first, then use default)
BASEOUTPUTDIR = Path(
    os.getenv(
        'ICESAT2_OUTPUT',
        'E:/Processing/ICeSat-2/output_data'  # Windows default
    )
)

# Raw data input directory
INPUTDIR = Path(
    os.getenv(
        'ICESAT2_INPUT',
        'E:/Processing/ICeSat-2/raw_data'  # Windows default
    )
)

# InSAR validation data
INSARDIR = Path(
    os.getenv(
        'INSAR_DATA',
        'C:/bcosta/InSAR_for_ICESat-2'  # Windows default - note inconsistent drive!
    )
)

# Ensure directories exist
BASEOUTPUTDIR.mkdir(parents=True, exist_ok=True)
INPUTDIR.mkdir(parents=True, exist_ok=True)

# Don't require INSARDIR to exist (optional data)
if INSARDIR.exists():
    print(f"✓ InSAR directory found: {INSARDIR}")
else:
    print(f"⚠ InSAR directory not found: {INSARDIR}")    

CANVEC_WATERBODY_SHAPEFILE = Path(
    os.getenv(
        'CANVEC_SHAPEFILE',
        r'E:\GIS\canvec_50K_NT_Hydro_shp\waterbody_2_2.shp'  # Default
    )
)

# Check availability and report
if CANVEC_WATERBODY_SHAPEFILE.exists():
    print(f"✓ CanVec shapefile found: {CANVEC_WATERBODY_SHAPEFILE}")
else:
    print(f"⚠ CanVec shapefile not found: {CANVEC_WATERBODY_SHAPEFILE}")
    print(f"  Lake detection will fall back to elevation-only method (reduced accuracy)")

# ============================================================================
# EXPORT FOR IMPORT
# ============================================================================
__all__ = ['BASEOUTPUTDIR', 'INPUTDIR', 'INSARDIR']

# ============================================================================
# PRINT CONFIGURATION (for debugging)
# ============================================================================
if __name__ == '__main__':
    print("=" * 80)
    print("ICESat-2 WORKFLOW CONFIGURATION")
    print("=" * 80)
    print(f"BASEOUTPUTDIR: {BASEOUTPUTDIR}")
    print(f"  Exists: {BASEOUTPUTDIR.exists()}")
    print(f"\nINPUTDIR: {INPUTDIR}")
    print(f"  Exists: {INPUTDIR.exists()}")
    print(f"\nINSARDIR: {INSARDIR}")
    print(f"  Exists: {INSARDIR.exists()}")
    print("=" * 80)