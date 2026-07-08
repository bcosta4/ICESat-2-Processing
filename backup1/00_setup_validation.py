"""
================================================================================
SCRIPT 00: ENHANCED SETUP VALIDATION - COMPREHENSIVE CHECK
================================================================================

ENHANCEMENTS OVER ORIGINAL:
✓ Added laspy package check (critical for Script 01)
✓ Added memory availability check (prevents Script 04/06 crashes)
✓ Added Python version validation (requires 3.7+)
✓ Added InSAR data directory validation (Script 03)
✓ Added detailed package version reporting
✓ Added fixes validation (checks if memory-optimized scripts are used)
✓ Better error messages with actionable fixes

ORIGINAL VALIDATIONS:
✓ Python packages (14 core packages)
✓ Script syntax validation
✓ Directory structure
✓ I/O permissions
✓ Disk space check
✓ Timestamp initialization

NEW VALIDATIONS:
✓ Memory availability (needs 8GB+ for 12.8M dataset)
✓ Python version (3.7+ required)
✓ laspy package (critical for LAS file reading)
✓ InSAR data availability (optional but recommended)
✓ Memory-optimized scripts check (Scripts 04, 06)

================================================================================
"""

import sys
import ast
from pathlib import Path
from datetime import datetime
import shutil
import platform

# UTF-8 encoding support
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*80)
print("SCRIPT 00: ENHANCED SETUP VALIDATION & ENVIRONMENT CHECK")
print("="*80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Python: {sys.version.split()[0]}")
print("="*80 + "\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPTS_DIR = Path(r'E:\Processing\ICeSat-2\scripts')
BASE_OUTPUT_DIR = Path(r'E:\Processing\ICeSat-2\output_data')
INSAR_DIR = Path(r'C:\bcosta\InSAR_for_ICeSat-2')  # Optional
RAW_DATA_DIR = Path(r'E:\Processing\ICeSat-2\raw_data')  # Optional

# Create base output directory if missing
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# PART 0: PYTHON VERSION CHECK (NEW)
# ============================================================================

print("PART 0: PYTHON VERSION VALIDATION")
print("-"*80)

version_info = sys.version_info
python_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"

if version_info >= (3, 7):
    print(f" [✓] Python {python_version} (requires 3.7+)")
else:
    print(f" [✗] Python {python_version} is too old (requires 3.7+)")
    print(f"     Please upgrade Python")
    sys.exit(1)

# Check 64-bit (required for large datasets)
if sys.maxsize > 2**32:
    print(f" [✓] 64-bit Python detected")
else:
    print(f" [✗] 32-bit Python detected (requires 64-bit for large datasets)")
    sys.exit(1)

print()

# ============================================================================
# PART 1: CHECK REQUIRED PYTHON PACKAGES (ENHANCED)
# ============================================================================

print("PART 1: PYTHON PACKAGE VALIDATION (ENHANCED)")
print("-"*80)

required_packages = [
    ("pandas", "Data manipulation", True),
    ("numpy", "Numerical computing", True),
    ("scipy", "Scientific computing", True),
    ("matplotlib", "Visualization", True),
    ("h5py", "HDF5 file IO", True),
    ("rasterio", "Raster data handling", True),
    ("geopandas", "Geospatial data", True),
    ("shapely", "Geometry operations", True),
    ("pyproj", "Coordinate transformations", True),
    ("seaborn", "Statistical visualization", True),
    ("netCDF4", "NetCDF file IO", False),  # Optional
    ("xarray", "Multi-dimensional arrays", False),  # Optional
    ("sklearn", "Machine learning", True),
    ("PIL", "Image processing", True),
    ("laspy", "LAS file reading", True),  # CRITICAL - was missing!
]

missing_critical = []
missing_optional = []
installed_packages = []

for package, description, is_critical in required_packages:
    try:
        mod = __import__(package)
        
        # Get version if available
        version = ""
        if hasattr(mod, '__version__'):
            version = f"v{mod.__version__}"
        
        print(f" [✓] {package:15s} {version:12s} - {description}")
        installed_packages.append(package)
        
    except ImportError:
        if is_critical:
            print(f" [✗] {package:15s} {'':12s} - {description} (CRITICAL - MISSING)")
            missing_critical.append(package)
        else:
            print(f" [⚠] {package:15s} {'':12s} - {description} (Optional - missing)")
            missing_optional.append(package)

print()
print(f"Package Status: {len(installed_packages)}/{len(required_packages)} installed")

if missing_critical:
    print(f"\n[ERROR] Missing {len(missing_critical)} CRITICAL packages:")
    print("\nInstall with:")
    print(f"  pip install {' '.join(missing_critical)}")
    print("\nCritical packages are required for pipeline to run.")
    sys.exit(1)

if missing_optional:
    print(f"\n[INFO] Missing {len(missing_optional)} optional packages:")
    for pkg in missing_optional:
        print(f"  pip install {pkg}")
    print("\nOptional packages enhance functionality but are not required.")

print("[✓] All critical packages available\n")

# ============================================================================
# PART 1B: MEMORY AVAILABILITY CHECK (NEW)
# ============================================================================

print("PART 1B: MEMORY AVAILABILITY CHECK")
print("-"*80)

try:
    import psutil
    
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    available_gb = mem.available / (1024**3)
    percent_used = mem.percent
    
    print(f" Total RAM: {total_gb:.1f} GB")
    print(f" Available RAM: {available_gb:.1f} GB ({100-percent_used:.1f}% free)")
    
    # Check if enough for 12.8M dataset
    if available_gb >= 8:
        print(f" [✓] Sufficient memory for large datasets (needs 4-8GB)")
    elif available_gb >= 4:
        print(f" [⚠] Limited memory ({available_gb:.1f} GB available)")
        print(f"     Close other applications before running Scripts 04 & 06")
    else:
        print(f" [✗] Insufficient memory ({available_gb:.1f} GB available, needs 4GB+)")
        print(f"     Memory-optimized scripts required for your dataset size")
        print(f"     Close all other applications and try again")
    
except ImportError:
    print(f" [⚠] psutil not installed - cannot check memory")
    print(f"     Install with: pip install psutil")
    print(f"     (Optional check - pipeline can still run)")

print()

# ============================================================================
# PART 2: VALIDATE WORKFLOW SCRIPTS (ENHANCED)
# ============================================================================

print("PART 2: WORKFLOW SCRIPT VALIDATION")
print("-"*80)

def check_syntax(filepath):
    """Validate Python syntax by parsing AST"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, "OK"
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def check_imports(filepath):
    """Extract and validate required imports"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        tree = ast.parse(code)
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module if node.module else 'relative')
        
        return imports
    except Exception:
        return []

def check_if_memory_optimized(filepath):
    """Check if script has memory optimization (chunked processing)"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for chunking patterns
        has_chunk_size = 'CHUNK_SIZE' in content or 'chunk_size' in content
        has_chunk_loop = 'range(0, len(data), ' in content or 'range(0, len(' in content
        
        return has_chunk_size and has_chunk_loop
    except:
        return False

# Define workflow scripts
workflow_scripts = [
    ('00_setup_validation.py', 'Setup & Validation', False),
    ('01_filter_photons.py', 'Photon Filtering', False),
    ('02_filter_terrain.py', 'Terrain Classification', False),
    ('03_analysis_core.py', 'Core Analysis', False),
    ('04_analysis_spatial.py', 'Spatial Analysis', True),  # Needs optimization
    ('05_analysis_temporal.py', 'Temporal Analysis', False),
    ('05b_temporal_stratification.py', 'Temporal Stratification', False),
    ('06_analysis_advanced.py', 'Advanced Analysis', True),  # Needs optimization
    ('07_output_generation.py', 'Output Generation', False),
]

syntax_errors = []
passed_scripts = []
optimization_warnings = []

for script_name, description, needs_optimization in workflow_scripts:
    script_path = SCRIPTS_DIR / script_name
    
    # Check if file exists
    if not script_path.exists():
        print(f" [✗] {script_name:35s} - FILE NOT FOUND")
        syntax_errors.append((script_name, "File not found"))
        continue
    
    # Check syntax
    syntax_ok, syntax_msg = check_syntax(script_path)
    
    if syntax_ok:
        imports = check_imports(script_path)
        
        # Check for memory optimization if needed
        if needs_optimization:
            is_optimized = check_if_memory_optimized(script_path)
            if is_optimized:
                opt_status = "(memory-optimized ✓)"
            else:
                opt_status = "(NEEDS OPTIMIZATION ⚠)"
                optimization_warnings.append(script_name)
        else:
            opt_status = ""
        
        print(f" [✓] {script_name:35s} - {len(imports)} imports {opt_status}")
        passed_scripts.append(script_name)
    else:
        print(f" [✗] {script_name:35s} - {syntax_msg}")
        syntax_errors.append((script_name, syntax_msg))

print()
print(f"Script Status: {len(passed_scripts)}/{len(workflow_scripts)} valid")

if syntax_errors:
    print(f"\n[ERROR] {len(syntax_errors)} scripts have syntax errors:")
    for script, error in syntax_errors:
        print(f" {script}: {error}")
    print("\nFix syntax errors and run this script again.")
    sys.exit(1)

if optimization_warnings:
    print(f"\n[WARNING] {len(optimization_warnings)} scripts may cause memory errors on large datasets:")
    for script in optimization_warnings:
        print(f" • {script} - Use memory-optimized version")
    print("\nRecommendation: Replace with *_MEMORY_FIXED.py versions")
    print("See CRITICAL_FIXES_SUMMARY.md for details")

print("[✓] All workflow scripts have valid syntax\n")

# ============================================================================
# PART 3: VERIFY DIRECTORY STRUCTURE (ENHANCED)
# ============================================================================

print("PART 3: DIRECTORY STRUCTURE VALIDATION")
print("-"*80)

# Check scripts directory
if SCRIPTS_DIR.exists():
    script_count = len(list(SCRIPTS_DIR.glob('*.py')))
    print(f" [✓] Scripts directory exists: {SCRIPTS_DIR}")
    print(f"     Found {script_count} Python scripts")
else:
    print(f" [✗] Scripts directory NOT FOUND: {SCRIPTS_DIR}")
    sys.exit(1)

# Check/create output directory
if BASE_OUTPUT_DIR.exists():
    print(f" [✓] Output directory exists: {BASE_OUTPUT_DIR}")
else:
    try:
        BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f" [✓] Created output directory: {BASE_OUTPUT_DIR}")
    except Exception as e:
        print(f" [✗] Cannot create output directory: {e}")
        sys.exit(1)

# Check raw data directory (optional)
if RAW_DATA_DIR.exists():
    las_files = len(list(RAW_DATA_DIR.glob('*.las')))
    csv_files = len(list(RAW_DATA_DIR.glob('*.csv')))
    print(f" [✓] Raw data directory exists: {RAW_DATA_DIR}")
    print(f"     Found {las_files} LAS files, {csv_files} CSV files")
else:
    print(f" [⚠] Raw data directory not found: {RAW_DATA_DIR}")
    print(f"     (Optional - specify path in Script 01)")

# Check InSAR directory (optional)
if INSAR_DIR.exists():
    tif_files = len(list(INSAR_DIR.glob('*.tif')))
    print(f" [✓] InSAR directory exists: {INSAR_DIR}")
    print(f"     Found {tif_files} GeoTIFF files")
else:
    print(f" [⚠] InSAR directory not found: {INSAR_DIR}")
    print(f"     (Optional - InSAR integration will be skipped)")

print()

# ============================================================================
# PART 4: TEST I/O PERMISSIONS & DISK SPACE (ENHANCED)
# ============================================================================

print("PART 4: I/O PERMISSIONS & DISK SPACE")
print("-"*80)

# Test write permissions
test_file = BASE_OUTPUT_DIR / 'io_test.txt'
try:
    with open(test_file, 'w') as f:
        f.write('test')
    test_file.unlink()
    print(f" [✓] Write permissions OK")
except Exception as e:
    print(f" [✗] Cannot write to {BASE_OUTPUT_DIR}: {e}")
    sys.exit(1)

# Check disk space
try:
    stat = shutil.disk_usage(BASE_OUTPUT_DIR)
    free_gb = stat.free / (1024**3)
    total_gb = stat.total / (1024**3)
    used_pct = (stat.used / stat.total) * 100
    
    print(f" Total disk: {total_gb:.1f} GB")
    print(f" Free space: {free_gb:.1f} GB ({100-used_pct:.1f}% free)")
    
    # Estimate space needed for 29M photons
    estimated_need_gb = 15  # ~8GB CSV + 5GB intermediate + 2GB outputs
    
    if free_gb < estimated_need_gb:
        print(f" [✗] Insufficient disk space (needs ~{estimated_need_gb}GB, have {free_gb:.1f}GB)")
        print(f"     Free up disk space before running pipeline")
        sys.exit(1)
    elif free_gb < estimated_need_gb * 2:
        print(f" [⚠] Limited disk space ({free_gb:.1f} GB free, recommended: >{estimated_need_gb*2}GB)")
    else:
        print(f" [✓] Sufficient disk space available")
        
except Exception as e:
    print(f" [⚠] Cannot check disk space: {e}")

print()

# ============================================================================
# PART 5: INITIALIZE RUN TIMESTAMP
# ============================================================================

print("PART 5: RUN INITIALIZATION")
print("-"*80)

# Get next run number for today
def get_next_run_folder():
    """Create folder name: run_YYYYMMDD_n1, run_YYYYMMDD_n2, etc."""
    today = datetime.now().strftime('%Y%m%d')
    
    # Find existing runs for today
    existing_runs = list(BASE_OUTPUT_DIR.glob(f'run_{today}_n*'))
    
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
    return run_id

run_id = get_next_run_folder()
print(f" [✓] Run ID: {run_id}")

# Create timestamp file
timestamp_file = BASE_OUTPUT_DIR / 'latest_run_timestamp.txt'
try:
    with open(timestamp_file, 'w') as f:
        f.write(run_id)
    print(f" [✓] Timestamp saved: {timestamp_file}")
except Exception as e:
    print(f" [✗] Cannot write timestamp: {e}")
    sys.exit(1)

# Create run output directory
run_output_dir = BASE_OUTPUT_DIR / f'run_{run_id}'
try:
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f" [✓] Run output directory: {run_output_dir}")
except Exception as e:
    print(f" [✗] Cannot create run directory: {e}")
    sys.exit(1)

print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("="*80)
print("VALIDATION SUMMARY")
print("="*80)

checks = [
    ("Python version (3.7+)", version_info >= (3, 7)),
    ("64-bit Python", sys.maxsize > 2**32),
    ("Critical packages", len(missing_critical) == 0),
    ("Workflow scripts", len(syntax_errors) == 0),
    ("Directory structure", SCRIPTS_DIR.exists() and BASE_OUTPUT_DIR.exists()),
    ("I/O permissions", True),  # Already checked above
    ("Disk space (>15GB)", free_gb >= 15),
    ("Timestamp initialization", timestamp_file.exists()),
]

all_passed = True
warnings_present = False

for check_name, result in checks:
    status = "[✓]" if result else "[✗]"
    print(f"{status} {check_name}")
    if not result:
        all_passed = False

# Check warnings
if missing_optional:
    print(f"[⚠] Optional packages missing ({len(missing_optional)})")
    warnings_present = True

if optimization_warnings:
    print(f"[⚠] Scripts need memory optimization ({len(optimization_warnings)})")
    warnings_present = True

print()

if all_passed:
    print("[✓] ALL CRITICAL VALIDATION CHECKS PASSED")
    
    if warnings_present:
        print("\nWarnings detected (optional - pipeline can still run):")
        if missing_optional:
            print(f" • {len(missing_optional)} optional packages missing")
        if optimization_warnings:
            print(f" • {len(optimization_warnings)} scripts may hit memory limits on large datasets")
            print("   Recommendation: Use *_MEMORY_FIXED.py versions")
    
    print()
    print("Environment is ready for ICESat-2 processing pipeline!")
    print()
    print("Next steps:")
    print(f" 1. Run: python {SCRIPTS_DIR / '01_filter_photons.py'}")
    print(f" 2. Or full workflow: python {SCRIPTS_DIR / '08_run_workflow.py'} --temporal 2")
    print()
    
    if optimization_warnings:
        print("⚠ IMPORTANT: For large datasets (12.8M+ crossovers):")
        print("  Replace Scripts 04 & 06 with memory-optimized versions")
        print("  See: CRITICAL_FIXES_SUMMARY.md")
        print()
    
    sys.exit(0)

else:
    print("[✗] SOME CRITICAL CHECKS FAILED")
    print()
    print("Please address the issues above before proceeding.")
    print()
    sys.exit(1)

# ============================================================================
