"""
================================================================================
SCRIPT 00: ENHANCED SETUP VALIDATION - WITH REPRODUCIBILITY INFRASTRUCTURE
================================================================================

ENHANCEMENTS OVER ORIGINAL:

✓ Added laspy package check (critical for Script 01)
✓ Added memory availability check (prevents Script 04/06 crashes)
✓ Added Python version validation (requires 3.7+)
✓ Added InSAR data directory validation (Script 03)
✓ Added detailed package version reporting
✓ Added fixes validation (checks if memory-optimized scripts are used)
✓ Better error messages with actionable fixes
✓ UTF-8 ENCODING FIX (NO UNICODE ERRORS!)
✓ GIT HASH TRACKING (reproducibility)
✓ RUN METADATA LOGGING (audit trail)
✓ COMPLETE LOGGING INFRASTRUCTURE (file + console)

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
✓ Complete reproducibility infrastructure (logging, metadata, git hash)

================================================================================
"""

import sys
import ast
import io
import json
import logging
from pathlib import Path
from datetime import datetime
import shutil
import platform
import subprocess

# UTF-8 ENCODING FIX (CRITICAL - FIXES YOUR UNICODE ERROR!)
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================================
# GIT HASH TRACKING
# ============================================================================

def get_git_hash():
    """Get git commit hash for reproducibility"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip()[:8] if result.returncode == 0 else 'unknown'
    except Exception:
        return 'unknown'


# ============================================================================
# LOGGING SETUP (DUAL OUTPUT: FILE + CONSOLE)
# ============================================================================

def setup_logging(output_dir):
    """Configure logging to both file and console"""
    
    log_file = output_dir / 'script_00_run.log'
    
    logger = logging.getLogger('script_00')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    # File handler - UTF-8 encoding, full details
    fh = logging.FileHandler(log_file, encoding='utf-8', mode='w')
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(fh_formatter)
    
    # Console handler - INFO level, clean output
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ============================================================================
# RUN METADATA CLASS (AUDIT TRAIL)
# ============================================================================

class RunMetadata:
    """Capture complete run parameters for reproducibility"""
    
    def __init__(self, run_id, git_hash):
        self.run_id = run_id
        self.script_name = '00_setup_validation'
        self.script_version = 'v1.0-reproducible'
        self.git_hash = git_hash
        self.start_time = datetime.now().isoformat()
        self.end_time = None
        self.validation_results = {}
        self.environment_info = {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'platform': platform.system(),
            'platform_release': platform.release(),
            'is_64bit': sys.maxsize > 2**32,
        }
    
    def finalize(self):
        """Mark run as complete"""
        self.end_time = datetime.now().isoformat()
    
    def save(self, output_dir):
        """Save metadata as JSON"""
        metadata_file = output_dir / f'{self.script_name}_metadata.json'
        
        metadata = {
            'run_id': self.run_id,
            'script': self.script_name,
            'version': self.script_version,
            'git_hash': self.git_hash,
            'environment': self.environment_info,
            'timestamps': {
                'start': self.start_time,
                'end': self.end_time,
            },
            'validation_results': self.validation_results,
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str, ensure_ascii=False)
        
        return metadata_file


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
# PRINT HEADER & SETUP LOGGING
# ============================================================================

print("="*80)
print("SCRIPT 00: ENHANCED SETUP VALIDATION & ENVIRONMENT CHECK")
print("="*80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Python: {sys.version.split()[0]}")
print("="*80 + "\n")

# Get git hash for reproducibility
git_hash = get_git_hash()
print(f"Git Hash: {git_hash}\n")

# Get next run folder and setup logging
def get_next_run_folder():
    """Create folder name: run_YYYYMMDD_n1, run_YYYYMMDD_n2, etc."""
    today = datetime.now().strftime('%Y%m%d')
    existing_runs = list(BASE_OUTPUT_DIR.glob(f'run_{today}_n*'))
    if len(existing_runs) == 0:
        run_number = 1
    else:
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
run_output_dir = BASE_OUTPUT_DIR / f'run_{run_id}'
run_output_dir.mkdir(parents=True, exist_ok=True)

# Setup logging
logger = setup_logging(run_output_dir)

# Create metadata tracker
metadata = RunMetadata(run_id, git_hash)

logger.info("="*80)
logger.info("SCRIPT 00: SETUP VALIDATION")
logger.info("="*80)
logger.info(f"Run ID: {run_id}")
logger.info(f"Git Hash: {git_hash}")
logger.info(f"Output directory: {run_output_dir}")

print(f"\nRun ID: {run_id}\n")

# ============================================================================
# PART 0: PYTHON VERSION VALIDATION
# ============================================================================

print("PART 0: PYTHON VERSION VALIDATION")
print("-"*80)

logger.info("\nPART 0: PYTHON VERSION VALIDATION")

version_info = sys.version_info
python_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"

if version_info >= (3, 7):
    print(f" [✓] Python {python_version} (requires 3.7+)")
    logger.info(f"Python {python_version} - OK")
else:
    print(f" [✗] Python {python_version} is too old (requires 3.7+)")
    logger.error(f"Python {python_version} - FAILED (requires 3.7+)")
    print(f" Please upgrade Python")
    sys.exit(1)

# Check 64-bit (required for large datasets)
if sys.maxsize > 2**32:
    print(f" [✓] 64-bit Python detected")
    logger.info("64-bit Python - OK")
else:
    print(f" [✗] 32-bit Python detected (requires 64-bit for large datasets)")
    logger.error("32-bit Python - FAILED (requires 64-bit)")
    sys.exit(1)

print()

# ============================================================================
# PART 1: CHECK REQUIRED PYTHON PACKAGES (ENHANCED)
# ============================================================================

print("PART 1: PYTHON PACKAGE VALIDATION (ENHANCED)")
print("-"*80)

logger.info("\nPART 1: PYTHON PACKAGE VALIDATION (ENHANCED)")

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
        version = ""
        if hasattr(mod, '__version__'):
            version = f"v{mod.__version__}"
        print(f" [✓] {package:15s} {version:12s} - {description}")
        logger.info(f"{package:15s} {version:12s} - {description}")
        installed_packages.append(package)
    except ImportError:
        if is_critical:
            print(f" [✗] {package:15s} {'':12s} - {description} (CRITICAL - MISSING)")
            logger.error(f"{package:15s} - CRITICAL - MISSING")
            missing_critical.append(package)
        else:
            print(f" [⚠] {package:15s} {'':12s} - {description} (Optional - missing)")
            logger.warning(f"{package:15s} - Optional - missing")
            missing_optional.append(package)

print()

print(f"Package Status: {len(installed_packages)}/{len(required_packages)} installed")
logger.info(f"Package Status: {len(installed_packages)}/{len(required_packages)} installed")

if missing_critical:
    print(f"\n[ERROR] Missing {len(missing_critical)} CRITICAL packages:")
    logger.error(f"Missing {len(missing_critical)} CRITICAL packages: {missing_critical}")
    print("\nInstall with:")
    print(f" pip install {' '.join(missing_critical)}")
    print("\nCritical packages are required for pipeline to run.")
    sys.exit(1)

if missing_optional:
    print(f"\n[INFO] Missing {len(missing_optional)} optional packages:")
    logger.info(f"Missing {len(missing_optional)} optional packages: {missing_optional}")
    for pkg in missing_optional:
        print(f" pip install {pkg}")
    print("\nOptional packages enhance functionality but are not required.")

print("[✓] All critical packages available\n")
logger.info("[✓] All critical packages available")

# ============================================================================
# PART 1B: MEMORY AVAILABILITY CHECK (NEW)
# ============================================================================

print("PART 1B: MEMORY AVAILABILITY CHECK")
print("-"*80)

logger.info("\nPART 1B: MEMORY AVAILABILITY CHECK")

try:
    import psutil
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024**3)
    available_gb = mem.available / (1024**3)
    percent_used = mem.percent
    
    print(f" Total RAM: {total_gb:.1f} GB")
    print(f" Available RAM: {available_gb:.1f} GB ({100-percent_used:.1f}% free)")
    logger.info(f"Memory - Total: {total_gb:.1f}GB, Available: {available_gb:.1f}GB ({100-percent_used:.1f}% free)")
    
    if available_gb >= 8:
        print(f" [✓] Sufficient memory for large datasets (needs 4-8GB)")
        logger.info("Memory check - OK (sufficient)")
    elif available_gb >= 4:
        print(f" [⚠] Limited memory ({available_gb:.1f} GB available)")
        print(f" Close other applications before running Scripts 04 & 06")
        logger.warning(f"Memory check - LIMITED ({available_gb:.1f}GB available)")
    else:
        print(f" [✗] Insufficient memory ({available_gb:.1f} GB available, needs 4GB+)")
        print(f" Memory-optimized scripts required for your dataset size")
        print(f" Close all other applications and try again")
        logger.error(f"Memory check - INSUFFICIENT ({available_gb:.1f}GB available, needs 4GB+)")
except ImportError:
    print(f" [⚠] psutil not installed - cannot check memory")
    print(f" Install with: pip install psutil")
    print(f" (Optional check - pipeline can still run)")
    logger.warning("psutil not installed - memory check skipped (optional)")

print()

# ============================================================================
# PART 2: VALIDATE WORKFLOW SCRIPTS (ENHANCED)
# ============================================================================

print("PART 2: WORKFLOW SCRIPT VALIDATION")
print("-"*80)

logger.info("\nPART 2: WORKFLOW SCRIPT VALIDATION")

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
        logger.warning(f"{script_name} - FILE NOT FOUND")
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
        logger.info(f"{script_name:35s} - {len(imports)} imports {opt_status}")
        passed_scripts.append(script_name)
    else:
        print(f" [✗] {script_name:35s} - {syntax_msg}")
        logger.error(f"{script_name} - {syntax_msg}")
        syntax_errors.append((script_name, syntax_msg))

print()

print(f"Script Status: {len(passed_scripts)}/{len(workflow_scripts)} valid")
logger.info(f"Script Status: {len(passed_scripts)}/{len(workflow_scripts)} valid")

if syntax_errors:
    print(f"\n[ERROR] {len(syntax_errors)} scripts have syntax errors:")
    logger.error(f"{len(syntax_errors)} scripts have syntax errors:")
    for script, error in syntax_errors:
        print(f" {script}: {error}")
        logger.error(f" {script}: {error}")
    print("\nFix syntax errors and run this script again.")
    sys.exit(1)

if optimization_warnings:
    print(f"\n[WARNING] {len(optimization_warnings)} scripts may cause memory errors on large datasets:")
    logger.warning(f"{len(optimization_warnings)} scripts may cause memory errors:")
    for script in optimization_warnings:
        print(f" • {script} - Use memory-optimized version")
        logger.warning(f" {script} - Use memory-optimized version")
    print("\nRecommendation: Replace with *_MEMORY_FIXED.py versions")
    print("See CRITICAL_FIXES_SUMMARY.md for details")

print("[✓] All workflow scripts have valid syntax\n")
logger.info("[✓] All workflow scripts have valid syntax")

# ============================================================================
# PART 3: VERIFY DIRECTORY STRUCTURE (ENHANCED)
# ============================================================================

print("PART 3: DIRECTORY STRUCTURE VALIDATION")
print("-"*80)

logger.info("\nPART 3: DIRECTORY STRUCTURE VALIDATION")

# Check scripts directory
if SCRIPTS_DIR.exists():
    script_count = len(list(SCRIPTS_DIR.glob('*.py')))
    print(f" [✓] Scripts directory exists: {SCRIPTS_DIR}")
    print(f" Found {script_count} Python scripts")
    logger.info(f"Scripts directory - OK ({script_count} scripts)")
else:
    print(f" [✗] Scripts directory NOT FOUND: {SCRIPTS_DIR}")
    logger.error(f"Scripts directory NOT FOUND: {SCRIPTS_DIR}")
    sys.exit(1)

# Check/create output directory
if BASE_OUTPUT_DIR.exists():
    print(f" [✓] Output directory exists: {BASE_OUTPUT_DIR}")
    logger.info(f"Output directory - OK")
else:
    try:
        BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f" [✓] Created output directory: {BASE_OUTPUT_DIR}")
        logger.info(f"Output directory - CREATED")
    except Exception as e:
        print(f" [✗] Cannot create output directory: {e}")
        logger.error(f"Cannot create output directory: {e}")
        sys.exit(1)

# Check raw data directory (optional)
if RAW_DATA_DIR.exists():
    las_files = len(list(RAW_DATA_DIR.glob('*.las')))
    csv_files = len(list(RAW_DATA_DIR.glob('*.csv')))
    print(f" [✓] Raw data directory exists: {RAW_DATA_DIR}")
    print(f" Found {las_files} LAS files, {csv_files} CSV files")
    logger.info(f"Raw data directory - OK ({las_files} LAS, {csv_files} CSV)")
else:
    print(f" [⚠] Raw data directory not found: {RAW_DATA_DIR}")
    print(f" (Optional - specify path in Script 01)")
    logger.info(f"Raw data directory - NOT FOUND (optional)")

# Check InSAR directory (optional)
if INSAR_DIR.exists():
    tif_files = len(list(INSAR_DIR.glob('*.tif')))
    print(f" [✓] InSAR directory exists: {INSAR_DIR}")
    print(f" Found {tif_files} GeoTIFF files")
    logger.info(f"InSAR directory - OK ({tif_files} GeoTIFF files)")
else:
    print(f" [⚠] InSAR directory not found: {INSAR_DIR}")
    print(f" (Optional - InSAR integration will be skipped)")
    logger.info(f"InSAR directory - NOT FOUND (optional)")

print()

# ============================================================================
# PART 4: TEST I/O PERMISSIONS & DISK SPACE (ENHANCED)
# ============================================================================

print("PART 4: I/O PERMISSIONS & DISK SPACE")
print("-"*80)

logger.info("\nPART 4: I/O PERMISSIONS & DISK SPACE")

# Test write permissions
test_file = BASE_OUTPUT_DIR / 'io_test.txt'
try:
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write('test')
    test_file.unlink()
    print(f" [✓] Write permissions OK")
    logger.info("Write permissions - OK")
except Exception as e:
    print(f" [✗] Cannot write to {BASE_OUTPUT_DIR}: {e}")
    logger.error(f"Cannot write to {BASE_OUTPUT_DIR}: {e}")
    sys.exit(1)

# Check disk space
try:
    stat = shutil.disk_usage(BASE_OUTPUT_DIR)
    free_gb = stat.free / (1024**3)
    total_gb = stat.total / (1024**3)
    used_pct = (stat.used / stat.total) * 100
    
    print(f" Total disk: {total_gb:.1f} GB")
    print(f" Free space: {free_gb:.1f} GB ({100-used_pct:.1f}% free)")
    logger.info(f"Disk space - Total: {total_gb:.1f}GB, Free: {free_gb:.1f}GB ({100-used_pct:.1f}% free)")
    
    estimated_need_gb = 15
    if free_gb < estimated_need_gb:
        print(f" [✗] Insufficient disk space (needs ~{estimated_need_gb}GB, have {free_gb:.1f}GB)")
        logger.error(f"Insufficient disk space: needs ~{estimated_need_gb}GB, have {free_gb:.1f}GB")
        print(f" Free up disk space before running pipeline")
        sys.exit(1)
    elif free_gb < estimated_need_gb * 2:
        print(f" [⚠] Limited disk space ({free_gb:.1f} GB free, recommended: >{estimated_need_gb*2}GB)")
        logger.warning(f"Limited disk space: {free_gb:.1f}GB free")
    else:
        print(f" [✓] Sufficient disk space available")
        logger.info("Disk space - OK (sufficient)")
except Exception as e:
    print(f" [⚠] Cannot check disk space: {e}")
    logger.warning(f"Cannot check disk space: {e}")

print()

# ============================================================================
# PART 5: INITIALIZE RUN TIMESTAMP
# ============================================================================

print("PART 5: RUN INITIALIZATION")
print("-"*80)

logger.info("\nPART 5: RUN INITIALIZATION")

# Create timestamp file
timestamp_file = BASE_OUTPUT_DIR / 'latest_run_timestamp.txt'

try:
    with open(timestamp_file, 'w', encoding='utf-8') as f:
        f.write(run_id)
    print(f" [✓] Run ID: {run_id}")
    print(f" [✓] Timestamp saved: {timestamp_file}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Timestamp file created")
except Exception as e:
    print(f" [✗] Cannot write timestamp: {e}")
    logger.error(f"Cannot write timestamp: {e}")
    sys.exit(1)

print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("="*80)
print("VALIDATION SUMMARY")
print("="*80)

logger.info("\nVALIDATION SUMMARY")

checks = [
    ("Python version (3.7+)", version_info >= (3, 7)),
    ("64-bit Python", sys.maxsize > 2**32),
    ("Critical packages", len(missing_critical) == 0),
    ("Workflow scripts", len(syntax_errors) == 0),
    ("Directory structure", SCRIPTS_DIR.exists() and BASE_OUTPUT_DIR.exists()),
    ("I/O permissions", True),
    ("Disk space (>15GB)", free_gb >= 15 if 'free_gb' in locals() else True),
    ("Timestamp initialization", timestamp_file.exists()),
]

all_passed = True
warnings_present = False

for check_name, result in checks:
    status = "[✓]" if result else "[✗]"
    print(f"{status} {check_name}")
    logger.info(f"{status} {check_name}")
    if not result:
        all_passed = False

print()

# Check warnings
if missing_optional:
    print(f"[⚠] Optional packages missing ({len(missing_optional)})")
    logger.warning(f"Optional packages missing ({len(missing_optional)})")
    warnings_present = True

if optimization_warnings:
    print(f"[⚠] Scripts need memory optimization ({len(optimization_warnings)})")
    logger.warning(f"Scripts need memory optimization ({len(optimization_warnings)})")
    warnings_present = True

print()

if all_passed:
    print("[✓] ALL CRITICAL VALIDATION CHECKS PASSED")
    logger.info("[✓] ALL CRITICAL VALIDATION CHECKS PASSED")
    if warnings_present:
        print("\nWarnings detected (optional - pipeline can still run):")
        logger.info("Warnings present but pipeline can still run")
        if missing_optional:
            print(f" • {len(missing_optional)} optional packages missing")
        if optimization_warnings:
            print(f" • {len(optimization_warnings)} scripts may hit memory limits on large datasets")
            print(" Recommendation: Use *_MEMORY_FIXED.py versions")
    
    print()
    print("Environment is ready for ICESat-2 processing pipeline!")
    logger.info("Environment is ready for ICESat-2 processing pipeline!")
    print()
    print("Next steps:")
    print(f" 1. Run: python {SCRIPTS_DIR / '01_filter_photons.py'}")
    print(f" 2. Or full workflow: python {SCRIPTS_DIR / '08_run_workflow.py'} --temporal 2")
    print()
    
    if optimization_warnings:
        print("⚠ IMPORTANT: For large datasets (12.8M+ crossovers):")
        print(" Replace Scripts 04 & 06 with memory-optimized versions")
        print(" See: CRITICAL_FIXES_SUMMARY.md")
        print()
    
    # Finalize metadata and save
    metadata.finalize()
    metadata.validation_results = {
        'passed_checks': len([c for c in checks if c[1]]),
        'total_checks': len(checks),
        'all_passed': all_passed,
    }
    metadata_file = metadata.save(run_output_dir)
    logger.info(f"Metadata saved: {metadata_file}")
    
    sys.exit(0)
else:
    print("[✗] SOME CRITICAL CHECKS FAILED")
    logger.error("[✗] SOME CRITICAL CHECKS FAILED")
    print()
    print("Please address the issues above before proceeding.")
    logger.error("Please address the issues above before proceeding.")
    print()
    sys.exit(1)

# ============================================================================
