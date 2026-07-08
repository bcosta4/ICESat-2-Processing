
# =============================================================================
# HEADER / LOGGING BLOCKS - ICESat-2 Permafrost Workflow (Scripts 00-08)
# Reproducibility standard: unified schema via repro_utils.py
# Console banner stays clean; git hash, command line, checksums, and full
# environment detail are written ONLY to the .log file and *_metadata.json
# =============================================================================


# =============================================================================
# SCRIPT 00: SETUP & VALIDATION
# =============================================================================
"""
SCRIPT 00: SETUP AND VALIDATION

Purpose
-------
Initialise a new run: generate RUN_TIMESTAMP, create the output directory,
validate the environment, and write the initial reproducibility metadata
record (git commit hash, command line, environment fingerprint) that all
downstream scripts (01-08) will extend.

Reproducibility
----------------
- Full provenance (git hash, command line, environment) is captured via
  repro_utils.RunMetadata and written to 00_metadata.json and script00_run.log.
- Console output remains a clean progress banner only.
"""

import sys
from pathlib import Path
from datetime import datetime
from repro_utils import setup_logging, RunMetadata

SCRIPT_NAME = "00_setup_validation"
SCRIPT_VERSION = "v1.0"

print("=" * 80, flush=True)
print("SCRIPT 00: SETUP AND VALIDATION", flush=True)
print("=" * 80, flush=True)

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(BASE_OUTPUT_DIR) / f"run_{RUN_TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger = setup_logging(OUTPUT_DIR, SCRIPT_NAME)
metadata = RunMetadata(run_id=RUN_TIMESTAMP, script_name=SCRIPT_NAME, script_version=SCRIPT_VERSION)

logger.info("=" * 80)
logger.info("SCRIPT 00: SETUP AND VALIDATION")
logger.info("=" * 80)
logger.info(f"Run ID: {RUN_TIMESTAMP}")
logger.info(f"Git commit: {metadata.git_commit_hash}")
logger.info(f"Command line: {metadata.command_line}")
logger.info(f"Environment: {metadata.environment}")
print(f"Run ID: {RUN_TIMESTAMP}", flush=True)
print(f"Output directory: {OUTPUT_DIR}", flush=True)

# ... setup/validation logic unchanged ...

# At end of script:
# metadata.finalize()
# metadata.save(OUTPUT_DIR)


# =============================================================================
# SCRIPT 01: PHOTON/SEGMENT FILTERING (MULTI-PRODUCT)
# =============================================================================
"""
SCRIPT 01: ICESat-2 PHOTON/SEGMENT FILTERING - MULTI-PRODUCT (ATL03/ATL06/ATL08)

Purpose
-------
Load raw ICESat-2 data and apply quality + study-area filters to extract
high-confidence ground returns. Study area is defined by the polygon in
processing_area.gpkg (point-in-polygon test, not a rectangular bbox).

Reproducibility
----------------
- Input file list (raw ATL03/06/08 files) is checksummed via
  repro_utils.describe_input_files() and stored in 01_metadata.json.
- Study-area GeoPackage path and description are stored under
  study_area_reference in the metadata JSON.
- Git commit hash and full command line are captured automatically and
  written only to script01_run.log / 01_metadata.json (not printed to console).
"""

import sys
import io
import argparse
import warnings
from pathlib import Path
from datetime import datetime
import geopandas as gpd
from repro_utils import setup_logging, RunMetadata, describe_input_files

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

from config import INPUT_DIR, BASE_OUTPUT_DIR, VALID_PRODUCTS, DEFAULT_PRODUCT

SCRIPT_NAME = "01_filter_photons"
SCRIPT_VERSION = "v2.0-polygon-study-area"

parser = argparse.ArgumentParser(description="ICESat-2 Multi-Product Filtering")
parser.add_argument("--product", type=str, choices=VALID_PRODUCTS, default=DEFAULT_PRODUCT)
args = parser.parse_args()
PRODUCT = args.product

STUDY_AREA_GPKG = Path(r"E:\Processing\ICESat-2\raw_data\processing_area.gpkg")

print("=" * 80, flush=True)
print("SCRIPT 01: ICESat-2 DATA FILTERING - MULTI-PRODUCT", flush=True)
print("=" * 80, flush=True)
print(f"Product selected: {PRODUCT}", flush=True)

timestamp_file = Path(BASE_OUTPUT_DIR) / "latest_RUN_TIMESTAMP.txt"
if not timestamp_file.exists():
    print(f"Timestamp file not found: {timestamp_file}")
    print("Script 00 must run first!")
    sys.exit(1)
RUN_TIMESTAMP = timestamp_file.read_text(encoding="utf-8").strip()
OUTPUT_DIR = Path(BASE_OUTPUT_DIR) / f"run_{RUN_TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger = setup_logging(OUTPUT_DIR, SCRIPT_NAME)

study_gdf = gpd.read_file(STUDY_AREA_GPKG).to_crs("EPSG:4326") if STUDY_AREA_GPKG.exists() else None
if study_gdf is not None:
    minx, miny, maxx, maxy = study_gdf.total_bounds
    study_area_reference = {
        "source_file": str(STUDY_AREA_GPKG),
        "n_features": len(study_gdf),
        "bounds_wgs84": {"lat_min": miny, "lat_max": maxy, "lon_min": minx, "lon_max": maxx},
    }
else:
    logger.error(f"Study area GeoPackage not found: {STUDY_AREA_GPKG}")
    print(f"ERROR: Study area GeoPackage not found: {STUDY_AREA_GPKG}", flush=True)
    sys.exit(1)

metadata = RunMetadata(
    run_id=RUN_TIMESTAMP, script_name=SCRIPT_NAME, script_version=SCRIPT_VERSION,
    config_parameters={"product": PRODUCT, "confidence_threshold": 3},
    study_area_reference=study_area_reference,
)

logger.info("=" * 80)
logger.info(f"SCRIPT 01: FILTERING - product={PRODUCT}")
logger.info(f"Git commit: {metadata.git_commit_hash}")
logger.info(f"Command line: {metadata.command_line}")
logger.info(f"Study area source: {STUDY_AREA_GPKG}")
logger.info(f"Study area reference: {study_area_reference}")
print(f"Run ID: {RUN_TIMESTAMP}", flush=True)
print(f"Output folder: {OUTPUT_DIR.name}", flush=True)
print("=" * 80, flush=True)

# ... filtering logic unchanged (see previously delivered full script) ...

# Before saving metadata, attach raw input file checksums, e.g.:
# input_files = sorted(Path(INPUT_DIR).glob(f"*{PRODUCT}*"))
# metadata.add_input_files(input_files)
# metadata.set_processing_results({"total_rows_output": len(icesat2_data)})
# metadata.finalize()
# metadata.save(OUTPUT_DIR)


# =============================================================================
# SCRIPT 02: TERRAIN CLASSIFICATION (LAND/OCEAN/LAKES)
# =============================================================================
"""
SCRIPT 02: TERRAIN CLASSIFICATION - LAND / OCEAN / LAKES

Purpose
-------
Classify filtered ground photons using Statistics Canada land boundaries
and NRCan CanVec waterbody polygons, refined by elevation thresholds.

Reproducibility
----------------
- Checksums of CANADA_LAND_SHAPEFILE and CANVEC_WATERBODY_SHAPEFILE are
  recorded under input_files in 02_metadata.json.
- Git commit hash and command line captured to script02_run.log /
  02_metadata.json only (console banner unchanged).
"""

import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import geopandas as gpd
from config import BASE_OUTPUT_DIR, VALID_PRODUCTS, DEFAULT_PRODUCT
from repro_utils import setup_logging, RunMetadata, describe_input_files

SCRIPT_NAME = "02_filter_terrain"
SCRIPT_VERSION = "v1.3"

CANADA_LAND_SHAPEFILE = Path(r"E:\...\000b21ae.shp")
CANVEC_WATERBODY_SHAPEFILE = Path(r"E:\...\hd_1480009_0.shp")

parser = argparse.ArgumentParser(description="ICESat-2 Terrain Classification")
parser.add_argument("--product", type=str, choices=VALID_PRODUCTS, default=DEFAULT_PRODUCT)
args = parser.parse_args()
PRODUCT = args.product

print("=" * 80, flush=True)
print("SCRIPT 02: TERRAIN CLASSIFICATION WITH ELEVATION FILTERING", flush=True)
print("=" * 80, flush=True)
print(f"Product: {PRODUCT}", flush=True)

timestamp_file = Path(BASE_OUTPUT_DIR) / "latest_RUN_TIMESTAMP.txt"
if not timestamp_file.exists():
    print(f"ERROR: Timestamp file not found: {timestamp_file}")
    sys.exit(1)
RUN_TIMESTAMP = timestamp_file.read_text(encoding="utf-8").strip()
OUTPUT_DIR = Path(BASE_OUTPUT_DIR) / f"run_{RUN_TIMESTAMP}"

logger = setup_logging(OUTPUT_DIR, SCRIPT_NAME)
metadata = RunMetadata(
    run_id=RUN_TIMESTAMP, script_name=SCRIPT_NAME, script_version=SCRIPT_VERSION,
    config_parameters={"product": PRODUCT, "LAND_MIN_ELEV": -7.0, "LAKE_MIN_ELEV": -2.0, "OCEAN_MAX_ELEV": 1.5},
)
metadata.add_input_files([CANADA_LAND_SHAPEFILE, CANVEC_WATERBODY_SHAPEFILE])

logger.info("=" * 80)
logger.info(f"SCRIPT 02: TERRAIN CLASSIFICATION - product={PRODUCT}")
logger.info(f"Git commit: {metadata.git_commit_hash}")
logger.info(f"Command line: {metadata.command_line}")
logger.info(f"Input GIS layers: {[str(p) for p in [CANADA_LAND_SHAPEFILE, CANVEC_WATERBODY_SHAPEFILE]]}")
print(f"Run ID: {RUN_TIMESTAMP}", flush=True)
print("=" * 80, flush=True)

# ... classification logic unchanged ...
# metadata.set_processing_results({...})
# metadata.finalize(); metadata.save(OUTPUT_DIR)


# =============================================================================
# SCRIPT 03: CORE ANALYSIS - CROSSOVER DETECTION + INSAR INTEGRATION
# =============================================================================
"""
SCRIPT 03: CORE ANALYSIS - CROSSOVER DETECTION / INSAR INTEGRATION
WITH PHASE 2A MULTI-TIER OUTLIER FILTERING (cm/yr units)

Sign convention: dH = h_late - h_early. Negative = subsidence,
Positive = uplift. Enforced by explicit early/late swap after crossover pairing.

Reproducibility
----------------
- Full Config.as_dict() logged under config_parameters in 03_metadata.json.
- Input LAND CSV file checksummed under input_files.
- --outlier-filter, --outlier-sigma, --temporal, --product flags captured
  automatically via command_line field (sys.argv) - no need to log manually.
- Git commit hash written only to log/JSON, not console.
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
warnings.filterwarnings("ignore")

from config import BASE_OUTPUT_DIR, INSAR_DIR
from repro_utils import setup_logging, RunMetadata

SCRIPT_NAME = "03_analysis_core"
SCRIPT_VERSION = "v3.9-MULTIPRODUCT-OPTIONAL-OUTLIER-REPRO"

class Config:
    CROSSOVER_DISTANCE_THRESHOLD_M = 5
    TIME_SEPARATION_THRESHOLD_DAYS = 10
    TIME_SEPARATION_MAX_DAYS = 2000
    RATE_PHYSICAL_MAX_CMYR = 200.0
    DH_PHYSICAL_MAX_M = 2.0
    OUTLIER_SIGMA_TIERS = [5.0, 2.0, 1.5]
    MEASUREMENT_UNCERTAINTY_CM = 4.2
    TEMPORAL_UNCERTAINTY_DAYS = 15
    BOOTSTRAP_SAMPLES = 1000
    BOOTSTRAP_CI_PERCENT = 95
    BOOTSTRAP_SEED = 42

    @classmethod
    def as_dict(cls):
        return {k: v for k, v in vars(cls).items() if not k.startswith("_") and not callable(v)}

parser = argparse.ArgumentParser(description="ICESat-2 Core Crossover Analysis with Phase 2A Filtering")
parser.add_argument("--temporal", type=str, choices=["1", "2", "3", "4"], default="2")
parser.add_argument("--skip-insar", action="store_true")
parser.add_argument("--product", type=str, default="ATL08")
parser.add_argument("--outlier-filter", type=str, choices=["on", "off"], default="on")
parser.add_argument("--outlier-sigma", type=str, default="5,2,1.5")
args = parser.parse_args()

TEMPORAL_OPTION = args.temporal
SKIP_INSAR = args.skip_insar
PRODUCT = args.product
OUTLIER_FILTER_ENABLED = args.outlier_filter == "on"
Config.OUTLIER_SIGMA_TIERS = [float(x) for x in args.outlier_sigma.split(",")] if OUTLIER_FILTER_ENABLED else []

print("=" * 80, flush=True)
print("SCRIPT 03: CORE ANALYSIS - CROSSOVER DETECTION / INSAR INTEGRATION", flush=True)
print("WITH PHASE 2A MULTI-TIER OUTLIER FILTERING (cm/yr units)", flush=True)
print("=" * 80, flush=True)

timestamp_file = Path(BASE_OUTPUT_DIR) / "latest_RUN_TIMESTAMP.txt"
if not timestamp_file.exists():
    print("ERROR: Run Script 00 first.")
    sys.exit(1)
RUN_TIMESTAMP = timestamp_file.read_text(encoding="utf-8").strip()
OUTPUT_DIR = Path(BASE_OUTPUT_DIR) / f"run_{RUN_TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger = setup_logging(OUTPUT_DIR, SCRIPT_NAME)
metadata = RunMetadata(
    run_id=RUN_TIMESTAMP, script_name=SCRIPT_NAME, script_version=SCRIPT_VERSION,
    config_parameters=Config.as_dict(),
)

logger.info("=" * 80)
logger.info("SCRIPT 03: CORE ANALYSIS")
logger.info(f"Git commit: {metadata.git_commit_hash}")
logger.info(f"Command line: {metadata.command_line}")
logger.info(f"Temporal option: {TEMPORAL_OPTION}, skip_insar: {SKIP_INSAR}, product: {PRODUCT}")
logger.info(f"Outlier filter enabled: {OUTLIER_FILTER_ENABLED}, tiers: {Config.OUTLIER_SIGMA_TIERS}")
print(f"Run ID: {RUN_TIMESTAMP}", flush=True)
print(f"Temporal option: {TEMPORAL_OPTION}", flush=True)
if SKIP_INSAR:
    print("InSAR integration SKIPPED (--skip-insar flag)", flush=True)
print("=" * 80, flush=True)

# ... crossover / InSAR logic unchanged ...
# INPUT_FILE = OUTPUT_DIR / "icesat2_filtered_ground_photons_LAND.csv"
# metadata.add_input_files([INPUT_FILE])
# metadata.set_processing_results({"filtering_statistics": filter_stats, "insar_validation_stats": insar_stats})
# metadata.finalize(); metadata.save(OUTPUT_DIR)


# =============================================================================
# SCRIPT 04: SPATIAL ANALYSIS
# =============================================================================
"""
SCRIPT 04: SPATIAL GRID ANALYSIS

Purpose
-------
Aggregate crossover-derived subsidence rates onto a spatial grid for
regional pattern analysis.

Reproducibility
----------------
- Input crossover CSV checksummed under input_files in 04_metadata.json.
- Grid resolution / aggregation parameters logged under config_parameters.
- Git commit hash / command line: log + JSON only, not console.
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from config import BASE_OUTPUT_DIR
from repro_utils import setup_logging, RunMetadata

SCRIPT_NAME = "04_analysis_spatial"
SCRIPT_VERSION = "v1.4"

GRID_RESOLUTION_M = 500.0
MIN_POINTS_PER_CELL = 3

print("=" * 80, flush=True)
print("SCRIPT 04: SPATIAL GRID ANALYSIS", flush=True)
print("=" * 80, flush=True)

timestamp_file = Path(BASE_OUTPUT_DIR) / "latest_RUN_TIMESTAMP.txt"
if not timestamp_file.exists():
    print("ERROR: Run Script 00 first.")
    sys.exit(1)
RUN_TIMESTAMP = timestamp_file.read_text(encoding="utf-8").strip()
OUTPUT_DIR = Path(BASE_OUTPUT_DIR) / f"run_{RUN_TIMESTAMP}"

logger = setup_logging(OUTPUT_DIR, SCRIPT_NAME)
metadata = RunMetadata(
    run_id=RUN_TIMESTAMP, script_name=SCRIPT_NAME, script_version=SCRIPT_VERSION,
    config_parameters={"grid_resolution_m": GRID_RESOLUTION_M, "min_points_per_cell": MIN_POINTS_PER_CELL},
)

logger.info("=" * 80)
logger.info("SCRIPT 04: SPATIAL GRID ANALYSIS")
logger.info(f"Git commit: {metadata.git_commit_hash}")
logger.info(f"Command line: {metadata.command_line}")
print(f"Run ID: {RUN_TIMESTAMP}", flush=True)
print("=" * 80, flush=True)

# ... spatial gridding logic unchanged ...
# metadata.add_input_files([input_file])
# metadata.set_processing_results({...})
# metadata.finalize(); metadata.save(OUTPUT_DIR)


# =============================================================================
# SCRIPT 05: TEMPORAL ANALYSIS
# =============================================================================
"""
SCRIPT 05: TEMPORAL ANALYSIS - SUBSIDENCE RATE TIME SERIES

Purpose
-------
Compute year-by-year and multi-year subsidence rate statistics from
crossover-derived elevation change pairs. Units: cm/yr, negative=subsidence,
positive=uplift.

Reproducibility
----------------
- Input crossover/InSAR-integrated CSV checksummed under input_files.
- Git commit hash / command line written to log + 05_metadata.json only.
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from config import BASE_OUTPUT_DIR
from repro_utils import setup_logging, RunMetadata

SCRIPT_NAME = "05_analysis_temporal"
SCRIPT_VERSION = "v1.5"

print("=" * 80, flush=True)
print("SCRIPT 05: TEMPORAL ANALYSIS - SUBSIDENCE RATE TIME SERIES", flush=True)
print("=" * 80, flush=True)

timestamp_file = Path(BASE_OUTPUT_DIR) / "latest_RUN_TIMESTAMP.txt"
if not timestamp_file.exists():
    print("ERROR: Run Script 00 first.")
    sys.exit(1)
RUN_TIMESTAMP = timestamp_file.read_text(encoding="utf-8").strip()
OUTPUT_DIR = Path(BASE_OUTPUT_DIR) / f"run_{RUN_TIMESTAMP}"

logger = setup_logging(OUTPUT_DIR, SCRIPT_NAME)
metadata = RunMetadata(run_id=RUN_TIMESTAMP, script_name=SCRIPT_NAME, script_version=SCRIPT_VERSION)

logger.info("=" * 80)
logger.info("SCRIPT 05: TEMPORAL ANALYSIS")
logger.info(f"Git commit: {metadata.git_commit_hash}")
logger.info(f"Command line: {metadata.command_line}")
print(f"Run ID: {RUN_TIMESTAMP}", flush=True)
print("=" * 80, flush=True)

# ... temporal rate logic unchanged ...
# metadata.add_input_files([input_file])
# metadata.set_processing_results({...})
# metadata.finalize(); metadata.save(OUTPUT_DIR)


# =============================================================================
# SCRIPT 05b: TEMPORAL STRATIFICATION / ACCELERATION METRICS
# =============================================================================
"""
SCRIPT 05b: TEMPORAL STRATIFICATION - ACCELERATION METRICS

Purpose
-------
Early (2019-2021) vs Late (2022-2024) period comparison and acceleration
trend analysis on subsidence rates. Units cm/yr, negative=subsidence,
positive=uplift.

Reproducibility
----------------
- EARLY_CUTOFF_YEAR and input file checksum logged under config_parameters /
  input_files in 05b_metadata.json.
- Git commit hash / command line: log + JSON only.
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import linregress
from config import BASE_OUTPUT_DIR
from repro_utils import setup_logging, RunMetadata

SCRIPT_NAME = "05b_temporal_stratification"
SCRIPT_VERSION = "v2.3-all-fixes-repro"
EARLY_CUTOFF_YEAR = 2022

print("=" * 80, flush=True)
print("SCRIPT 05b: TEMPORAL STRATIFICATION - ACCELERATION METRICS", flush=True)
print("WITH EARLY/LATE PERIOD COMPARISON", flush=True)
print("=" * 80, flush=True)

timestamp_file = Path(BASE_OUTPUT_DIR) / "latest_RUN_TIMESTAMP.txt"
try:
    RUN_TIMESTAMP = timestamp_file.read_text(encoding="utf-8").strip()
except FileNotFoundError:
    print("ERROR: Run Script 00 first (need latest_RUN_TIMESTAMP.txt)")
    sys.exit(1)
OUTPUT_DIR = Path(BASE_OUTPUT_DIR) / f"run_{RUN_TIMESTAMP}"

logger = setup_logging(OUTPUT_DIR, SCRIPT_NAME)
metadata = RunMetadata(
    run_id=RUN_TIMESTAMP, script_name=SCRIPT_NAME, script_version=SCRIPT_VERSION,
    config_parameters={"early_cutoff_year": EARLY_CUTOFF_YEAR, "sign_convention": "negative=subsidence,positive=uplift"},
)

logger.info("=" * 80)
logger.info("SCRIPT 05b: TEMPORAL STRATIFICATION")
logger.info(f"Git commit: {metadata.git_commit_hash}")
logger.info(f"Command line: {metadata.command_line}")
print(f"Run ID: {RUN_TIMESTAMP}", flush=True)
print("=" * 80, flush=True)

# ... early/late stratification logic unchanged ...
# metadata.add_input_files([input_file])
# metadata.set_processing_results({"early_stats": early_stats, "late_stats": late_stats})
# metadata.finalize(); metadata.save(OUTPUT_DIR)


# =============================================================================
# SCRIPT 06: SPATIAL CLUSTERING / ADVANCED ANALYSIS
# =============================================================================
"""
SCRIPT 06: SPATIAL CLUSTERING SUBSIDENCE ANALYSIS

Purpose
-------
k-NN graph + BFS spatial clustering of subsidence points, inverse-variance
weighted rate estimation, baseline/year-range stratification, robustness
assessment, vector exports (shp/gpkg). Units cm/yr, negative=subsidence,
positive=uplift.

Reproducibility
----------------
- Config.as_dict() (cluster_radius_m, min_cluster_size, weight_method, etc.)
  logged under config_parameters in 06_metadata.json.
- Input CSV (05_subsidence_rates.csv or fallback) checksummed under input_files.
- Git commit hash / command line: log + JSON only.
"""

import sys
import io
import warnings
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

stdout_enc = getattr(sys.stdout, "encoding", None) or ""
if stdout_enc.lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except AttributeError:
        pass

try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("Warning: geopandas not available. Vector exports will be skipped.")

from config import BASE_OUTPUT_DIR
from repro_utils import setup_logging, RunMetadata

SCRIPT_NAME = "06_analysis_advanced"
SCRIPT_VERSION = "v2.1-repro"

class Config:
    cluster_radius_m: float = 50.0
    min_cluster_size: int = 3
    weight_method: str = "inverse_variance"
    baseline_dt_bins: List[Tuple] = [(0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, None)]
    robustness_threshold_pct: float = 20.0
    utm_zone_epsg: int = 3155
    wgs84_epsg: int = 4326
    export_all_points: bool = False

    @classmethod
    def as_dict(cls):
        return {k: v for k, v in vars(cls).items() if not k.startswith("_") and not callable(v)}

print("=" * 80, flush=True)
print("SCRIPT 06: SPATIAL CLUSTERING SUBSIDENCE ANALYSIS", flush=True)
print("=" * 80, flush=True)

timestamp_file = Path(BASE_OUTPUT_DIR) / "latest_RUN_TIMESTAMP.txt"
if not timestamp_file.exists():
    print("ERROR: Run Script 00 first.")
    sys.exit(1)
RUN_TIMESTAMP = timestamp_file.read_text(encoding="utf-8").strip()
OUTPUT_DIR = Path(BASE_OUTPUT_DIR) / f"run_{RUN_TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger = setup_logging(OUTPUT_DIR, SCRIPT_NAME)
metadata = RunMetadata(
    run_id=RUN_TIMESTAMP, script_name=SCRIPT_NAME, script_version=SCRIPT_VERSION,
    config_parameters=Config.as_dict(),
)

logger.info("=" * 80)
logger.info("SCRIPT 06: SPATIAL CLUSTERING")
logger.info(f"Git commit: {metadata.git_commit_hash}")
logger.info(f"Command line: {metadata.command_line}")
print(f"Run ID: {RUN_TIMESTAMP}", flush=True)
print(f"Output: {OUTPUT_DIR}", flush=True)

# ... clustering logic unchanged ...
# metadata.add_input_files([input_file])
# metadata.set_processing_results({"n_clusters": n_clusters, "pct_clustered": pct_clustered})
# metadata.finalize(); metadata.save(OUTPUT_DIR)


# =============================================================================
# SCRIPT 07: OUTPUT GENERATION (FIGURES / REPORTS)
# =============================================================================
"""
SCRIPT 07: OUTPUT GENERATION - FIGURES AND SUMMARY REPORTS

Purpose
-------
Generate final maps, time-series plots, and summary reports from all
upstream analysis outputs (Scripts 02-06).

Reproducibility
----------------
- Records which upstream output files (with checksums) were used to build
  each figure, under input_files in 07_metadata.json.
- Colour scale ranges, sign convention, and figure parameters logged under
  config_parameters.
- Git commit hash / command line: log + JSON only.
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import BASE_OUTPUT_DIR
from repro_utils import setup_logging, RunMetadata

SCRIPT_NAME = "07_output_generation"
SCRIPT_VERSION = "v1.6"

print("=" * 80, flush=True)
print("SCRIPT 07: OUTPUT GENERATION - FIGURES AND SUMMARY REPORTS", flush=True)
print("=" * 80, flush=True)

timestamp_file = Path(BASE_OUTPUT_DIR) / "latest_RUN_TIMESTAMP.txt"
if not timestamp_file.exists():
    print("ERROR: Run Script 00 first.")
    sys.exit(1)
RUN_TIMESTAMP = timestamp_file.read_text(encoding="utf-8").strip()
OUTPUT_DIR = Path(BASE_OUTPUT_DIR) / f"run_{RUN_TIMESTAMP}"

logger = setup_logging(OUTPUT_DIR, SCRIPT_NAME)
metadata = RunMetadata(
    run_id=RUN_TIMESTAMP, script_name=SCRIPT_NAME, script_version=SCRIPT_VERSION,
    config_parameters={"sign_convention": "negative=subsidence,positive=uplift"},
)

logger.info("=" * 80)
logger.info("SCRIPT 07: OUTPUT GENERATION")
logger.info(f"Git commit: {metadata.git_commit_hash}")
logger.info(f"Command line: {metadata.command_line}")
print(f"Run ID: {RUN_TIMESTAMP}", flush=True)
print("=" * 80, flush=True)

# ... figure/report generation logic unchanged ...
# metadata.add_input_files([f04, f05, f05b, f06])
# metadata.set_processing_results({"figures_generated": [...]})
# metadata.finalize(); metadata.save(OUTPUT_DIR)


# =============================================================================
# SCRIPT 08: MASTER ORCHESTRATION
# =============================================================================
"""
SCRIPT 08: MASTER WORKFLOW ORCHESTRATION

Purpose
-------
Sequentially execute Scripts 00-07 for a full workflow run, capturing
top-level run provenance (git hash, command line, environment) that ties
together the entire pipeline execution.

Reproducibility
----------------
- Writes a top-level 08_workflow_metadata.json summarising which sub-scripts
  ran, their exit codes, and the shared RUN_TIMESTAMP/git commit hash that
  applies to the whole run.
- Console shows only a clean step-by-step progress banner; git hash and
  command line are written to script08_run.log / 08_workflow_metadata.json.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
from repro_utils import setup_logging, RunMetadata

SCRIPT_NAME = "08_run_workflow"
SCRIPT_VERSION = "v1.2"

print("=" * 80, flush=True)
print("SCRIPT 08: MASTER WORKFLOW ORCHESTRATION", flush=True)
print("=" * 80, flush=True)

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(BASE_OUTPUT_DIR) / f"run_{RUN_TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger = setup_logging(OUTPUT_DIR, SCRIPT_NAME)
metadata = RunMetadata(run_id=RUN_TIMESTAMP, script_name=SCRIPT_NAME, script_version=SCRIPT_VERSION)

logger.info("=" * 80)
logger.info("SCRIPT 08: MASTER WORKFLOW ORCHESTRATION")
logger.info(f"Git commit: {metadata.git_commit_hash}")
logger.info(f"Command line: {metadata.command_line}")
print(f"Run ID: {RUN_TIMESTAMP}", flush=True)
print("=" * 80, flush=True)

# ... sequential script invocation logic unchanged ...
# metadata.set_processing_results({"scripts_run": [...], "exit_codes": {...}})
# metadata.finalize(); metadata.save(OUTPUT_DIR, filename="08_workflow_metadata.json")
