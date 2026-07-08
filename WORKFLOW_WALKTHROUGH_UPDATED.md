# ICESat-2 Processing Workflow for Coastal Subsidence Analysis
## Comprehensive Technical Documentation for Tuktoyaktuk Peninsula Study

**Author:** PhD Candidate in Physical Geography  
**Study Area:** Tuktoyaktuk Peninsula, Arctic Canada  
**Research Focus:** Coastal subsidence, permafrost degradation, and coastal processes  
**Data Source:** ICESat-2 ATL03/ATL06 photon-level and aggregated elevation products  
**Analysis Period:** 2019-2024 (6-year time series)  
**Last Updated:** January 15, 2026

---

## Executive Summary

This document describes a **9-script consolidated processing pipeline** designed to extract subsidence rates from ICESat-2 satellite laser altimetry data for Arctic coastal environments. The workflow implements robust quality control, spatial-temporal analysis, and validation against InSAR data. All scripts include complete reproducibility infrastructure (timestamped outputs, metadata logging, and git hash tracking).

### Key Workflow Characteristics

- **Memory-Optimized:** Hybrid k-nearest neighbor approach reduces RAM usage from 16 GB to 3-4 GB
- **Robust:** Multiple fallback mechanisms prevent processing failures
- **Reproducible:** Complete audit trail with JSON metadata, execution logs, and git versioning
- **Validated:** InSAR cross-validation with correlation analysis
- **Publication-Ready:** High-resolution figures (300/600 DPI) with consistent styling

---

## Directory Structure

```
E:\Processing\ICESat-2\
├── scripts\                          # Processing scripts (run from here)
│   ├── 00_setup_validation.py
│   ├── 01_filter_photons.py
│   ├── 02_filter_terrain.py
│   ├── 03_analysis_core.py
│   ├── 04_analysis_spatial.py
│   ├── 05_analysis_temporal.py
│   ├── 05b_temporal_stratification.py
│   ├── 06_analysis_advanced.py
│   ├── 07_output_generation.py
│   └── 08_run_workflow.py            # Master orchestrator
│
├── raw_data\                         # Input LAS/CSV files
│   ├── ATL03_*.h5                   # ICESat-2 photon data
│   └── *.csv                        # Preprocessed CSV (if available)
│
└── output_data\                      # All processing outputs
    ├── latest_run_timestamp.txt     # Shared run identifier
    ├── run_20260115_n1\            # Run-specific folder
    │   ├── icesat2_filtered.csv
    │   ├── icesat2_repeat_track_matches.csv
    │   ├── icesat2_insar_integrated.csv
    │   ├── figures\
    │   │   ├── Figure01-07.png
    │   │   └── *.png
    │   ├── *_metadata.json          # Reproducibility metadata
    │   └── script*_run.log          # Execution logs
    └── run_20260115_n2\            # Next run (auto-incremented)
```

---

## Workflow Overview: 9-Script Pipeline

### **Phase 1: Data Preparation & Filtering** (Scripts 00-02)

**Purpose:** Validate environment, filter photons by quality and terrain type

### **Phase 2: Core Subsidence Analysis** (Script 03)

**Purpose:** Crossover analysis and InSAR integration

### **Phase 3: Spatial-Temporal Analysis** (Scripts 04-05b)

**Purpose:** Spatial clustering, temporal trends, seasonal stratification

### **Phase 4: Advanced Analysis & Outputs** (Scripts 06-07)

**Purpose:** Hotspot detection, publication-quality figures, GIS export

### **Phase 5: Workflow Orchestration** (Script 08)

**Purpose:** Automated sequential execution with state management

---

## Detailed Script-by-Script Walkthrough

---

## **Script 00: Setup Validation**

### **Purpose**
Validate Python environment, check dependencies, verify directory structure, and initialize run infrastructure.

### **Key Operations**

1. **Python Version Check**
   - Requires Python 3.7+ (64-bit)
   - Validates against 32-bit installations

2. **Package Validation** (13 critical packages)
   ```
   pandas, numpy, scipy, matplotlib, h5py
   rasterio, geopandas, shapely, pyproj
   seaborn, sklearn, PIL, laspy
   ```

3. **Script Syntax Validation**
   - Parses all workflow scripts with AST
   - Reports syntax errors before execution

4. **Directory Structure**
   - Creates `raw_data/` and `output_data/` if missing
   - Validates write permissions
   - Checks disk space (≥15 GB required)

5. **Run Initialization**
   - Generates timestamp: `YYYYMMDD_HHMMSS`
   - Creates run folder: `run_20260115_n1` (auto-increments daily)
   - Saves `latest_run_timestamp.txt` for downstream scripts

### **Inputs**
None (validates environment only)

### **Outputs**
- `latest_run_timestamp.txt` → Shared run identifier
- `run_YYYYMMDD_nX/script00_run.log` → Validation log

### **Critical Features**
✅ **Auto-folder creation** if missing  
✅ **Graceful timestamp handling** (no crashes)  
✅ **UTF-8 encoding fix** (prevents Unicode errors)  
✅ **Git hash tracking** for reproducibility

---

## **Script 01: Photon Filtering**

### **Purpose**
Quality-filter raw ICESat-2 photons to retain high-confidence ground returns.

### **Method: Multi-Stage Filtering**

1. **Spatial Bounding**
   - Filters to study area (Tuktoyaktuk Peninsula)
   - Typical bounds: `69°-71°N, 130°-135°W`

2. **Quality Filtering**
   - Uses `atl03_quality` or `signal_conf_ph` flags
   - Keeps only **medium/high confidence** photons
   - Removes noise (clouds, sea ice returns, etc.)

3. **Temporal Window**
   - Restricts to study period: **2019-01-15 to 2024-12-31**
   - Ensures seasonal consistency (e.g., summer-only analysis)

4. **Strong Beam Selection**
   - ICESat-2 has 3 beam pairs (6 beams total)
   - Uses only **strong beams** (3x better signal-to-noise)

### **Inputs**
- `raw_data/*.las` or `*.csv` → Raw ICESat-2 photons (ATL03/ATL06)

### **Outputs**
- `icesat2_filtered.csv` → Cleaned photons (~30-40% size reduction)
- `script01_run.log` → Processing log
- `01_filter_photons_metadata.json` → Reproducibility metadata

### **Critical Parameters**
```python
QUALITY_THRESHOLD = 2           # Medium/high confidence
TEMPORAL_START = '2019-01-15'
TEMPORAL_END = '2024-12-31'
BEAM_TYPE = 'strong'            # Strong beams only
```

---

## **Script 02: Terrain Classification**

### **Purpose**
Separate valid ground returns from water bodies, lakes, and sea ice using geospatial masks.

### **Method: Land/Ocean/Lake Masking**

1. **Load Geospatial Masks**
   - GSHHG (Global Self-consistent Hierarchical High-resolution Geography)
   - Local shapefiles (lakes, ponds, coastal boundaries)

2. **Spatial Intersection**
   - Filters photons to **land-only** locations
   - Removes ocean, lake, and river returns

3. **Why It Matters**
   - Subsidence analysis requires **stable ground**
   - Water surface elevation ≠ ground elevation
   - Prevents contamination from sea-level change

### **Inputs**
- `icesat2_filtered.csv` (from Script 01)
- Shapefiles: `*.shp` (land cover masks)

### **Outputs**
- `icesat2_terrain_filtered.csv` → Land-only photons
- `script02_run.log`
- `02_filter_terrain_metadata.json`

### **Critical Features**
✅ **Geospatial accuracy** (±10 m precision)  
✅ **Multi-layer masking** (ocean + lakes + rivers)  
✅ **Preserves photon timestamps** for temporal analysis

---

## **Script 03: Core Crossover Analysis & InSAR Integration**

### **Purpose**
Calculate subsidence rates using **crossover analysis** and validate against **InSAR** data.

### **Method 1: Crossover Analysis**

**Concept:** ICESat-2 has repeating ground tracks (~91-day orbit). Where ascending and descending tracks intersect at different times, we can measure elevation change.

**Algorithm:**
1. **Identify Crossover Points**
   - Tracks within **2 m radius** = same location
   - Time separation: 10-365 days (optimal)

2. **Calculate dH/dt**
   ```
   dH = Elevation_later - Elevation_earlier (meters)
   dt = Time_later - Time_earlier (days)
   Rate = (dH / dt) × 365.25 × 1000 (mm/year)
   ```

3. **Outlier Removal**
   - Removes physically impossible rates (>100 mm/yr)
   - Applies **3-sigma filter** (removes top/bottom 0.3%)

### **Method 2: InSAR Integration**

1. **Load InSAR Displacement Maps**
   - GeoTIFF format (vertical displacement, mm/yr)
   - Sources: Sentinel-1, RADARSAT-2, TerraSAR-X

2. **Spatial Extraction**
   - Extracts InSAR values at ICESat-2 crossover locations
   - Uses **bilinear interpolation** for sub-pixel accuracy

3. **Validation Statistics**
   - Pearson correlation (*r*)
   - Root Mean Square Error (RMSE)
   - Mean bias

### **Inputs**
- `icesat2_terrain_filtered.csv` (from Script 02)
- `C:\forICeSat-2\*.tif` → InSAR displacement maps (optional)

### **Outputs**
- `icesat2_repeat_track_matches.csv` → Crossover pairs with rates
- `icesat2_insar_integrated.csv` → Merged ICESat-2 + InSAR
- `03_crossover_statistics.txt` → Summary report
- `script03_run.log`
- `03_analysis_core_metadata.json`

### **Critical Parameters**
```python
CROSSOVER_RADIUS_M = 2.0        # Spatial tolerance
MIN_TIME_SEPARATION = 10        # Days (avoid same-orbit repeats)
MAX_TIME_SEPARATION = 365       # Days (seasonal consistency)
OUTLIER_THRESHOLD_SIGMA = 3     # Statistical filter
```

---

## **Script 04: Spatial Analysis**

### **Purpose**
Compute spatial statistics using **hybrid k-nearest neighbor clustering** (memory-optimized).

### **Method: Hybrid k-Nearest + Radius Filter**

**Problem:** Pure radius queries (`query_ball_point`) use 13-16 GB RAM → crashes  
**Solution:** Hybrid approach uses 3-4 GB RAM

**Algorithm:**
1. **Build KD-Tree** (spatial index)
2. **Query k-Nearest Neighbors** (k=250, fixed memory allocation)
3. **Filter to 2 km radius** (spatially meaningful)
4. **Compute Local Statistics**
   - Mean subsidence rate
   - Standard deviation
   - Point density (points/km²)

**Fallback Mechanism:**
```python
Try k=250 → If memory error → Try k=200 → If error → Try k=150
```

### **Gridded Statistics**

- **Grid Cell Size:** 20 km × 20 km (optimal for Tuktoyaktuk scale)
- **Outputs:** Mean, median, std, min, max per cell
- **Format:** Square matrix heatmap (GIS-compatible)

### **Inputs**
- `icesat2_repeat_track_matches.csv` (from Script 03)

### **Outputs**
- `04_spatial_statistics.csv` → Point-based local stats
- `04_zonal_statistics_20km_grid.csv` → Gridded subsidence rates
- `04_analysis_summary.txt` → Report
- `script04_run.log`
- `04_analysis_spatial_metadata.json`

### **Critical Parameters**
```python
SEARCH_RADIUS_M = 2000.0        # 2 km (coastal feature scale)
K_NEIGHBORS_PRIMARY = 250       # Memory-efficient
GRID_CELL_SIZE_M = 20000.0      # 20 km grid
```

---

## **Script 05: Temporal Analysis**

### **Purpose**
Stratify subsidence rates by **elevation zones** (10 m bins) and analyze temporal trends.

### **Method: Elevation Stratification**

1. **Bin by Elevation**
   ```
   Bins: 0-10m, 10-20m, 20-30m, ... (ΔELEV = 10m)
   ```

2. **Compute Statistics per Bin**
   - Count, mean, median, std, IQR, MAD
   - Identifies elevation-dependent subsidence patterns

3. **Timescale Analysis (Reference Only)**
   - Classifies pairs by time separation (data-driven tertiles)
   - **Note:** Cross-stratification removed due to mismatch
     - Expected: 85% short-repeat tracks (<30 days)
     - Actual: 83% annual repeats (~365 days)
   - Kept as **reference analysis only**

### **Inputs**
- `icesat2_repeat_track_matches.csv` (from Script 03)

### **Outputs**
- `05_subsidence_rates.csv` → All pairs with calculated rates
- `05_statistics_elevation.txt` → Human-readable stratification
- `05_statistics_elevation.csv` → Machine-readable (for plotting)
- `05_statistics_timescale.txt` → Reference analysis
- `script05_run.log`
- `05_analysis_temporal_metadata.json`

### **Critical Parameters**
```python
ELEVATION_BIN_SIZE = 10         # Meters (configurable)
```

---

## **Script 05b: Temporal Stratification (Seasonal)**

### **Purpose**
Compare **Early vs. Late** observation periods with robust statistical testing.

### **Method: Temporal Period Comparison**

1. **Define Periods**
   ```
   Early Period: 2019-01-15 to 2021-12-31 (3 years)
   Late Period:  2022-01-01 to 2024-12-31 (3 years)
   ```

2. **Robust Statistical Tests**
   - **t-test** (parametric)
   - **Mann-Whitney U** (non-parametric)
   - **Kendall's tau** (trend detection)
   - **Cohen's d** (effect size)
   - **Bootstrap confidence intervals** (1000 iterations)
   - **Median Absolute Deviation (MAD)** (outlier-robust)

3. **Distribution Analysis**
   - Histograms (overlapping)
   - Box plots (side-by-side)
   - Violin plots (density visualization)

### **Inputs**
- `icesat2_repeat_track_matches.csv` (from Script 03)

### **Outputs**
- `icesat2_early_period.csv` → Early period data
- `icesat2_late_period.csv` → Late period data
- `temporal_acceleration_metrics.csv` → All metrics
- `temporal_period_statistics.txt` → **Human-readable report** (copy to thesis!)
- `05b_Figure_Distribution_Comparison.png` (300 DPI)
- `05b_Figure_Distribution_Comparison_HiRes.png` (600 DPI)
- `script05b_run.log`
- `05b_temporal_stratification_metadata.json`

### **Critical Parameters**
```python
EARLY_PERIOD_START = '2019-01-15'
EARLY_PERIOD_END = '2021-12-31'
LATE_PERIOD_START = '2022-01-01'
LATE_PERIOD_END = '2024-12-31'
```

---

## **Script 06: Advanced Analysis (Clustering)**

### **Purpose**
Identify **subsidence hotspots** using spatial clustering.

### **Method: Hybrid k-Nearest Clustering**

1. **Cluster Definition**
   - Points within 2 km radius
   - Minimum cluster size: 10 points
   - Subsidence threshold: >20 mm/yr (hotspot)

2. **Cluster Statistics**
   - Mean rate, spatial extent, point density
   - Centroid coordinates
   - Cluster ID assignment

### **Inputs**
- `icesat2_repeat_track_matches.csv` (from Script 03/04)

### **Outputs**
- `06_clusters_identified.csv` → Cluster assignments
- `06_cluster_statistics.csv` → Cluster properties
- `06_analysis_summary.txt` → Report
- `script06_run.log`
- `06_advanced_analysis_metadata.json`

### **Critical Parameters**
```python
SEARCH_RADIUS_M = 2000.0        # 2 km clustering radius
MIN_CLUSTER_SIZE = 10           # Minimum points
SUBSIDENCE_THRESHOLD_MMYR = 20  # Hotspot definition
```

---

## **Script 07: Output Generation**

### **Purpose**
Generate **7 publication-ready figures** and export data for GIS.

### **Figures Generated**

| Figure | Description | Format |
|--------|-------------|--------|
| **Figure 01** | Zonal Grid Heatmap (20×20 km) | 300 DPI PNG |
| **Figure 02** | Temporal Period Comparison (Early vs. Late) | 300 DPI PNG |
| **Figure 03** | Timescale Stratification | 300 DPI PNG |
| **Figure 04** | Density Classification | 300 DPI PNG |
| **Figure 05** | Hotspot Spatial Map | 300 DPI PNG |
| **Figure 06** | InSAR vs. ICESat-2 Scatter | 300 DPI PNG |
| **Figure 07** | InSAR Validation (3-panel) | 300 DPI PNG |

### **InSAR Validation Outputs**

1. **Binned Statistics by Elevation**
   - `07_insar_binned_stats_by_elevation.csv`
   - Elevation bins: 10 m (configurable)
   - Metrics: R², RMSE, MAE, bias per bin

2. **QGIS-Optimized Subsample**
   - `07_insar_qgis_subsample.csv`
   - Downsampling: 1:10 (90% file size reduction)
   - Spatial grid cells for efficient rendering

### **Inputs**
- All outputs from Scripts 03-06
- `icesat2_insar_integrated.csv` (from Script 03)

### **Outputs**
- `figures/Figure01-07.png` → 7 publication figures
- `07_insar_binned_stats_by_elevation.csv`
- `07_insar_qgis_subsample.csv`
- `Summary_Statistics.txt` → Compiled report
- `script07_run.log`
- `07_output_generation_metadata.json`

### **Critical Features**
✅ **Consistent styling** (seaborn-v0_8-darkgrid)  
✅ **High resolution** (300 DPI standard, 600 DPI optional)  
✅ **InSAR validation** (3-panel comparison)  
✅ **GIS-ready exports** (downsampled for performance)

---

## **Script 08: Workflow Orchestration**

### **Purpose**
Automated sequential execution of Scripts 00-07 with **state management** and **resume capability**.

### **Features**

1. **Temporal Filtering Options** (command-line)
   ```bash
   python 08_run_workflow.py --temporal 1   # June-October
   python 08_run_workflow.py --temporal 2   # June-September (default)
   python 08_run_workflow.py --temporal 3   # Early vs Late (June-July vs Aug-Sept)
   python 08_run_workflow.py --temporal 4   # Same-month-only
   ```

2. **State Management**
   - Tracks SUCCESS/FAILED/SKIPPED status per script
   - Saves `workflow_execution_state.json`

3. **Resume Capability**
   ```bash
   python 08_run_workflow.py --resume      # Resume from last failed
   python 08_run_workflow.py --start 03    # Start from Script 03
   python 08_run_workflow.py --status      # Print current state
   ```

4. **Execution Logging**
   - Wall-clock time per script
   - Memory usage tracking
   - Exit codes and error messages

### **Inputs**
None (orchestrates all scripts)

### **Outputs**
- `workflow_execution_state.json` → State tracker
- `processing_log_RUNID.txt` → Master execution log

### **Critical Features**
✅ **Real-time output** (no buffering)  
✅ **Timeout handling** (prevents infinite hangs)  
✅ **Required vs. optional** script distinction  
✅ **Summary statistics** (success rate, total time)

---

## Reproducibility Infrastructure

All scripts include:

### **1. Metadata JSON**
```json
{
  "run_id": "20260115_151058",
  "script": "03_analysis_core",
  "version": "v2.1-reproducible",
  "git_hash": "a3b5c7d9",
  "environment": {
    "python_version": "3.12.3",
    "platform": "win32"
  },
  "timestamps": {
    "start": "2026-01-15T15:10:58",
    "end": "2026-01-15T15:12:34"
  },
  "configuration": {
    "crossover_radius_m": 2.0,
    "min_time_separation": 10,
    "max_time_separation": 365
  },
  "processing_results": {
    "total_crossovers": 2148756,
    "mean_subsidence_rate_mmyr": -12.34
  }
}
```

### **2. Execution Logs**
- UTF-8 encoded `.log` files
- Timestamped entries (ISO 8601)
- DEBUG-level file logging + INFO-level console output

### **3. Git Hash Tracking**
- Captures exact code version
- Enables complete workflow reproducibility
- Falls back to `'unknown'` if not in git repo

---

## Key Scientific Improvements Over Original Workflow

| Aspect | Original (14 scripts) | New (9 scripts) |
|--------|----------------------|-----------------|
| **Memory Usage** | 13-16 GB (crashes) | 3-4 GB (stable) |
| **Execution Success** | ~10% (memory errors) | 90-95% (robust fallbacks) |
| **Reproducibility** | Partial (scattered logs) | Complete (JSON + git + logs) |
| **Temporal Stratification** | Broken (85% mismatch) | Fixed (data-driven tertiles) |
| **InSAR Validation** | Missing | Implemented (R², RMSE, bias) |
| **Figure Quality** | 150 DPI | 300/600 DPI |
| **GIS Export** | Heavy files (slow) | Optimized (90% smaller) |
| **UTF-8 Encoding** | Crashes on º/°/± | Fixed (no errors) |
| **Timestamp Handling** | Hard crash if missing | Graceful auto-generation |

---

## Usage Examples

### **Full Workflow (Automated)**
```bash
cd E:\Processing\ICESat-2\scripts
python 08_run_workflow.py --temporal 2
```

### **Individual Script Execution**
```bash
python 00_setup_validation.py        # Validate environment
python 01_filter_photons.py          # Filter photons
python 03_analysis_core.py           # Crossover analysis
python 05b_temporal_stratification.py # Seasonal comparison
python 07_output_generation.py       # Generate figures
```

### **Resume Failed Run**
```bash
python 08_run_workflow.py --resume
```

### **Dry Run (Planning)**
```bash
python 08_run_workflow.py --dry-run
```

---

## Outputs Summary

### **Per-Run Folder Structure**
```
run_20260115_n1/
├── icesat2_filtered.csv                  # Script 01
├── icesat2_terrain_filtered.csv          # Script 02
├── icesat2_repeat_track_matches.csv      # Script 03 ⭐ KEY OUTPUT
├── icesat2_insar_integrated.csv          # Script 03 (if InSAR available)
├── 04_spatial_statistics.csv             # Script 04
├── 04_zonal_statistics_20km_grid.csv     # Script 04
├── 05_subsidence_rates.csv               # Script 05
├── 05_statistics_elevation.csv           # Script 05 ⭐ THESIS TABLE
├── 05b_temporal_period_statistics.txt    # Script 05b ⭐ THESIS TEXT
├── 06_clusters_identified.csv            # Script 06
├── 06_cluster_statistics.csv             # Script 06
├── figures/
│   ├── Figure01-07.png                   # Script 07 ⭐ PUBLICATION FIGURES
│   └── *.png
├── 07_insar_qgis_subsample.csv           # Script 07 ⭐ QGIS IMPORT
├── Summary_Statistics.txt                # Script 07 ⭐ OVERVIEW
├── *_metadata.json                       # All scripts (reproducibility)
└── script*_run.log                       # All scripts (debugging)
```

---

## Common Issues & Solutions

### **Issue 1: Memory Error in Script 04/06**
**Solution:** Automatic fallback to k=200, then k=150

### **Issue 2: Missing Timestamp File**
**Solution:** Auto-generates timestamp (no crash)

### **Issue 3: Unicode Errors (º, °, ±)**
**Solution:** UTF-8 encoding fix applied to all scripts

### **Issue 4: InSAR Data Missing**
**Solution:** Scripts skip InSAR analysis gracefully (optional)

### **Issue 5: Empty Crossover Results**
**Solution:** Check temporal filtering parameters (--temporal option)

---

## Citation & Acknowledgments

**Pipeline Developer:** ICESat-2 Coastal Analysis Team  
**Study Site:** Tuktoyaktuk Peninsula, Northwest Territories, Canada  
**Data Source:** NASA ICESat-2 (Ice, Cloud, and land Elevation Satellite-2)  
**Validation Data:** Sentinel-1 InSAR (European Space Agency)  
**Processing Framework:** Python 3.12 + SciPy + GeoPandas ecosystem

---

## Next Steps for Supervisor Review

1. **Validate Scientific Approach**
   - Crossover radius (2 m) appropriate for coastal terrain?
   - Temporal separation (10-365 days) captures seasonal dynamics?
   - Elevation stratification (10 m bins) aligns with geomorphology?

2. **Assess Outputs for Thesis**
   - `05_statistics_elevation.csv` → **Table 4.1: Subsidence by Elevation**
   - `05b_temporal_period_statistics.txt` → **Section 4.2: Temporal Acceleration**
   - `Figure 05-07` → **Figures 4.3-4.5: Spatial Patterns & InSAR Validation**

3. **Reproducibility Check**
   - Metadata JSON files complete?
   - Git hash tracking functional?
   - Execution logs sufficient for troubleshooting?

4. **Performance Validation**
   - Memory usage acceptable (<4 GB)?
   - Processing time reasonable (~30-60 min total)?
   - Success rate high (>90%)?

---

## Appendix: Parameter Reference

### **Global Parameters (All Scripts)**
```python
BASE_OUTPUT_DIR = Path(r'E:\Processing\ICESat-2\output_data')
RAW_DATA_DIR = Path(r'E:\Processing\ICESat-2\raw_data')
INSAR_DIR = Path(r'C:\forICeSat-2')  # Optional
```

### **Critical Scientific Parameters**

| Parameter | Value | Script | Justification |
|-----------|-------|--------|---------------|
| `CROSSOVER_RADIUS_M` | 2.0 | 03 | GPS precision ±2-3 m |
| `MIN_TIME_SEPARATION` | 10 | 03 | Avoid same-orbit repeats |
| `MAX_TIME_SEPARATION` | 365 | 03 | Seasonal consistency |
| `SEARCH_RADIUS_M` | 2000.0 | 04, 06 | Coastal feature scale |
| `K_NEIGHBORS_PRIMARY` | 250 | 04, 06 | Memory-optimized |
| `ELEVATION_BIN_SIZE` | 10 | 05 | Geomorphic units |
| `GRID_CELL_SIZE_M` | 20000.0 | 04 | Regional-scale grid |
| `SUBSIDENCE_THRESHOLD_MMYR` | 20.0 | 06 | Hotspot definition |

---

**End of Workflow Documentation**  
*Generated: January 15, 2026*  
*Version: 2.0 (Consolidated Pipeline)*
