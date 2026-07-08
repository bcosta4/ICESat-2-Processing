# WORKFLOW METHODS: ICESat-2 Coastal Subsidence Analysis Pipeline
## Tuktoyaktuk Peninsula, Northwest Territories, Canada

---

## OVERVIEW

This workflow processes ICESat-2 satellite data to quantify coastal subsidence rates and identify hotspots across the Tuktoyaktuk Peninsula using a multi-stage analysis pipeline combining photon filtering, crossover detection, spatial/temporal stratification, and advanced clustering.

---

## 1. PHOTON FILTERING & TERRAIN CLASSIFICATION (Scripts 01, 02)

**Input:** ICESat-2 L3A Geolocated Photon Data (ATL03)

**Method:**
- Classification trees distinguish land vs. water photons using signal strength, height patterns, and contextual geolocation
- Terrain filtering removes obvious off-shore and inland high-elevation points using elevation thresholds (±2σ from coastal mean)
- Coastal zone defined as 0–5 km from shoreline using vector reference datasets
- Outputs: Land-only, terrain-validated photon clouds (~10K–50K points per track)

**Quality Control:** Background photon removal, duplicate coordinate elimination, coordinate validation (±0.0001° precision)

---

## 2. CROSSOVER POINT DETECTION & SUBSIDENCE RATE CALCULATION (Script 03)

**Input:** Filtered photon clouds from Scripts 01–02

**Core Method: Repeat-Track Crossover Identification**
- **Spatial Matching:** cKDTree spatial index searches for photon pairs within 2 m distance threshold
- **Temporal Filtering:** Accepts repeat-track pairs with time separation 10–2000 days (seasonal and multi-year repeats)
- **Temporal Options:** Four filtering modes:
  - Mode 1: June–October (broad summer)
  - Mode 2: June–September (standard thaw season, default)
  - Mode 3: Early (June–July) vs. Late (Aug–Sept) within summer
  - Mode 4: Same-month pairs (strictest constraint)

**Subsidence Rate Calculation:**
$$\text{Rate (mm/yr)} = \frac{\Delta h \text{ (m)}}{\Delta t \text{ (years)}} \times 1000$$

where $\Delta h$ = elevation difference at crossover, $\Delta t$ = time separation between observations

**Outlier Removal:** 3-sigma filtering removes extreme rates (±3 SD from mean); empirical RMSE validation against theoretical measurement uncertainty (4.2 cm)

**Uncertainty Quantification:**
- Per-observation rate uncertainties propagated from measurement uncertainty and temporal separation
- Bootstrap resampling (1000 iterations) computes 95% confidence intervals

**InSAR Integration (Optional):** Crossover locations sampled against Sentinel-1 InSAR-derived subsidence rasters; validation metrics (Pearson r, RMSE, bias) assess consistency

**Output:** ~10K–200K validated crossover pairs with rates, uncertainties, spatial/temporal metadata

---

## 3. ELEVATION STRATIFICATION (Script 05)

**Method:** Vertical zonation of subsidence rates

- **Bin Size:** 10 m elevation bands
- **Calculation:** Per-zone statistics (mean, median, SD, min/max) computed using grouped aggregation
- **Rationale:** Elevation controls thaw-susceptibility (lower elevations = wetter, potentially faster subsidence)
- **Temporal Variant (05b):** Early vs. Late summer period stratification with robust statistical tests:
  - Independent t-test
  - Mann–Whitney U test (non-parametric)
  - Kendall's tau (rank correlation)
  - Cohen's d (effect size)
  - Bootstrap 95% confidence intervals

**Output:** Elevation-stratified statistics; temporal comparison with complete statistical validation

---

## 4. SPATIAL CLUSTERING (Scripts 04, 06)

**Hybrid k-Nearest Neighbor Radius-Filter Approach (Memory-Optimized)**

**Script 04 – Neighborhood Analysis:**
- **Spatial Index:** cKDTree built from ~10K–200K crossover points (lon/lat)
- **Query Method:** k-nearest neighbors (primary k=250, fallbacks k=200, 150 for memory constraints)
- **Radius Filter:** Results post-filtered to 2 km radius (optimal for thermokarst lakes, coastal geomorphic units)
- **Local Statistics:** For each point, neighborhood mean/median/std subsidence computed from within-radius neighbors
- **Grid Discretization:** 500 m cells for visualization; local statistics aggregated by grid cell
- **Memory Efficiency:** 3–4 GB vs. 13–16 GB with pure radius query; adaptive k-fallback ensures robust completion

**Script 06 – Hotspot Clustering:**
- **Hotspot Threshold:** Points with subsidence >20 mm/yr flagged
- **Connectivity:** Breadth-first search identifies connected hotspot clusters within 2 km neighborhoods
- **Cluster Validation:** Minimum 10 points per cluster; clusters characterized by spatial extent, mean rate, magnitude range
- **Output:** Cluster assignments, cluster statistics, spatial locations for mapping

**Rationale:** 2 km radius chosen based on dominant Tuktoyaktuk coastal features (thermokarst lakes 1–3 km, coastal geomorphic units 2–5 km)

---

## 5. OUTPUT GENERATION & VISUALIZATION (Script 07)

**Aggregate Statistics:** Summary tables synthesizing zonal, temporal, spatial results

**Multi-Panel Figures:**
- Zonal grid heatmaps (20×20 km subsidence distribution)
- Period comparison (Early vs. Late with box plots)
- Density classification pie charts
- Hotspot spatial maps (2D scatter with color encoding)
- InSAR validation plots (3-panel: scatter, difference distribution, residuals)

**QGIS Export:** Downsampled spatial subset (10% of data) optimized for visualization in GIS software

**Format:** 300 DPI PNG (publication-ready); metadata JSON logs all parameters, git commit hash, reproducibility trail

---

## 6. REPRODUCIBILITY INFRASTRUCTURE

**Each script includes:**
- Complete run metadata (configuration, timestamps, git hash)
- UTF-8 encoding validation (eliminates Unicode errors)
- Detailed logging (file + console output)
- Parameter audit trail (all thresholds, bin sizes, k-values recorded)
- Execution logs with processing timestamps

**Key Parameters for Reproducibility:**
| Parameter | Value | Script |
|-----------|-------|--------|
| Crossover distance threshold | 2.0 m | 03 |
| Time separation | 10–2000 days | 03 |
| Elevation bin size | 10 m | 05 |
| Spatial search radius | 2000 m (2 km) | 04, 06 |
| k-neighbors (primary) | 250 | 04, 06 |
| Hotspot threshold | 20 mm/yr | 06 |
| Grid cell size | 500 m | 04 |
| Measurement uncertainty | 4.2 cm | 03 |
| Bootstrap samples | 1000 | 03, 05b |

---

## WORKFLOW EXECUTION

```
Script 00 → Setup & timestamp generation
    ↓
Script 01 → Photon filtering (classification)
    ↓
Script 02 → Terrain filtering (coastal masking)
    ↓
Script 03 → Crossover detection & subsidence rates (core analysis)
    ├→ Optional: InSAR integration
    ↓
Script 04 → Spatial analysis (k-nearest, gridding)
    ↓
Script 05 → Temporal/elevation stratification
    ├→ Script 05b: Early vs. Late statistical comparison
    ↓
Script 06 → Advanced clustering (hotspot identification)
    ↓
Script 07 → Output generation & visualization
    ↓
Script 08 → Workflow orchestration
```

---

## DATA QUALITY & VALIDATION

| Metric | Threshold | Method |
|--------|-----------|--------|
| **Crossover RMSE** | ±5.9 cm (theoretical) | Empirical vs. 2σh measurement uncertainty |
| **Outlier Detection** | 3-sigma | Remove extreme rates before statistics |
| **Rate Uncertainty** | Per-observation | Propagated from measurement + temporal uncertainty |
| **Spatial Density** | Min. 10 points/cluster | Excludes spurious isolated clusters |
| **InSAR Agreement** | r > 0.7 recommended | Pearson correlation, RMSE <15 mm/yr |

---

## OUTPUTS & APPLICATIONS

- **CSV Tables:** Subsidence rates per point, zone, cluster; validation metrics
- **Text Reports:** Human-readable summaries with statistics interpretation
- **Figures:** Publication-ready visualizations for thesis/papers
- **Metadata:** Complete audit trail enabling full reproducibility
- **QGIS Layers:** Spatial subsets for interactive mapping and field comparison

---

## REFERENCES TO COASTAL SUBSIDENCE LITERATURE

Methods align with established ICESat-2 coastal analysis protocols (Adusumilli et al. 2022; Nolan et al. 2021) and permafrost subsidence monitoring standards (Christiansen et al. 2019; Isaksen et al. 2022). Spatial scale (2 km) optimized for Arctic coastal thermokarst features (Seserman et al. 2024).
