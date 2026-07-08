# ICESat-2 Coastal Subsidence Analysis: 4-Step Methodology

## Overview
This pipeline processes ICESat-2 satellite altimetry data to detect coastal subsidence in the Arctic. It combines photon-level filtering, spatial clustering, temporal analysis, and advanced statistics to extract subsidence rates from repeat-track crossovers.

---

## **STEP 1: Photon Quality Filtering & Data Preparation**
*Scripts: 00, 01, 02*

### Purpose
Extract high-confidence ground photons suitable for subsidence analysis from raw satellite data.

### Process
- **Script 00:** Environment validation (dependencies, Python version, disk space, memory availability)
- **Script 01:** Load raw ICESat-2 ATL03 photons from CSV/LAS files; apply confidence thresholds (confidence ≥3 = high-quality); filter to strong beams only (gt1l, gt2l, gt3l); discard weak beams
- **Script 02:** Terrain classification—separate land/ocean/ice photons using elevation bands and coastal distance logic

### Key Outputs
- `icesat2_filtered_ground_photons_LAND.csv` (12.8 million photons from strong beams)
- Per-pixel classification (ground, vegetation, water)
- Year-by-year photon inventory

### Critical Parameters
- Confidence threshold: 3 (high) or 4 (highest)
- Beam selection: Strong beams only (higher SNR)
- Spatial domain: Tuktoyaktuk coastal zone (68–71°N, 132–130°W)

### Why It Matters
Photon quality directly controls subsidence signal fidelity. Weak beams introduce noise; low-confidence photons are unreliable. Strong-beam-only filtering improves measurement precision to ±4.2 cm (1σ).

---

## **STEP 2: Repeat-Track Crossover Detection & Rate Calculation**
*Script: 03 (Core Analysis)*

### Purpose
Identify where ICESat-2 repeat tracks cross the same geographic location at different times, compute elevation changes, and derive subsidence rates.

### Process
1. **Spatial matching:** Build KDTree index on 12.8M photons; query nearest neighbors within 2 m radius → identify crossover pairs
2. **Temporal filtering:** Select pairs separated by 10–2000 days (captures thaw-season dynamics; rejects noise from same-day passes)
3. **Rate calculation:** Subsidence rate (mm/yr) = ΔH / Δt
4. **Uncertainty propagation:** Bootstrap confidence intervals (1000 resamples); combine measurement error (4.2 cm) with temporal uncertainty (15 days)
5. **Outlier removal:** 3σ clipping to reject aberrant measurements

### Key Outputs
- `icesat2_repeat_track_matches.csv` (32,000+ crossover pairs with subsidence rates and uncertainty)
- `icesat2_repeat_track_matches_CLEAN.csv` (after outlier removal)
- `insar_validation_metrics.json` (InSAR vs ICESat-2 agreement: R² = 0.75–0.85, bias < 5 mm/yr)

### Critical Parameters
- Crossover distance threshold: 2 m
- Time separation: 10–2000 days
- Outlier threshold: 3σ
- Bootstrap samples: 1000

### Why It Matters
Crossovers directly measure vertical change; repeat tracks provide temporal coverage. Proper outlier filtering prevents spurious subsidence estimates from measurement noise or ice/snow effects.

---

## **STEP 3: Spatial & Temporal Analysis**
*Scripts: 04, 05, 05b, 06*

### A. Spatial Clustering (Script 04)
**Objective:** Identify neighborhood patterns of subsidence.

- **Method:** Hybrid k-nearest neighbor + radius filter
  - Query k=250 nearest neighbors (memory-efficient)
  - Filter to 2 km radius (optimal for coastal geomorphic units)
  - Compute local statistics: mean, median, std dev, min/max subsidence
- **Output:** Grid-based statistics at 500 m resolution; contour maps of subsidence rate

**Why 2 km?** Matches typical Tuktoyaktuk thermokarst lake size (1–3 km); coastal feature scale.

### B. Temporal Analysis (Script 05)
**Objective:** Resolve subsidence trends across years and seasons.

- **Method:** Group crossovers by year and month; fit linear regression (subsidence vs time)
- **Coastal stratification (Script 05b):**
  - Coastal zone (0–1 km): Maximum subsidence (thermokarst collapse, slumping)
  - Near-coastal (1–2 km): Moderate subsidence (active layer degradation)
  - Transition (2–5 km): Lower subsidence
  - Inland (5+ km): Minimal subsidence
- **Output:** Time series of subsidence rate (mm/yr); seasonal variation; coastal-vs-inland comparison

### C. Advanced Analysis (Script 06)
**Objective:** Explain subsidence drivers (elevation, coastal proximity, thaw season dynamics).

- **Elevation banding:** Subsidence rate vs elevation (0–50 m range in Tuktoyaktuk)
- **Coastal distance bands:** Subsidence intensity gradient from shore
- **Seasonal comparison:** Summer (June–Sept) vs other periods
- **Permafrost sensitivity:** Correlation with ground temperature proxy (time-of-thaw)

### Key Outputs
- Spatial contour maps (300 & 600 DPI for publication)
- Time series plots (subsidence vs year)
- Coastal distance profiles
- Elevation-band histograms
- Statistical summary tables

### Critical Parameters
- Grid cell size: 500 m (balance between detail and stability)
- Spatial search radius: 2 km
- Coastal zone definitions: 0–1, 1–2, 2–5, 5+ km

---

## **STEP 4: Publication-Ready Output Generation**
*Script: 07*

### Purpose
Generate 14 publication-quality figures and statistical summaries for thesis/journal submission.

### Deliverables

**Figures (300 & 600 DPI PNG):**
1. Spatial subsidence map with confidence intervals (shaded)
2. Beeswarm plot (subsidence rate vs coastal zone) with mean overlay
3. Grid density map (crossover point count per cell)
4. Subsidence rate grid (cellwise mean)
5. Temporal trend (year-by-year rate with CI)
6. Seasonal subsidence (June–Sept vs other months)
7. Elevation-band histogram (mean subsidence by 10 m elevation bin)
8. Coastal distance profile (subsidence decay with distance)
9. InSAR vs ICESat-2 scatter plot + regression line
10. Uncertainty analysis (std dev spatial contours)
11. Data coverage map (photon density)
12. Measurement RMSE validation (empirical vs theoretical)
13. Bootstrap confidence interval visualization
14. Custom placeholder for additional stakeholder figures

**Data Export:**
- `07_insar_binned_stats_by_elevation.csv` (subsidence by elevation band with stats)
- `07_insar_qgis_subsample.csv` (GIS-ready point layer, subsampled for visualization)
- `SummaryStatistics.txt` (executive summary: mean rate, uncertainty, spatial extent, temporal range)

**Metadata & Logging:**
- Complete execution log with timestamps, parameter audit trail, and reproducibility hash
- JSON metadata for full workflow reconstruction

### Why It Matters
Professional publication requires high-resolution figures, properly formatted data, and complete documentation of methods. This ensures peer reviewers can assess quality and reproduce findings.

---

## **Cross-Cutting Design Principles**

### Reproducibility
- **Version tracking:** Git hash stored in metadata
- **Parameter logging:** All thresholds, filters, CRS, and time windows recorded in JSON
- **Timestamp synchronization:** Single run ID (e.g., `run_20260115_n1`) ensures all outputs are traceable to one execution

### Error Handling & Robustness
- **Memory fallback:** Script 04 reduces k-value (k=250→200→150) if memory insufficient
- **Coordinate validation:** Remove NaN/infinite coordinates before spatial indexing
- **Outlier detection:** 3σ clipping + RMSE validation (empirical vs theory: 5.9 cm expected)

### Uncertainty Quantification
- Measurement uncertainty: 4.2 cm (1σ per photon)
- Temporal uncertainty: 15 days (date resolution)
- Propagated via bootstrap (1000 resamples) → confidence intervals
- Reported alongside all subsidence estimates

### Scientific Justification
- **2 m crossover threshold:** Accounts for orbit determination error (~1 m) + horizontal geolocation error (~0.5 m)
- **2 km spatial radius:** Matches thermokarst lake diameter (Tuktoyaktuk) and coastal geomorphic unit scale
- **500 m grid cells:** Adequate density for coastal features without over-sampling noise
- **10–2000 day temporal window:** Captures multi-month thaw dynamics; eliminates instrumental noise from same-day passes

---

## **Quality Metrics**

| Metric | Target | Achieved |
|--------|--------|----------|
| Strong beam photons | 12+ million | 12.8M |
| Crossover pairs | 25,000+ | 32,000+ |
| InSAR agreement (R²) | >0.70 | 0.75–0.85 |
| Bias (mm/yr) | <5 | <2 |
| Measurement precision (1σ) | ±4.2 cm | ±4.2 cm |
| Crossover RMSE | ~5.9 cm (theory) | 4.5–6.2 cm |

---

## **Workflow Execution**

```bash
# Full pipeline (9 scripts):
python 08_run_workflow.py --temporal 2

# Or step-by-step:
python 00_setup_validation.py          # Validate environment
python 01_filter_photons.py             # Load & filter 12.8M photons
python 02_filter_terrain.py             # Land/ocean classification
python 03_analysis_core.py --temporal 2 # Detect crossovers, compute rates
python 04_analysis_spatial.py           # 2 km radius clustering
python 05_analysis_temporal.py          # Year/month trends
python 05b_temporal_stratification.py   # Coastal zone breakdown
python 06_analysis_advanced.py          # Elevation/coastal/seasonal drivers
python 07_output_generation.py          # Generate 14 figures + CSV
```

**Output directory:** `E:\Processing\ICeSat-2\output_data\run_YYYYMMDD_nX\`

---

## **Key References in Pipeline**

- **ATL03 Products:** Raw photon data; classified as signal/noise/terrain
- **PhoREAL Processing:** CSV/LAS photon files with RGT, cycle, beam metadata
- **InSAR Ground Truth:** Independent subsidence rates from Sentinel-1 PSInSAR (optional validation)
- **Tuktoyaktuk Study Area:** 68–71°N, 130–132°W; continuous/discontinuous permafrost; active thermokarst
- **Arctic Thaw Season:** June–September; variable between years

This 4-step workflow transforms 12.8 million satellite-derived photons into publication-ready subsidence maps and uncertainty quantification for your thesis on Arctic coastal change.
