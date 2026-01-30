# ICESat-2 Workflow Improvement Implementation Guide

**Date**: January 16, 2026  
**Status**: Critical fixes - implement before publication  
**Priority**: Immediate (Tier 1: Required for scientific validity)

---

## Executive Summary

This guide provides step-by-step instructions to implement **6 critical improvements** to the ICESat-2 permafrost subsidence analysis workflow. These fixes address:

1. **Sign convention errors** (Scripts 04, 06) - affecting all spatial statistics
2. **Temporal calculation inflation** (Script 03) - rates inflated 24-50×
3. **Insufficient outlier filtering** (Scripts 01-03) - extreme values persisting
4. **Figure generation errors** (Script 07) - dimension and colormap issues
5. **Metadata incompleteness** (All scripts) - reproducibility gaps

**Expected outcomes**:
- Subsidence rates: -1063 mm/yr → **-20 to -50 mm/yr** ✓
- Spatial maps: all red (uplift) → **blue (subsidence) dominant** ✓
- RMSE ratio: 4.067 → **< 2.0** ✓
- Figures: dimension warnings → **clean generation** ✓

---

## Implementation Timeline

| Phase | Duration | Tasks | Scripts |
|-------|----------|-------|---------|
| **Phase 1** | 1 week | Fix temporal calculation, fix sign convention | 03, 04, 06 |
| **Phase 2** | 2-3 weeks | Multi-tier outlier filtering | 01, 02, 03 |
| **Phase 3** | 1 week | Enhanced visualizations, metadata | 05, 07 |
| **Phase 4** | 1 week | Testing, validation, documentation | All |

---

## Phase 1: Critical Sign Convention & Temporal Fixes (1 week)

### Task 1.1: Fix Temporal Calculation in Script 03 (1-2 hours)

**File**: `03_analysis_core.py`

**Problem**: 
```python
# WRONG: Uses fixed 15-day parameter
rate_mm_yr = dH_m / (TEMPORAL_UNCERTAINTY_DAYS / 365.25) * 1000
# Result: rates inflated by factor of (actual_days / 15)
```

**Solution**: Use actual temporal separation between paired observations

**Implementation**:

1. **Open** `03_analysis_core.py`

2. **Locate** the crossover pair calculation section (~line 300-400)

3. **Find** where rate is calculated - typically in a loop processing pairs

4. **Replace** the rate calculation with:
```python
# CRITICAL FIX: Use actual temporal separation, not fixed 15-day window
dH_m = photons.iloc[j]['h'] - photons.iloc[i]['h']
dt_days = abs((photons.iloc[j]['datetime'] - photons.iloc[i]['datetime']).days)

# Correct formula uses ACTUAL days between observations
dt_years = dt_days / 365.25
rate_mm_yr = (dH_m / dt_years) * 1000.0

# Save actual temporal separation for metadata
crossover_pair = {
    ...
    'dt_days_actual': dt_days,  # NEW: Actual separation
    'dt_years': dt_years,
    'subsidence_rate_mm_per_year': rate_mm_yr,  # Correct rate
    ...
}
```

5. **Update** metadata to include `dt_days_actual`

6. **Add** verification logging:
```python
mean_rate = df_clean['subsidence_rate_mm_per_year'].mean()
logger.info(f"Mean subsidence rate: {mean_rate:.2f} mm/yr")
logger.info(f"Expected range: -20 to -50 mm/yr for Tuktoyaktuk permafrost")

# Critical assertion
assert -100 < mean_rate < 20, f"Rate {mean_rate} outside expected range!"
```

**Verification**:
- ✓ Log shows mean rate in range -50 to -20 mm/yr
- ✓ No assertion error
- ✓ Output file includes `dt_days_actual` column

---

### Task 1.2: Fix Sign Convention in Script 04 (30 minutes)

**File**: `04_analysis_spatial.py`

**Problem**: Column detection picks `rate_uncertainty_mm_yr` instead of subsidence rate

**Solution**: Implement priority-based column detection

**Implementation**:

1. **Open** `04_analysis_spatial.py`

2. **Locate** column detection section (~line 200-220):
```python
# WRONG: Picks first 'rate' match
for col in df.columns:
    if 'rate' in col.lower():
        rate_col = col  # Could be uncertainty!
```

3. **Replace** with prioritized detection:
```python
# Priority 1: Explicit subsidence rate column
rate_col = None
for col in df.columns:
    if col.lower() == 'subsidence_rate_mm_per_year':
        rate_col = col
        break

# Priority 2: Any subsidence column
if rate_col is None:
    for col in df.columns:
        if 'subsidence' in col.lower() and 'rate' in col.lower():
            rate_col = col
            break

# Priority 3: Generic rate (but not uncertainty!)
if rate_col is None:
    for col in df.columns:
        if 'rate' in col.lower() and 'uncertainty' not in col.lower():
            rate_col = col
            break

# Validate detection
assert rate_col is not None, "Cannot find rate column!"
logger.info(f"Using rate column: {rate_col}")
```

4. **Add sign convention verification**:
```python
# Verify sign convention
mean_rate = df[rate_col].mean()
negative_count = (df[rate_col] < 0).sum()
negative_pct = 100.0 * negative_count / len(df)

logger.info(f"Sign verification:")
logger.info(f"  Mean rate: {mean_rate:.2f} mm/yr")
logger.info(f"  Negative (subsidence): {negative_count:,} ({negative_pct:.1f}%)")
logger.info(f"  Positive (uplift): {(df[rate_col] > 0).sum():,}")

assert negative_count > 0, "No subsidence values found!"
assert negative_pct > 20, f"Only {negative_pct:.1f}% show subsidence!"
```

5. **Fix dimension validation** (around line 280):
```python
# Validate coordinate ranges
lon_min, lat_min = coords.min(axis=0)
lon_max, lat_max = coords.max(axis=0)

assert lon_max > lon_min, f"Invalid lon range: {lon_min} to {lon_max}"
assert lat_max > lat_min, f"Invalid lat range: {lat_min} to {lat_max}"

logger.info(f"Grid extent: lon {lon_min:.4f}°-{lon_max:.4f}°, lat {lat_min:.4f}°-{lat_max:.4f}°")
```

**Verification**:
- ✓ Log shows correct rate column name
- ✓ Log shows >50% negative values
- ✓ No assertion errors
- ✓ Grid dimensions printed without errors

---

### Task 1.3: Fix Sign Convention in Script 06 (30 minutes)

**File**: `06_analysis_advanced.py`

**Same procedure as Script 04**, but with additional hotspot threshold check:

1. **Apply same column detection** as Task 1.2

2. **Fix hotspot identification** (~line 280):
```python
# OLD (WRONG - picks uplift as hotspots)
hotspot_mask = subsidence_rates >= SUBSIDENCE_THRESHOLD_MM_YR  # ✗ Wrong sign!

# NEW (CORRECT - for negative threshold, use <=)
# Note: SUBSIDENCE_THRESHOLD_MM_YR should be NEGATIVE (e.g., -20.0)
hotspot_mask = subsidence_rates <= SUBSIDENCE_THRESHOLD_MM_YR  # ✓ Correct

logger.info(f"Hotspot threshold: subsidence >= {abs(SUBSIDENCE_THRESHOLD_MM_YR)} mm/yr")
logger.info(f"(Represented as <= {SUBSIDENCE_THRESHOLD_MM_YR} in negative convention)")
```

3. **Verify cluster statistics are negative**:
```python
if len(df_cluster_stats) > 0:
    mean_cluster = df_cluster_stats['mean_subsidence'].mean()
    logger.info(f"Cluster mean subsidence: {mean_cluster:.2f} mm/yr")
    
    if mean_cluster > 0:
        logger.warning(f"WARNING: Cluster mean is positive - may indicate wrong column!")
        logger.warning(f"Column used: {rate_col}")
```

**Verification**:
- ✓ Log shows correct hotspot threshold interpretation
- ✓ Cluster statistics show NEGATIVE means
- ✓ No positive uplift clusters

---

## Phase 2: Multi-Tier Outlier Filtering (2-3 weeks)

### Task 2.1: Enhance Script 03 Outlier Filtering (1-2 weeks)

**File**: `03_analysis_core.py`

**Problem**: Current 3-sigma filtering leaves ±80 m/yr extremes; RMSE ratio = 4.067

**Solution**: Implement multi-tier approach (Nature 2024 methodology)

**Implementation**:

1. **Add multi-tier filtering function** to Script 03:
```python
def filter_outliers_multitier(df, rate_col, config, logger=None):
    """
    Multi-tier outlier filtering:
    Tier 1: Physical plausibility
    Tier 2: Progressive sigma (5σ → 3σ → 2σ)
    """
    
    initial_count = len(df)
    logger.info(f"Starting multi-tier filtering with {initial_count:,} points")
    
    # TIER 1: Physical plausibility
    logger.info("TIER 1: Physical plausibility filtering...")
    
    physical_filter = (
        (np.abs(df[rate_col]) < config.RATE_PHYSICAL_MAX_MM_YR) &
        (df['dH_m'] < config.DH_PHYSICAL_MAX_M)
    )
    df = df[physical_filter].copy()
    tier1_removed = initial_count - len(df)
    logger.info(f"  Removed {tier1_removed:,} physically implausible points")
    
    # TIER 2: Progressive sigma filtering
    logger.info("TIER 2: Progressive sigma filtering...")
    
    for sigma_level in [5.0, 3.0, 2.0]:
        mean_rate = df[rate_col].mean()
        std_rate = df[rate_col].std()
        
        sigma_filter = np.abs(df[rate_col] - mean_rate) < sigma_level * std_rate
        removed = (~sigma_filter).sum()
        df = df[sigma_filter].copy()
        
        logger.info(f"  σ={sigma_level}: Removed {removed:,}, remaining {len(df):,}")
    
    return df, {
        'initial': initial_count,
        'tier1_removed': tier1_removed,
        'final': len(df),
        'removal_pct': 100 * (initial_count - len(df)) / initial_count,
    }
```

2. **Add configuration parameters** to Config class:
```python
class Config:
    # ... existing parameters ...
    
    # TIER 1: Physical plausibility limits
    RATE_PHYSICAL_MAX_MM_YR = 200.0  # Absolute max subsidence
    DH_PHYSICAL_MAX_M = 2.0          # Max elevation change per pair
    
    # TIER 2: Progressive sigma filtering
    OUTLIER_SIGMA_TIERS = [5.0, 3.0, 2.0]
```

3. **Apply filtering** in main execution:
```python
logger.info("Applying multi-tier outlier filtering...")
df_clean, filter_stats = filter_outliers_multitier(
    df_pairs,
    'subsidence_rate_mm_per_year',
    config,
    logger
)

metadata.filtering_statistics = filter_stats
logger.info(f"Filtered results: {filter_stats['final']:,} points ({filter_stats['removal_pct']:.1f}% removed)")
```

**Verification**:
- ✓ Log shows tier-by-tier removal statistics
- ✓ RMSE ratio < 2.0
- ✓ No ±80 m/yr extremes remain
- ✓ Filtering statistics in metadata

---

### Task 2.2: Add Quality Filtering to Scripts 01 & 02 (1 week)

**Files**: `01_filter_photons.py`, `02_filter_terrain.py`

**Problem**: Poor quality photons retained, causing outliers downstream

**Solution**: Add ATL03 quality metrics filtering

**Implementation for Script 01**:

1. **Locate** photon quality filtering section

2. **Add quality checks**:
```python
# Apply quality filters (ATL03 standards)
logger.info("Applying ATL03 quality filters...")

quality_filters = (
    # Signal confidence (0-4, higher is better)
    (photons['signal_conf_ph'] >= 3) &  # Medium-high confidence
    # Quality flag (0 = good)
    (photons['quality_ph'] == 0) &
    # Elevation plausibility (compared to reference DEM)
    (np.abs(photons['h_ph'] - reference_dem_height) < 10)  # Within 10m of DEM
)

photons = photons[quality_filters].copy()
logger.info(f"After quality filtering: {len(photons):,} photons")

# Track removal statistics
metadata.filtering_statistics['quality_filters_removed'] = (
    len(photons) - quality_filters.sum()
)
```

**Implementation for Script 02**:

Similar approach - add confidence and DEM consistency checks

---

## Phase 3: Enhanced Visualizations (1 week)

### Task 3.1: Fix Figures in Script 07 (1 week)

**File**: `07_output_generation.py`

**Problem**: Dimension errors, no diverging colormap, uniform red maps

**Solution**: Implement dimension validation and TwoSlopeNorm

**Implementation**:

1. **Add imports**:
```python
from matplotlib.colors import TwoSlopeNorm, BoundaryNorm
import matplotlib.pyplot as plt
```

2. **Create spatial contour function**:
```python
def create_spatial_contour_map_fixed(spatial_grid, output_dir, logger=None):
    """Generate contour map with proper dimension handling and diverging colormap"""
    
    # Validate data
    if len(spatial_grid) == 0 or 'rate_mm_yr' not in spatial_grid.columns:
        logger.warning("Insufficient data for contour maps")
        return None
    
    # Create pivot table with dimension checks
    pivot = spatial_grid.pivot_table(
        values='rate_mm_yr',
        index='latitude',
        columns='longitude',
        aggfunc='mean'
    )
    
    if pivot.empty or pivot.shape[0] < 2 or pivot.shape[1] < 2:
        logger.warning("Pivot table has insufficient dimensions")
        return None
    
    # Get rate statistics
    rate_min = spatial_grid['rate_mm_yr'].quantile(0.05)
    rate_max = spatial_grid['rate_mm_yr'].quantile(0.95)
    
    # Create figure with TwoSlopeNorm
    fig, ax = plt.subplots(figsize=(12, 8))
    
    norm = TwoSlopeNorm(vmin=rate_min, vcenter=0, vmax=rate_max)
    
    cs = ax.contourf(
        pivot.columns, pivot.index, pivot.values,
        levels=20,
        cmap='RdBu_r',  # Blue=subsidence, Red=uplift
        norm=norm
    )
    
    cbar = plt.colorbar(cs, ax=ax, extend='both')
    cbar.set_label('Subsidence Rate (mm/yr)', fontsize=11)
    
    ax.set_xlabel('Longitude (°W)', fontsize=11)
    ax.set_ylabel('Latitude (°N)', fontsize=11)
    ax.set_title('Permafrost Subsidence Distribution (Blue=Subsidence, Red=Uplift)',
                fontsize=12, fontweight='bold')
    
    # Save
    fig.savefig(
        output_dir / '07_Figure_Spatial_Contours.png',
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    
    return fig
```

3. **Create 4-panel figure**:
```python
# Use the comprehensive function from 07_visualization_FIXES.py
create_enhanced_multipanel_figure(spatial_grid, clusters_df, output_dir, logger)
```

4. **Replace old figure calls**:
```python
# OLD (remove these)
# create_spatial_contour_map_fig04(...)
# create_hotspot_multipanel_fig06(...)

# NEW (replace with)
create_spatial_contour_map_fixed(spatial_grid, output_dir, logger)
create_enhanced_multipanel_figure(spatial_grid, clusters_df, output_dir, logger)
```

**Verification**:
- ✓ No dimension error messages
- ✓ Figures generated successfully
- ✓ Spatial maps show BLUE (subsidence) dominant
- ✓ Color scale symmetric around 0

---

## Phase 4: Testing & Validation (1 week)

### Verification Checklist

Run through each verification step to confirm fixes:

```markdown
## Sign Convention Verification

- [ ] Script 03: Log shows "subsidence_rate_mm_per_year" column detected
- [ ] Script 03: Log shows mean rate in range -50 to -20 mm/yr
- [ ] Script 03: Log shows >50% negative values
- [ ] Script 04: Log shows correct rate column name
- [ ] Script 04: Log shows grid extent (lon/lat ranges)
- [ ] Script 04: Output spatial_statistics.csv has negative mean values
- [ ] Script 06: Log shows hotspot threshold interpretation (>=XX mm/yr subsidence)
- [ ] Script 06: Cluster statistics show NEGATIVE mean subsidence
- [ ] Script 07: Figures generate without dimension warnings
- [ ] Script 07: Spatial maps show BLUE (subsidence) and RED (uplift) zones

## Temporal Calculation Verification

- [ ] Script 03: Column 'dt_days_actual' in output CSV
- [ ] Script 03: Subsidence rate values reasonable (not inflated)
- [ ] Script 03: Log confirms "TEMPORAL CALCULATION FIXED"
- [ ] Metadata includes filtering statistics

## Outlier Filtering Verification

- [ ] Script 03: Log shows tier-by-tier filtering results
- [ ] Script 03: RMSE ratio < 2.0
- [ ] Script 03: No extreme outliers (±80 m/yr) remain
- [ ] Script 01/02: Quality filters applied
- [ ] Script 01/02: Log shows photons removed by quality

## Figure Generation Verification

- [ ] Script 07: No dimension error messages
- [ ] Script 07: All figures generated successfully
- [ ] Spatial contours show correct subsidence/uplift distinction
- [ ] 4-panel figure includes all panels
- [ ] Color scales are sensible (symmetric around 0)

## Metadata Verification

- [ ] All metadata files include filtering statistics
- [ ] Temporal metadata includes dt_days_actual
- [ ] Column names logged in all spatial scripts
- [ ] Git hashes recorded for reproducibility
```

### Test Run Procedure

1. **Run Script 03**:
```bash
python 03_analysis_core.py --temporal 2
```
Expected output:
- Mean rate: -25 to -45 mm/yr
- dt_days_actual column in output
- TEMPORAL CALCULATION FIXED confirmation

2. **Run Script 04**:
```bash
python 04_analysis_spatial.py
```
Expected output:
- Correct rate column detected
- >50% negative values verified
- No dimension warnings

3. **Run Script 06**:
```bash
python 06_analysis_advanced.py
```
Expected output:
- Correct rate column detected
- Cluster mean subsidence NEGATIVE
- Hotspot threshold correctly interpreted

4. **Run Script 07**:
```bash
python 07_output_generation.py
```
Expected output:
- All figures generated
- No dimension errors
- Spatial maps show blue/red distinction

---

## Quick Reference: Key File Changes

| Script | File | Changes | Priority |
|--------|------|---------|----------|
| 03 | analysis_core.py | Temporal calc, multi-tier filtering | **CRITICAL** |
| 04 | analysis_spatial.py | Column detection, dimension validation | **CRITICAL** |
| 06 | analysis_advanced.py | Column detection, hotspot threshold | **CRITICAL** |
| 07 | output_generation.py | Dimension validation, TwoSlopeNorm | **HIGH** |
| 01 | filter_photons.py | Quality metrics filtering | **HIGH** |
| 02 | filter_terrain.py | Quality metrics filtering | **HIGH** |
| 05 | analysis_temporal.py | Update metadata | **MEDIUM** |

---

## Support Files

The following support files have been created:

- **03_analysis_core_FIXED.py** - Complete fixed version (copy code)
- **04_analysis_spatial_COLUMN_FIX.py** - Detailed patch instructions
- **06_analysis_advanced_COLUMN_FIX.py** - Detailed patch instructions
- **07_visualization_FIXES.py** - Complete visualization functions
- **IMPLEMENTATION_PLAN.md** - Overview of all issues

---

## Questions & Troubleshooting

**Q: How do I know if the sign convention is fixed?**  
A: After running Script 04, check the output log for "Negative values (subsidence): X% (should be >50%)". Also verify spatial maps show BLUE zones.

**Q: What if temporal calculation shows strange rates?**  
A: Check Script 03 log for "TEMPORAL CALCULATION FIXED" message. Verify output CSV has 'dt_days_actual' column with actual time separations.

**Q: Figures still showing all red?**  
A: Check Script 04 log for detected rate column. If it shows "rate_uncertainty_mm_yr", the column detection fix didn't apply correctly.

**Q: How do I revert if something breaks?**  
A: Keep backups of original scripts. Changes are isolated to specific functions, so you can revert individual changes if needed.

---

## Next Steps After Implementation

1. **Document changes** in workflow README with version notes
2. **Update methodology section** of thesis/paper with correct rate calculations
3. **Regenerate all figures** using fixed scripts
4. **Recalculate statistics** for publication
5. **Archive complete run** with metadata for reproducibility
6. **Update supplementary materials** with corrected data/figures

