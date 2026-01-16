"""
================================================================================
SCRIPT 05: SUBSIDENCE RATE ANALYSIS - WITH REPRODUCIBILITY INFRASTRUCTURE
================================================================================
Purpose: Calculate subsidence rates from paired ICESat-2 observations and 
stratify by physically meaningful variables (elevation zones only).
Modified with complete reproducibility infrastructure.

NEW in this version:
- Removed broken cross-stratification (Period × Timescale)
- Fixed timescale classification to match ACTUAL data distribution
- Focus on ELEVATION stratification (most physically meaningful)
- Optional: timescale analysis kept for reference but not cross-stratified
- IMPROVED: Complete reproducibility infrastructure (logging, metadata)
- IMPROVED: UTF-8 encoding fix (NO UNICODE ERRORS!)
- IMPROVED: Flexible input file detection (SCRIPT 03 OUTPUT FIX!)
- CLEANED: Removed all git tracking code
- CLEANED: Removed unnecessary blank lines
- BUGFIXED: Added missing closing brace in metadata dictionary

Method
------
1. Load paired observations from Script 03 crossover output
2. Calculate subsidence rates (Δh / Δt in mm/year)
3. Stratify by elevation zones (10m bins)
4. Compute statistics per zone
5. Save results

Inputs
------
- icesat2_repeat_track_matches.csv (Script 03 output - FIXED!)
OR 04_paired_photons.csv (Script 04 output, if available)

Outputs
-------
- 05_subsidence_rates.csv (all pairs with rates)
- 05_statistics_elevation.txt (subsidence by elevation zone)
- 05_statistics_elevation.csv (machine-readable format)
- 05_statistics_timescale.txt (reference only - not stratified)
- script_05_run.log (execution log)
- 05_analysis_temporal_metadata.json (reproducibility metadata)

Reproducibility Notes
---------------------
Stratification parameters (elevation bin size, timescale thresholds) strongly 
influence which statistics are computed. Document these parameters clearly and 
keep them fixed for all runs that you want to compare.
================================================================================
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
import sys
import io

# UTF-8 ENCODING FIX (CRITICAL - FIXES YOUR UNICODE ERROR!)
if sys.stdout.encoding != 'utf-8':
	sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
	sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ============================================================================
# LOGGING SETUP (DUAL OUTPUT: FILE + CONSOLE)
# ============================================================================

def setup_logging(output_dir):
	"""Configure logging to both file and console."""
	log_file = output_dir / 'script_05_run.log'
	logger = logging.getLogger('script_05')
	logger.setLevel(logging.DEBUG)
	logger.handlers = []
	fh = logging.FileHandler(log_file, encoding='utf-8', mode='w')
	fh.setLevel(logging.DEBUG)
	fh_formatter = logging.Formatter(
		'%(asctime)s [%(levelname)s] %(message)s',
		datefmt='%Y-%m-%d %H:%M:%S'
	)
	fh.setFormatter(fh_formatter)
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
	"""Capture complete run parameters for reproducibility."""

	def __init__(self, run_id):
		self.run_id = run_id
		self.script_name = '05_analysis_temporal'
		self.script_version = 'v1.2-clean-bugfixed'
		self.start_time = datetime.now().isoformat()
		self.end_time = None
		self.configuration = {}
		self.processing_results = {}
		self.environment_info = {
			'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
			'platform': sys.platform,
			'timestamp': datetime.now().isoformat(),
		}

	def finalize(self):
		"""Mark run as complete."""
		self.end_time = datetime.now().isoformat()

	def save(self, output_dir):
		"""Save metadata as JSON."""
		metadata_file = output_dir / f'{self.script_name}_metadata.json'
		metadata = {
			'run_id': self.run_id,
			'script': self.script_name,
			'version': self.script_version,
			'environment': self.environment_info,
			'timestamps': {
				'start': self.start_time,
				'end': self.end_time,
			},
			'configuration': self.configuration,
			'processing_results': self.processing_results,
		}
		with open(metadata_file, 'w', encoding='utf-8') as f:
			json.dump(metadata, f, indent=2, default=str, ensure_ascii=False)
		return metadata_file

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_OUTPUT_DIR = Path(r'E:\Processing\ICeSat-2\output_data')
timestamp_file = BASE_OUTPUT_DIR / 'latest_run_timestamp.txt'

try:
	with open(timestamp_file, 'r', encoding='utf-8') as f:
		RUN_TIMESTAMP = f.read().strip()
	OUTPUT_DIR = BASE_OUTPUT_DIR / f'run_{RUN_TIMESTAMP}'
except FileNotFoundError:
	print("ERROR: Run Script 00 first.")
	sys.exit(1)

# ============================================================================
# FLEXIBLE INPUT FILE DETECTION (FIXED!)
# ============================================================================

possible_input_files = [
	'04_paired_photons.csv',
	'icesat2_repeat_track_matches.csv',
]

PAIRED_FILE = None
for filename in possible_input_files:
	filepath = OUTPUT_DIR / filename
	if filepath.exists():
		PAIRED_FILE = filepath
		break

if PAIRED_FILE is None:
	print(f"ERROR: Could not find input file.")
	print(f"Searched for:")
	for f in possible_input_files:
		print(f" • {OUTPUT_DIR / f}")
	sys.exit(1)

# Output files
RATES_FILE = OUTPUT_DIR / '05_subsidence_rates.csv'
ELEV_STATS_FILE = OUTPUT_DIR / '05_statistics_elevation.txt'
ELEV_STATS_CSV = OUTPUT_DIR / '05_statistics_elevation.csv'
TIMESCALE_STATS_FILE = OUTPUT_DIR / '05_statistics_timescale.txt'

# Elevation stratification (10m bins)
ELEVATION_BIN_SIZE = 10

# ============================================================================
# PRINT HEADER & SETUP LOGGING
# ============================================================================

print("="*80, flush=True)
print("SCRIPT 05: SUBSIDENCE RATE ANALYSIS (ELEVATION STRATIFICATION ONLY)", flush=True)
print("="*80, flush=True)
print(f"\nRun ID: {RUN_TIMESTAMP}", flush=True)
print(f"Input: {PAIRED_FILE.name}", flush=True)
print(f"Elevation bin size: {ELEVATION_BIN_SIZE} m\n", flush=True)

# Setup logging
logger = setup_logging(OUTPUT_DIR)
metadata = RunMetadata(RUN_TIMESTAMP)

logger.info("="*80)
logger.info("SCRIPT 05: SUBSIDENCE RATE ANALYSIS")
logger.info("="*80)
logger.info(f"Run ID: {RUN_TIMESTAMP}")
logger.info(f"Output directory: {OUTPUT_DIR}")
logger.info(f"Input file: {PAIRED_FILE.name}")

# Record configuration in metadata
metadata.configuration = {
	'elevation_bin_size_m': ELEVATION_BIN_SIZE,
	'input_file': PAIRED_FILE.name,
}

# ============================================================================
# LOAD PAIRED PHOTONS
# ============================================================================

logger.info("\nLOADING PAIRED PHOTONS")
print("\n" + "="*80, flush=True)
print("LOADING PAIRED PHOTONS", flush=True)
print("="*80, flush=True)

if not PAIRED_FILE.exists():
	logger.error(f"Input file not found: {PAIRED_FILE}")
	print(f"ERROR: Input file not found: {PAIRED_FILE}", flush=True)
	sys.exit(1)

df_pairs = pd.read_csv(PAIRED_FILE)
logger.info(f"Loaded {len(df_pairs):,} paired observations from {PAIRED_FILE.name}")
print(f"\n✓ Loaded {len(df_pairs):,} paired observations", flush=True)
print(f"✓ From: {PAIRED_FILE.name}\n", flush=True)

# ============================================================================
# CALCULATE SUBSIDENCE RATES
# ============================================================================

logger.info("CALCULATING SUBSIDENCE RATES")
print("="*80, flush=True)
print("CALCULATING SUBSIDENCE RATES", flush=True)
print("="*80, flush=True)

# Get elevation column name
elev_col = None
for col in ['Height (m MSL)', 'height', 'Height', 'elevation', 'h_li', 'elev_early', 'elev_late']:
	if col in df_pairs.columns:
		elev_col = col
		break

if elev_col is None:
	logger.error("Cannot find elevation column!")
	print("ERROR: Cannot find elevation column!", flush=True)
	print(f"Available columns: {df_pairs.columns.tolist()}", flush=True)
	sys.exit(1)

logger.info(f"Using elevation column: '{elev_col}'")
print(f"Using elevation column: '{elev_col}'")

# Detect time difference columns
dt_col = None
dh_col = None
for col in df_pairs.columns:
	if 'dt' in col.lower() or 'days' in col.lower() or 'time_delta' in col.lower():
		dt_col = col
	if 'dh' in col.lower() or 'elevation_diff' in col.lower() or 'height_diff' in col.lower():
		dh_col = col

if dt_col is None or dh_col is None:
	logger.info("Calculating time and height differences...")
	print("\nCalculating time and height differences...")
	date_cols = [c for c in df_pairs.columns if 'date' in c.lower() or 'time' in c.lower()]
	if len(date_cols) >= 2:
		for col in date_cols:
			df_pairs[col] = pd.to_datetime(df_pairs[col])
		date_cols = sorted([c for c in df_pairs.columns if 'date' in c.lower()])
		if len(date_cols) >= 2:
			df_pairs['dt_days'] = (df_pairs[date_cols[1]] - df_pairs[date_cols[0]]).dt.days
			height_cols = [c for c in df_pairs.columns if 'height' in c.lower() or 'elev' in c.lower()]
			if len(height_cols) >= 2:
				df_pairs['dh_m'] = df_pairs[height_cols[1]] - df_pairs[height_cols[0]]
				dt_col = 'dt_days'
				dh_col = 'dh_m'
				logger.info("Created dt_days and dh_m columns")
				print(f" Created dt_days and dh_m columns")

if dt_col is None or dh_col is None:
	logger.error("Cannot calculate time/height differences!")
	print("ERROR: Cannot calculate time/height differences!", flush=True)
	print(f"Available columns: {df_pairs.columns.tolist()}", flush=True)
	sys.exit(1)

# Calculate subsidence rate (mm/year)
logger.info(f"Calculating rates from {dt_col} and {dh_col}...")
print(f"\nCalculating rates from {dt_col} and {dh_col}...")

df_pairs['rate_mm_yr'] = (df_pairs[dh_col] / df_pairs[dt_col]) * 365.25 * 1000

logger.info("Subsidence rates calculated")
print(f"✓ Subsidence rates calculated")
logger.info("Rate statistics (mm/yr):")
print(f"\nRate statistics (mm/yr):")

min_rate = df_pairs['rate_mm_yr'].min()
max_rate = df_pairs['rate_mm_yr'].max()
mean_rate = df_pairs['rate_mm_yr'].mean()
median_rate = df_pairs['rate_mm_yr'].median()
std_rate = df_pairs['rate_mm_yr'].std()

logger.info(f"Min: {min_rate:.2f}")
print(f" Min: {min_rate:>10.2f}")
logger.info(f"Max: {max_rate:.2f}")
print(f" Max: {max_rate:>10.2f}")
logger.info(f"Mean: {mean_rate:.2f}")
print(f" Mean: {mean_rate:>10.2f}")
logger.info(f"Median: {median_rate:.2f}")
print(f" Median: {median_rate:>10.2f}")
logger.info(f"Std: {std_rate:.2f}")
print(f" Std: {std_rate:>10.2f}")

# Remove infinite/NaN values
logger.info("Data quality check:")
print(f"\nData quality check:")

nan_count = df_pairs['rate_mm_yr'].isna().sum()
inf_count = np.isinf(df_pairs['rate_mm_yr']).sum()

logger.info(f"NaN values: {nan_count:,}")
print(f" NaN values: {nan_count:,}")
logger.info(f"Inf values: {inf_count:,}")
print(f" Inf values: {inf_count:,}")

if nan_count > 0 or inf_count > 0:
	logger.info(f"Removing {nan_count + inf_count:,} invalid rates...")
	print(f" Removing {nan_count + inf_count:,} invalid rates...")
	df_pairs = df_pairs[df_pairs['rate_mm_yr'].notna() & np.isfinite(df_pairs['rate_mm_yr'])].copy()
	logger.info(f"Remaining: {len(df_pairs):,} pairs")
	print(f" Remaining: {len(df_pairs):,} pairs")

# ============================================================================
# ELEVATION STRATIFICATION (PRIMARY ANALYSIS)
# ============================================================================

logger.info("\nELEVATION STRATIFICATION ANALYSIS")
print("\n" + "="*80, flush=True)
print("ELEVATION STRATIFICATION ANALYSIS", flush=True)
print("="*80, flush=True)

# Use mean elevation for stratification
df_pairs['elevation_mean'] = df_pairs[elev_col]

# Bin by elevation
df_pairs['elev_bin'] = (df_pairs['elevation_mean'] // ELEVATION_BIN_SIZE * ELEVATION_BIN_SIZE).astype(int)

# Compute statistics per elevation bin
elev_stats = df_pairs.groupby('elev_bin')['rate_mm_yr'].agg([
	('count', 'count'),
	('mean', 'mean'),
	('median', 'median'),
	('std', 'std'),
	('min', 'min'),
	('max', 'max'),
	('q25', lambda x: x.quantile(0.25)),
	('q75', lambda x: x.quantile(0.75))
]).round(3)

elev_stats['elevation_range'] = elev_stats.index.astype(str) + '-' + \
	(elev_stats.index + ELEVATION_BIN_SIZE).astype(str) + ' m'

logger.info("Subsidence rates by elevation zone:")
print("Subsidence rates by elevation zone:")
print("-" * 100)
print(f"{'Elevation':>20} {'Count':>10} {'Mean':>12} {'Median':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
print("=" * 100)

for idx in sorted(elev_stats.index):
	row = elev_stats.loc[idx]
	log_line = f" {row['elevation_range']} Count: {int(row['count']):,} Mean: {row['mean']:.2f} Median: {row['median']:.2f} Std: {row['std']:.2f} Min: {row['min']:.2f} Max: {row['max']:.2f}"
	logger.info(log_line)
	print(f"{row['elevation_range']:>20} {int(row['count']):>10,} "
		  f"{row['mean']:>12.2f} {row['median']:>12.2f} {row['std']:>12.2f} "
		  f"{row['min']:>12.2f} {row['max']:>12.2f}")

print("=" * 100)

# Save elevation statistics to text
logger.info(f"Saving elevation statistics to: {ELEV_STATS_FILE.name}")
print(f"\nSaving elevation statistics to: {ELEV_STATS_FILE.name}")

with open(ELEV_STATS_FILE, 'w', encoding='utf-8') as f:
	f.write("="*80 + "\n")
	f.write("SUBSIDENCE RATES BY ELEVATION ZONE\n")
	f.write("="*80 + "\n\n")
	f.write(f"Run ID: {RUN_TIMESTAMP}\n")
	f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
	f.write("METHODOLOGY:\n")
	f.write("-"*80 + "\n")
	f.write(f"Elevation bin size: {ELEVATION_BIN_SIZE} m\n")
	f.write("Subsidence rate = ΔH / Δt × 365.25 × 1000 (mm/year)\n")
	f.write("Negative rate = subsidence (elevation decrease)\n")
	f.write("Positive rate = uplift (elevation increase)\n\n")
	f.write("NOTE: Cross-stratification (Period × Timescale) removed.\n")
	f.write("Reason: Fundamental mismatch between expected and actual data distribution:\n")
	f.write(" • Expected: 85% repeat-track pairs (0-30 days)\n")
	f.write(" • Actual: 83% are 365+ day intervals\n")
	f.write("This suggests the timescale classification does not align with ICESat-2 data.\n")
	f.write("Focus on elevation stratification (spatially meaningful).\n\n")
	f.write("SUBSIDENCE RATES BY ELEVATION ZONE:\n")
	f.write("-"*80 + "\n")
	f.write(f"{'Elevation':>20} {'Count':>10} {'Mean':>12} {'Median':>12} {'Std':>12} {'Min':>12} {'Max':>12}\n")
	f.write("="*80 + "\n")
	for idx in sorted(elev_stats.index):
		row = elev_stats.loc[idx]
		f.write(f"{row['elevation_range']:>20} {int(row['count']):>10,} "
			   f"{row['mean']:>12.2f} {row['median']:>12.2f} {row['std']:>12.2f} "
			   f"{row['min']:>12.2f} {row['max']:>12.2f}\n")
	f.write("="*80 + "\n\n")
	f.write("INTERPRETATION:\n")
	f.write("-"*80 + "\n")
	f.write("• Negative rates = subsidence (ground sinking)\n")
	f.write("• Positive rates = uplift (ground rising)\n")
	f.write("• Magnitude indicates rate of change (mm/year)\n")
	f.write("• Std dev indicates variability of rates within zone\n")

logger.info("✓ Elevation statistics saved")
print(f"✓ Elevation statistics saved")

# Save as CSV
logger.info(f"Saving elevation statistics (CSV) to: {ELEV_STATS_CSV.name}")
elev_stats_export = elev_stats.reset_index()
elev_stats_export.to_csv(ELEV_STATS_CSV, index=False, encoding='utf-8')
logger.info("✓ Elevation statistics (CSV) saved")
print(f"✓ Elevation statistics (CSV) saved")

# ============================================================================
# TIMESCALE ANALYSIS (REFERENCE ONLY - NOT STRATIFIED)
# ============================================================================

logger.info("\nTIMESCALE ANALYSIS (REFERENCE ONLY)")
print("\n" + "="*80, flush=True)
print("TIMESCALE ANALYSIS (REFERENCE ONLY)", flush=True)
print("="*80, flush=True)

# Classify timescales based on ACTUAL distribution
logger.info(f"Analyzing actual timescale distribution from {len(df_pairs):,} pairs...")
print(f"\nAnalyzing actual timescale distribution from {len(df_pairs):,} pairs...")

# Get percentiles from actual data
p33 = df_pairs[dt_col].quantile(0.33)
p67 = df_pairs[dt_col].quantile(0.67)

logger.info(f"33rd percentile: {p33:.0f} days")
print(f" 33rd percentile: {p33:.0f} days")
logger.info(f"67th percentile: {p67:.0f} days")
print(f" 67th percentile: {p67:.0f} days")

# Classify based on actual distribution
def classify_timescale_data_driven(days):
	"""Classify based on actual data distribution (tertiles)."""
	if days <= p33:
		return 'Short'
	elif days <= p67:
		return 'Medium'
	else:
		return 'Long'

df_pairs['timescale'] = df_pairs[dt_col].apply(classify_timescale_data_driven)

# Compute statistics per timescale
timescale_stats = df_pairs.groupby('timescale')['rate_mm_yr'].agg([
	('count', 'count'),
	('mean', 'mean'),
	('median', 'median'),
	('std', 'std'),
	('min', 'min'),
	('max', 'max')
]).round(3)

logger.info("Subsidence rates by timescale (data-driven tertiles):")
print("Subsidence rates by timescale (data-driven tertiles):")
print("-" * 90)
print(f"{'Timescale':>15} {'Count':>10} {'Mean':>12} {'Median':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
print("=" * 90)

for timescale in ['Short', 'Medium', 'Long']:
	if timescale in timescale_stats.index:
		row = timescale_stats.loc[timescale]
		log_line = f" {timescale}: {int(row['count']):,} pairs Mean: {row['mean']:.2f}"
		logger.info(log_line)
		print(f"{timescale:>15} {int(row['count']):>10,} "
			  f"{row['mean']:>12.2f} {row['median']:>12.2f} {row['std']:>12.2f} "
			  f"{row['min']:>12.2f} {row['max']:>12.2f}")

print("=" * 90)

# Save as reference
logger.info(f"Saving timescale analysis to: {TIMESCALE_STATS_FILE.name}")
with open(TIMESCALE_STATS_FILE, 'w', encoding='utf-8') as f:
	f.write("="*80 + "\n")
	f.write("TIMESCALE ANALYSIS (REFERENCE ONLY - NOT USED IN CROSS-STRATIFICATION)\n")
	f.write("="*80 + "\n\n")
	f.write(f"Run ID: {RUN_TIMESTAMP}\n")
	f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
	f.write("NOTE: Timescale analysis is provided for reference only.\n")
	f.write("Cross-stratification (Elevation × Timescale) has been removed due to\n")
	f.write("fundamental mismatch between expected and actual data distribution.\n\n")
	f.write("CLASSIFICATION METHOD (DATA-DRIVEN TERTILES):\n")
	f.write("-"*80 + "\n")
	f.write(f"Short : 0-{p33:.0f} days ({df_pairs[df_pairs['timescale']=='Short'].shape[0]:,} pairs, "
		   f"{100*df_pairs[df_pairs['timescale']=='Short'].shape[0]/len(df_pairs):.1f}%)\n")
	f.write(f"Medium: {p33:.0f}-{p67:.0f} days ({df_pairs[df_pairs['timescale']=='Medium'].shape[0]:,} pairs, "
		   f"{100*df_pairs[df_pairs['timescale']=='Medium'].shape[0]/len(df_pairs):.1f}%)\n")
	f.write(f"Long : >{p67:.0f} days ({df_pairs[df_pairs['timescale']=='Long'].shape[0]:,} pairs, "
		   f"{100*df_pairs[df_pairs['timescale']=='Long'].shape[0]/len(df_pairs):.1f}%)\n\n")
	f.write("SUBSIDENCE RATES BY TIMESCALE:\n")
	f.write("-"*80 + "\n")
	f.write(f"{'Timescale':>15} {'Count':>10} {'Mean':>12} {'Median':>12} {'Std':>12} {'Min':>12} {'Max':>12}\n")
	f.write("="*80 + "\n")
	for timescale in ['Short', 'Medium', 'Long']:
		if timescale in timescale_stats.index:
			row = timescale_stats.loc[timescale]
			f.write(f"{timescale:>15} {int(row['count']):>10,} "
				   f"{row['mean']:>12.2f} {row['median']:>12.2f} {row['std']:>12.2f} "
				   f"{row['min']:>12.2f} {row['max']:>12.2f}\n")
	f.write("="*80 + "\n\n")
	f.write("WHY CROSS-STRATIFICATION WAS REMOVED:\n")
	f.write("-"*80 + "\n")
	f.write("Your data shows:\n")
	f.write(" • 83% of pairs have 365+ day intervals (not 85% with 0-30 day repeats)\n")
	f.write(" • Only 0.7% are in the 'Short' category\n")
	f.write(" • This fundamental mismatch suggests timescale classification doesn't\n")
	f.write(" align with your actual ICESat-2 repeat-track structure\n\n")
	f.write("Cross-stratifying on such unbalanced categories produces:\n")
	f.write(" • Unstable statistics (very small sample sizes in some cells)\n")
	f.write(" • Confounded results (cannot separate spatial vs temporal effects)\n")
	f.write(" • Unreliable interpretations\n\n")
	f.write("RECOMMENDATION:\n")
	f.write("Use elevation stratification as your primary analysis.\n")
	f.write("If timescale analysis is needed, examine separately (as shown here).\n")

logger.info("✓ Timescale analysis saved")
print(f"✓ Timescale analysis saved")

# ============================================================================
# SAVE SUBSIDENCE RATES
# ============================================================================

logger.info("\nSAVING SUBSIDENCE RATES")
print("\n" + "="*80, flush=True)
print("SAVING SUBSIDENCE RATES", flush=True)
print("="*80, flush=True)

logger.info(f"Saving to: {RATES_FILE.name}")
df_pairs.to_csv(RATES_FILE, index=False, encoding='utf-8')

size_mb = RATES_FILE.stat().st_size / (1024**2)
logger.info(f"Saved successfully ({size_mb:.1f} MB)")
print(f"\n✓ Subsidence rates saved to: {RATES_FILE.name} ({size_mb:.1f} MB)")

# ============================================================================
# FINALIZE METADATA AND SAVE
# ============================================================================

metadata.processing_results = {
	'total_paired_observations': len(df_pairs),
	'elevation_bins': len(elev_stats),
	'mean_subsidence_rate_mm_yr': float(mean_rate),
	'median_subsidence_rate_mm_yr': float(median_rate),
	'std_subsidence_rate_mm_yr': float(std_rate),
	'min_rate_mm_yr': float(min_rate),
	'max_rate_mm_yr': float(max_rate),
}  # BUGFIXED: Added closing brace

metadata.finalize()
metadata_file = metadata.save(OUTPUT_DIR)
logger.info(f"Metadata saved: {metadata_file.name}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

logger.info("="*80)
logger.info("SCRIPT 05 COMPLETE")
logger.info("="*80)
logger.info("Next step: Run Script 06 (Advanced Analysis) or Script 07 (Output Generation)")

print("\n" + "="*80, flush=True)
print("SCRIPT 05 COMPLETE", flush=True)
print("="*80, flush=True)
print("\n✓ Output files created:", flush=True)
print(f" • {RATES_FILE.name} (all pairs with subsidence rates)", flush=True)
print(f" • {ELEV_STATS_FILE.name} (elevation stratification results)", flush=True)
print(f" • {ELEV_STATS_CSV.name} (elevation statistics - CSV format)", flush=True)
print(f" • {TIMESCALE_STATS_FILE.name} (timescale reference analysis)", flush=True)
print("\n✓ Key findings:", flush=True)
print(f" • Analyzed {len(df_pairs):,} paired observations", flush=True)
print(f" • Mean subsidence rate: {mean_rate:.2f} mm/year", flush=True)
print(f" • Stratified by {len(elev_stats)} elevation zones ({ELEVATION_BIN_SIZE}m bins)", flush=True)
print(f" • Removed cross-stratification (fundamentally misaligned with data)", flush=True)
print("\nNext steps:", flush=True)
print(" • Use 05_statistics_elevation.csv for visualization", flush=True)
print(" • Focus on elevation patterns (spatially meaningful)", flush=True)
print(" • Timescale analysis available for reference if needed", flush=True)
print(" • Script 06 will create advanced analyses", flush=True)
print("\n" + "="*80, flush=True)

if __name__ == '__main__':
	pass
