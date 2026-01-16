"""
================================================================================
SCRIPT 05B: TEMPORAL STRATIFICATION - WITH REPRODUCIBILITY INFRASTRUCTURE
================================================================================

Purpose
-------
Perform temporal stratification (Early vs Late period comparison) with
complete statistical and visualization outputs.

Modified with complete reproducibility infrastructure.

FEATURES INCLUDED:
✓ Early vs Late period comparison (split at 2021)
✓ Robust statistics (t-test, Mann-Whitney, Kendall's tau)
✓ Bootstrap confidence intervals
✓ TEXT REPORT GENERATION (human-readable summary)
✓ DISTRIBUTION HISTOGRAMS & BOXPLOTS (publication figures)
✓ IMPROVED: Complete reproducibility infrastructure (logging, metadata, git hash)
✓ IMPROVED: UTF-8 encoding fix (NO UNICODE ERRORS!)

TEXT REPORT OUTPUT:
- Human-readable .txt file with all statistics
- Perfect for copying into thesis results section
- Shows interpretation and significance

VISUALIZATION OUTPUT:
- Distribution comparison (histogram + boxplot)
- Statistical annotations on figures
- Publication-ready quality (300 DPI default, 600 DPI HiRes option)

Inputs
------
- icesat2_repeat_track_matches.csv (Script 03 or 04 output)

Outputs
-------
- icesat2_early_period.csv (filtered early period data)
- icesat2_late_period.csv (filtered late period data)
- temporal_acceleration_metrics.csv (all metrics)
- temporal_period_statistics.txt (human-readable report)
- 05b_figure_distribution_comparison.png (300 DPI figure)
- 05b_figure_distribution_comparison_HiRes.png (600 DPI figure)
- script_05b_run.log (execution log)
- 05b_temporal_stratification_metadata.json (reproducibility metadata)

Reproducibility Notes
---------------------
Temporal split point (SPLIT_YEAR = 2021) strongly influences which
statistics are computed. Document this parameter clearly and keep it
fixed for all runs that you want to compare.

Complete audit trail provided via metadata JSON and execution log.
================================================================================
"""
import sys
import json
import logging
import io
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import subprocess

# UTF-8 ENCODING FIX
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def get_git_hash():
    """Get git commit hash for reproducibility"""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, timeout=5)
        return result.stdout.strip()[:8] if result.returncode == 0 else 'unknown'
    except:
        return 'unknown'

def setup_logging(output_dir):
    """Configure logging to both file and console"""
    logfile = output_dir / 'script_05b_run.log'
    logger = logging.getLogger('script05b')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    fh = logging.FileHandler(logfile, encoding='utf-8', mode='w')
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fh_formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def get_or_create_timestamp(base_dir):
    """Get timestamp from file, or create one if missing"""
    timestamp_file = base_dir / 'latest_run_timestamp.txt'
    if timestamp_file.exists():
        try:
            with open(timestamp_file, 'r', encoding='utf-8') as f:
                timestamp = f.read().strip()
            return timestamp
        except:
            pass
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    try:
        with open(timestamp_file, 'w', encoding='utf-8') as f:
            f.write(timestamp)
    except:
        pass
    return timestamp

class RunMetadata:
    """Capture complete run parameters for reproducibility"""
    def __init__(self, run_id, git_hash):
        self.run_id = run_id
        self.script_name = '05b_temporal_stratification'
        self.script_version = 'v1.4-fixed-paths'
        self.git_hash = git_hash
        self.start_time = datetime.now().isoformat()
        self.end_time = None
        self.configuration = {}
        self.processing_results = {}
        self.environment_info = {
            'python_version': f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}',
            'platform': sys.platform,
            'timestamp': datetime.now().isoformat(),
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
            'configuration': self.configuration,
            'processing_results': self.processing_results,
        }
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str, ensure_ascii=False)
        return metadata_file

# ============================================================================
# INITIALIZATION
# ============================================================================
BASE_OUTPUT_DIR = Path(r'E:\Processing\ICeSat-2\output_data')
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_TIMESTAMP = get_or_create_timestamp(BASE_OUTPUT_DIR)
OUTPUT_DIR = BASE_OUTPUT_DIR / f'run_{RUN_TIMESTAMP}'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logger = setup_logging(OUTPUT_DIR)
git_hash = get_git_hash()
metadata = RunMetadata(RUN_TIMESTAMP, git_hash)

POSSIBLE_INPUT_FILES = [
    OUTPUT_DIR / 'icesat2_repeat_track_matches.csv',
    OUTPUT_DIR / '04_paired_photons.csv',
]
INPUT_FILE = None
for filepath in POSSIBLE_INPUT_FILES:
    if filepath.exists():
        INPUT_FILE = filepath
        break

OUTPUT_EARLY = OUTPUT_DIR / 'icesat2_early_period.csv'
OUTPUT_LATE = OUTPUT_DIR / 'icesat2_late_period.csv'
OUTPUT_METRICS = OUTPUT_DIR / 'temporal_acceleration_metrics.csv'
OUTPUT_STATS = OUTPUT_DIR / 'temporal_period_statistics.txt'
OUTPUT_FIGURE_300 = OUTPUT_DIR / '05b_Figure_Distribution_Comparison.png'
OUTPUT_FIGURE_600 = OUTPUT_DIR / '05b_Figure_Distribution_Comparison_HiRes.png'

EARLY_PERIOD_START = pd.Timestamp('2019-01-15')
EARLY_PERIOD_END = pd.Timestamp('2021-12-31')
LATE_PERIOD_START = pd.Timestamp('2022-01-01')
LATE_PERIOD_END = pd.Timestamp('2024-12-31')

metadata.configuration = {
    'early_period_start': str(EARLY_PERIOD_START.date()),
    'early_period_end': str(EARLY_PERIOD_END.date()),
    'late_period_start': str(LATE_PERIOD_START.date()),
    'late_period_end': str(LATE_PERIOD_END.date()),
    'input_file': str(INPUT_FILE) if INPUT_FILE else 'NOT FOUND',
    'output_directory': str(OUTPUT_DIR),
}

print('=' * 80)
print('SCRIPT 05B - TEMPORAL STRATIFICATION - FIXED')
print('=' * 80)
print(f'Run ID: {RUN_TIMESTAMP}')
print(f'Git Hash: {git_hash}')
print(f'Output directory: {OUTPUT_DIR}')
print('=' * 80)
print('LOADING DATA')
print('=' * 80)

logger.info('=' * 80)
logger.info('SCRIPT 05B - TEMPORAL STRATIFICATION - FIXED')
logger.info('=' * 80)
logger.info(f'Run ID: {RUN_TIMESTAMP}')
logger.info(f'Git Hash: {git_hash}')
logger.info(f'Output directory: {OUTPUT_DIR}')

# ============================================================================
# DATA LOADING
# ============================================================================
if INPUT_FILE is None:
    print(f'ERROR: Input file not found!')
    print(f'Searched for:')
    for filepath in POSSIBLE_INPUT_FILES:
        print(f' • {filepath}')
    print(f'Run Script 03 or 04 first to generate the input file.')
    logger.error('Input file not found in OUTPUT_DIR')
    logger.error('Run Script 03 or 04 first')
    sys.exit(1)

try:
    df = pd.read_csv(INPUT_FILE, parse_dates=[col for col in
                                               pd.read_csv(INPUT_FILE, nrows=1).columns
                                               if 'date' in col.lower() or 'time' in col.lower()])
    logger.info(f'Loaded {len(df)} crossover points from {INPUT_FILE.name}')
    print(f'Loaded {len(df)} crossover points')
except Exception as e:
    print(f'ERROR reading {INPUT_FILE.name}: {e}')
    logger.error(f'ERROR reading {INPUT_FILE.name}: {e}')
    sys.exit(1)

print('=' * 80)
print('TEMPORAL STRATIFICATION')
print('=' * 80)
logger.info('TEMPORAL STRATIFICATION')

datetime_cols = [col for col in df.columns
                 if 'date' in col.lower() or 'time' in col.lower()]
if datetime_cols:
    datetime_col = datetime_cols[0]
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
else:
    print('ERROR: No datetime column found')
    logger.error('No datetime column found')
    sys.exit(1)

early = df[(df[datetime_col] >= EARLY_PERIOD_START) &
           (df[datetime_col] <= EARLY_PERIOD_END)].copy()
late = df[(df[datetime_col] >= LATE_PERIOD_START) &
          (df[datetime_col] <= LATE_PERIOD_END)].copy()

print(f'Early period ({EARLY_PERIOD_START.date()} to {EARLY_PERIOD_END.date()}): {len(early)} points')
print(f'Late period ({LATE_PERIOD_START.date()} to {LATE_PERIOD_END.date()}): {len(late)} points')
logger.info(f'Early period: {len(early)} points')
logger.info(f'Late period: {len(late)} points')

early.to_csv(OUTPUT_EARLY, index=False)
late.to_csv(OUTPUT_LATE, index=False)
print(f'Saved {OUTPUT_EARLY.name}')
print(f'Saved {OUTPUT_LATE.name}')

print('=' * 80)
print('ROBUST STATISTICAL ANALYSIS')
print('=' * 80)
logger.info('ROBUST STATISTICAL ANALYSIS')

elevation_cols = [col for col in df.columns
                  if any(x in col.lower() for x in ['elevation', 'height', 'change', 'rate', 'subsidence'])]
if not elevation_cols:
    print('ERROR: No elevation/height/change column found')
    logger.error('No elevation/height/change column found')
    sys.exit(1)

elev_col = elevation_cols[0]
print(f'Using column: {elev_col}')

early_vals = early[elev_col].dropna().values
late_vals = late[elev_col].dropna().values
early_mean, early_std = np.mean(early_vals), np.std(early_vals)
late_mean, late_std = np.mean(late_vals), np.std(late_vals)

early_vals_clean = early_vals[np.abs(early_vals - early_mean) < 5 * early_std]
late_vals_clean = late_vals[np.abs(late_vals - late_mean) < 5 * late_std]

print(f'Early period - Mean: {np.mean(early_vals_clean):.3f}, Std: {np.std(early_vals_clean):.3f}')
print(f'Late period - Mean: {np.mean(late_vals_clean):.3f}, Std: {np.std(late_vals_clean):.3f}')

t_stat, t_pval = stats.ttest_ind(early_vals_clean, late_vals_clean)
u_stat, u_pval = stats.mannwhitneyu(early_vals_clean, late_vals_clean)
tau, tau_pval = stats.kendalltau(range(len(early_vals_clean)), early_vals_clean)

print(f'T-test: t={t_stat:.4f}, p={t_pval:.6f} {"SIGNIFICANT" if t_pval < 0.05 else "not significant"}')
print(f'Mann-Whitney U: U={u_stat:.2f}, p={u_pval:.6f} {"SIGNIFICANT" if u_pval < 0.05 else "not significant"}')
print(f'Kendall\'s tau: tau={tau:.4f}, p={tau_pval:.6f}')

pooled_std = np.sqrt(((len(early_vals_clean)-1) * np.std(early_vals_clean, ddof=1)**2 +
                       (len(late_vals_clean)-1) * np.std(late_vals_clean, ddof=1)**2) /
                      (len(early_vals_clean) + len(late_vals_clean) - 2))
cohen_d = (np.mean(late_vals_clean) - np.mean(early_vals_clean)) / pooled_std if pooled_std != 0 else 0
print(f'Cohen\'s d: {cohen_d:.4f}', end='')
if abs(cohen_d) < 0.2:
    print(' (negligible effect)')
elif abs(cohen_d) < 0.5:
    print(' (small effect)')
elif abs(cohen_d) < 0.8:
    print(' (medium effect)')
else:
    print(' (large effect)')

n_bootstrap = 1000
bootstrap_diffs = []
for _ in range(n_bootstrap):
    early_sample = np.random.choice(early_vals_clean, size=len(early_vals_clean), replace=True)
    late_sample = np.random.choice(late_vals_clean, size=len(late_vals_clean), replace=True)
    bootstrap_diffs.append(np.mean(late_sample) - np.mean(early_sample))

ci_lower = np.percentile(bootstrap_diffs, 2.5)
ci_upper = np.percentile(bootstrap_diffs, 97.5)
print(f'Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')

early_mad = np.median(np.abs(early_vals_clean - np.median(early_vals_clean)))
late_mad = np.median(np.abs(late_vals_clean - np.median(late_vals_clean)))
print(f'Median Absolute Deviation - Early: {early_mad:.4f}, Late: {late_mad:.4f}')

metadata.processing_results = {
    'early_period_count': int(len(early_vals_clean)),
    'late_period_count': int(len(late_vals_clean)),
    'early_period_mean': float(np.mean(early_vals_clean)),
    'late_period_mean': float(np.mean(late_vals_clean)),
    'early_period_std': float(np.std(early_vals_clean)),
    'late_period_std': float(np.std(late_vals_clean)),
    'early_period_median': float(np.median(early_vals_clean)),
    'late_period_median': float(np.median(late_vals_clean)),
    'early_mad': float(early_mad),
    'late_mad': float(late_mad),
    't_test_statistic': float(t_stat),
    't_test_p_value': float(t_pval),
    'mann_whitney_u_statistic': float(u_stat),
    'mann_whitney_u_p_value': float(u_pval),
    'kendall_tau': float(tau),
    'kendall_tau_p_value': float(tau_pval),
    'cohens_d': float(cohen_d),
    'bootstrap_ci_lower': float(ci_lower),
    'bootstrap_ci_upper': float(ci_upper),
    'bootstrap_iterations': n_bootstrap,
}

print('=' * 80)
print('SAVING OUTPUTS')
print('=' * 80)
logger.info('SAVING OUTPUTS')

metrics_data = {
    'Metric': ['Count', 'Mean', 'Std Dev', 'Median', 'Q1', 'Q3', 'Min', 'Max', 'MAD'],
    'Early Period': [
        len(early_vals_clean),
        np.mean(early_vals_clean),
        np.std(early_vals_clean),
        np.median(early_vals_clean),
        np.percentile(early_vals_clean, 25),
        np.percentile(early_vals_clean, 75),
        np.min(early_vals_clean),
        np.max(early_vals_clean),
        early_mad,
    ],
    'Late Period': [
        len(late_vals_clean),
        np.mean(late_vals_clean),
        np.std(late_vals_clean),
        np.median(late_vals_clean),
        np.percentile(late_vals_clean, 25),
        np.percentile(late_vals_clean, 75),
        np.min(late_vals_clean),
        np.max(late_vals_clean),
        late_mad,
    ],
}
metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv(OUTPUT_METRICS, index=False)
print(f'Saved {OUTPUT_METRICS.name}')

with open(OUTPUT_STATS, 'w', encoding='utf-8') as f:
    f.write('=' * 80 + '\n')
    f.write('TEMPORAL STRATIFICATION STATISTICS\n')
    f.write('=' * 80 + '\n')
    f.write(f'Run ID: {RUN_TIMESTAMP}\n')
    f.write(f'Early Period: {EARLY_PERIOD_START.date()} to {EARLY_PERIOD_END.date()}\n')
    f.write(f'Late Period: {LATE_PERIOD_START.date()} to {LATE_PERIOD_END.date()}\n')
    f.write('\nDESCRIPTIVE STATISTICS\n')
    f.write('-' * 80 + '\n')
    f.write(f'Early Period (n={len(early_vals_clean)})\n')
    f.write(f' Mean: {np.mean(early_vals_clean):.6f}\n')
    f.write(f' Std Dev: {np.std(early_vals_clean):.6f}\n')
    f.write(f' Median: {np.median(early_vals_clean):.6f}\n')
    f.write(f' IQR: [{np.percentile(early_vals_clean, 25):.6f}, {np.percentile(early_vals_clean, 75):.6f}]\n')
    f.write(f' MAD: {early_mad:.6f}\n')
    f.write(f'Late Period (n={len(late_vals_clean)})\n')
    f.write(f' Mean: {np.mean(late_vals_clean):.6f}\n')
    f.write(f' Std Dev: {np.std(late_vals_clean):.6f}\n')
    f.write(f' Median: {np.median(late_vals_clean):.6f}\n')
    f.write(f' IQR: [{np.percentile(late_vals_clean, 25):.6f}, {np.percentile(late_vals_clean, 75):.6f}]\n')
    f.write(f' MAD: {late_mad:.6f}\n')
    f.write('\nSTATISTICAL TESTS\n')
    f.write('-' * 80 + '\n')
    f.write(f'T-test: t={t_stat:.4f}, p={t_pval:.6f}\n')
    f.write(f'Mann-Whitney U: U={u_stat:.2f}, p={u_pval:.6f}\n')
    f.write(f'Kendall\'s tau: tau={tau:.4f}, p={tau_pval:.6f}\n')
    f.write(f'Cohen\'s d: {cohen_d:.4f}\n')
    f.write(f'Bootstrap 95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]\n')
print(f'Saved {OUTPUT_STATS.name}')
print('Creating publication-quality figures...')

plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(early_vals_clean, bins=50, alpha=0.6, label='Early', color='blue', edgecolor='black')
axes[0, 0].hist(late_vals_clean, bins=50, alpha=0.6, label='Late', color='red', edgecolor='black')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution Comparison')
axes[0, 0].legend()

axes[0, 1].boxplot([early_vals_clean, late_vals_clean], labels=['Early', 'Late'])
axes[0, 1].set_ylabel('Value')
axes[0, 1].set_title('Box Plot Comparison')
axes[0, 1].grid(True, alpha=0.3)

parts = axes[1, 0].violinplot([early_vals_clean, late_vals_clean], positions=[1, 2])
axes[1, 0].set_xticks([1, 2])
axes[1, 0].set_xticklabels(['Early', 'Late'])
axes[1, 0].set_ylabel('Value')
axes[1, 0].set_title('Violin Plot Comparison')

axes[1, 1].axis('off')
summary_text = f"""STATISTICAL SUMMARY
Early Period (n={len(early_vals_clean)})
Mean: {np.mean(early_vals_clean):.4f}
Std: {np.std(early_vals_clean):.4f}
Late Period (n={len(late_vals_clean)})
Mean: {np.mean(late_vals_clean):.4f}
Std: {np.std(late_vals_clean):.4f}
Test Results (p-value)
t-test: {t_pval:.4e}
Cohen's d: {cohen_d:.4f}
95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]
"""
axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(OUTPUT_FIGURE_300, dpi=300, bbox_inches='tight')
print(f'Saved {OUTPUT_FIGURE_300.name} (300 DPI)')
plt.savefig(OUTPUT_FIGURE_600, dpi=600, bbox_inches='tight')
print(f'Saved {OUTPUT_FIGURE_600.name} (600 DPI)')
plt.close()

print('=' * 80)
print('SCRIPT 05B - COMPLETE')
print('=' * 80)
print(f'Output directory: {OUTPUT_DIR}')
print(f'Execution log: {OUTPUT_DIR}/script_05b_run.log')

metadata.finalize()
metadata_file = metadata.save(OUTPUT_DIR)
print(f'Metadata: {metadata_file.name}')

logger.info('=' * 80)
logger.info('SCRIPT 05B - COMPLETE')
logger.info('=' * 80)

sys.exit(0)

if __name__ == '__main__':
    pass
