# ICESat-2 Consolidation - Master Workflow Executor

## File: 07_run_workflow.py

This is the flexible master orchestrator for the consolidated workflow. It can run the complete pipeline or resume from any point.

```python
"""
Master Workflow Executor for Consolidated ICESat-2 Processing Pipeline

This script orchestrates execution of the consolidated workflow with flexible options:
- Run full pipeline (all 8 scripts)
- Run partial pipeline (starting from any script)
- Resume from last failed script
- Generate detailed execution report

Usage:
    python 07_run_workflow.py              # Run all scripts
    python 07_run_workflow.py --start 04   # Start from script 04
    python 07_run_workflow.py --resume     # Resume from last failure
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import argparse
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_OUTPUT_DIR = Path(r'E:\Processing\ICESat-2\output_data')
SCRIPTS_DIR = Path(r'E:\Processing\ICESat-2\scripts')

# Define all scripts in execution order (consolidated workflow)
SCRIPTS = [
    ('00', '00_setup_validation.py', 'Environment Setup & Validation'),
    ('01', '01_filter_photons.py', 'ICESat-2 Photon Filtering'),
    ('01b', '01b_filter_terrain.py', 'Land/Water Classification'),
    ('02', '02_analysis_core.py', 'Crossover Analysis & InSAR Integration'),
    ('03', '03_analysis_spatial.py', 'Spatial Analysis & Visualization'),
    ('04', '04_analysis_temporal.py', 'Temporal & Multi-Scale Analysis'),
    ('05', '05_analysis_advanced.py', 'Spatial Density Clustering'),
    ('06', '06_output_generation.py', 'Publication Figures & GIS Export'),
]

# Execution state file
STATE_FILE = BASE_OUTPUT_DIR / 'workflow_execution_state.json'

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_state():
    """Load execution state from last run"""
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {'last_completed': None, 'last_failed': None, 'execution_history': []}

def save_state(state):
    """Save execution state"""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def find_script_index(script_num):
    """Find script index by number"""
    for i, (num, name, desc) in enumerate(SCRIPTS):
        if num == script_num:
            return i
    return None

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(text)
    print("="*80)

def print_step(num, description):
    """Print step information"""
    print(f"\n[{num:02d}] {description}")
    print("-"*40)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_workflow(start_from=None, resume=False):
    """Execute the consolidation workflow"""
    
    print_header("ICESat-2 CONSOLIDATION WORKFLOW - OPTION A")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Scripts directory: {SCRIPTS_DIR}")
    print(f"Output directory: {BASE_OUTPUT_DIR}")
    print(f"Total scripts: {len(SCRIPTS)}")
    
    # Determine start index
    state = load_state()
    
    if resume and state['last_failed']:
        print(f"\nRESUME MODE: Resuming from {state['last_failed']}")
        start_idx = find_script_index(state['last_failed'])
    elif start_from:
        print(f"\nSTART MODE: Starting from {start_from}")
        start_idx = find_script_index(start_from)
        if start_idx is None:
            print(f"ERROR: Script {start_from} not found")
            sys.exit(1)
    else:
        print("\nFULL MODE: Running complete pipeline")
        start_idx = 0
    
    # Execute scripts
    results = {}
    total_start = time.time()
    
    for i, (script_num, script_name, description) in enumerate(SCRIPTS):
        if i < start_idx:
            continue  # Skip scripts before start
        
        print_step(script_num, description)
        print(f"Script: {script_name}")
        
        script_path = SCRIPTS_DIR / script_name
        
        # Verify script exists
        if not script_path.exists():
            print(f"[ERROR] Script not found: {script_path}")
            results[script_num] = ('MISSING', 0)
            state['last_failed'] = script_num
            save_state(state)
            continue
        
        # Run script
        script_start = time.time()
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=SCRIPTS_DIR,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout per script
            )
            
            script_time = time.time() - script_start
            
            if result.returncode == 0:
                print(f"[OK] Completed in {script_time:.1f}s")
                results[script_num] = ('SUCCESS', script_time)
                state['last_completed'] = script_num
            else:
                print(f"[FAILED] Exit code {result.returncode}")
                if result.stderr:
                    print(f"Error output:\n{result.stderr[:500]}")
                results[script_num] = ('FAILED', script_time)
                state['last_failed'] = script_num
                save_state(state)
                print("\nTo resume: python 07_run_workflow.py --resume")
                return results
        
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] Script exceeded 1 hour limit")
            results[script_num] = ('TIMEOUT', 3600)
            state['last_failed'] = script_num
            save_state(state)
            return results
        
        except Exception as e:
            print(f"[ERROR] {str(e)}")
            results[script_num] = ('ERROR', 0)
            state['last_failed'] = script_num
            save_state(state)
            return results
    
    # Update state on successful completion
    if not state['last_failed']:
        state['last_completed'] = SCRIPTS[-1][0]
    state['execution_history'].append({
        'timestamp': datetime.now().isoformat(),
        'results': results
    })
    save_state(state)
    
    # Print summary
    total_time = time.time() - total_start
    print_header("EXECUTION SUMMARY")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Total scripts run: {len(results)}")
    print("-"*80)
    
    for script_num, script_name, description in SCRIPTS:
        if script_num not in results:
            status, duration = "SKIPPED", 0
        else:
            status, duration = results[script_num]
        status_symbol = "[OK]" if status == "SUCCESS" else f"[{status[0:2]}]"
        print(f"{script_num:4} : {status:10} {status_symbol:6} ({duration:.1f}s)")
    
    print("="*80)
    
    # Final status
    failed = sum(1 for status, _ in results.values() if status != 'SUCCESS')
    if failed > 0:
        print(f"\n⚠ {failed} script(s) failed or had errors")
        print(f"Run with --resume to continue from failure point")
        return results
    else:
        print(f"\n✓ All scripts completed successfully!")
        
        # Log final timestamp
        try:
            timestamp_file = BASE_OUTPUT_DIR / 'latest_run_timestamp.txt'
            if timestamp_file.exists():
                with open(timestamp_file, 'r') as f:
                    run_id = f.read().strip()
                log_file = BASE_OUTPUT_DIR / f'run_{run_id}' / f'processing_log_{run_id}.txt'
                
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write("="*80 + "\n")
                    f.write("ICESat-2 Processing Pipeline Log\n")
                    f.write("="*80 + "\n")
                    f.write(f"Run ID: {run_id}\n")
                    f.write(f"Start time: {datetime.now()}\n")
                    f.write(f"Total time: {total_time/60:.1f} minutes\n\n")
                    for script_num, script_name, description in SCRIPTS:
                        if script_num in results:
                            status, duration = results[script_num]
                            f.write(f"{script_num}: {script_name}\n")
                            f.write(f" Status: {status}\n")
                            f.write(f" Duration: {duration:.1f}s\n\n")
                
                print(f"Log saved to: {log_file}")
        except Exception as e:
            print(f"Warning: Could not save log: {e}")
    
    return results

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Parse arguments and execute workflow"""
    parser = argparse.ArgumentParser(
        description='ICESat-2 Consolidation Workflow Executor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 07_run_workflow.py              # Run all scripts
  python 07_run_workflow.py --start 04   # Start from script 04
  python 07_run_workflow.py --resume     # Resume from last failure
  python 07_run_workflow.py --status     # Show execution status
        """
    )
    
    parser.add_argument('--start', type=str, default=None,
                        help='Start from specific script number (e.g., 04)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last failed script')
    parser.add_argument('--status', action='store_true',
                        help='Show execution status and exit')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be executed without running')
    
    args = parser.parse_args()
    
    # Show status
    if args.status:
        state = load_state()
        print_header("WORKFLOW EXECUTION STATUS")
        print(f"Last completed: {state.get('last_completed', 'None')}")
        print(f"Last failed: {state.get('last_failed', 'None')}")
        print(f"Execution history entries: {len(state.get('execution_history', []))}")
        return
    
    # Dry run
    if args.dry_run:
        print_header("DRY RUN - Scripts to Execute")
        start_idx = find_script_index(args.start) if args.start else 0
        for i, (num, name, desc) in enumerate(SCRIPTS):
            if i >= start_idx:
                print(f"[{num}] {name:30} - {desc}")
        return
    
    # Normal execution
    run_workflow(start_from=args.start, resume=args.resume)

if __name__ == '__main__':
    main()
```

---

## Usage Examples

### Run Complete Pipeline
```bash
python 07_run_workflow.py
```

### Start from Specific Script
```bash
python 07_run_workflow.py --start 04
```

### Resume from Failure
```bash
python 07_run_workflow.py --resume
```

### Check Status
```bash
python 07_run_workflow.py --status
```

### Dry Run (Preview)
```bash
python 07_run_workflow.py --dry-run
```

---

## State Management

The workflow maintains an execution state file (`workflow_execution_state.json`) that tracks:
- Last completed script
- Last failed script
- Full execution history
- Timestamps for each run

This enables:
- Resuming from failures
- Tracking workflow health
- Auditing processing history
- Debugging issues

---

## Error Handling

The executor provides robust error handling:
- **Script not found** → Report and skip
- **Execution failure** → Stop and save state
- **Timeout** → Stop after 1 hour per script
- **Exception** → Catch and report

All errors include helpful messages suggesting recovery steps.
