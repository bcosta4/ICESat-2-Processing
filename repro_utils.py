
"""
repro_utils.py

Shared reproducibility utilities for the ICESat-2 permafrost elevation-change
workflow (Scripts 00-08). Import this module from every script to guarantee a
consistent metadata schema, consistent logging setup, and consistent
provenance capture (git commit hash, command line, input file checksums).

This module has NO side effects on import - all logging/metadata objects are
created explicitly by the calling script.
"""

import hashlib
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


# =============================================================================
# GIT COMMIT HASH
# =============================================================================
def get_git_commit_hash(repo_path: Optional[Path] = None) -> str:
    """Return the current git commit hash (short) for the script's repo.
    Returns 'unknown' if git is unavailable or the directory is not a repo."""
    try:
        cwd = str(repo_path) if repo_path else str(Path(__file__).resolve().parent)
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            commit = result.stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd, capture_output=True, text=True, timeout=5,
            )
            if dirty.stdout.strip():
                commit += "-dirty"
            return commit
    except Exception:
        pass
    return "unknown"


# =============================================================================
# INPUT FILE CHECKSUMS
# =============================================================================
def compute_file_sha256(filepath: Path, chunk_size: int = 8192) -> str:
    """Compute SHA-256 checksum of a file. Returns 'unavailable' on failure."""
    try:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "unavailable"


def describe_input_files(filepaths: List[Path]) -> List[Dict[str, Any]]:
    """Build a list of {path, sha256, size_bytes, modified} dicts for a set
    of input files, for inclusion in run metadata."""
    records = []
    for fp in filepaths:
        fp = Path(fp)
        if not fp.exists():
            records.append({"path": str(fp), "sha256": "FILE_NOT_FOUND",
                             "size_bytes": None, "modified": None})
            continue
        stat = fp.stat()
        records.append({
            "path": str(fp),
            "sha256": compute_file_sha256(fp),
            "size_bytes": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })
    return records


# =============================================================================
# ENVIRONMENT INFO
# =============================================================================
def get_environment_info() -> Dict[str, Any]:
    """Lightweight environment fingerprint. Full pinned dependency versions
    are tracked separately in requirements.txt (see repo root)."""
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
    }


# =============================================================================
# UNIFIED LOGGING SETUP
# =============================================================================
def setup_logging(output_dir: Path, script_name: str) -> logging.Logger:
    """Configure logging to both file (<output_dir>/<script_name>_run.log,
    DEBUG level, UTF-8) and console (INFO level, clean format).
    Consistent across all scripts 00-08."""
    log_file = output_dir / f"{script_name}_run.log"
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    fh = logging.FileHandler(log_file, encoding="utf-8", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# =============================================================================
# UNIFIED RUN METADATA SCHEMA
# =============================================================================
class RunMetadata:
    """Unified reproducibility metadata object, shared schema across all
    scripts (00-08). Console output stays clean - all provenance detail
    (git hash, command line, checksums, environment) is written ONLY to the
    per-script log file and the metadata JSON, never to the console banner.

    JSON schema (top-level keys):
        run_id, script_name, script_version, git_commit_hash, command_line,
        timestamps {start, end}, environment, input_files, config_parameters,
        study_area_reference, processing_results
    """

    def __init__(self, run_id: str, script_name: str, script_version: str,
                 config_parameters: Optional[Dict[str, Any]] = None,
                 study_area_reference: Optional[Dict[str, Any]] = None):
        self.run_id = run_id
        self.script_name = script_name
        self.script_version = script_version
        self.git_commit_hash = get_git_commit_hash()
        self.command_line = " ".join(sys.argv)
        self.start_time = datetime.now().isoformat()
        self.end_time = None
        self.environment = get_environment_info()
        self.input_files: List[Dict[str, Any]] = []
        self.config_parameters = config_parameters or {}
        self.study_area_reference = study_area_reference or {}
        self.processing_results: Dict[str, Any] = {}

    def add_input_files(self, filepaths: List[Path]) -> None:
        self.input_files = describe_input_files(filepaths)

    def set_processing_results(self, results: Dict[str, Any]) -> None:
        self.processing_results = results

    def finalize(self) -> None:
        self.end_time = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "script_name": self.script_name,
            "script_version": self.script_version,
            "git_commit_hash": self.git_commit_hash,
            "command_line": self.command_line,
            "timestamps": {"start": self.start_time, "end": self.end_time},
            "environment": self.environment,
            "input_files": self.input_files,
            "config_parameters": self.config_parameters,
            "study_area_reference": self.study_area_reference,
            "processing_results": self.processing_results,
        }

    def save(self, output_dir: Path, filename: Optional[str] = None) -> Path:
        filename = filename or f"{self.script_name}_metadata.json"
        metadata_file = Path(output_dir) / filename
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str, ensure_ascii=False)
        return metadata_file
