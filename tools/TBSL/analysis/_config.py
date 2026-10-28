"""Portable paths and exact sampling alignment shared by the paper recipes."""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("TBSL_ANALYSIS_DATA_ROOT", HERE / "data")).resolve()
OUTPUT_ROOT = Path(os.environ.get("TBSL_ANALYSIS_OUTPUT_ROOT", HERE / "artifacts")).resolve()
RESULTS_DIR = OUTPUT_ROOT / "figs" / "results"
APPENDIX_DIR = OUTPUT_ROOT / "figs" / "appendix"
PERFORMANCE_DIR = OUTPUT_ROOT / "figs" / "performance"
REPORTS_DIR = OUTPUT_ROOT / "reports"

_map_file = os.environ.get("TBSL_ANALYSIS_INPUT_MAP")
INPUT_MAP = json.loads(Path(_map_file).read_text()) if _map_file else {}
if not isinstance(INPUT_MAP, dict) or not all(
    isinstance(k, str) and isinstance(v, str) for k, v in INPUT_MAP.items()
):
    raise ValueError("Input map must be a JSON object mapping historical paths to new paths")


def data_path(*parts):
    """Resolve a historical relative input name, optionally remapped to a new run."""
    key = Path(*parts).as_posix()
    value = Path(INPUT_MAP.get(key, key))
    return str(value if value.is_absolute() else DATA_ROOT / value)


def align_steps(global_steps, distribution_steps):
    """Reject absent/ambiguous samples instead of indexing past the thermo array."""
    import numpy as np

    global_steps = np.asarray(global_steps)
    distribution_steps = np.asarray(distribution_steps)
    if global_steps.size == 0 or np.any(np.diff(global_steps) <= 0):
        raise ValueError("Global steps must be nonempty and strictly increasing")
    indices = np.searchsorted(global_steps, distribution_steps)
    if np.any(indices >= len(global_steps)):
        raise ValueError("Distribution steps do not align with global output")
    if np.any(global_steps[indices] != distribution_steps):
        raise ValueError("Distribution steps do not align with global output")
    return indices
