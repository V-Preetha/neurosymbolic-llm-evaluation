"""Reproducible paper analysis with the repository's legacy API preserved."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


_legacy_path = Path(__file__).resolve().parents[1] / "analysis.py"
_legacy_spec = spec_from_file_location("final_experiment._legacy_analysis", _legacy_path)
if _legacy_spec is None or _legacy_spec.loader is None:
    raise ImportError(f"cannot load legacy analysis API from {_legacy_path}")
_legacy = module_from_spec(_legacy_spec)
sys.modules[_legacy_spec.name] = _legacy
_legacy_spec.loader.exec_module(_legacy)

classification_metrics = _legacy.classification_metrics
bridge_delta = _legacy.bridge_delta
mcnemar = _legacy.mcnemar
translation = _legacy.translation
metrics = _legacy.metrics
paired = _legacy.paired
bootstrap = _legacy.bootstrap
depth = _legacy.depth
analyze = _legacy.analyze
