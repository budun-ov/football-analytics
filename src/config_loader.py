"""
Configuration loader.

Reads config.yaml and exposes a plain namespace so the rest of
the codebase can do:  cfg.detection.confidence
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path = "config.yaml") -> SimpleNamespace:
    """
    Parse *config_path* and return a deeply-nested SimpleNamespace.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        SimpleNamespace with the same hierarchy as the YAML.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file cannot be parsed.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Make sure 'config.yaml' is in the project root directory."
        )

    with config_path.open("r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}

    return _dict_to_namespace(raw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dict_to_namespace(obj: Any) -> Any:
    """Recursively convert dicts to SimpleNamespace."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_dict_to_namespace(item) for item in obj]
    return obj
