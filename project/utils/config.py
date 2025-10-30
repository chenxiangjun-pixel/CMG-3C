"""Configuration loading utilities."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path("/project")


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if "inherit" in data:
        base_path = ROOT / data["inherit"]
        base = load_yaml_config(base_path)
        override = {k: v for k, v in data.items() if k != "inherit"}
        return deep_update(base, override)
    return data


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result
