"""Hierarchical YAML composition and ``key=value`` command-line overrides."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


CONFIG_ROOT = Path(__file__).resolve().parent / "configs"
GROUP_KEYS = ("universe", "anchor_selection", "grouping", "attributes", "residual", "jacobian", "solver", "aggregation", "acceptance", "evaluation", "benchmark")


def _merge(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return data


def _parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def _set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    target = config
    for key in keys[:-1]:
        existing = target.setdefault(key, {})
        if not isinstance(existing, dict):
            raise ValueError(f"cannot assign below scalar config key: {dotted_key}")
        target = existing
    target[keys[-1]] = value


def compose_config(arguments: list[str], *, config_root: Path = CONFIG_ROOT) -> dict[str, Any]:
    """Compose defaults, named fragments, and dotted overrides.

    Bare ``grouping=knn_3d`` selects ``configs/grouping/knn_3d.yaml`` when
    present.  Otherwise it is treated as a scalar assignment.
    """

    config = _load_yaml(config_root / "default.yaml")
    deferred: list[tuple[str, Any]] = []
    for argument in arguments:
        if "=" not in argument:
            raise ValueError(f"expected key=value argument, got: {argument}")
        key, raw = argument.split("=", 1)
        value = _parse_value(raw)
        if key == "experiment":
            fragment = config_root / "experiment" / f"{value}.yaml"
            if not fragment.exists():
                raise FileNotFoundError(fragment)
            _merge(config, _load_yaml(fragment))
        elif key in GROUP_KEYS and "." not in key:
            fragment = config_root / key / f"{value}.yaml"
            if fragment.exists():
                _merge(config, _load_yaml(fragment))
            else:
                deferred.append((f"{key}.name", value))
        else:
            deferred.append((key, value))
    for key, value in deferred:
        _set_dotted(config, key, value)
    return config


def config_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def save_resolved_config(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
