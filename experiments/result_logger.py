"""Machine-readable artifact writer with stable experiment identifiers."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .config import config_hash, save_resolved_config


class ResultLogger:
    def __init__(self, config: dict[str, Any]) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.config = config
        self.config_hash = config_hash(config)
        self.experiment_id = f"{config['experiment_name']}-{timestamp}-{self.config_hash[:8]}"
        self.directory = Path(config["output_root"]) / self.experiment_id
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "plots").mkdir()
        save_resolved_config(config, self.directory / "resolved_config.yaml")
        self.write_json("environment.json", self.environment())

    def environment(self) -> dict[str, Any]:
        method_root = Path(__file__).resolve().parent.parent

        def repository_state(directory: Path) -> dict[str, Any]:
            try:
                commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=directory, capture_output=True, text=True, check=True).stdout.strip()
                branch = subprocess.run(["git", "branch", "--show-current"], cwd=directory, capture_output=True, text=True, check=True).stdout.strip()
                dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=directory, capture_output=True, text=True, check=True).stdout)
                return {"commit": commit, "branch": branch, "dirty": dirty}
            except (OSError, subprocess.CalledProcessError):
                return {"commit": "unknown", "branch": "unknown", "dirty": None}

        return {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "nerficg_git": repository_state(method_root.parent.parent.parent),
            "method_git": repository_state(method_root),
            "config_hash": self.config_hash,
        }

    def identifiers(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "scene": self.config.get("scene"),
            "seed": self.config.get("seed"),
            "grouping": self.config["grouping"]["name"],
            "anchor_selection": self.config["anchor_selection"]["name"],
            "group_size": self.config["grouping"].get("group_size"),
            "attribute_block": self.config["attributes"]["name"],
            "solver": self.config["solver"]["name"],
            "damping": self.config["solver"].get("initial_damping"),
            "aggregation": self.config["aggregation"]["name"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config_hash": self.config_hash,
        }

    def write_json(self, filename: str, data: Any) -> None:
        with (self.directory / filename).open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, default=str)

    def append_jsonl(self, filename: str, row: dict[str, Any]) -> None:
        with (self.directory / filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({**self.identifiers(), **row}, default=str) + "\n")

    def write_csv(self, filename: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            (self.directory / filename).touch()
            return
        enriched = [{**self.identifiers(), **row} for row in rows]
        fields = list(dict.fromkeys(key for row in enriched for key in row))
        with (self.directory / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(enriched)

    def initialize_expected_artifacts(self) -> None:
        for filename in ("anchors.jsonl", "groups.jsonl", "group_diagnostics.csv", "coupling_metrics.csv", "one_step_results.csv", "rollout_metrics.csv", "timing.csv", "memory.csv", "failures.jsonl"):
            (self.directory / filename).touch(exist_ok=True)
