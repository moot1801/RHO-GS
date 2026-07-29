"""Explicit, selective Cartesian sweep launcher with a dry-run mode."""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path

import yaml


RUNNERS = {"analyze_checkpoint", "one_step_benchmark", "rollout_benchmark"}


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    values = dict(argument.split("=", 1) for argument in arguments)
    runner = values.get("runner", "analyze_checkpoint")
    if runner not in RUNNERS:
        raise ValueError(f"runner must be one of {sorted(RUNNERS)}")
    matrix_path = Path(values["matrix"])
    with matrix_path.open("r", encoding="utf-8") as handle:
        specification = yaml.safe_load(handle)
    base = [str(value) for value in specification.get("base", [])]
    matrix = specification.get("matrix", {})
    keys = list(matrix)
    combinations = [dict(zip(keys, choice, strict=True)) for choice in itertools.product(*(matrix[key] for key in keys))]
    commands = [
        [sys.executable, "-m", f"experiments.runners.{runner}", *base, *(f"{key}={value}" for key, value in combination.items())]
        for combination in combinations
    ]
    manifest = Path(values.get("manifest", "sweep_manifest.json"))
    manifest.write_text(json.dumps({"runner": runner, "commands": commands}, indent=2), encoding="utf-8")
    if values.get("dry_run", "true").lower() == "true":
        print(manifest)
        return 0
    for command in commands:
        subprocess.run(command, check=True)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
