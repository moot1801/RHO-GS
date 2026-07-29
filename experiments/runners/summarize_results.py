"""Summarize experiment directories without changing source artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def _plots(directory: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    plot_directory = directory / "plots"
    plot_directory.mkdir(exist_ok=True)
    created: list[str] = []
    coupling = _read_csv(directory / "coupling_metrics.csv")
    values = [value for row in coupling if (value := _float(row, "c_ij_fro")) is not None]
    if values:
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.hist(values, bins=min(50, max(10, int(len(values) ** 0.5))))
        axis.set(xlabel="normalized coupling Frobenius norm", ylabel="pair count", title="Sampled pair coupling")
        figure.tight_layout()
        path = plot_directory / "coupling_histogram.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        created.append(str(path))
    one_step = _read_csv(directory / "one_step_results.csv")
    if one_step:
        predicted = _float(one_step[0], "predicted_reduction")
        actual = _float(one_step[0], "actual_reduction")
        if predicted is not None and actual is not None:
            figure, axis = plt.subplots(figsize=(5, 4))
            axis.bar(["predicted", "actual"], [predicted, actual])
            axis.set(ylabel="loss reduction", title="One-step quadratic agreement")
            figure.tight_layout()
            path = plot_directory / "one_step_reduction.png"
            figure.savefig(path, dpi=160)
            plt.close(figure)
            created.append(str(path))
    rollout = _read_csv(directory / "rollout_metrics.csv")
    iterations = [_float(row, "rollout_iteration") for row in rollout]
    losses = [_float(row, "sampled_loss_after") for row in rollout]
    if rollout and all(value is not None for value in iterations + losses):
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(iterations, losses)
        axis.set(xlabel="rollout iteration", ylabel="sampled RGB L2", title="Fixed-topology rollout")
        figure.tight_layout()
        path = plot_directory / "rollout_loss.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        created.append(str(path))
    return created


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments:
        raise ValueError("provide one or more experiment output directories")
    summaries = []
    for raw in arguments:
        directory = Path(raw)
        with (directory / "summary.json").open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        summaries.append({
            "directory": str(directory),
            "summary": summary,
            "coupling_rows": len(_read_csv(directory / "coupling_metrics.csv")),
            "one_step_rows": len(_read_csv(directory / "one_step_results.csv")),
            "rollout_rows": len(_read_csv(directory / "rollout_metrics.csv")),
            "plots": _plots(directory),
        })
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
