"""Evaluate simple non-local sanity checks on the local-defect dataset."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "generated" / "local_defects.npz"
OUTPUT = ROOT / "results"


def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return 100.0 * float((predictions == labels).mean())


def global_mean_rule(images: np.ndarray) -> np.ndarray:
    """A deliberately weak baseline using only global brightness."""
    means = images.mean(axis=(1, 2, 3))
    bins = np.quantile(means, [0.25, 0.5, 0.75])
    return np.digitize(means, bins).astype(np.int64)


def random_rule(images: np.ndarray) -> np.ndarray:
    """Chance baseline for four balanced classes."""
    return np.zeros(len(images), dtype=np.int64)


def main() -> None:
    data = np.load(DATA_PATH)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []

    for model_name, predictor in (
        ("constant_class_rule", random_rule),
        ("global_mean_rule", global_mean_rule),
    ):
        for split in ("train", "test"):
            images = data[f"{split}_images"]
            labels = data[f"{split}_labels"]
            rows.append(
                {
                    "model": model_name,
                    "split": split,
                    "accuracy": accuracy(predictor(images), labels),
                }
            )

    with (OUTPUT / "baseline_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "split", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "property": "fine-grained local defect recognition with identical global shape",
        "results": rows,
    }
    with (OUTPUT / "baseline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Results written to {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
