"""Run lightweight sanity checks on the Controlled Local Patterns dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
        )
        self.classifier = nn.Linear(32, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs).flatten(1)
        return self.classifier(features)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "generated" / "local_patterns.npz",
    )
    parser.add_argument("--output", type=Path, default=root / "results")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def to_tensor(images: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float()
    return tensor / 255.0


def load_data(path: Path, batch_size: int) -> dict[str, DataLoader]:
    data = np.load(path)
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "test_iid", "test_ood_location"):
        dataset = TensorDataset(
            to_tensor(data[f"{split}_images"]),
            torch.from_numpy(data[f"{split}_labels"]).long(),
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=0,
        )
    return loaders


@torch.no_grad()
def accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        predictions = model(inputs).argmax(dim=1)
        correct += int((predictions == labels).sum())
        total += labels.numel()
    return 100.0 * correct / total


def train_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> tuple[dict[str, float], float]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    start = time.perf_counter()

    for _ in range(epochs):
        model.train()
        for inputs, labels in loaders["train"]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

    duration = time.perf_counter() - start
    metrics = {
        split: accuracy(model, loader, device)
        for split, loader in loaders.items()
    }
    metrics["generalization_gap"] = (
        metrics["test_iid"] - metrics["test_ood_location"]
    )
    return metrics, duration


def summarize(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | str]]:
    summaries: list[dict[str, float | str]] = []
    for model_name in ("random", "mlp", "small_cnn"):
        selected = [row for row in rows if row["model"] == model_name]
        for metric in (
            "train_accuracy",
            "iid_accuracy",
            "ood_accuracy",
            "generalization_gap",
            "training_seconds",
        ):
            values = np.asarray([float(row[metric]) for row in selected])
            summaries.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                }
            )
    return summaries


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    rows: list[dict[str, float | int | str]] = []

    for seed in args.seeds:
        set_seed(seed)
        loaders = load_data(args.data, args.batch_size)

        rows.append(
            {
                "model": "random",
                "seed": seed,
                "train_accuracy": 25.0,
                "iid_accuracy": 25.0,
                "ood_accuracy": 25.0,
                "generalization_gap": 0.0,
                "training_seconds": 0.0,
            }
        )

        for model_name, model_factory in (("mlp", MLP), ("small_cnn", SmallCNN)):
            set_seed(seed)
            metrics, duration = train_model(
                model_factory(),
                loaders,
                device,
                args.epochs,
                args.learning_rate,
            )
            row = {
                "model": model_name,
                "seed": seed,
                "train_accuracy": metrics["train"],
                "iid_accuracy": metrics["test_iid"],
                "ood_accuracy": metrics["test_ood_location"],
                "generalization_gap": metrics["generalization_gap"],
                "training_seconds": duration,
            }
            rows.append(row)
            print(
                f"{model_name} seed={seed}: "
                f"IID={row['iid_accuracy']:.2f}%, "
                f"OOD={row['ood_accuracy']:.2f}%, "
                f"gap={row['generalization_gap']:.2f} pp"
            )

    fieldnames = [
        "model",
        "seed",
        "train_accuracy",
        "iid_accuracy",
        "ood_accuracy",
        "generalization_gap",
        "training_seconds",
    ]
    with (args.output / "baseline_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "settings": {
            "data": args.data.name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seeds": args.seeds,
            "device": str(device),
            "torch_version": torch.__version__,
        },
        "summary": summarize(rows),
    }
    with (args.output / "baseline_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    print(f"Results written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
