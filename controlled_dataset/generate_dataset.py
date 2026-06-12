"""Generate a controlled dataset for local-pattern and position generalization."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PATTERNS = {
    "horizontal": [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)],
    "vertical": [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)],
    "diagonal_down": [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
    "diagonal_up": [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "generated")
    parser.add_argument("--seed", type=int, default=4205)
    parser.add_argument("--train-per-class", type=int, default=500)
    parser.add_argument("--test-per-class", type=int, default=100)
    return parser.parse_args()


def position_for_split(rng: np.random.Generator, split: str) -> tuple[int, int]:
    x = int(rng.integers(1, 23))
    if split in {"train", "test_iid"}:
        y = int(rng.integers(1, 11))
    else:
        y = int(rng.integers(20, 23))
    return x, y


def render_sample(
    rng: np.random.Generator,
    pattern: list[tuple[int, int]],
    split: str,
) -> np.ndarray:
    image = rng.normal(loc=32.0, scale=8.0, size=(32, 32, 3))
    image = np.clip(image, 0, 255).astype(np.uint8)

    x0, y0 = position_for_split(rng, split)
    intensity = int(rng.integers(190, 256))
    color = np.array([intensity, intensity, intensity], dtype=np.uint8)

    for cell_x, cell_y in pattern:
        x = x0 + 2 * cell_x
        y = y0 + 2 * cell_y
        image[y : y + 2, x : x + 2] = color

    return image


def make_split(
    rng: np.random.Generator,
    split: str,
    samples_per_class: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    names: list[str] = []

    for label, (name, pattern) in enumerate(PATTERNS.items()):
        for _ in range(samples_per_class):
            images.append(render_sample(rng, pattern, split))
            labels.append(label)
            names.append(name)

    order = rng.permutation(len(images))
    return (
        np.stack(images)[order],
        np.asarray(labels, dtype=np.int64)[order],
        [names[index] for index in order],
    )


def save_examples(
    output: Path,
    split_data: dict[str, tuple[np.ndarray, np.ndarray, list[str]]],
) -> None:
    tile_size = 64
    row_height = 82
    canvas = Image.new("RGB", (4 * 96, 3 * row_height), "white")
    draw = ImageDraw.Draw(canvas)

    for row, split in enumerate(("train", "test_iid", "test_ood_location")):
        images, labels, _ = split_data[split]
        for column, class_name in enumerate(PATTERNS):
            label = list(PATTERNS).index(class_name)
            image = images[np.flatnonzero(labels == label)[0]]
            tile = Image.fromarray(image).resize(
                (tile_size, tile_size), Image.Resampling.NEAREST
            )
            x = column * 96 + 16
            y = row * row_height
            canvas.paste(tile, (x, y))
            draw.text((column * 96 + 2, y + 65), class_name, fill="black")

    canvas.save(output / "examples.png")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    split_data = {
        "train": make_split(rng, "train", args.train_per_class),
        "test_iid": make_split(rng, "test_iid", args.test_per_class),
        "test_ood_location": make_split(
            rng, "test_ood_location", args.test_per_class
        ),
    }

    np.savez_compressed(
        args.output / "local_patterns.npz",
        class_names=np.asarray(list(PATTERNS)),
        train_images=split_data["train"][0],
        train_labels=split_data["train"][1],
        test_iid_images=split_data["test_iid"][0],
        test_iid_labels=split_data["test_iid"][1],
        test_ood_location_images=split_data["test_ood_location"][0],
        test_ood_location_labels=split_data["test_ood_location"][1],
    )

    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "samples", "samples_per_class"])
        for split, (images, _, _) in split_data.items():
            writer.writerow([split, len(images), len(images) // len(PATTERNS)])

    save_examples(args.output, split_data)
    print(f"Dataset written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
