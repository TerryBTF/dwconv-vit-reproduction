"""Generate a controlled local-defect dataset for fine-detail recognition."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


CLASSES = ("horizontal_defect", "vertical_defect", "diagonal_down_defect", "diagonal_up_defect")
IMAGE_SIZE = 64
OBJECT_BOX = (14, 14, 49, 49)
DEFECT_PATTERNS = {
    "horizontal_defect": [(-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0)],
    "vertical_defect": [(0, -2), (0, -1), (0, 0), (0, 1), (0, 2)],
    "diagonal_down_defect": [(-2, -2), (-1, -1), (0, 0), (1, 1), (2, 2)],
    "diagonal_up_defect": [(-2, 2), (-1, 1), (0, 0), (1, -1), (2, -2)],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "generated")
    parser.add_argument("--seed", type=int, default=4206)
    parser.add_argument("--train-per-class", type=int, default=150)
    parser.add_argument("--test-per-class", type=int, default=200)
    return parser.parse_args()


def object_mask() -> np.ndarray:
    mask = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(OBJECT_BOX, fill=255)
    return np.asarray(mask) > 0


MASK = object_mask()
VALID_CENTERS = np.argwhere(MASK)[
    (np.argwhere(MASK)[:, 0] >= 19)
    & (np.argwhere(MASK)[:, 0] <= 44)
    & (np.argwhere(MASK)[:, 1] >= 19)
    & (np.argwhere(MASK)[:, 1] <= 44)
]


def render_sample(rng: np.random.Generator, label: int) -> np.ndarray:
    image = rng.normal(loc=25.0, scale=6.0, size=(IMAGE_SIZE, IMAGE_SIZE, 3))
    image = np.clip(image, 0, 255).astype(np.uint8)

    object_intensity = int(rng.integers(118, 139))
    object_noise = rng.normal(loc=0.0, scale=3.0, size=(IMAGE_SIZE, IMAGE_SIZE))
    object_values = np.clip(object_intensity + object_noise, 0, 255).astype(np.uint8)
    image[MASK] = np.stack([object_values[MASK]] * 3, axis=1)

    center_y, center_x = VALID_CENTERS[int(rng.integers(0, len(VALID_CENTERS)))]
    defect_intensity = int(rng.integers(218, 246))
    pattern = DEFECT_PATTERNS[CLASSES[label]]
    for dx, dy in pattern:
        x = int(center_x + dx)
        y = int(center_y + dy)
        image[y - 1 : y + 2, x - 1 : x + 2] = defect_intensity

    return image


def make_split(
    rng: np.random.Generator,
    samples_per_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    for label in range(len(CLASSES)):
        for _ in range(samples_per_class):
            images.append(render_sample(rng, label))
            labels.append(label)

    order = rng.permutation(len(images))
    return np.stack(images)[order], np.asarray(labels, dtype=np.int64)[order]


def save_examples(output: Path, images: np.ndarray, labels: np.ndarray) -> None:
    tile_size = 96
    canvas = Image.new("RGB", (4 * 132, 124), "white")
    draw = ImageDraw.Draw(canvas)
    for column, class_name in enumerate(CLASSES):
        index = int(np.flatnonzero(labels == column)[0])
        tile = Image.fromarray(images[index]).resize(
            (tile_size, tile_size), Image.Resampling.NEAREST
        )
        x = column * 132 + 18
        canvas.paste(tile, (x, 4))
        draw.text((column * 132 + 4, 104), class_name, fill="black")
    canvas.save(output / "examples.png")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    train_images, train_labels = make_split(rng, args.train_per_class)
    test_images, test_labels = make_split(rng, args.test_per_class)

    np.savez_compressed(
        args.output / "local_defects.npz",
        class_names=np.asarray(CLASSES),
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
    )

    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "samples", "samples_per_class", "label_signal"])
        writer.writerow(["train", len(train_images), args.train_per_class, "tiny local defect orientation"])
        writer.writerow(["test", len(test_images), args.test_per_class, "tiny local defect orientation"])

    save_examples(args.output, train_images, train_labels)
    print(f"Dataset written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
