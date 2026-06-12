# Controlled Local Patterns Dataset

This synthetic dataset tests whether an image classifier can learn a local spatial pattern from limited data and recognize it at previously unseen image locations.

## Property

The target property is **location-generalized recognition of local spatial arrangements**.

Each image contains one of four five-cell patterns:

- horizontal
- vertical
- descending diagonal
- ascending diagonal

Every class has the same foreground area, grayscale intensity distribution, background distribution, image size, and number of samples. The label is determined only by the relative arrangement of the five cells.

## Splits

| Split | Samples | Position distribution |
|---|---:|---|
| `train` | 2,000 | Random horizontal position, upper image region |
| `test_iid` | 400 | Same position distribution as training |
| `test_ood_location` | 400 | Random horizontal position, unseen lower image region |

The IID split measures ordinary local-pattern recognition. The OOD-location split tests whether the learned pattern transfers to absolute positions absent from training.

## Generate

```bash
python controlled_dataset/generate_dataset.py
```

This creates:

- `generated/local_patterns.npz`
- `generated/manifest.csv`
- `generated/examples.png`

The default seed is `4205`. Dataset sizes can be changed through command-line arguments.

## Load

```python
import numpy as np

data = np.load("controlled_dataset/generated/local_patterns.npz")
train_images = data["train_images"]
train_labels = data["train_labels"]
```

Images use `uint8` RGB format with shape `[N, 32, 32, 3]`.

