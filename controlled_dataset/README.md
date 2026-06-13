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

## Sanity-Check Baselines

The evaluation script trains an MLP and a small CNN, and includes a 25%
random-classification baseline:

```bash
python controlled_dataset/evaluate_baselines.py
```

Default settings:

- 20 epochs
- batch size 64
- Adam with learning rate `1e-3`
- seeds 0, 1, and 2
- automatic CUDA/CPU selection

To run explicitly on CPU:

```bash
python controlled_dataset/evaluate_baselines.py --device cpu
```

Results are written to:

- `controlled_dataset/results/baseline_runs.csv`
- `controlled_dataset/results/baseline_summary.json`

The reported location-generalization gap is:

```text
IID accuracy - OOD-location accuracy
```

A useful model should have high IID accuracy before its OOD gap is interpreted.

### Observed Results

Mean accuracy plus or minus sample standard deviation over seeds 0, 1, and 2:

| Model | IID accuracy | OOD-location accuracy | Gap |
|---|---:|---:|---:|
| Random | 25.00 | 25.00 | 0.00 |
| MLP | 84.17 +/- 10.32 | 26.50 +/- 1.00 | 57.67 +/- 9.46 |
| Small CNN | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 0.00 +/- 0.00 |

The MLP learns the task at familiar positions but falls to approximately
chance accuracy in the held-out image region. The CNN transfers the local
patterns across positions for every tested seed. This confirms that the
dataset is learnable and that the OOD split detects position-dependent
solutions.

