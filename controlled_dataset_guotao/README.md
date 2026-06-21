# Controlled Local Defects Dataset

This synthetic dataset tests whether an image classifier can recognize a class that is defined only by a tiny local defect while the global object shape is identical for every class.

## Property

The target property is **fine-grained local detail recognition from small data**.

The DWConv-ViT paper argues that ViTs trained from scratch on small datasets can miss local details, and that a depth-wise convolution shortcut can add local inductive bias with little overhead. This dataset isolates that claim: the global image is uninformative, and the label can only be inferred from a small local pattern.

## Classes

Every image is a 64x64 RGB image with the same centered circular object on a noisy dark background. The class is the orientation of a small bright defect inside the circle:

- horizontal defect
- vertical defect
- descending diagonal defect
- ascending diagonal defect

The defect appears at a random valid location inside the circle. The global shape, object size, background distribution, foreground intensity range, and number of samples per class are controlled.

## Splits

| Split | Samples | Label signal |
|---|---:|---|
| `train` | 600 | tiny local defect orientation |
| `test` | 800 | tiny local defect orientation |

The train set is deliberately small: 150 samples per class. The test set uses new random defect locations and noise.

## Generate

```bash
python controlled_dataset_guotao/generate_dataset.py
```

This creates:

- `generated/local_defects.npz`
- `generated/manifest.csv`
- `generated/examples.png`

## Sanity Checks

```bash
python controlled_dataset_guotao/evaluate_baselines.py
```

Observed simple non-local baselines:

| Model | Train accuracy | Test accuracy |
|---|---:|---:|
| Constant class rule | 25.0 | 25.0 |
| Global mean rule | 33.0 | 25.0 |

These baselines show that class balance and global brightness do not solve the task.

## Learned Baselines

```bash
python controlled_dataset_guotao/evaluate_learned_models.py
```

Default settings:

- 25 epochs
- batch size 64
- Adam with learning rate `1e-3`
- seeds 0, 1, and 2
- CPU evaluation

Observed mean accuracy plus or minus sample standard deviation:

| Model | Train accuracy | Test accuracy | Gap |
|---|---:|---:|---:|
| MLP | 36.56 +/- 13.02 | 25.00 +/- 0.00 | 11.56 +/- 13.02 |
| Small CNN | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 0.00 +/- 0.00 |

The MLP does not learn the local defect rule from this small dataset. The small CNN solves the task perfectly, showing that a local convolutional inductive bias is useful for this controlled property.
