# Environment Fix Log

Date: 2026-05-23

This note records repository changes made to run the DWConv ViT reproduction in the local Python/PyTorch environment.

## Added Contributor Guide

- File: `AGENTS.md`
- Change: Added repository contributor guidelines.
- Reason: Document project layout, setup commands, coding style, testing expectations, and PR/commit practices for future contributors.

## Added CIFAR Data Loader

- Files:
  - `official_code/data/__init__.py`
  - `official_code/data/build.py`
- Change: Added a `build_loader(config)` implementation for CIFAR-10 and CIFAR-100 using `torchvision.datasets`.
- Reason: `official_code/main.py` imports `from data import build_loader`, but the repository did not contain the `official_code/data/` module. Training could not start without this loader.
- Details:
  - Supports `DATA.DATASET: cifar10` and `DATA.DATASET: cifar100`.
  - Uses `download=False`, matching the README expectation that CIFAR data is prepared in advance.
  - Adds train/validation transforms for 224x224 ViT input.
  - Uses distributed sampling during launched training and a normal random sampler for local smoke tests.
  - Creates a `timm.data.Mixup` object when mixup/cutmix is enabled in config.

## Fixed Git Ignore Rule

- File: `.gitignore`
- Change: Changed `data/` to `/data/`.
- Reason: The previous rule ignored every directory named `data`, including the source package `official_code/data/`. The new rule only ignores the repository-root dataset directory.

## Fixed PyTorch Compatibility

- File: `official_code/utils.py`
- Change: Replaced:

```python
from torch._six import inf
```

with:

```python
from math import inf
```

- Reason: `torch._six` was removed from newer PyTorch versions. The code only needed the infinity constant, which is available from Python's standard library.

## Fixed Distributed Launch Compatibility

- File: `official_code/main.py`
- Change: Updated local rank argument parsing to accept `--local_rank`, `--local-rank`, and the `LOCAL_RANK` environment variable.
- Reason: Newer PyTorch launch utilities pass local rank differently from the older code. Without this change, training failed with `the following arguments are required: --local_rank`.

## Fixed timm Scheduler Compatibility

- File: `official_code/lr_scheduler.py`
- Change: Replaced the old `CosineLRScheduler` argument:

```python
t_mul=1.
```

with:

```python
cycle_mul=1.
```

- Reason: In `timm 1.0.27`, the cosine scheduler constructor uses `cycle_mul` instead of the older `t_mul` name.

## Added ViT DWConv Toggle

- Files:
  - `official_code/config.py`
  - `official_code/models/build.py`
  - `official_code/models/vit.py`
  - `reproduction_configs/vit_tiny_16_224_cifar10_baseline_100ep.yaml`
  - `reproduction_configs/vit_tiny_16_224_cifar100_baseline_100ep.yaml`
- Change: Added `MODEL.ViT.USE_DWCONV` and `MODEL.ViT_S.USE_DWCONV` config flags. The Transformer block now only creates and applies depth-wise convolution shortcuts when the flag is true.
- Reason: The vendored ViT implementation always included DWConv shortcuts, so baseline ViT runs were not possible from YAML. The toggle allows fair baseline vs DWConv comparisons using the same training pipeline.
- Verification: A local forward-pass check produced valid outputs for both configs. The DWConv model has 5,547,082 trainable parameters and the baseline model has 5,519,434, confirming the shortcut modules are disabled in baseline mode.

## Environment Notes

- The Python environment is located at `/media/terry/Data/venvs/dwconv-vit`.
- CIFAR data is located at `/media/terry/Data/datasets/cifar`.
- Before training, run:

```bash
source /media/terry/Data/venvs/dwconv-vit/bin/activate
unset PYTHONPATH
cd /home/terry/Documents/fundamental_project/dwconv-vit-reproduction/official_code
```

- `unset PYTHONPATH` avoids ROS paths leaking into the virtual environment.
