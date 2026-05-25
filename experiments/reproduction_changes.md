# Reproduction Changes

Date: 2026-05-23

This note records repository changes made to run the DWConv ViT reproduction in the local Python/PyTorch environment.

## Added CIFAR Data Loader

- Files:
  - `official_code/data/__init__.py`
  - `official_code/data/build.py`
- Change: Added a `build_loader(config)` implementation for CIFAR-10 and CIFAR-100 using `torchvision.datasets`.

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
