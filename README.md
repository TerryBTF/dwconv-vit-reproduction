# DWConv ViT Reproduction

Reproducing ViT-Tiny with depth-wise convolution shortcuts on CIFAR-10 and CIFAR-100.

## Paper

- Paper: Depth-Wise Convolutions in Vision Transformers for Efficient Training on Small Datasets
- arXiv: https://arxiv.org/abs/2407.19394
- Official code: https://github.com/ZTX-100/Efficient_ViT_with_DW
- Vendored official commit: `d7ae645bedec54b4850ff659889c0588164aaac2`

The original implementation is stored in [`official_code/`](official_code/). We keep our course-specific configs, logs, and documentation outside that directory unless a reproduction change requires editing the implementation.

## Project Scope

This project is an existing-code reproduction for the Fundamental Research in Machine and Deep Learning final project.

We focus on the main small-dataset classification claim from Table 1: adding depth-wise convolution shortcuts should improve ViT-Tiny accuracy on CIFAR-10 and CIFAR-100.

We do not aim to reproduce the full paper. We exclude ImageNet, COCO, Swin, CaiT, and Tiny-ImageNet from the initial scope because the course project is expected to fit a reduced single-GPU compute budget.

## Reproduction Criteria

We plan to cover four criteria:

| Criterion | Question | Planned evidence |
|---|---|---|
| Reproduced | Does DWConv improve ViT-Tiny on CIFAR-10/CIFAR-100? | Main Table 1 subset: ViT-Tiny baseline vs ViT-Tiny + DWConv |
| Ablation study | Does DWConv reduce dependence on positional embeddings? | With/without positional embedding, with/without DWConv |
| Hyperparams check | Is the DWConv improvement sensitive to training budget or kernel size? | 50 vs 100 epochs, or DWConv kernel size 3 vs 5 vs 7 |
| New algorithm variant | Does a slightly different DWConv variant preserve or improve the reported effect? | Evaluate a small variant such as changing DWConv placement, kernel composition, or shortcut frequency |

## Ruifang(Terry) Zhang Scope

Ruifang(Terry) Zhang is responsible for the `Reproduced` criterion.

Ruifang(Terry) Zhang's target is the main Table 1 subset:

| Dataset | Paper baseline | Paper DWConv | Our baseline | Our DWConv |
|---|---:|---:|---:|---:|
| CIFAR-10 | 94.01 | 96.41 | TBD | TBD |
| CIFAR-100 | 73.68 | 78.05 | TBD | TBD |

The official paper used 300 epochs and 4 NVIDIA P100 GPUs. Our initial configs use 100 epochs to support a reduced single-GPU reproduction. We will report the compute difference clearly and evaluate whether the qualitative trend holds rather than claiming exact numerical reproduction.

## Repository Layout

```text
official_code/                    Official implementation from the authors
reproduction_configs/             Course-specific reduced-compute configs
experiments/                      Experiment plans and result logs
requirements.txt                  Python package requirements
```

## Setup

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

Install a PyTorch build that matches your CUDA version if the generic install does not select the right wheel.

The official loader expects CIFAR data to already exist under the path passed with `--data-path` and uses `download=False`.

## Ruifang(Terry) Zhang Training Commands

Run from `official_code/`.

CIFAR-10 DWConv run:

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 main.py \
  --cfg ../reproduction_configs/vit_tiny_16_224_cifar10_100ep.yaml \
  --data-path /path/to/cifar-data \
  --batch-size 128 \
  --output ../outputs \
  --tag member1-cifar10-dw
```

CIFAR-100 DWConv run:

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12346 main.py \
  --cfg ../reproduction_configs/vit_tiny_16_224_cifar100_100ep.yaml \
  --data-path /path/to/cifar-data \
  --batch-size 128 \
  --output ../outputs \
  --tag member1-cifar100-dw
```

On newer PyTorch versions, `torchrun` may be preferred, but the official code currently requires a `--local_rank` argument. If switching to `torchrun`, update `main.py` argument parsing first.

## Current Implementation Note

The official `official_code/models/vit.py` currently implements the DWConv shortcut directly inside the ViT Transformer block. The baseline without DWConv is therefore not yet configurable from YAML.

Before running the baseline rows, we need to add either:

- a config flag such as `MODEL.ViT.USE_DWCONV`, or
- a separate baseline ViT model variant.

This should be done as a small, documented reproduction change so that baseline and DWConv runs share the same training pipeline.

## Reporting Checklist

For every run, record:

- dataset
- config file
- exact command
- GPU model
- epochs
- batch size
- seed
- best Acc@1
- final Acc@1
- training time
- commit hash
