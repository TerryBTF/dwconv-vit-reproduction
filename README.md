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

## Reproduction Criteria

We plan to cover four criteria:

| Criterion | Question | Planned evidence |
|---|---|---|
| Reproduced | Does DWConv improve ViT-Tiny on CIFAR-10/CIFAR-100? | Main Table 1 subset: ViT-Tiny baseline vs ViT-Tiny + DWConv |
| Ablation study | Does DWConv reduce dependence on positional embeddings? | With/without positional embedding, with/without DWConv |
| Hyperparams check |  |  |
| New algorithm variant |  |  |

## Criterion 1: Reproduced 

Ruifang(Terry) Zhang is responsible for the `Reproduced` criterion.

The target is the main Table 1 subset:

| Dataset | Paper baseline | Paper DWConv | Our baseline | Our DWConv |
|---|---:|---:|---:|---:|
| CIFAR-10 | 94.01 | 96.41 | 90.95 | 94.72 |
| CIFAR-100 | 73.68 | 78.05 | 67.16 | 74.73 |

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

## Training Commands

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

On newer PyTorch versions, `torchrun` may be preferred. The local reproduction code accepts both `--local_rank` and `--local-rank` for compatibility.

## Current Results

Completed reduced-budget runs:

| Run ID | Dataset | Model | Epochs | Batch size | Best Acc@1 | Final Acc@1 | Training time |
|---|---|---|---:|---:|---:|---:|---:|
| M1-C10-DW | CIFAR-10 | ViT-Tiny + DWConv | 100 | 128 | 94.72 | 94.67 | 2:33:12 |
| M1-C100-DW | CIFAR-100 | ViT-Tiny + DWConv | 100 | 128 | 74.73 | 74.73 | 2:37:07 |
| M1-C10-BASE | CIFAR-10 | ViT-Tiny baseline | 100 | 128 | 90.95 | 90.85 | 2:18:28 |
| M1-C100-BASE | CIFAR-100 | ViT-Tiny baseline | 100 | 128 | 67.16 | 67.11 | 2:17:57 |

Run details:

- Config: `reproduction_configs/vit_tiny_16_224_cifar10_100ep.yaml`
- Tag: `member1-cifar10-dw`
- Output log: `outputs/ViT_Tiny_16_224_cifar10_100ep/member1-cifar10-dw/log_rank0.txt`
- Checkpoint: `outputs/ViT_Tiny_16_224_cifar10_100ep/member1-cifar10-dw/ckpt_epoch_99.pth`
- Commit: `29ae20c`

CIFAR-100 DWConv:

- Config: `reproduction_configs/vit_tiny_16_224_cifar100_100ep.yaml`
- Tag: `member1-cifar100-dw`
- Output log: `outputs/ViT_Tiny_16_224_cifar100_100ep/member1-cifar100-dw/log_rank0.txt`
- Final Acc@5: 91.40
- GFLOPs: 1.263017664G
- Commit: `29ae20c`

CIFAR-10 baseline:

- Config: `reproduction_configs/vit_tiny_16_224_cifar10_baseline_100ep.yaml`
- Tag: `member1-cifar10-baseline`
- Output log: `outputs/ViT_Tiny_16_224_cifar10_baseline_100ep/member1-cifar10-baseline/log_rank0.txt`
- Final Acc@5: 99.27
- GFLOPs: 1.25803296G
- Commit: `29ae20c`

CIFAR-100 baseline:

- Config: `reproduction_configs/vit_tiny_16_224_cifar100_baseline_100ep.yaml`
- Tag: `member1-cifar100-baseline`
- Output log: `outputs/ViT_Tiny_16_224_cifar100_baseline_100ep/member1-cifar100-baseline/log_rank0.txt`
- Final Acc@5: 87.38
- GFLOPs: 1.25805024G
- Commit: `29ae20c`

## Implementation Note

The local reproduction code adds `MODEL.ViT.USE_DWCONV` and `MODEL.ViT_S.USE_DWCONV` so baseline and DWConv variants can share the same training pipeline. Baseline configs are available in `reproduction_configs/` with `baseline_100ep` in the file name.

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
