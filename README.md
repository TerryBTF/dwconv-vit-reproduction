# DWConv ViT Reproduction

This repository contains a reproducibility project for **Depth-Wise Convolutions in Vision Transformers for Efficient Training on Small Datasets**. The project evaluates whether depth-wise convolution (DWConv) shortcuts improve ViT-Tiny on CIFAR-10 and CIFAR-100 under a reduced single-GPU training budget.

- Paper: <https://arxiv.org/abs/2407.19394>
- Official code: <https://github.com/ZTX-100/Efficient_ViT_with_DW>
- Vendored official commit: `d7ae645bedec54b4850ff659889c0588164aaac2`
- Report: [`report/fundamental_report.md`](report/fundamental_report.md)
- PDF report: [`report/fundamental_report.pdf`](report/fundamental_report.pdf)

## Reproducibility Criteria

This is an existing-code reproduction. The project addresses three criteria:

| Criterion | Responsible member | Evidence |
|---|---|---|
| Reproduced | Ruifang Zhang | ViT-Tiny baseline vs ViT-Tiny + DWConv on CIFAR-10 and CIFAR-100 |
| Ablation study | Guotao Gou | Positional embedding on/off crossed with DWConv on/off |
| New algorithm variant | Radoslaw Majer | Learned per-layer fusion weights for the DWConv shortcut |

## Main Results

The original paper trained for 300 epochs on four NVIDIA P100 GPUs. These runs use 100 epochs on a single GPU, so the goal is to reproduce the qualitative trend rather than match the paper's exact accuracies.

| Dataset | Paper baseline | Paper DWConv | Our baseline | Our DWConv | DWConv gain |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | 94.01 | 96.41 | 90.95 | 94.72 | +3.77 |
| CIFAR-100 | 73.68 | 78.05 | 67.16 | 74.73 | +7.57 |

The reproduced results uphold the paper's main conclusion: DWConv improves ViT-Tiny on both datasets, with a larger gain on CIFAR-100.

## Ablation Summary

The ablation study evaluates whether DWConv reduces dependence on learned positional embeddings (PE).

| Dataset | PE | DWConv | Best Acc@1 | Final Acc@1 | GFLOPs | Time |
|---|---|---|---:|---:|---:|---:|
| CIFAR-10 | yes | no | 92.44 | 92.4 | 1.2580 | 4:50:52 |
| CIFAR-10 | yes | yes | 96.31 | 96.1 | 1.2630 | 5:36:44 |
| CIFAR-10 | no | no | 84.54 | 84.4 | 1.2580 | 4:50:19 |
| CIFAR-10 | no | yes | 96.11 | 96.0 | 1.2630 | 5:32:53 |
| CIFAR-100 | yes | no | 68.46 | 68.4 | 1.2581 | 4:54:14 |
| CIFAR-100 | yes | yes | 77.71 | 77.7 | 1.2630 | 5:38:15 |
| CIFAR-100 | no | no | 60.12 | 60.1 | 1.2581 | 4:49:57 |
| CIFAR-100 | no | yes | 79.42 | 79.3 | 1.2630 | 5:31:57 |

Removing PE strongly hurts the baseline, while the DWConv model remains accurate without PE. This supports the interpretation that DWConv supplies useful local spatial information.

## Repository Layout

```text
official_code/             Vendored implementation adapted for the reproduction
reproduction_configs/      YAML configs for main and ablation runs
experiments/               Experiment notes and result logs
report/                    Markdown and PDF report
controlled_dataset/        Individual controlled-dataset assignment assets
controlled_dataset_guotao/ Individual controlled-dataset assignment assets
requirements.txt           Python package requirements
```

## Setup

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

Install a PyTorch build matching your CUDA version if the generic wheel is not appropriate. CIFAR data must already exist under the path passed with `--data-path`; the loader uses `download=False`.

## Running Main Experiments

Run commands from `official_code/`.

CIFAR-10 DWConv:

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 main.py \
  --cfg ../reproduction_configs/vit_tiny_16_224_cifar10_100ep.yaml \
  --data-path /path/to/cifar-data \
  --batch-size 128 \
  --output ../outputs \
  --tag member1-cifar10-dw
```

CIFAR-100 DWConv:

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12346 main.py \
  --cfg ../reproduction_configs/vit_tiny_16_224_cifar100_100ep.yaml \
  --data-path /path/to/cifar-data \
  --batch-size 128 \
  --output ../outputs \
  --tag member1-cifar100-dw
```

Baseline runs use the corresponding `baseline_100ep.yaml` configs in `reproduction_configs/`.

## Running Ablations

The ablation configs cross `MODEL.ViT.USE_DWCONV` with `MODEL.ViT.USE_PE`. Run the provided script from the repository root:

```bash
DATA_PATH=/path/to/cifar BATCH_SIZE=32 ./ablation_study.sh
```

Outputs are written under `outputs/<config-name>/<tag>/`.

## Implementation Notes

The vendored ViT implementation was adapted to expose `MODEL.ViT.USE_DWCONV` and `MODEL.ViT.USE_PE`, enabling controlled baseline, DWConv, and no-positional-embedding runs from YAML configs. Compatibility fixes for newer PyTorch and timm versions are documented in [`experiments/reproduction_changes.md`](experiments/reproduction_changes.md).
