# Ruifang(Terry) Zhang Main Reproduction Log

Owner: Ruifang(Terry) Zhang

Criterion: Reproduced

## Research Question

Does adding depth-wise convolution shortcuts improve ViT-Tiny classification accuracy on CIFAR-10 and CIFAR-100 under a reduced single-GPU training budget?

## Planned Runs

| Run ID | Dataset | Model | Config | Status | Acc@1 | Training Time |
|---|---|---|---|---|---:|---|
| M1-C10-DW | CIFAR-10 | ViT-Tiny + DWConv | `reproduction_configs/vit_tiny_16_224_cifar10_100ep.yaml` | planned | TBD | TBD |
| M1-C100-DW | CIFAR-100 | ViT-Tiny + DWConv | `reproduction_configs/vit_tiny_16_224_cifar100_100ep.yaml` | planned | TBD | TBD |
| M1-C10-BASE | CIFAR-10 | ViT-Tiny baseline without DWConv | TBD after baseline toggle | planned | TBD | TBD |
| M1-C100-BASE | CIFAR-100 | ViT-Tiny baseline without DWConv | TBD after baseline toggle | planned | TBD | TBD |

## Notes

- The official `models/vit.py` currently includes the DWConv shortcut in the ViT Transformer block.
- A baseline without DWConv requires a small code/config switch or a separate baseline model variant before running the two baseline rows.
- Record exact command, GPU model, batch size, seed, best Acc@1, final Acc@1, and wall-clock training time for every run.
