# Ruifang(Terry) Zhang Main Reproduction Log

Owner: Ruifang(Terry) Zhang

Criterion: Reproduced

## Research Question

Does adding depth-wise convolution shortcuts improve ViT-Tiny classification accuracy on CIFAR-10 and CIFAR-100 under a reduced single-GPU training budget?

## Planned Runs

| Run ID | Dataset | Model | Config | Status | Acc@1 | Training Time |
|---|---|---|---|---|---:|---|
| M1-C10-DW | CIFAR-10 | ViT-Tiny + DWConv | `reproduction_configs/vit_tiny_16_224_cifar10_100ep.yaml` | done | 94.72 | 2:33:12 |
| M1-C100-DW | CIFAR-100 | ViT-Tiny + DWConv | `reproduction_configs/vit_tiny_16_224_cifar100_100ep.yaml` | done | 74.73 | 2:37:07 |
| M1-C10-BASE | CIFAR-10 | ViT-Tiny baseline without DWConv | `reproduction_configs/vit_tiny_16_224_cifar10_baseline_100ep.yaml` | done | 90.95 | 2:18:28 |
| M1-C100-BASE | CIFAR-100 | ViT-Tiny baseline without DWConv | `reproduction_configs/vit_tiny_16_224_cifar100_baseline_100ep.yaml` | done | 67.16 | 2:17:57 |

## Completed Run Details

### M1-C10-DW

- Dataset: CIFAR-10
- Model: ViT-Tiny + DWConv
- Epochs: 100
- Batch size: 128
- Seed: 0
- Best Acc@1: 94.72%
- Final Acc@1: 94.67%
- Final Acc@5: 99.74%
- Training time: 2:33:12
- GFLOPs: 1.263000384G
- Commit: `29ae20c`
- Output log: `outputs/ViT_Tiny_16_224_cifar10_100ep/member1-cifar10-dw/log_rank0.txt`
- Checkpoint: `outputs/ViT_Tiny_16_224_cifar10_100ep/member1-cifar10-dw/ckpt_epoch_99.pth`

### M1-C100-DW

- Dataset: CIFAR-100
- Model: ViT-Tiny + DWConv
- Epochs: 100
- Batch size: 128
- Seed: 0
- Best Acc@1: 74.73%
- Final Acc@1: 74.73%
- Final Acc@5: 91.40%
- Training time: 2:37:07
- GFLOPs: 1.263017664G
- Commit: `29ae20c`
- Output log: `outputs/ViT_Tiny_16_224_cifar100_100ep/member1-cifar100-dw/log_rank0.txt`

### M1-C10-BASE

- Dataset: CIFAR-10
- Model: ViT-Tiny baseline without DWConv
- Epochs: 100
- Batch size: 128
- Seed: 0
- Best Acc@1: 90.95%
- Final Acc@1: 90.85%
- Final Acc@5: 99.27%
- Training time: 2:18:28
- GFLOPs: 1.25803296G
- Commit: `29ae20c`
- Output log: `outputs/ViT_Tiny_16_224_cifar10_baseline_100ep/member1-cifar10-baseline/log_rank0.txt`

### M1-C100-BASE

- Dataset: CIFAR-100
- Model: ViT-Tiny baseline without DWConv
- Epochs: 100
- Batch size: 128
- Seed: 0
- Best Acc@1: 67.16%
- Final Acc@1: 67.11%
- Final Acc@5: 87.38%
- Training time: 2:17:57
- GFLOPs: 1.25805024G
- Commit: `29ae20c`
- Output log: `outputs/ViT_Tiny_16_224_cifar100_baseline_100ep/member1-cifar100-baseline/log_rank0.txt`

