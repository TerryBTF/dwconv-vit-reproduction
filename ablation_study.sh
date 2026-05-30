#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFICIAL_DIR="$ROOT_DIR/official_code"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/data/cifar}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-12360}"

if [[ ! -d "$DATA_PATH" ]]; then
  echo "Dataset path does not exist: $DATA_PATH" >&2
  echo "Set DATA_PATH=/path/to/cifar or download CIFAR into $ROOT_DIR/data/cifar." >&2
  exit 1
fi

experiments=(
  "cifar10-baseline-pe|vit_tiny_16_224_cifar10_baseline_100ep.yaml|ablation-cifar10-baseline-pe"
  "cifar10-dwconv-pe|vit_tiny_16_224_cifar10_100ep.yaml|ablation-cifar10-dwconv-pe"
  "cifar10-baseline-nope|vit_tiny_16_224_cifar10_baseline_nope_100ep.yaml|ablation-cifar10-baseline-nope"
  "cifar10-dwconv-nope|vit_tiny_16_224_cifar10_nope_100ep.yaml|ablation-cifar10-dwconv-nope"
  "cifar100-baseline-pe|vit_tiny_16_224_cifar100_baseline_100ep.yaml|ablation-cifar100-baseline-pe"
  "cifar100-dwconv-pe|vit_tiny_16_224_cifar100_100ep.yaml|ablation-cifar100-dwconv-pe"
  "cifar100-baseline-nope|vit_tiny_16_224_cifar100_baseline_nope_100ep.yaml|ablation-cifar100-baseline-nope"
  "cifar100-dwconv-nope|vit_tiny_16_224_cifar100_nope_100ep.yaml|ablation-cifar100-dwconv-nope"
)

cd "$OFFICIAL_DIR"

for index in "${!experiments[@]}"; do
  IFS="|" read -r run_id cfg_file tag <<< "${experiments[$index]}"
  port=$((MASTER_PORT_BASE + index))

  echo "============================================================"
  echo "Running ablation: $run_id"
  echo "Config: $ROOT_DIR/reproduction_configs/$cfg_file"
  echo "Tag: $tag"
  echo "Master port: $port"
  echo "============================================================"

  "$PYTHON_BIN" -m torch.distributed.launch \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_port "$port" \
    main.py \
    --cfg "$ROOT_DIR/reproduction_configs/$cfg_file" \
    --data-path "$DATA_PATH" \
    --batch-size "$BATCH_SIZE" \
    --output "$OUTPUT_DIR" \
    --tag "$tag"
done
