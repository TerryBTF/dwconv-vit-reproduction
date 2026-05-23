# Repository Guidelines

## Project Structure & Module Organization

This repository is a course reproduction of DWConv ViT results on small image datasets.

- `official_code/`: vendored author implementation. Core training entry point is `official_code/main.py`; model code lives in `official_code/models/`; YAML model configs live in `official_code/configs/`.
- `reproduction_configs/`: reduced-compute configs for course experiments, currently ViT-Tiny CIFAR-10 and CIFAR-100 100-epoch runs.
- `experiments/`: experiment plans, run notes, and result logs.
- `official_code/figures/`: paper architecture figures used by the upstream README.
- `requirements.txt`: Python dependencies.

Prefer adding course-specific configs, logs, and documentation outside `official_code/`. Only edit `official_code/` when a reproduction change requires modifying the implementation.

## Build, Test, and Development Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training from `official_code/`:

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 main.py \
  --cfg ../reproduction_configs/vit_tiny_16_224_cifar10_100ep.yaml \
  --data-path /path/to/cifar-data \
  --batch-size 128 \
  --output ../outputs \
  --tag member1-cifar10-dw
```

The CIFAR loader expects data to already exist under `--data-path` and uses `download=False`.

Run the upstream CUDA window-process test, if the extension and CUDA environment are available:

```bash
cd official_code/kernels/window_process
python unit_test.py
```

## Coding Style & Naming Conventions

Python code follows the existing PyTorch style in `official_code/`: 4-space indentation, `snake_case` functions and variables, and `CamelCase` classes. YAML configs should use descriptive names matching model, image size, dataset, and budget, for example `vit_tiny_16_224_cifar10_100ep.yaml`.

Keep changes small and reproducible. Avoid broad refactors of vendored official code unless needed for a documented experiment.

## Testing Guidelines

There is no project-wide test suite yet. For implementation changes, run the smallest relevant smoke test: model import/build checks, a short training launch if compute permits, or `official_code/kernels/window_process/unit_test.py` for CUDA kernel changes. Record commands, GPU, seed, dataset, config, and observed result in `experiments/`.

## Commit & Pull Request Guidelines

Existing commits use concise imperative summaries, such as `Document algorithm variant criterion` and `Set up DWConv ViT reproduction scaffold`. Follow that style: one clear sentence, focused on the change.

Pull requests should include the reproduction goal, changed configs or code paths, exact commands run, data path assumptions, hardware used, and any result table updates. Link related issues or course tasks when applicable.

## Security & Configuration Tips

Do not commit datasets, model checkpoints, virtual environments, or large output directories. Keep local paths such as `/path/to/cifar-data` out of committed configs unless they are portable placeholders.
