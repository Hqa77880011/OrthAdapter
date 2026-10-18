# OrthAdapter

Implementation of **OrthAdapter: SVD-Free Factor-Orthogonal Low-Rank Adaptation for Replay-Free Class-Incremental Learning**.

## Installation

Use Python `>=3.10`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

`prepare` downloads [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), reserves a class-balanced validation set and preserves the test split. `pretrained=true` downloads the torchvision ViT weights. For [Tiny-ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip), extract to `data/tiny-imagenet-200` and use `configs/tinyimagenet.json`.

```bash
python -m orthadapter prepare --config configs/cifar100.json
```

## Training and execution

`configs/cifar100.json` fixes seed `2026`. Use `--seed` for a new run and keep its data/config paired. Paths are configurable with the CLI or JSON.

```bash
python -m orthadapter train --config configs/cifar100.json --output outputs/orthadapter
```

## Evaluation

```bash
python -m orthadapter evaluate --config configs/cifar100.json --checkpoint outputs/orthadapter/stage_10.pt --output outputs/orthadapter
```

## Comparisons and ablations

`compare` runs the configured variants; it includes model fitting where required. Keep comparison outputs separate from the main run.

```bash
python -m orthadapter compare --config configs/cifar100.json --output outputs/comparison
python -m orthadapter plot --config configs/cifar100.json --output outputs/comparison
```

## Outputs

Run directories contain `history.csv`, `stages.csv`, `task_accuracy.csv`, `metrics.json`, `evaluation.csv` as produced by the corresponding steps. `config.json` records effective settings. Comparison results are in `outputs/comparison/comparison.csv`; plotting reads those computed files. Each incremental stage saves `stage_N.pt`.

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, torchvision 0.22.1, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
