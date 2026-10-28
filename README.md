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

`split_seed=2026` and `class_order_seed=2026` are independent of the training seed, keeping the split and class order fixed across repeats. CIFAR-100 uses 45,000/5,000/10,000 train/validation/test images. At stage t the training loader exposes only current-class images; it does not replay old images or feature vectors. Tiny-ImageNet uses a deterministic half/half split of its labeled official validation set for local validation/test; its original experimental split is not supplied. Inputs use fixed ImageNet normalization.

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

`configs/factorial.json` selects the equal-capacity four arms `lora`, `alignment`, `no_alignment`, `orthadapter`; use the same prepare/train/compare commands with that config. All retain query/value adapters in the last six ViT blocks, rank 4, alpha 8, and 20 epochs per stage. `lambda_orth=0.05` applies to the two orthogonality arms. `--stages 20` selects the five-class CIFAR-100 protocol. `--set rank=...` supports rank ablations. `linear` and `finetune` are supported baseline controls; EWC/LwF/CODA-Prompt/RanPAC/InfLoRA integrations are not supplied.

To measure current-stage step cost from an existing checkpoint:

```bash
python -m orthadapter profile --config configs/cifar100.json --checkpoint outputs/orthadapter/stage_10.pt --output outputs/profile
```

`timing_warmup` defaults to 200 and `timing_steps` to 1000. The two measurements reload the same cumulative checkpoint, use the same fixed current-stage training batch and fresh optimizer, and toggle only the orthogonality loss. Alignment/evaluation/data loading are outside the timed region. `adapter_bytes` is retained tensor storage, distinct from `peak_cuda_bytes`.

## Outputs

Run directories contain `history.csv`, `stages.csv`, `task_accuracy.csv`, `metrics.json`, `evaluation.csv` as produced by the corresponding steps. `config.json` records effective settings. Comparison results are in `outputs/comparison/comparison.csv`; plotting reads those computed files. Each incremental stage saves `stage_N.pt`.

## Verification

```bash
python tests/smoke.py
```

Passed on Windows / Python 3.13.5 / NumPy 2.1.3 (OpenBLAS) / PyTorch 2.7.1+cu118 / torchvision 0.22.1: four-class folder preparation, two small-ViT stages with two updates each, finite current-factor/head gradients and updates, exact preservation of historical factors/classifier rows, checkpoint prediction consistency, evaluation/plotting, matched step profiling and hand factor-correlation/forgetting/storage checks. This tiny randomly initialized model checks mechanics; full pretrained ViT experiments and reported results were not rerun.

| Paper module or metric | Code entry | Configuration or command |
|---|---|---|
| Frozen cumulative query/value LoRA, unified seen-class head | `method.CumulativeAttention`, `OrthAdapter` | `adapter_blocks=6`, `rank=4`, `alpha=8` |
| Normalized factor-only orthogonality | `CumulativeAttention.orthogonality` | `lambda_orth=0.05`; no feature covariance/SVD |
| Old-row freezing and new-row alignment | `OrthAdapter.add_stage/align` | `alignment` / `orthadapter` |
| Seen/old/new accuracy and task-by-stage matrix | `cli.train`, `cli.measure` | `stages.csv`, `task_accuracy.csv` |
| Average/final incremental accuracy, forgetting | `metrics.incremental_metrics` | `metrics.json` |
| Trainable parameters, retained adapter bytes | `cli.train`, `metrics.adapter_storage` | `stages.csv` |
| Orthogonality step overhead and peak CUDA memory | `cli.profile` | `profile.csv` |

Forgetting uses each old task's maximum before the final stage, then subtracts its final accuracy and averages old tasks. It is not clamped to zero. Accuracies/forgetting use percentage units; average incremental accuracy weights complete stages equally. The cumulative transformer implementation predates this update; this update adds fixed protocol seeds, the missing alignment-only arm, cleared historical gradients, explicit storage/forgetting helpers, profiling and functional checks. No generated verification score is a paper result.
