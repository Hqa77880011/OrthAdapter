# OrthAdapter

Implementation of **OrthAdapter: SVD-Free Factor-Orthogonal Low-Rank Adaptation for Replay-Free Class-Incremental Learning**. A frozen vision transformer accumulates query/value LoRA branches across class increments. New factors are softly orthogonalized against historical factors, then new classifier weights are aligned to old weight norms.

## Installation

Use Python 3.10 or newer. Install matching PyTorch and torchvision builds for your device, then install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. For NVIDIA GPUs, choose a CUDA build from the [PyTorch installer](https://pytorch.org/get-started/locally/) in place of the CPU command. The supported dependency range is PyTorch 2.4–2.7 with its matching torchvision release.

## Data and incremental training

`prepare` downloads [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) through torchvision and creates a seeded, class-balanced validation split of 5,000 training images. The remaining 45,000 images form the training pool; the official test split is untouched. The prepared directory contains `train.npz`, `val.npz`, `test.npz` and `meta.json`. Array indices identify the original training examples.

For [Tiny-ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip), extract the archive to `data/tiny-imagenet-200`. Its annotated validation images are split deterministically into validation and test halves because the released test images have no labels. `prepare` writes image-path manifests and does not copy the images. A custom image collection can use `dataset="folder"` with identical class directories under `root/train`, `root/val` and `root/test`.

```bash
python -m orthadapter prepare --config configs/cifar100.json
python -m orthadapter train --config configs/cifar100.json --output outputs/orthadapter
python -m orthadapter evaluate --config configs/cifar100.json --checkpoint outputs/orthadapter/stage_10.pt --output outputs/orthadapter
python -m orthadapter plot --config configs/cifar100.json --output outputs/orthadapter
```

The default class order is a seeded permutation, divided into ten groups of ten classes. `--stages 20` selects twenty groups of five; use the corresponding `stage_20.pt` checkpoint when evaluating that run. Tiny-ImageNet uses `configs/tinyimagenet.json` and ten groups of twenty classes.

## Baselines and ablations

```bash
python -m orthadapter compare --config configs/cifar100.json --output outputs/comparison
python -m orthadapter plot --config configs/cifar100.json --output outputs/comparison
python -m orthadapter train --config configs/cifar100.json --set rank=8 --output outputs/rank8
```

| Variant | Behavior |
| --- | --- |
| `linear` | Frozen backbone; train only the new classifier rows |
| `finetune` | Train the backbone and new classifier rows |
| `lora` | Cumulative LoRA without orthogonalization or alignment |
| `no_alignment` | LoRA with factor orthogonality |
| `orthadapter` | Factor orthogonality and norm alignment |

The executable comparison covers these controlled baselines; the paper's additional continual-learning systems are not implemented under alternative names.

## Configuration and outputs

Commands read the supplied JSON configuration. `--data`, `--output`, `--device`, `--seed`, `--batch-size`, `--epochs`, `--max-steps` and `--limit` override individual fields. `--device auto` chooses CUDA when available. `max_steps=0` and `limit=0` mean no limit. `max_steps` stops optimization after the requested number of updates; it does not produce a trained benchmark model. For other fields use `--set KEY=JSON`, for example `--set lr=0.01` or `--set 'pretrained=false'`. String values require JSON quotes.

`backbone="vit_b_16"`, `pretrained=true` and `image_size=224` use torchvision's ImageNet-pretrained ViT-B/16. LoRA modifies query and value projections in the last six blocks, with rank 4, alpha 8 and orthogonality weight 0.05. Training uses AdamW; its learning rate, weight decay and per-stage epoch count are explicit configuration choices. `backbone="small"`, `pretrained=false`, `image_size=32` selects a two-block, 64-feature transformer for bounded development runs.

`history.csv` records per-stage training. `stages.csv` reports all-seen, old-class and new-class accuracy, alignment scale and parameter growth. `task_accuracy.csv` stores the lower-triangular task accuracy matrix. `metrics.json` reports average incremental accuracy, final accuracy and average forgetting. Forgetting compares each old task's best accuracy before the final stage with its final accuracy. `stage_N.pt` saves the model, class order, stage boundaries and accuracy matrix. `evaluation.csv` evaluates a saved stage independently.

## Method conventions

At every stage, the old adapters, backbone and old classifier heads are frozen. Cross entropy still includes logits over all seen classes. Only current-stage images are accessed for optimization. Historical model parameters remain in the cumulative inference path; no historical image or feature buffer is retained. The factor penalty uses row-normalized A and column-normalized B with denominator `(previous_stages * rank^2)` per adapted projection. Alignment is skipped at the first stage and subsequently rescales only the newest classifier weights. Classifier heads have no bias so the norm-alignment operation has a single defined target. Cumulative attention computes the summed effective projections for every sample and needs no task identifier.

Data, weights and generated outputs are ignored by Git. Run directories store the effective `config.json`, CSV measurements and JSON metrics. Plots read the generated measurement files.
