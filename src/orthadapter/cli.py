from pathlib import Path
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Subset

from .common import (
    Images,
    loader,
    device_of,
    seed_everything,
    prepare,
    parse_config,
    save_json,
    save_rows,
    plot_csv,
)
from .method import OrthAdapter, VARIANTS
from .metrics import incremental_metrics, adapter_storage


def stage_groups(classes, stages, seed):
    if stages < 1 or stages > classes or classes % stages:
        raise ValueError("stages must divide the number of classes")
    order = np.random.default_rng(seed).permutation(classes).tolist()
    return order, np.asarray(order).reshape(stages, -1).tolist()


@torch.no_grad()
def measure(model, dataset, classes, remap, config):
    indices = np.flatnonzero(np.isin(dataset.labels, classes))
    model.eval()
    correct = total = 0
    for x, y in loader(Subset(dataset, indices.tolist()), config):
        logits, _ = model(x.to(device_of(config)))
        labels = remap[y]
        correct += (logits.argmax(1).cpu() == labels).sum().item()
        total += len(y)
    return 100 * correct / total


def train(config):
    if config["variant"] not in VARIANTS:
        raise ValueError(f"Unknown variant: {config['variant']}")
    seed_everything(config["seed"])
    data = Images(config, "train", augment=True)
    test = Images(config, "test")
    order, groups = stage_groups(len(data.classes), config["stages"], config.get("class_order_seed", 2026))
    remap = torch.tensor(np.argsort(order), dtype=torch.long)
    model = OrthAdapter(config).to(device_of(config))
    history, stage_rows, accuracy_matrix = [], [], []
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    for stage, classes in enumerate(groups):
        model.add_stage(len(classes))
        current = Subset(data, np.flatnonzero(np.isin(data.labels, classes)).tolist())
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            parameters, lr=config["lr"], weight_decay=config["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, config["epochs"]
        )
        steps = 0
        for epoch in range(config["epochs"]):
            model.train()
            losses = []
            for x, labels in loader(current, config, shuffle=True):
                x = x.to(device_of(config))
                y = remap[labels].to(device_of(config))
                logits, _ = model(x)
                loss = F.cross_entropy(logits, y)
                if config["variant"] in ["orthadapter", "no_alignment"]:
                    loss = loss + config["lambda_orth"] * model.orthogonality()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                steps += 1
                if config["max_steps"] and steps >= config["max_steps"]:
                    break
            row = {
                "variant": config["variant"],
                "stage": stage + 1,
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            scheduler.step()
            if config["max_steps"] and steps >= config["max_steps"]:
                break
        gamma = model.align() if config["variant"] in ["orthadapter", "alignment"] else 1.0
        seen = [label for group in groups[: stage + 1] for label in group]
        task_accuracy = [
            measure(model, test, group, remap, config) for group in groups[: stage + 1]
        ]
        accuracy_matrix.append(task_accuracy)
        row = {
            "variant": config["variant"],
            "seed": config["seed"],
            "stage": stage + 1,
            "accuracy": measure(model, test, seen, remap, config),
            "new_accuracy": task_accuracy[-1],
            "old_accuracy": float(np.mean(task_accuracy[:-1])) if stage else None,
            "alignment": gamma,
            "trainable_parameters": sum(p.numel() for p in parameters),
            "adapter_bytes": adapter_storage(model),
            "adapter_parameters": sum(
                p.numel()
                for layer in model.adapted_layers()
                for adapters in [layer.query_adapters, layer.value_adapters]
                for p in adapters.parameters()
            ),
        }
        stage_rows.append(row)
        torch.save(
            {
                "model": model.state_dict(),
                "classes": data.classes,
                "groups": groups[: stage + 1],
                "order": order,
                "config": config,
                "accuracy_matrix": accuracy_matrix,
            },
            output / f"stage_{stage + 1}.pt",
        )
    summary = {
        "variant": config["variant"],
        "seed": config["seed"],
        **incremental_metrics(accuracy_matrix, [r["accuracy"] for r in stage_rows]),
        "adapter_parameters": stage_rows[-1]["adapter_parameters"],
        "adapter_bytes": adapter_storage(model),
    }
    save_rows(output / "history.csv", history)
    save_rows(output / "stages.csv", stage_rows)
    save_rows(
        output / "task_accuracy.csv",
        [
            {"stage": i + 1, "task": j + 1, "accuracy": value}
            for i, row in enumerate(accuracy_matrix)
            for j, value in enumerate(row)
        ],
    )
    save_json(output / "metrics.json", summary)
    save_json(output / "config.json", config)
    print(json.dumps(summary), flush=True)
    return summary


def load_checkpoint(config):
    """Reconstruct cumulative stage modules before loading a saved state."""
    state = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    trained = dict(state["config"], pretrained=False)
    model = OrthAdapter(trained).to(device_of(config))
    for group in state["groups"]:
        model.add_stage(len(group))
    model.load_state_dict(state["model"])
    return model, state


def evaluate(config):
    model, state = load_checkpoint(config)
    remap = torch.tensor(np.argsort(state["order"]), dtype=torch.long)
    data = Images(config, "test")
    rows = [
        {"task": i + 1, "accuracy": measure(model, data, group, remap, config)}
        for i, group in enumerate(state["groups"])
    ]
    save_rows(Path(config["output"]) / "evaluation.csv", rows)
    print(json.dumps(rows), flush=True)


def profile(config):
    """Measure matched current-stage training steps, excluding stage-end alignment."""
    rows = []
    for weight in [0., config["lambda_orth"]]:
        model, state = load_checkpoint(config)
        data = Images(config, "train", augment=False)
        current = Subset(data, np.flatnonzero(np.isin(data.labels, state["groups"][-1])).tolist())
        x, labels = next(iter(loader(current, config)))
        x = x.to(device_of(config))
        y = torch.tensor(np.argsort(state["order"]), dtype=torch.long)[labels].to(x.device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["lr"], weight_decay=config["weight_decay"])
        model.train()
        timings = []
        for i in range(config.get("timing_warmup", 200) + config.get("timing_steps", 1000)):
            if x.is_cuda:
                torch.cuda.synchronize()
                if i == config.get("timing_warmup", 200):
                    torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x)[0], y)
            if weight:
                loss = loss + weight * model.orthogonality()
            loss.backward()
            optimizer.step()
            if x.is_cuda:
                torch.cuda.synchronize()
            if i >= config.get("timing_warmup", 200):
                timings.append((time.perf_counter() - start) * 1000)
        rows.append({"stage": len(state["groups"]), "lambda_orth": weight,
                     "batch_size": len(x), "step_ms": float(np.mean(timings)),
                     "adapter_bytes": adapter_storage(model),
                     "peak_cuda_bytes": torch.cuda.max_memory_allocated() if x.is_cuda else None})
    save_rows(Path(config["output"]) / "profile.csv", rows)
    return rows


def compare(config):
    rows = []
    for seed in config["seeds"]:
        for variant in config["variants"]:
            current = dict(
                config,
                variant=variant,
                seed=seed,
                output=str(Path(config["output"]) / variant / str(seed)),
            )
            rows.append(train(current))
    save_rows(Path(config["output"]) / "comparison.csv", rows)


def main():
    command, config = parse_config(["prepare", "train", "evaluate", "compare", "profile", "plot"])
    if command == "prepare":
        prepare(config)
    elif command == "train":
        train(config)
    elif command == "evaluate":
        evaluate(config)
    elif command == "compare":
        compare(config)
    elif command == "profile":
        profile(config)
    else:
        root = Path(config["output"])
        if (root / "stages.csv").exists():
            plot_csv(
                root / "stages.csv",
                root / "incremental_accuracy.pdf",
                "stage",
                "accuracy",
            )
        else:
            plot_csv(
                root / "comparison.csv",
                root / "accuracy_forgetting.pdf",
                "forgetting",
                "average_accuracy",
            )
