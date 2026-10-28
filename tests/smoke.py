"""Two small ViT stages: frozen history, factor gradients, alignment and metrics."""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from orthadapter.common import prepare, plot_csv
from orthadapter.cli import train, evaluate, load_checkpoint, profile
from orthadapter.method import CumulativeAttention
from orthadapter.metrics import incremental_metrics, adapter_storage


def main():
    torch.set_num_threads(1)
    out = Path("outputs/smoke").resolve()
    rng = np.random.default_rng(71)
    for split in ["train", "val", "test"]:
        for label in ["a", "b", "c", "d"]:
            path = out / "raw" / split / label
            path.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(path / f"{i}.png")
    config = json.loads(Path("configs/cifar100.json").read_text())
    config.update(dataset="folder", root=str(out / "raw"), data=str(out / "data"), output=str(out),
                  device="cpu", image_size=16, backbone="small", pretrained=False,
                  adapter_blocks=2, rank=2, alpha=4., stages=2, epochs=1, max_steps=2,
                  batch_size=2, timing_warmup=1, timing_steps=2)
    prepare(config)
    summary = train(config)
    first = torch.load(out / "stage_1.pt", weights_only=False)
    final = torch.load(out / "stage_2.pt", weights_only=False)
    for name, old in first["model"].items():
        if ".query_adapters.0." in name or ".value_adapters.0." in name or name.startswith("classifiers.0."):
            assert torch.equal(old, final["model"][name]), name
    current = dict(config, checkpoint=str(out / "stage_2.pt"))
    model, _ = load_checkpoint(current)
    other, _ = load_checkpoint(current)
    model.eval(); other.eval()
    x = torch.rand(2, 3, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(model(x)[0], other(x)[0])
    model.train()
    before = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
    loss = torch.nn.functional.cross_entropy(model(x)[0], torch.tensor([2, 3])) + .05 * model.orthogonality()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    optimizer.step()
    assert all(not torch.equal(before[name], p) for name, p in model.named_parameters() if p.requires_grad)
    evaluate(current)
    timings = profile(current)
    assert all(np.isfinite(r["step_ms"]) for r in timings)
    plot_csv(out / "stages.csv", out / "stages.pdf", "stage", "accuracy")
    assert adapter_storage(model) == 8192 and summary["adapter_bytes"] == 8192
    assert incremental_metrics([[80], [75, 90], [60, 70, 85]], [80, 82, 72])["forgetting"] == 20
    attention = CumulativeAttention(nn.MultiheadAttention(2, 1, batch_first=True), 1, 1)
    attention.add_stage(); attention.add_stage()
    with torch.no_grad():
        for factors in list(attention.query_adapters) + list(attention.value_adapters):
            factors.a.copy_(torch.tensor([[1., 0.]])); factors.b.copy_(torch.tensor([[1.], [0.]]))
    assert attention.orthogonality().item() == 4.
    with torch.no_grad():
        attention.query_adapters[-1].a.mul_(3); attention.query_adapters[-1].b.div_(3)
    assert attention.orthogonality().item() == 4.
    print("PASS: prepared 4-class folders, 2 stages x 2 updates, frozen history, current factor/head gradients and updates, checkpoint reload/evaluation, matched step timing, hand orthogonality/forgetting/bytes")


if __name__ == "__main__":
    main()
