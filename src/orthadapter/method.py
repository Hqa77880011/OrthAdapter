import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models.vision_transformer import VisionTransformer

VARIANTS = ["orthadapter", "lora", "alignment", "no_alignment", "linear", "finetune"]


class Factors(nn.Module):
    def __init__(self, dimension, rank):
        super().__init__()
        self.a = nn.Parameter(torch.empty(rank, dimension))
        self.b = nn.Parameter(torch.zeros(dimension, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def residual(self, alpha):
        return self.b @ self.a * alpha / len(self.a)


class CumulativeAttention(nn.Module):
    def __init__(self, attention, rank, alpha):
        super().__init__()
        self.base = attention
        self.dimension = attention.embed_dim
        self.heads = attention.num_heads
        self.rank = rank
        self.alpha = alpha
        self.query_adapters = nn.ModuleList()
        self.value_adapters = nn.ModuleList()

    def add_stage(self):
        device = self.base.in_proj_weight.device
        self.query_adapters.append(Factors(self.dimension, self.rank).to(device))
        self.value_adapters.append(Factors(self.dimension, self.rank).to(device))

    def forward(self, query, key, value, need_weights=False, attn_mask=None):
        d = self.dimension
        q_weight, k_weight, v_weight = self.base.in_proj_weight.split(d, dim=0)
        for adapter in self.query_adapters:
            q_weight = q_weight + adapter.residual(self.alpha)
        for adapter in self.value_adapters:
            v_weight = v_weight + adapter.residual(self.alpha)
        bias = self.base.in_proj_bias
        q_bias, k_bias, v_bias = (
            bias.split(d) if bias is not None else (None, None, None)
        )
        q = F.linear(query, q_weight, q_bias)
        k = F.linear(key, k_weight, k_bias)
        v = F.linear(value, v_weight, v_bias)
        batch, tokens, _ = q.shape
        q, k, v = [
            x.reshape(batch, tokens, self.heads, d // self.heads).transpose(1, 2)
            for x in [q, k, v]
        ]
        attention = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.base.dropout if self.training else 0.0,
        )
        merged = attention.transpose(1, 2).reshape(batch, tokens, d)
        return self.base.out_proj(merged), None

    def orthogonality(self):
        loss = self.base.in_proj_weight.new_zeros(())
        for adapters in [self.query_adapters, self.value_adapters]:
            if len(adapters) < 2:
                continue
            current = adapters[-1]
            a = F.normalize(current.a, dim=1, eps=1e-8)
            b = F.normalize(current.b, dim=0, eps=1e-8)
            old_a = torch.cat(
                [
                    F.normalize(item.a.detach(), dim=1, eps=1e-8)
                    for item in adapters[:-1]
                ],
                0,
            )
            old_b = torch.cat(
                [
                    F.normalize(item.b.detach(), dim=0, eps=1e-8)
                    for item in adapters[:-1]
                ],
                1,
            )
            denominator = (len(adapters) - 1) * self.rank**2
            loss = (
                loss
                + ((a @ old_a.T).square().sum() + (b.T @ old_b).square().sum())
                / denominator
            )
        return loss


class OrthAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config["backbone"] == "small":
            self.backbone = VisionTransformer(
                image_size=config["image_size"],
                patch_size=4,
                num_layers=2,
                num_heads=4,
                hidden_dim=64,
                mlp_dim=128,
                num_classes=1,
            )
            self.dimension = 64
        elif config["backbone"] == "vit_b_16":
            self.backbone = models.vit_b_16(
                weights=models.ViT_B_16_Weights.DEFAULT
                if config["pretrained"]
                else None
            )
            self.dimension = 768
        else:
            raise ValueError("backbone must be vit_b_16 or small")
        self.backbone.heads = nn.Identity()
        self.classifiers = nn.ModuleList()
        if config["variant"] in ["orthadapter", "lora", "alignment", "no_alignment"]:
            blocks = list(self.backbone.encoder.layers)
            for block in blocks[-config["adapter_blocks"] :]:
                block.self_attention = CumulativeAttention(
                    block.self_attention, config["rank"], config["alpha"]
                )
        self.backbone.requires_grad_(False)

    def adapted_layers(self):
        return [
            module
            for module in self.backbone.modules()
            if isinstance(module, CumulativeAttention)
        ]

    def add_stage(self, classes):
        self.requires_grad_(False)
        for parameter in self.parameters():
            parameter.grad = None
        for layer in self.adapted_layers():
            layer.add_stage()
        if self.config["variant"] == "finetune":
            self.backbone.requires_grad_(True)
        self.classifiers.append(
            nn.Linear(self.dimension, classes, bias=False).to(
                next(self.backbone.parameters()).device
            )
        )

    def forward(self, x):
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        features = self.backbone((x - mean) / std)
        return torch.cat([head(features) for head in self.classifiers], 1), features

    def orthogonality(self):
        return sum(
            (layer.orthogonality() for layer in self.adapted_layers()),
            next(self.parameters()).new_zeros(()),
        )

    @torch.no_grad()
    def align(self):
        if len(self.classifiers) < 2:
            return 1.0
        old = torch.cat([head.weight for head in self.classifiers[:-1]], 0)
        new = self.classifiers[-1].weight
        gamma = old.norm(dim=1).mean() / (new.norm(dim=1).mean() + 1e-8)
        new.mul_(gamma)
        return gamma.item()
