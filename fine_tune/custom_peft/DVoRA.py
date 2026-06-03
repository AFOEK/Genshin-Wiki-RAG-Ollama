import torch.nn as nn
import torch.nn.functional as F

import torch
import hashlib
import math
import logging

from pathlib import Path

from ..utils_peft import get_target_modules, freeze_all_parameters, get_parent_module

log = logging.getLogger(__name__)

class DVoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, *, r: int, alpha: float, eps: float = 1e-6, seed: int = 1337, init_b: float = 0.0, init_d: float = 1.0):
        super().__init__()
        
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"DVoRALinear only supports nn.Linear, got {type(linear)}")
        
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(r)
        self.eps = float(eps)

        weight = linear.weight.detach().clone()
        self.register_buffer("base_weight", weight)

        if linear.bias is not None:
            self.register_buffer("base_bias", linear.bias.detach().clone())
        else:
            self.base_bias = None

        magnitude = torch.linalg.vector_norm(weight.float(), dim=1)
        self.magnitude = nn.Parameter(magnitude.to(dtype=weight.dtype))

        shape_key = f"{seed}:{self.out_features}:{self.in_features}:{self.r}"
        shape_hash = int(hashlib.sha256(shape_key.encode("utf-8")).hexdigest()[:8], 16)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(shape_hash)

        A = torch.randn(self.r, self.in_features, generator=gen, dtype=torch.float32)
        B = torch.randn(self.out_features, self.r, generator=gen, dtype=torch.float32)

        A = A / math.sqrt(max(1, self.in_features))
        B = B / math.sqrt(max(1, self.r))

        self.register_buffer("vera_A", A.to(dtype=weight.dtype))
        self.register_buffer("vera_B", B.to(dtype=weight.dtype))

        self.vera_d = nn.Parameter(torch.full((self.r,), float(init_d), dtype=weight.dtype))
        self.vera_b = nn.Parameter(torch.full((self.out_features,), float(init_b), dtype=weight.dtype))

    def delta_weight(self) -> torch.Tensor:
        B_scaled = self.vera_B * self.vera_b[:, None]
        A_scaled = self.vera_A * self.vera_d[:, None]
        return self.scaling * (B_scaled @ A_scaled)

    def effective_weight(self) -> torch.Tensor:
        direction = self.base_weight + self.delta_weight()
        direction_norm = torch.linalg.vector_norm(direction.float(), dim=1, keepdim=True).clamp_min(self.eps).to(dtype=direction.dtype)
        return self.magnitude[:, None] * direction / direction_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.base_bias)
    
def apply_dvora_adapters(model: nn.Module, train_cfg: dict) -> nn.Module:
    c = train_cfg.get("dvora", {}) or {}

    r = int(c.get("r", 256))
    alpha = float(c.get("alpha", 32))
    eps = float(c.get("eps", 1e-6))
    seed = int(c.get("seed", 1337))
    init_b = float(c.get("init_b", 0.0))
    init_d = float(c.get("init_d", 1.0))

    target_modules = set(get_target_modules(train_cfg, c))

    if not target_modules:
        raise RuntimeError("[DVoRA] No target modules configured")

    freeze_all_parameters(model)

    replaced = 0

    for name, module in list(model.named_modules()):
        leaf_name = name.split(".")[-1]

        if leaf_name not in target_modules:
            continue

        if not isinstance(module, nn.Linear):
            continue

        parent, child_name = get_parent_module(model, name)
        setattr(parent, child_name, DVoRALinear(module, r=r, alpha=alpha, eps=eps, seed=seed, init_b=init_b, init_d=init_d))
        replaced += 1

    if replaced == 0:
        raise RuntimeError(f"[DVoRA] Replaced 0 modules. Check target_modules={sorted(target_modules)}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    log.info("[DVoRA] replaced=%d trainable=%d total=%d trainable_pct=%.6f", replaced, trainable, total, 100.0 * trainable / max(1, total))
    return model

def save_dvora_adapter(model: nn.Module, output_dir: Path, train_cfg: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_state = {}
    for name, tensor in model.state_dict().items():
        if any(k in name for k in ["magnitude", "vera_b", "vera_d"]):
            adapter_state[name] = tensor.detach().cpu()

    payload = {"type": "dvora",  "config": train_cfg.get("dvora", {}), "state_dict": adapter_state}

    torch.save(payload, output_dir / "dvora_adapter.pt")
    log.info("[DVoRA] saved adapter state: %s", output_dir / "dvora_adapter.pt")