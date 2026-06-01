import torch.nn as nn
import torch.nn.functional as F

import torch
import hashlib
import math
import logging

from pathlib import Path

from ..utils_peft import get_target_modules, freeze_all_parameters, get_parent_module

log = logging.getLogger(__name__)

class BoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, *, r: int, alpha: float, eps: float = 1e-6, seed: int = 1337, init_col_scale: float = 1.0):
        super().__init__()

        if not isinstance(linear, nn.Linear):
            raise TypeError(f"BoRALinear only support nn.Linear, got {type(linear)}")
        
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(r)
        self.eps = float(eps)

        weight = linear.weight.detach().clone()
        device = weight.device
        dtype = weight.dtype

        self.register_buffer("base_weight", weight)

        if linear.bias is not None:
            self.register_buffer("base_bias", linear.bias.detach().clone())
        else:
            self.base_bias = None

        row_magnitude = torch.linalg.vector_norm(weight.float(), dim=1)
        self.row_magnitude = nn.Parameter(row_magnitude.to(device=device, dtype=dtype))
        self.col_scale = nn.Parameter(torch.full((self.in_features,), float(init_col_scale), device=device, dtype=dtype))

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))

        A = torch.empty(self.r, self.in_features, dtype=torch.float32)
        B = torch.zeros(self.out_features, self.r, dtype=torch.float32)

        nn.init.kaiming_uniform_(A, a=math.sqrt(5))

        self.lora_A = nn.Parameter(A.to(device=device, dtype=dtype))
        self.lora_B = nn.Parameter(B.to(device=device, dtype=dtype))

    def delta_weight(self) -> torch.Tensor:
        return self.scaling * (self.lora_B @ self.lora_A)
    
    def effective_weight(self) -> torch.Tensor:
        direction = self.base_weight + self.delta_weight()
        row_norm = torch.linalg.vector_norm(direction.float(), dim=1, keepdim=True).clamp_min(self.eps).to(dtype=direction.dtype)
        row_unit = direction / row_norm
        return self.row_magnitude[:, None] * row_unit * self.col_scale[None, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.base_bias)


def apply_bora_adapters(model: nn.Module, train_cfg: dict) -> nn.Module:
    c = train_cfg.get("bora", {}) or {}

    r = int(c.get("r", 16))
    alpha = float(c.get("alpha", 32))
    eps = float(c.get("eps", 1e-6))
    seed = int(c.get("seed", 1337))
    init_col_scale = float(c.get("init_col_scale", 1.0))

    target_modules = set(get_target_modules(train_cfg, c))

    if not target_modules:
        raise RuntimeError("[BoRA] No target modules configured")

    freeze_all_parameters(model)

    replaced = 0

    for name, module in list(model.named_modules()):
        leaf_name = name.split(".")[-1]

        if leaf_name not in target_modules:
            continue

        if not isinstance(module, nn.Linear):
            continue

        parent, child_name = get_parent_module(model, name)

        setattr(
            parent,
            child_name,
            BoRALinear(
                module,
                r=r,
                alpha=alpha,
                eps=eps,
                seed=seed,
                init_col_scale=init_col_scale,
            ),
        )

        replaced += 1

    if replaced == 0:
        raise RuntimeError(f"[BoRA] Replaced 0 modules. Check target_modules={sorted(target_modules)}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    log.info("[BoRA] replaced=%d trainable=%d total=%d trainable_pct=%.6f", replaced, trainable, total, 100.0 * trainable / max(1, total))

    return model

def save_bora_adapter(model: nn.Module, output_dir: Path, train_cfg: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_state = {}

    for name, tensor in model.state_dict().items():
        if any(k in name for k in ["row_magnitude", "col_scale", "lora_A", "lora_B"]):
            adapter_state[name] = tensor.detach().cpu()

    payload = {
        "type": "bora",
        "config": train_cfg.get("bora", {}),
        "state_dict": adapter_state,
    }

    torch.save(payload, output_dir / "bora_adapter.pt")
    log.info("[BoRA] saved adapter state: %s", output_dir / "bora_adapter.pt")