"""
Experimental Hybrid-Quantum LoRA
"""

import torch
import math
import logging
import pennylane as qml
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .VQC import VariationalQuantumLayer

log = logging.getLogger(__name__)

class HybridQuantumLoRAAdapter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        rank: int = 4,
        quantum_layers: int = 1,
        alpha: float = 8.0,
        dropout: float = 0.05,
        quantum_dropout: float = 0.10,
        shots: int | None = None,
        diff_method: str = "parameter-shift",
        quantum_backend=None,
    ):
        super().__init__()
        self.rank = rank
        self.scale = alpha / rank
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.quantum = VariationalQuantumLayer(
            n_qubits=rank,
            n_layers=quantum_layers,
            shots=shots,
            diff_method=diff_method,
            quantum_dropout=quantum_dropout,
            device=quantum_backend,
        )
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-3)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        angles = math.pi * torch.tanh(self.down(hidden))
        quantum_features = self.quantum(angles).to(device=hidden.device, dtype=hidden.dtype)
        update = self.up(self.dropout(quantum_features))
        return hidden + self.scale * update


class ModuleQuantumLoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 4,
        quantum_layers: int = 1,
        alpha: float = 8.0,
        dropout: float = 0.05,
        quantum_dropout: float = 0.10,
        shots: int | None = None,
        diff_method: str = "parameter-shift",
        token_policy: str = "cls",
        adapter_device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        if token_policy not in {"cls", "all"}:
            raise ValueError("token_policy must be 'cls' or 'all'.")
        self.base_layer = base_layer
        self.rank = rank
        self.scale = alpha / rank
        self.token_policy = token_policy
        self.adapter_device = adapter_device
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False
        self.down = nn.Linear(in_features, rank, bias=False).to(adapter_device)
        self.quantum = VariationalQuantumLayer(
            n_qubits=rank,
            n_layers=quantum_layers,
            shots=shots,
            diff_method=diff_method,
            quantum_dropout=quantum_dropout,
            device=None,
        ).to(adapter_device)
        self.up = nn.Linear(rank, out_features, bias=False).to(adapter_device)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-3)

    def adapter_update(self, hidden: torch.Tensor) -> torch.Tensor:
        original_device = hidden.device
        original_dtype = hidden.dtype
        hidden = hidden.to(self.adapter_device).float()
        angles = math.pi * torch.tanh(self.down(hidden))
        quantum_features = self.quantum(angles).to(device=self.adapter_device, dtype=torch.float32)
        update = self.up(self.dropout(quantum_features))
        return update.to(device=original_device, dtype=original_dtype)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(hidden)
        if hidden.dim() == 3 and self.token_policy == "cls":
            cls_hidden = hidden[:, 0, :]
            cls_update = self.adapter_update(cls_hidden)
            output = base_output.clone()
            output[:, 0, :] = output[:, 0, :] + self.scale * cls_update
            return output
        flat_hidden = hidden.reshape(-1, hidden.size(-1))
        flat_update = self.adapter_update(flat_hidden)
        update = flat_update.reshape(*hidden.shape[:-1], base_output.size(-1))
        return base_output + self.scale * update