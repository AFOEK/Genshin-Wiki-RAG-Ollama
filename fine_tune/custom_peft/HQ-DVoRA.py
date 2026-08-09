"""
Experimental Hybrid-Quantum DVoRA
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

class HybridQuantumDVoRAAdapter(nn.Module):
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
        self.vera_b = nn.Parameter(torch.ones(rank))
        self.vera_d = nn.Parameter(torch.ones(hidden_size))
        self.log_magnitude = nn.Parameter(torch.zeros(hidden_size))
        self.quantum_rank_gate_strength = nn.Parameter(torch.tensor(0.10))
        self.quantum_output_gate_strength = nn.Parameter(torch.tensor(0.10))
        self.quantum_magnitude_strength = nn.Parameter(torch.tensor(0.10))
        self.magnitude_up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.zeros_(self.magnitude_up.weight)
        nn.init.normal_(self.down.weight, mean=0.0, std=1.0 / math.sqrt(hidden_size))
        nn.init.normal_(self.up.weight, mean=0.0, std=1.0 / math.sqrt(rank))
        self.down.weight.requires_grad_(False)
        self.up.weight.requires_grad_(False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        z = self.down(hidden)
        # Static VeRA rank scaling
        z = z * self.vera_b   
        # Quantum bottleneck
        angles = math.pi * torch.tanh(z)  
        quantum_features = self.quantum(angles).to(device=hidden.device, dtype=hidden.dtype)
        # Quantum-conditioned rank gate
        quantum_rank_gate = 1.0 + self.quantum_rank_gate_strength * torch.tanh( quantum_features)
        quantum_features = quantum_features * quantum_rank_gate
        # Frozen random projection: rank -> hidden_size
        directional_update = self.up( self.dropout(quantum_features))
        # Static VeRA output scaling
        directional_update = directional_update * self.vera_d
        # Quantum-conditioned output gate
        quantum_output_gate = 1.0 + self.quantum_output_gate_strength * torch.tanh( directional_update)
        directional_update = directional_update * quantum_output_gate
        # DoRA-style direction branch
        direction = hidden + self.scale * directional_update
        direction = F.normalize(direction, p=2, dim=-1)
        # Static DoRA magnitude
        base_magnitude = self.log_magnitude.view(1, -1)
        base_magnitude = base_magnitude.to(device=hidden.device, dtype=hidden.dtype)
        # Quantum-conditioned magnitude correction
        quantum_magnitude_delta = self.magnitude_up(quantum_features)
        quantum_magnitude_delta = self.quantum_magnitude_strength * torch.tanh(quantum_magnitude_delta)
        magnitude = torch.exp(base_magnitude + quantum_magnitude_delta)
        return magnitude * direction

class ModuleQuantumDVoRALinear(nn.Module):
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
        self.vera_b = nn.Parameter(torch.ones(rank, device=adapter_device))
        self.vera_d = nn.Parameter(torch.ones(out_features, device=adapter_device))
        self.log_magnitude = nn.Parameter(torch.zeros(out_features, device=adapter_device))
        self.quantum_rank_gate_strength = nn.Parameter(torch.tensor(0.10, device=adapter_device))
        self.quantum_output_gate_strength = nn.Parameter(torch.tensor(0.10, device=adapter_device))
        self.quantum_magnitude_strength = nn.Parameter(torch.tensor(0.10, device=adapter_device))
        self.magnitude_up = nn.Linear(rank, out_features, bias=False).to(adapter_device)
        nn.init.zeros_(self.magnitude_up.weight)
        nn.init.normal_(self.down.weight, mean=0.0, std=1.0 / math.sqrt(in_features))
        nn.init.normal_(self.up.weight, mean=0.0, std=1.0 / math.sqrt(rank))
        self.down.weight.requires_grad_(False)
        self.up.weight.requires_grad_(False)

    def adapter_update(self, hidden: torch.Tensor) -> torch.Tensor:
        original_device = hidden.device
        original_dtype = hidden.dtype
        hidden = hidden.to(self.adapter_device).float()
        # Frozen random projection: hidden -> rank
        z = self.down(hidden)
        # Static VeRA rank scaling
        z = z * self.vera_b
        # Quantum bottleneck
        angles = math.pi * torch.tanh(z)
        quantum_features = self.quantum(angles).to(device=self.adapter_device, dtype=torch.float32)
        # Quantum-conditioned rank gate
        quantum_rank_gate = 1.0 + self.quantum_rank_gate_strength * torch.tanh(quantum_features)
        quantum_features = quantum_features * quantum_rank_gate
        # Frozen random projection: rank -> output dimension
        directional_update = self.up(self.dropout(quantum_features))
        # Static VeRA output scaling
        directional_update = directional_update * self.vera_d
        # Quantum-conditioned output gate
        quantum_output_gate = 1.0 + self.quantum_output_gate_strength * torch.tanh(directional_update)
        directional_update = directional_update * quantum_output_gate
        # Direction branch
        direction = F.normalize(directional_update, p=2, dim=-1)
        # Static DoRA magnitude
        base_magnitude = self.log_magnitude.view(1, -1)
        # Quantum-conditioned magnitude correction
        quantum_magnitude_delta = self.magnitude_up(quantum_features)
        quantum_magnitude_delta = self.quantum_magnitude_strength * torch.tanh(quantum_magnitude_delta)
        magnitude = torch.exp(base_magnitude + quantum_magnitude_delta)
        update = magnitude * direction
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