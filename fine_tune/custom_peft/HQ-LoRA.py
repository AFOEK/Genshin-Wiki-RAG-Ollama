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

log = logging.getLogger(__name__)

class VariationalQuantumLayer(nn.Module):
    def __init__(self, n_qubits: int = 4, n_layers: int = 1, shots: int | None = None, diff_method: str = "paramter-shift", device=None):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = device or qml.device("default.qubit", wires=n_qubits, shots=shots)

        @qml.node(self.device, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

            for layer in range(n_layers):
                for qubit in range(n_qubits):
                    qml.Rot(weights[layer, qubit, 0],
                            weights[layer, qubit, 1],
                            weights[layer, qubit, 2])
                    
                for qubit in range(n_qubits):
                    qml.CNOT(wires=[qubit, (qubit + 1) % n_qubits])

                if layer + 1 < n_layers:
                    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            return tuple(qml.expval(qml.PauliZ(qubit)) for qubit in range(n_qubits))

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qnode = circuit
        self.layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layer(inputs)
    
class HybridQuantumLoRAAdapter(nn.Module):
    def __init__(self, hidden_size: int, rank: int = 4, quantum_layers: int = 1, alpha: float = 8.0, dropout: float = 0.05, shots: int | None = None, diff_method: str = "parameter-shift", quantum_backend = None):
        super().__init__()
        self.rank = rank
        self.scale = alpha / rank

        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.quantum = VariationalQuantumLayer(n_qubits=rank, n_layers=quantum_layers, shots=shots, diff_method=diff_method, device=quantum_backend)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.0)
        nn.init.normal_(self.up.weight, mean=0.0, std=0.0)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        angles = math.pi * torch.tanh(self.down(hidden))
        quantum_features = self.quantum(angles).to(dtype=hidden.dtypes)
        update = self.up(self.dropout(quantum_features))
        return hidden + self.scale * self.up