import torch
import logging
import pennylane as qml
import torch.nn as nn

log = logging.getLogger(__name__)

class VariationalQuantumLayer(nn.Module):
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, shots: int | None = None, diff_method: str = "parameter-shift", quantum_dropout: float = 0.0, quantum_backend_name: str = "lightning.qubit", device=None):
        super().__init__()
        if not 0.0 <= quantum_dropout < 1.0:
            raise ValueError("quantum_dropout must be in [0.0, 1.0).")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.quantum_dropout = quantum_dropout
        self.device = device or qml.device(quantum_backend_name, wires=n_qubits, shots=shots)
        self.weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, 3))

        @qml.qnode(self.device, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights, rotation_mask):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            for layer in range(n_layers):
                for qubit in range(n_qubits):
                    theta = weights[layer, qubit] * rotation_mask[layer, qubit]
                    qml.Rot(theta[0], theta[1], theta[2], wires=qubit)
                for qubit in range(n_qubits):
                    qml.CNOT(wires=[qubit, (qubit + 1) % n_qubits])
                if layer + 1 < n_layers:
                    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            return tuple(qml.expval(qml.PauliZ(qubit)) for qubit in range(n_qubits))
        self.qnode = circuit

    def make_rotation_mask(self) -> torch.Tensor:
        if not self.training or self.quantum_dropout <= 0.0:
            return torch.ones_like(self.weights)
        keep_probability = 1.0 - self.quantum_dropout
        mask = torch.bernoulli(torch.full_like(self.weights, keep_probability))
        return mask.detach()

    def run_single(self, inputs: torch.Tensor, rotation_mask: torch.Tensor) -> torch.Tensor:
        output = self.qnode(inputs, self.weights, rotation_mask)
        if isinstance(output, tuple):
            output = torch.stack(list(output))
        return output

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        rotation_mask = self.make_rotation_mask()
        if inputs.dim() == 1:
            return self.run_single(inputs, rotation_mask)
        outputs = [self.run_single(inputs[index], rotation_mask) for index in range(inputs.size(0))]
        return torch.stack(outputs, dim=0)
    