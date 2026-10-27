import torch
import logging
import pennylane as qml
import torch.nn as nn

from quantum_device import make_quantum_device
from configs import QUANTUM_INIT_STD, QUANTUM_BACKEND_NAME

log = logging.getLogger(__name__)

class VariationalQuantumLayer(nn.Module):
    def __init__(self, n_qubits: int = 4, n_layers: int = 2, shots: int | None = None, diff_method: str = "parameter-shift", quantum_dropout: float = 0.0, device=None, use_ibm_fake_noise: bool = False, ibm_fake_backend_name: str = "FakeSherbrooke", simulator_seed: int | None = None,):
        super().__init__()
        if not 0.0 <= quantum_dropout < 1.0:
            raise ValueError("quantum_dropout must be in [0.0, 1.0).")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.diff_method = diff_method
        self.quantum_dropout = quantum_dropout
        self.use_ibm_fake_noise = use_ibm_fake_noise
        self.ibm_fake_backend_name = ibm_fake_backend_name
        self.device = (
            device
            if device is not None
            else make_quantum_device(
                n_qubits=n_qubits,
                shots=shots,
                use_ibm_fake_noise=use_ibm_fake_noise,
                ibm_fake_backend_name=ibm_fake_backend_name,
                simulator_seed=simulator_seed,
            )
        )

        self.weights = nn.Parameter(QUANTUM_INIT_STD * torch.randn(n_layers, n_qubits, 3,))

        @qml.qnode(self.device, interface="torch", diff_method=diff_method, cache=True)
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

    def run_quantum_batch(self, inputs, rotation_mask):
        output = self.qnode(inputs, self.weights, rotation_mask)
        if isinstance(output, tuple):
            output = torch.stack(list(output), dim=-1)
        return output
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        rotation_mask = self.make_rotation_mask()
        if inputs.dim() == 1:
            return self.run_single(inputs, rotation_mask)

        needs_unbatched_grad = (self.training and self.diff_method == "parameter-shift")
    
        if needs_unbatched_grad:
            outputs = [self.run_single(x, rotation_mask) for x in inputs]
            return torch.stack(outputs, dim=0)
    
        return self.run_quantum_batch(inputs, rotation_mask)
    