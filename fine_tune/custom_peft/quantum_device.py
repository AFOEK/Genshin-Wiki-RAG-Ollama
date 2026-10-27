import os
import logging
from types import MethodType

import pennylane as qml
from qiskit import transpile
from qiskit_aer import AerSimulator

from configs import QUANTUM_BACKEND_NAME, NOISE_SIMULATION_METHOD, NOISE_SIMULATION_DEVICE, NOISE_BATCHED_SHOTS_GPU

log = logging.getLogger(__name__)

def enable_qiskit_routing(dev, seed=None):
    legacy = getattr(dev, "target_device", dev)
    if not hasattr(legacy, "transpile_args"):
        raise RuntimeError(
            "This PennyLane-Qiskit version does not expose "
            "the legacy transpile_args interface required "
            "by the training notebook's routing implementation."
        )

    def routed_compile(self):
        options = dict(self.transpile_args)
        options.pop("routing_method", None)
        options.setdefault("optimization_level", 1)
        options.setdefault("seed_transpiler", seed)
        return transpile(
            self._circuit,
            backend=self.compile_backend or self.backend,
            routing_method="sabre",
            **options,
        )

    legacy.compile = MethodType(routed_compile, legacy)
    log.info("[QUANTUM] SABRE routing enabled")
    return dev

class IBMFakeNoiseFactory:
    def __init__(
        self,
        backend_name="FakeSherbrooke",
        shots=1024,
        simulation_method="density_matrix",
        simulation_device="CPU",
        batched_shots_gpu=False,
        simulator_seed=None,
    ):
        self.backend_name = backend_name
        self.shots = shots
        self.simulation_method = simulation_method
        self.simulation_device = simulation_device.upper()
        self.batched_shots_gpu = batched_shots_gpu
        self.simulator_seed = simulator_seed

    def get_fake_backend(self):
        try:
            from qiskit_ibm_runtime import fake_provider
        except ImportError as exc:
            raise ImportError("Install qiskit-ibm-runtime") from exc

        backend_class = getattr(fake_provider, self.backend_name, None)
        if backend_class is None:
            raise ValueError(f"Unknown IBM fake backend: {self.backend_name}")
        return backend_class()

    def make_pennylane_device(self, n_qubits):
        if self.shots is None or int(self.shots) <= 0:
            raise ValueError("Noisy simulation requires positive finite shots")

        fake_backend = self.get_fake_backend()
        available = AerSimulator().available_devices()
        if self.simulation_device not in available:
            raise RuntimeError(f"Aer {self.simulation_device} unavailable. Available devices: {available}")

        batched_gpu = (self.batched_shots_gpu and self.simulation_device == "GPU" and self.simulation_method == "statevector")
        simulator = AerSimulator.from_backend(
            fake_backend,
            method=self.simulation_method,
            device=self.simulation_device,
            batched_shots_gpu=batched_gpu,
            seed_simulator=self.simulator_seed,
            max_parallel_threads=int(os.environ.get("AER_NUM_THREADS", str(os.cpu_count() or 1),)),
            max_parallel_experiments=0,
            max_parallel_shots=1,
        )

        dev = qml.device("qiskit.aer", wires=n_qubits, shots=int(self.shots), backend=simulator,)
        dev = enable_qiskit_routing(dev, seed=self.simulator_seed)
        log.info("[QUANTUM] backend=%s method=%s device=%s shots=%s", self.backend_name, self.simulation_method, self.simulation_device, self.shots,)
        return dev


def make_quantum_device(n_qubits: int, shots: int | None = None, use_ibm_fake_noise: bool = False, ibm_fake_backend_name: str = "FakeSherbrooke", simulator_seed: int | None = None):
    if use_ibm_fake_noise:
        factory = IBMFakeNoiseFactory(backend_name=ibm_fake_backend_name, shots=shots, simulation_method=NOISE_SIMULATION_METHOD, simulation_device=NOISE_SIMULATION_DEVICE, batched_shots_gpu=NOISE_BATCHED_SHOTS_GPU, simulator_seed=simulator_seed,)
        return factory.make_pennylane_device(n_qubits)

    return qml.device(QUANTUM_BACKEND_NAME, wires=n_qubits, shots=shots,)
