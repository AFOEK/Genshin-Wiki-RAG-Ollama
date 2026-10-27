import torch
import os

from pathlib import Path

SEEDS = [42, 72, 102, 114, 202, 568, 854, 902, 1096, 2026] #42, 72, 102, 114, 202, 568, 854, 902, 1096, 2026
MODEL_NAME = "Snowflake/snowflake-arctic-embed-l-v2.0" #Snowflake/snowflake-arctic-embed-l-v2.0, sentence-transformers/all-MiniLM-L6-v2
DATASET_MODE = "genshin"  # "genshin" or "wikiqa"
GENSHIN_SOURCE_TRAIN_PATH = "/home/fmontalfu/kaggle/input/datasets/afoek88/genshin-impact-peft-dataset/genshin_lora_train.jsonl"
GENSHIN_SOURCE_VALIDATION_PATH = "/home/fmontalfu/kaggle/input/datasets/afoek88/genshin-impact-peft-dataset/genshin_lora_val.jsonl"
GENSHIN_RETRIEVAL_DIR = Path("/home/fmontalfu/kaggle/working/genshin_retrieval")
GENSHIN_RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)
GENSHIN_TRAIN_PATH = str(GENSHIN_RETRIEVAL_DIR / "train.jsonl")
GENSHIN_VALIDATION_PATH = str(GENSHIN_RETRIEVAL_DIR / "validation.jsonl")
REBUILD_GENSHIN_RETRIEVAL = False
OUTPUT_DIR = "/home/fmontalfu/kaggle/working"
PLOT_DIR = Path(OUTPUT_DIR) / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
CIRCUIT_DIR = Path(OUTPUT_DIR) / "quantum_circuits"
CIRCUIT_DIR.mkdir(parents=True, exist_ok=True)
POOLED_ADAPTER_MODES = ["classical_lora", "classical_dvora", "hq_lora", "hq_dvora"]
MODULE_ADAPTER_MODES = ["module_hq_lora", "module_hq_dvora"]
CLASSICAL_ADAPTER_MODES = ["classical_lora", "classical_dvora"]
QUANTUM_POOLED_ADAPTER_MODES = ["hq_lora", "hq_dvora"]
QUANTUM_MODULE_ADAPTER_MODES = ["module_hq_lora", "module_hq_dvora"]
ALL_ADAPTER_MODES = POOLED_ADAPTER_MODES + MODULE_ADAPTER_MODES
ADAPTER_MODE_GROUPS = {
    "classical": CLASSICAL_ADAPTER_MODES,
    "quantum": QUANTUM_POOLED_ADAPTER_MODES,
    "quantum_module": QUANTUM_MODULE_ADAPTER_MODES,
    "module_quantum": QUANTUM_MODULE_ADAPTER_MODES,
    "all": ALL_ADAPTER_MODES,
}
VALID_ADAPTER_MODES = set(ALL_ADAPTER_MODES) | set(ADAPTER_MODE_GROUPS.keys())
MODULE_TARGET_PRESETS = {
    "query": [
        "attention.self.query",
    ],
    "key": [
        "attention.self.key",
    ],
    "value": [
        "attention.self.value",
    ],
    "qv": [
        "attention.self.query",
        "attention.self.value",
    ],
    "qkv": [
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
    ],
    "attention_output": [
        "attention.output.dense",
    ],
    "attention": [
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        "attention.output.dense",
    ],
    "mlp_intermediate": [
        "intermediate.dense",
    ],
    "mlp_output": [
        "output.dense",
    ],
    "mlp": [
        "intermediate.dense",
        "output.dense",
    ],
    "all_transformer_linear": [
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        "attention.output.dense",
        "intermediate.dense",
        "output.dense",
    ],
}
MODULE_TOKEN_POLICY = "cls" #all, cls
MODULE_TARGET_PRESET_RUNS = ["qkv",] #"query", "value", "qv", "qkv", "attention", "mlp",
MODULE_LAYER_POLICY = "last"  # "all", "last", "last_k", "selected"
MODULE_LAST_K = 4
MODULE_SELECTED_LAYERS = [21, 22, 23]
ADAPTER_MODE = "all" # all, classical, quantum, quantum_module, classical_lora, classical_dvora, hq_lora, hq_dvora, module_hq_lora, module_hq_dvora
RANK = 4
QUANTUM_LAYERS = 1
ALPHA = 4.0
DROPOUT = 0.1
QUANTUM_DROPOUT = 0.15
SHOTS = None #None | 1024
DIFF_METHOD = "parameter-shift" #adjoint | parameter-shift
GRADIENT_ACCUMULATION_STEPS = 2
BATCH_SIZE = 8
EPOCHS = 20 #10 >= x <= 25
HYBRID_LR = LR = 1e-4
VQC_LR = 5e-5 #1e-4 | 5e-5
CONTROL_LR = 5e-5 #1e-4 | 5e-5
HYBRID_MIN_LR = MIN_LR = 1e-6
VQC_MIN_LR = 5e-7
CONTROL_MIN_LR = 5e-7
HYBRID_WEIGHT_DECAY = 0.01
VQC_WEIGHT_DECAY = 0.0
CONTROL_WEIGHT_DECAY = 0.0
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1e-8
MAX_GRAD_NORM = 2.0
WARMUP_RATIO = 0.05
LORA_DOWN_INIT_STD = 0.02
LORA_UP_INIT_STD = 1e-3
QUANTUM_INIT_STD = 0.01
PATIENCE = 4
USE_LR_SCHEDULER = True
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 1
LR_SCHEDULER_THRESHOLD = 1e-3
USE_EMBEDDING_CACHE = True
REBUILD_EMBEDDING_CACHE = True
USE_MODULE_PREFIX_CACHE = True
REBUILD_MODULE_PREFIX_CACHE = True
MODULE_CACHE_BATCH_SIZE = 16
MODULE_CACHE_DIR = Path("/home/fmontalfu/kaggle/working/module_prefix_cache")
MODULE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODULE_ADAPTER_DEVICE = torch.device("cpu")
CPU_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))
QUERY_MAX_LENGTH = 80
DOCUMENT_MAX_LENGTH = 320
CACHE_BATCH_SIZE = 64
CACHE_DIR = Path("/home/fmontalfu/kaggle/working/embedding_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_CACHE_PATH = CACHE_DIR / "snowflake_genshin_train.pt"
VALIDATION_CACHE_PATH = CACHE_DIR / "snowflake_genshin_validation.pt"
MIN_DELTA = 1e-3
TEMPERATURE = 0.05
MAX_LENGTH = 256 # 256 | 512
QUANTUM_BACKEND_NAME = "lightning.qubit"
ENCODER_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUANTUM_DEVICE = torch.device("cpu")
USE_IBM_FAKE_NOISE = True # Set to true for QC noise
IBM_FAKE_BACKEND_NAME = "FakeSherbrooke"
NOISY_SHOTS = 1024
NOISY_DIFF_METHOD = "parameter-shift" # adjoint or parameter-shift
NOISE_SIMULATION_METHOD = "density_matrix"
NOISE_SIMULATION_DEVICE = "CPU"
NOISE_BATCHED_SHOTS_GPU = True
ACTIVE_SHOTS = NOISY_SHOTS if USE_IBM_FAKE_NOISE else SHOTS
ACTIVE_DIFF_METHOD = NOISY_DIFF_METHOD if USE_IBM_FAKE_NOISE else DIFF_METHOD
CPU_THREADS = int(os.environ.get("OMP_NUM_THREADS", "128"))
MODEL_PROFILES = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "pooling_mode": "mean",
        "query_prefix": "",
        "document_prefix": "",
        "max_length": MAX_LENGTH,
        "model_kwargs": {},
    },
    "Snowflake/snowflake-arctic-embed-l-v2.0": {
        "pooling_mode": "cls",
        "query_prefix": "query: ",
        "document_prefix": "",
        "max_length": MAX_LENGTH,
        "model_kwargs": {"add_pooling_layer": False},
    },
}