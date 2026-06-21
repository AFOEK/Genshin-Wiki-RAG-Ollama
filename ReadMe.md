# Genshin Impact Wiki Ollama RAG

A local Retrieval-Augmented Generation (RAG) pipeline for answering questions about **Genshin Impact** using crawled/wiki-style game data, local LLM backends, dense retrieval, lexical retrieval, reranking, and optional PEFT dataset generation.

The project currently supports data ingestion from multiple Genshin-related sources, stores processed chunks in SQLite, builds searchable vector indexes with FAISS or TurboVec, and queries the data through Ollama or llama.cpp.

---

## Table of Contents

- [Genshin Impact Wiki Ollama RAG](#genshin-impact-wiki-ollama-rag)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [Tech Stack](#tech-stack)
  - [Data Sources](#data-sources)
  - [Installation](#installation)
    - [Debian / Ubuntu Dependencies](#debian--ubuntu-dependencies)
    - [Clone the Repository](#clone-the-repository)
    - [Python Environment](#python-environment)
      - [Option 1: venv](#option-1-venv)
      - [Option 2: conda](#option-2-conda)
  - [Backend Setup](#backend-setup)
    - [Ollama Setup](#ollama-setup)
      - [Linux Installation](#linux-installation)
      - [Windows Installation](#windows-installation)
    - [llama.cpp Setup](#llamacpp-setup)
      - [CPU-only Build](#cpu-only-build)
      - [Vulkan Build](#vulkan-build)
      - [CUDA Build](#cuda-build)
      - [ARM KleidiAI Build](#arm-kleidiai-build)
      - [Combined Build](#combined-build)
  - [FAISS Installation](#faiss-installation)
    - [Install FAISS with pip](#install-faiss-with-pip)
    - [Build FAISS from Source](#build-faiss-from-source)
  - [Windows Setup](#windows-setup)
  - [Configuration](#configuration)
  - [Usage](#usage)
    - [Main Pipeline Flags](#main-pipeline-flags)
    - [Common Pipeline Commands](#common-pipeline-commands)
      - [Crawl + DB Audit](#crawl--db-audit)
      - [Migrate + Audit FAISS](#migrate--audit-faiss)
      - [Migrate + Audit TurboVec](#migrate--audit-turbovec)
      - [DB Repair + DB Audit](#db-repair--db-audit)
      - [Full Pipeline](#full-pipeline)
  - [Q\&A Testing](#qa-testing)
  - [Kaggle Embedding Support](#kaggle-embedding-support)
    - [Kaggle Setup](#kaggle-setup)
    - [Upload Chunks to Kaggle](#upload-chunks-to-kaggle)
  - [Automated Pipeline Script](#automated-pipeline-script)
  - [PEFT Dataset Generation](#peft-dataset-generation)
    - [Model Dependencies](#model-dependencies)
    - [Dataset Creation](#dataset-creation)
  - [To-do List](#to-do-list)
  - [Footnotes](#footnotes)

---

## Overview

Genshin Impact has a large and constantly growing set of characters, weapons, artifacts, books, regions, enemies, quests, and item records. This project builds a local RAG system over that information so a small local language model can answer game-related questions with retrieved context instead of relying only on its internal knowledge.

The default local setup uses:

- **Chat model:** `llama3.2:3b`
- **Embedding model:** `all-minilm`
- **Database:** SQLite
- **Vector search:** FAISS and/or TurboVec
- **Lexical search:** FTS5 / BM25
- **Backends:** Ollama and llama.cpp

The model choices are configurable. Larger models and stronger embedding models can be used if your hardware allows it.

---

## Features

- Multi-source Genshin data crawling.
- SQLite-based document, chunk, and embedding storage.
- FAISS dense vector search.
- TurboVec retrieval support.
- FTS5 / BM25 lexical search.
- Hybrid retrieval with dense + lexical ranking.
- Cross-encoder reranking.
- Parent-child chunk retrieval.
- Context expansion.
- Recency weighting.
- Ollama backend support.
- llama.cpp backend support.
- Kaggle-based embedding generation.
- PEFT dataset generation for SFT and retrieval-style training data.
- Cron-compatible pipeline runner.

---

## Tech Stack

Tested project stack:

| Component | Version / Tool |
|---|---|
| Python | 3.13.13 |
| SQLite | 3.46.1 |
| FAISS | 1.13.2 |
| Local LLM backend | Ollama / llama.cpp |
| Embedding backend | Ollama / external embedding models |
| Retrieval | SQLite, FAISS, BM25, hybrid, TurboVec |

---

## Data Sources

This project pulls data from the following sources:

- [Genshin Impact Fandom Wiki](https://genshin-impact.fandom.com/wiki/Genshin_Impact_Wiki)
- [KeqingMains Theorycrafting](https://keqingmains.com/)
- [Game8 Genshin Impact](https://game8.co/games/Genshin-Impact)
- [Genshin.gg](https://genshin.gg/)
- [Honey Hunter World](https://gensh.honeyhunterworld.com/?lang=EN)

Depending on the source, data may be collected through APIs, GitHub repositories, or web scraping.

> [!WARNING]
> Do not casually change source filtering settings. Some Genshin-related websites may contain leaked, datamined, beta, or unreleased content. The configuration includes filters intended to reduce that risk.

---

## Installation

### Debian / Ubuntu Dependencies

Install the required system packages:

```bash
sudo apt update
sudo apt install -y git curl ca-certificates python3 python3-venv python3-pip sqlite3 build-essential pkg-config libxml2-dev libxslt1-dev liblz4-dev zlib1g-dev libffi-dev libssl-dev unzip jq libvulkan-dev glslc libopenblas-dev
```

### Clone the Repository

```bash
git clone https://github.com/AFOEK/Genshin-Wiki-RAG-Ollama.git
cd Genshin-Wiki-RAG-Ollama
```

### Python Environment

You can use either `venv` or `conda`.

#### Option 1: venv

Create the environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

#### Option 2: conda

Create the environment from `environment.yml`:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate rag
```

---

## Backend Setup

### Ollama Setup

This project can use an Ollama server for chat generation and embeddings.

#### Linux Installation

Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Optional: enable Vulkan support.

```bash
sudo systemctl edit ollama.service
```

Add this override:

```ini
[Service]
Environment="OLLAMA_VULKAN=1"
```

Reload and restart Ollama:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

Pull the default models:

```bash
# Chat model
ollama pull llama3.2:3b

# Embedding model
ollama pull all-minilm
```

#### Windows Installation

Install Ollama from PowerShell:

```powershell
irm https://ollama.com/install.ps1 | iex
```

Optional: enable Vulkan support by setting a user environment variable:

```powershell
OLLAMA_VULKAN=1
```

Then close Ollama from the system tray and start it again.

Pull the default models:

```powershell
# Chat model
ollama pull llama3.2:3b

# Embedding model
ollama pull all-minilm
```

> [!TIP]
> The model is not fixed. You can use other Ollama models depending on your hardware and accuracy requirements. If you use different models, update the model parameters in [`rag/config.yaml`](rag/config.yaml).

---

### llama.cpp Setup

Clone the llama.cpp repository:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

#### CPU-only Build

```bash
cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
cmake --build build --config Release -j"$(nproc)"
```

#### Vulkan Build

```bash
cmake -B build -DGGML_VULKAN=1
cmake --build build --config Release -j"$(nproc)"
```

#### CUDA Build

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j"$(nproc)"
```

#### ARM KleidiAI Build

```bash
cmake -B build -DGGML_CPU_KLEIDIAI=ON
cmake --build build --config Release -j"$(nproc)"
```

#### Combined Build

```bash
cmake -B build -DGGML_CPU_KLEIDIAI=ON -DGGML_VULKAN=1 -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
cmake --build build --config Release -j"$(nproc)"
```

> [!TIP]
> Building llama.cpp with Vulkan or CUDA on Windows can be more fragile than Linux. For Windows, a CPU-only build is usually simpler. See the official [llama.cpp build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) for more details.

---

## FAISS Installation

There are two installation options:

1. Install FAISS from Python packages.
2. Build FAISS from source.

This project prefers a source build when possible, but the pip version is simpler and usually enough for CPU-only use.

### Install FAISS with pip

```bash
pip install -U faiss-cpu
```

### Build FAISS from Source

Install build dependencies:

```bash
sudo apt update
sudo apt install -y git cmake build-essential pkg-config python3-dev python3-venv libopenblas-dev liblapack-dev swig
```

Clone FAISS:

```bash
git clone https://github.com/facebookresearch/faiss.git
cd faiss
```

Configure a CPU build:

```bash
cmake -B build -DFAISS_ENABLE_PYTHON=ON -DFAISS_ENABLE_GPU=OFF -DBUILD_TESTING=OFF -DFAISS_ENABLE_C_API=OFF -DFAISS_ENABLE_OPENMP=ON -DBLA_VENDOR=OpenBLAS
```

For GPU support, change:

```bash
-DFAISS_ENABLE_GPU=ON
```

Build FAISS:

```bash
cmake --build build -j4
```

Install the Python bindings:

```bash
cd build/faiss/python
pip install .
```

Run a sanity check:

```bash
python -c "import faiss, numpy as np; print('faiss ok'); print('version:', getattr(faiss, 'version', '(no version)'))"
```

---

## Windows Setup

> [!CAUTION]
> Building FAISS on Windows can be fragile, especially for FAISS-GPU. Conda installation is recommended.

Check that `winget` is installed:

```powershell
winget -v
```

Install Miniconda and SQLite:

```powershell
winget install Anaconda.MiniConda3
winget install SQLite.SQLite
```

Initialize conda for PowerShell:

```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" init PowerShell
```

Create a conda environment:

```powershell
conda create -n rag python=3.13.4
```

Activate it:

```powershell
conda activate rag
```

Install FAISS:

```powershell
# CPU-only FAISS
conda install -c conda-forge faiss-cpu
```

Or, for GPU support:

```powershell
# GPU FAISS
conda install -c pytorch faiss-gpu
```

Install the remaining Python dependencies:

```powershell
pip install -r requirements.txt
```

---

## Configuration

The main configuration file is:

```text
rag/config.yaml
```

Configurable sections include:

- Storage paths.
- Ollama models.
- llama.cpp backend settings.
- FAISS index type and search metrics.
- Pipeline chunk size and overlap.
- Threading worker counts.
- Retrieval settings.
- Cross-encoder settings.
- Context expansion settings.
- Parent-child retrieval settings.
- Dataset generation settings.

> [!WARNING]
> Avoid changing source-level filtering unless you know what you are doing. These settings help control which sources are crawled and help filter leaked, beta, unreleased, or datamined information.

---

## Usage

The main pipeline entry point is:

```bash
python3 rag/main.py
```

### Main Pipeline Flags

Default-style pipeline options:

```bash
--DB_CRAWL=True
--DB_AUDIT=True
--DB_REPAIR=False
--FAISS_MIGRATE=False
--FAISS_AUDIT=False
--FAISS_OVERWRITE=False
--TURBOVEC_MIGRATE=False
--TURBOVEC_AUDIT=False
--TURBOVEC_OVERWRITE=False
--FTS_SYNC=True
--FTS_INIT=False
--FTS_REBUILD=False
--PARENT_REBUILD=False
--PARENT_SYNC=False
--PARENT_INIT=False
--BACKEND=ollama
```

| Flag | Purpose |
|---|---|
| `--DB_CRAWL` | Pull data from configured sources and store processed records in SQLite. |
| `--DB_AUDIT` | Check whether sources, documents, chunks, and embeddings were processed correctly. |
| `--DB_REPAIR` | Repair missing embeddings, missing chunks, or missing active records. |
| `--FAISS_MIGRATE` | Migrate embedding vectors from SQLite to a FAISS index. |
| `--FAISS_AUDIT` | Check whether FAISS records were processed correctly. |
| `--FAISS_OVERWRITE` | Overwrite the current FAISS vector database records. |
| `--TURBOVEC_MIGRATE` | Migrate embeddings to TurboVec. |
| `--TURBOVEC_AUDIT` | Audit TurboVec records. |
| `--TURBOVEC_OVERWRITE` | Overwrite the current TurboVec index records. |
| `--FTS_SYNC` | Sync newly added or changed lexical records to FTS5 / BM25. |
| `--FTS_INIT` | Initialize FTS5 for a first-time clean run. |
| `--FTS_REBUILD` | Force rebuild all FTS5 records. |
| `--PARENT_REBUILD` | Force rebuild all parent-child SQLite pairs. |
| `--PARENT_INIT` | Initialize parent-child pairs for a first-time clean run. |
| `--PARENT_SYNC` | Sync newly added or changed records to parent-child pairs. |
| `--BACKEND` | Select the backend, such as `ollama` or `llamacpp`. |

> [!WARNING]
> Running `--FTS_REBUILD` can take a long time. Depending on hardware I/O and CPU speed, it may require multiple days.

---

### Common Pipeline Commands

#### Crawl + DB Audit

```bash
python3 rag/main.py --DB_CRAWL=True --DB_AUDIT=True --FAISS_MIGRATE=False --FAISS_AUDIT=False --FTS_SYNC=True --PARENT_SYNC=True --TURBOVEC_MIGRATE=False --TURBOVEC_OVERWRITE=False --TURBOVEC_AUDIT=False --BACKEND=ollama
```

#### Migrate + Audit FAISS

```bash
python3 rag/main.py --DB_CRAWL=False --DB_AUDIT=False --FAISS_MIGRATE=True --FAISS_AUDIT=True --BACKEND=ollama
```

#### Migrate + Audit TurboVec

```bash
python3 rag/main.py --DB_CRAWL=False --DB_AUDIT=False --TURBOVEC_MIGRATE=True --TURBOVEC_OVERWRITE=True --TURBOVEC_AUDIT=True
```

#### DB Repair + DB Audit

```bash
python3 rag/main.py --DB_CRAWL=False --DB_AUDIT=True --DB_REPAIR=True --FTS_SYNC=True --PARENT_SYNC=True --TURBOVEC_MIGRATE=True --TURBOVEC_OVERWRITE=True --TURBOVEC_AUDIT=True --BACKEND=ollama
```

#### Full Pipeline

```bash
python3 rag/main.py --DB_CRAWL=True --DB_AUDIT=True --DB_REPAIR=True --FAISS_MIGRATE=True --FAISS_AUDIT=True --FAISS_OVERWRITE=True --TURBOVEC_MIGRATE=True --TURBOVEC_OVERWRITE=True --TURBOVEC_AUDIT=True --FTS_SYNC=True --PARENT_SYNC=True --BACKEND=ollama
```

---

## Q&A Testing

After crawling, chunking, embedding, and index migration are complete, test the RAG system with:

```bash
python3 rag/test.py --question "<YOUR_TEST_QUESTION>"
```

Example:

```bash
python3 rag/test.py --retriever hybrid --backend ollama --direct_top_k 20 --question "What is Zhongli's signature weapon?"
```

Available options include:

```bash
--config rag/config.yaml
--retriever {sqlite,faiss,bm25,hybrid,turbovec,hybrid_turbovec}
--direct_top_k 8-32
--board_top_k 50-80
--summarize_batch_size 4-16
--backend {ollama,llamacpp}
```

Default values:

| Option | Default |
|---|---|
| `--config` | `rag/config.yaml` |
| `--retriever` | `hybrid` |
| `--direct_top_k` | `12` |
| `--board_top_k` | `60` |
| `--summarize_batch_size` | `8` |
| `--backend` | `ollama` |

Additional retrieval behavior is controlled in [`rag/config.yaml`](rag/config.yaml), including:

- `cross_encoder`
- `context_expansion`
- `retrieval`
- `parent_child`

> [!NOTE]
> The `cross_encoder_model` value must be a valid `sentence_transformers` model string. It is not a generic Ollama model name and it is not a llama.cpp `.gguf` model.

---

## Kaggle Embedding Support

Local embedding generation can be slow or limited by hardware. This project includes Kaggle support so chunks can be embedded using larger models on a Kaggle T4 GPU.

The Kaggle upload workflow uses:

```text
kaggle_tools/upload.py
```

### Kaggle Setup

To use the Kaggle API, create API credentials:

1. Log in to Kaggle.
2. Go to account settings.
3. Create a legacy API key.
4. Download `kaggle.json`.
5. Place it in one of these locations:

Linux:

```text
~/.kaggle/kaggle.json
```

Windows:

```text
C:\Users\<YOUR_USERNAME>\.kaggle\kaggle.json
```

> [!TIP]
> Kaggle API tokens can also be used, but this guide uses the legacy API credential for simplicity.

### Upload Chunks to Kaggle

Extract chunks first:

```bash
python3 kaggle_tools/extract_chunks.py
```

Then upload them:

```bash
python3 kaggle_tools/upload.py --dataset_slug <YOUR_KAGGLE_USERNAME>/<GENSHIN_CHUNKS_NAME>
```

---

## Automated Pipeline Script

This project provides a general-purpose pipeline runner:

```bash
./run_pipeline.sh
```

It can be used manually or scheduled with cron.

Open your crontab:

```bash
crontab -e
```

Example cron job:

```cron
0 3 1,15 * * CRON_MODE=1 /home/<YOUR_USERNAME>/Documents/Genshin-Wiki-RAG-Ollama/run_pipeline.sh >> /home/<YOUR_USERNAME>/Documents/Genshin-Wiki-RAG-Ollama/rag/logs/pipeline_run.log 2>&1
```

This runs the pipeline at **3:00 AM on the 1st and 15th day of every month**.

> [!TIP]
> Cron format is:
>
> ```text
> minute hour day_of_month month day_of_week
> ```
>
> `CRON_MODE=1` acts as a lock so multiple instances of the same script do not run at the same time.

---

## PEFT Dataset Generation

The project currently focuses on crawling, chunking, indexing, and retrieval. It also includes planned and experimental support for creating training datasets for PEFT-style fine-tuning.

Planned or supported PEFT methods include:

- QLoRA: Quantized Low-Rank Adaptation.
- LoRA: Low-Rank Adaptation.
- QDoRA: Quantized Weight-Decomposed Low-Rank Adaptation.
- DoRA: Weight-Decomposed Low-Rank Adaptation.
- VeRA: Vector-based Random Matrix Adaptation.
- DVoRA: Weight-Decomposed Vector-based Random Matrix Adaptation.

The goal of fine-tuning is to:

- Improve answer quality.
- Reduce hallucinations.
- Improve citation targeting.
- Prepare for embedding fine-tuning.
- Improve domain-specific Genshin responses.

### Model Dependencies

Dataset generation uses three model roles configured in [`rag/config.yaml`](rag/config.yaml):

- `draft_model`
- `answer_model`
- `validation_model`

These models are used to generate:

- Supervised fine-tuning pairs.
- Hard negative pairs.
- Rejected examples.
- Answerability examples.

Example model pulls:

```bash
ollama pull gemma3:12b
ollama pull qwen3:8b
```

> [!WARNING]
> Make sure the model names in [`rag/config.yaml`](rag/config.yaml) match the models installed in Ollama. Larger models may produce better datasets, but they are not strictly required.

---

### Dataset Creation

Dataset generation is handled by:

```text
fine_tune/dataset_creation.py
```

The script is controlled by the `dataset_creation` section in [`rag/config.yaml`](rag/config.yaml). It supports multithreading through `workers` and `max_inflight`.

The `limit` value controls how many dataset examples are created.

Supported command-line arguments:

```bash
--config
--db
--out
--ollama-url
--model
--limit
--qa-per-chunk
--min-chars
--max-chars
--seed
--sources
--sleep
```

Generated files include:

| File | Purpose |
|---|---|
| `genshin_double_negative_pairs.jsonl` | Double-negative retrieval pairs. |
| `genshin_rag_sft_candidates.jsonl` | Candidate supervised fine-tuning examples. |
| `genshin_rejected.jsonl` | Rejected generated examples. |
| `genshin_retrieval_pairs.jsonl` | Retrieval training pairs. |
| `genshin_sft_negative_answerability.jsonl` | Negative answerability examples. |

The following files can be used for PEFT training:

- `genshin_double_negative_pairs.jsonl`
- `genshin_rag_sft_candidates.jsonl`
- `genshin_retrieval_pairs.jsonl`

The following files are mainly for filtering, debugging, or validation:

- `genshin_rejected.jsonl`
- `genshin_sft_negative_answerability.jsonl`

Example command:

```bash
python3 fine_tune/dataset_creation.py --model ollama --limit 10 --qa-per-chunk 5 --seed 102 --sources "genshin_wiki,kqm_tcl,kqm_news,honey,genshin_gg,game8"
```

---

## To-do List

- [x] Create JSONL datasets for QLoRA / QDoRA-style fine-tuning.
- [ ] Test stronger generator models such as Llama 3.1 8B, Llama 3.2 8B, Qwen 2.5 7B, Qwen 3.5 9B, or Mistral 7B.
- [x] Add Vulkan and accelerator support.
- [x] Add llama.cpp support.
- [x] Add support for stronger embedding models such as `mixedbread-ai/mxbai-embed-large-v1`, `BAAI/bge-large-en-v1.5`, and `nomic-ai/nomic-embed-text-v1.5`.[^1]
- [x] Add Kaggle embedding workflow.
- [x] Add cron job updates.
- [x] Pull from configured sources.
- [x] Add multithreading support.
- [x] Add FAISS support.
- [x] Add FTS5 / BM25 support.
- [x] Add cross-encoder support.
- [x] Add context expansion support.
- [x] Add reranker support.
- [x] Add dense similarity search.
- [x] Add hybrid BM25 + FAISS reranking.
- [x] Add TurboVec reranked support.[^4]
- [x] Add recency weighting.[^3]
- [x] Add parent-child retrieval.
- [ ] Add retriever cache layer.

---

## Footnotes

[^1]: Processed on Kaggle.
[^2]: ARM architectures only.
[^3]: Partial support; needs more testing.
[^4]: Beta version.
