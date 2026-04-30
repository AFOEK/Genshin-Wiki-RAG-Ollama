# Genshin Impact Wiki Ollama RAG (Retrieval-Augmented Generation)

## Introduction
With the growing records [Genshin Impact](https://genshin.hoyoverse.com/en/home?utm_source=fab&utm_medium=home) charaters, weapons, artifacts, books, environments, and items. Therefore with advancing of LLM (Large Language Model), this project utilized Llama3.2:3b (Llama 3.2 with 3 billions parameters) and MiniLM Embedding (all-minilm). This pipeline run using 3 main stacks:
- Python 3.14.5
- Sqlite3 3.46.1
- FAISS (Facebook AI Similarity Search)

This project data sources are pulled from [Genshin Impact Fandom Wiki](https://genshin-impact.fandom.com/wiki/Genshin_Impact_Wiki), [Keqing Main Theory Crafting](https://keqingmains.com/), [Game8](https://game8.co/games/Genshin-Impact), [Genshin GG](https://genshin.gg/) and [HoneyHunter](https://gensh.honeyhunterworld.com/?lang=EN), by using either API, Github repository and web scraping.

## Dependency
Before start to use this repos, this project required some packages (Debian based build):
```
sudo apt install git curl ca-certificates python3 python3-venv python3-pip sqlite3 build-essential pkg-config libxml2-dev libxslt1-dev liblz4-dev zlib1g-dev libffi-dev libssl-dev unzip jq libvulkan-dev glslc libopenblas-dev -y
```
After that clone this github:
```
git clone https://github.com/AFOEK/Genshin-Wiki-RAG-Ollama.git && cd Genshin-Wiki-RAG-Ollama
```
setup python3 virtual environment:
```
python3 -m venv .venv
```
and start the virtual environments:
```
source .venv/bin/activate
```
With python venv already activated install all python requirements by running:
```
pip install -r requirements.txt
```

### Ollama server setup
This project need Ollama server to assist on embeddings, for Linux installation:
```
curl -fsSL https://ollama.com/install.sh | sh
```
Enable Vulkan support:
```
sudo systemctl edit ollama.service
```
Put one line override:
```
[Service]
Environment="OLLAMA_VULKAN=1"
```
Reload the Ollama service:
```
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

For Windows installation:
```
irm https://ollama.com/install.ps1 | iex
```
For add Vulkan support by set user enviroment variable:
```
OLLAMA_VULKAN=1
```
Then close Ollama from the system tray and start it again.
After that pull appropiate Ollama model:
```
# Chat model (pick one)
ollama pull llama3.2:3b
# Embedding model
ollama pull all-minilm
```

> [!TIP]
The model isn't fix, this project can use other model depends with users requirements. If pulled model is differs with guide above, user need to change [config.yaml](rag/config.yaml) model params.

### Llama.cpp server setup
Clone the llama.cpp repository:
```
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
```
CPU build only:
```
cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
cmake --build build --config Release -j"$(nproc)"
```
Vulkan support:
```
cmake -B build -DGGML_VULKAN=1
cmake --build build --config Release -j"$(nproc)"
```
CUDA support:
```
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j"$(nproc)"
```
ARM® KleidiAI support [^2]:
```
cmake -B build -DGGML_CPU_KLEIDIAI=ON
cmake --build build --config Release -j"$(nproc)"
```
Combined build:
```
cmake -B build -DGGML_CPU_KLEIDIAI=ON -DGGML_VULKAN=1 -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
cmake --build build --config Release -j"$(nproc)"
```
> [!TIP]
For llama.cpp windows build for Vulkan or CUDA support are more tricky, it's better just build CPU only (without BLAS). More info how to install Llama.cpp using Vulkan or OpenBLAS can be seen in official [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) github guide.

### FAISS Installation
There are 2 options for installing FAISS:
- python pip installation
- source build

Both of the method are good to make it works, but this project preferred using source build FAISS.

#### Python FAISS installation
```
pip install -U faiss-cpu
```

#### Source build FAISS installation
Before running the installation make sure all packages are installed (Debian based build):
```
sudo apt update && sudo apt install -y git cmake build-essential pkg-config python3-dev python3-venv libopenblas-dev liblapack-dev swig
```
clone FAISS github:
```
git clone https://github.com/facebookresearch/faiss.git && cd faiss
```
and build the FAISS itself:
```
cmake -B build -DFAISS_ENABLE_PYTHON=ON -DFAISS_ENABLE_GPU=OFF -DBUILD_TESTING=OFF -DFAISS_ENABLE_C_API=OFF -DFAISS_ENABLE_OPENMP=ON -DBLA_VENDOR=OpenBLAS
```
if target build has GPU, FAISS itself support GPU by turn on:
```
...
-DFAISS_ENABLE_GPU=ON 
...
```
Run the build script by executing:
```
cmake --build build -j4
```
after it finish, build the python library:
```
cd build/faiss/python && pip install .
```
Sanity check for installation can be done by executing:
```
python -c "import faiss, numpy as np; print('faiss ok'); print('version:', getattr(faiss,'version','(no version)'))"
```

## Windows
> [!CAUTION]
Windows FAISS build is fragile especially for FAISS-GPU build

Before using the python script, make sure `Winget` already installed by running:
```
winget -v
```
After that install MiniConda:
```
winget install Anaconda.MiniConda3
winget install SQLite.SQLite
```
Make sure that `conda` is recognized by the powershell by running:
```
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" init powershell
```
Then create virtual environment using `conda`:
```
conda create -n rag python=3.12.4
```
activate the virtual environment:
```
conda activate rag
```
Install FAISS first:
```
# FAISS CPU only
conda install -c conda-forge faiss-cpu
```
or
```
# FAISS GPU Support
conda install -c pytorch faiss-gpu
```
Finally install the rest python dependency:
```
pip install -r requirements.txt
```

## Configuration
> [!WARNING]
Don't change sources configuration params, since some of the sources may or may not have leaked or datamined game data.

Before using the python script, inside [config.yaml](rag/config.yaml) there are some config can be changed, like `storage` path, `ollama` models, `faiss` index and metrics, `pipeline` chunks size, and `threading` workers. For other configurations are the best let them be, since it govern the data sources, and filters for `datamined` or `leaked` game informations.

## Usage
The main entry of the script is `main.py`, in the script it has options can be used, below is the default values:
```
--DB_CRAWL=True
--DB_AUDIT=True
--DB_REPAIR=False 
--FAISS_MIGRATE=False 
--FAISS_AUDIT=False
--FAISS_OVERWRITE=False
--FTS_SYNC=True
--FTS_INIT=False
--FTS_REBUILD=False
--BACKENDS=ollama   #options: ollama, llamacpp, llamma.cpp
```
Where `--DB_CRAWL` it will pull all the data from all datasource and store the embeddings inside Sqlite3, `--DB_AUDIT` it will check if the datasource is properly processed, `--DB_REPAIR` it repair missing embedding chunks or missing active chunks, `--FAISS_MIGRATE` it migrate the embedding vectors from Sqlite3 to FAISS, `--FAISS_AUDIT` it will check if the embedding is properly processed, `--FAISS_OVERWRITE` it will overwrite current FAISS vector database records, `--FTS_SYNC` it sync newly added or changed lexical source to `FST5/BM25` records, `--FTS_INIT` it uses for first time clean run assume taht previous run don't has `FTS5`, `--FTS_REBUILD` it force rebuild `FTS5` records and `--BACKENDS` it will pick backend type according user input.

> [!WARNING]
Running `--FTS_REBUILD` will take along time, it may or may not require 2-3 days to build it. Depends with hardware I/O and CPU clocks.

```
# Crawl + DB Audit
python3 rag/main.py --DB_CRAWL=True --DB_AUDIT=True --FAISS_MIGRATE=False --FAISS_AUDIT=False --FTS_SYNC=True --BACKEND=ollama
```
```
# Migrate + Audit FAISS
python3 rag/main.py --DB_CRAWL=False --DB_AUDIT=False --FAISS_MIGRATE=True --FAISS_AUDIT=True --BACKEND=ollama
```
```
# DB Repair + DB Audit
python3 rag/main.py --DB_CRAWL=False --DB_AUDIT=True --DB_REPAIR=True --FTS_SYNC=True --BACKEND=ollama
```
```
# Full pipeline
python3 rag/main.py --DB_CRAWL=True --DB_AUDIT=True --DB_REPAIR=True --FAISS_MIGRATE=True --FAISS_AUDIT=True --FAISS_OVERWRITE=True --FTS_INIT=True --BACKEND=ollama
```

## QnA Test
If all crawling, chunking, and embedding have done, user can test the RAG by running:
```
python3 rag/test.py --question "<YOUR_TEST_QUESTIONS>"
```
It can recieve query and generate output depends what user ask. In the `test.py` script has multiple flags such as:
```
--config rag/config.yaml                        #default value: rag/config.yaml
--retriever {sqlite, faiss, bm25, and hybrid}   #default value: hybrid
--direct_top_k 8-32                             #default value: 12
--board_top_k 50-80                             #default value: 60
--summarize_batch_size 4-16                     #default value: 8
--backend {ollama, llamacpp}                    #default value: ollama
```

Example usage:
```
python3 rag/test.py --retriever hybrid --backend ollama --direct_top_k 20 --question "What is ZhongLi signature weapon?"
```

Aside of CLI flags some settings are controlled by [config.yaml](rag/config.yaml), where it control:
- `cross_encoder`
- `context_expansion`
The values can be changed.

> [!INFO]
In reranker `cross_encoder_model` value, it must be string and it's not a generic ollama model or llama.cpp `.gguf` model. It require `sentence_transformers` model.

## Kaggle Embedding Support
Since local device may have limited computing power, free up that computing power for other task, or just try embedding to bigger or better embedding models with Kaggle T4 Nvidia GPU. With that this project utilized `Kaggle API` to send chunks to Kaggle, where it will be embeded to bigger model, by using [upload.py](kaggle_tools/upload.py) script.

### Kaggle setups
In order to access Kaggle using API, it required API credentials. It can be get by login to [Kaggle](www.kaggle.com/settings) -> Account -> Legacy Token Credential -> Create Legacy API Key. it will download a json file named `kaggle.json`, which it need to be place in `~/.kaggle/` or `C:\Users\<YOUR_USERNAME>\.kaggle\`

> [!TIP]
To access Kaggle API, it can use API Tokens but for the sake of simplicity , this guide use legacy API credential

### Upload to kaggle
After Kaggle API has been stored, before upload to the Kaggle itself. It required to run before hand:
```
python3 kaggle_tools/extract_chunks.py
```
It will extract all documents chunks and export it to `chunks.jsonl`. After that run:
```
python3 kaggle_tools/upload.py --dataset_slug <YOUR_KAGGLE_USERNAME>/<GENSHIN_CHUNKS_NAME>
```

## One-for-all script
This project provide a script to run as cron job or general usages, the script itself named [`./run_pipeline.sh`](./run_pipeline.sh). To setup the cron job:
```
crontab -e #choose your favorite text editor (use nano)

##Add to a new line assume that the repos is on Documents
0 3 1,15 * * CRON_MODE=1 /home/<YOUR_USERNAME>/Documents/Genshin-Wiki-RAG-Ollama/run_pipeline.sh >> /home/<YOUR_USERNAME>/Documents/Genshin-Wiki-RAG-Ollama/rag/logs/pipeline_run.log 2>&1
```

The script will run at 1st and 15th day every month at 3am.

> [!TIP]
The crontab entry can be change `0 3 1,15 * *` from the left to right order it detonates `minute hour day_of_month month day_of_week`. `CRON_MODE=1` is a lock so if 2 instance of same script running it will block.

## QLoRA / LoRA and DoRA fine-tuning
Since current project state is on crawling and embedding all the game data, isn't possible to do fine tuning, although it will be Q/LoRA (Quantization /Low-rank adaptation) or Q/DoRA (Quantization/Weight-Decomposed Low-Rank Adaptation) fine-tuning planned. This fine tuning aim for better answering, reduce hallucinations, targeted cite, and preparing for embedding fine tuning.

## To-do list
- [ ] JSONL for Q/LoRA (Quantization Low-rank adaptation) or Q/DoRA (Quantization/Weight-Decomposed Low-Rank Adaptation) fine-tuning.
- [ ] Use better generator model Llama3.2:8b, Qwen3.5:9b, Qwen 2.5:7b, Llama 3.1:8b, or Mistral 7b.
- [x] Add Vulkan support.
- [x] Llama.cpp support
- [x] Use better embedding model mixebread-ai/mxbai-embed-large-v1, BAAI/bge-large-en-v1.5, and nomic-ai/nomic-embed-text-v1.5 [^1].
- [x] Embedding using Kaggle.
- [x] Adding cron jobs updates.
- [x] Pulling from sources.
- [x] Add multithreading support.
- [x] FAISS support.

## Footenote
[^1]: It's get processed on Kaggle
[^2]: ARM architectures only