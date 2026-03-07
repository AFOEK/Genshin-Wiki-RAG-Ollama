# Genshin Impact Wiki Ollama RAG (Retrieval-Augmented Generation)

## Introduction
With the growing records [Genshin Impact](https://genshin.hoyoverse.com/en/home?utm_source=fab&utm_medium=home) charaters, weapons, artifacts, books, environments, and items. Therefore with advancing of LLM (Large Language Model), this project utilized Llama3.2:3b (Llamma 3.2 with 3 billions parameters) and MiniLM Embedding (all-minilm). This pipeline run using 3 main stacks:
- Python 3.14.5
- Sqlite3 3.46.1
- FAISS (Facebook AI Similarity Search)

This project data sources are pulled from [Genshin Impact Fandom Wiki](https://genshin-impact.fandom.com/wiki/Genshin_Impact_Wiki), [Keqing Main Theory Crafting](https://keqingmains.com/), and [HoneyHunter](https://gensh.honeyhunterworld.com/?lang=EN), by using either API, Github repository and web scrapping.

## Dependency
Before start to use this repos, this project required some packages (Debian based build):
```
sudo apt install git curl ca-certificates python3 python3-venv python3-pip sqlite3 build-essential pkg-config libxml2-dev libxslt1-dev liblz4-dev zlib1g-dev libffi-dev libssl-dev unzip jq -y
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
source .venv/bin/active
```
With python venv already activated install all python requirements by running:
```
pip -r requirements.txt
```

### FAISS Installation
There are 2 options for installing FAISS:
- python pip installation
- source build

Both of the method are good to make it works, but this project preferred using source build FAISS.

#### Python FAISS installation
```
pip install -U faiss-cpu
```

#### Souce build FAISS installation
Before running the installation make sure all packages are installed (Debian based build):
```
sudo apt update && sudo apt install -y git cmake build-essential pkg-config python3-dev python3-venv libopenblas-dev liblapack-dev swig
```
clone FAISS github:
```
git clone [https://github.com/facebookresearch/faiss.git](https://github.com/facebookresearch/faiss.git) && cd faiss
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

## Usage
The main entry of the script is `main.py`, in the script it has options can be used, below is the default values:
```
--DB_CRAWL=True
--DB_AUDIT=True 
--FAISS_migrate=False 
--FAISS_AUDIT=False
--FAISS_overwrite=False
```
Where `--DB_CRAWL` it will pull all the data from all datasource and store the embeddings inside Sqlite3, `--DB_AUDIT` it will check if the datasource is properly processed, `--FAISS_MIGRATE` it migrate the embedding vectors from Sqlite3 to FAISS, and `--FAISS_AUDIT` it will check if the embedding is properlly processed.

```
# Crawl + DB Audit
python3 rag/main.py --DB_CRAWL=True --DB_AUDIT=True --FAISS_migrate=False --FAISS_AUDIT=False
```
```
# Migrate + Audit FAISS
python3 rag/main.py --DB_CRAWL=False --DB_AUDIT=False --FAISS_migrate=True --FAISS_AUDIT=True
```
```
# Full pipeline
python3 rag/main.py --DB_CRAWL=True --DB_AUDIT=True --FAISS_migrate=True --FAISS_AUDIT=True --FAISS_overwrite=True
```