#!/usr/bin/env bash
set -evo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

sudo renice -n -10 -p $$

LOG="rag/log/pipeline_run.log"
mkdir -p rag/logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

log "Activating virtual environment"
source .venv/bin/activate

log "Starting crawl, repair, audit, and FAISS migrations"
python3 rag/main.py --DB_CRAWL=True --DB_AUDIT=True --DB_REPAIR=True --FAISS_MIGRATE=True --FAISS_AUDIT=True
log "Crawling, repair, audit, and FAISS. Done"

log "Extracting chunks"
python3 kaggle_tools/extract_chunks.py
log "Done extracting chunks"

log "uploading to Kaggle"
python3 kaggle_tools/upload.py --dataset-slug "AFOEK88/genshin-rag-chunks" --dataset-title "Genshin RAG Chunks Data" --chunks-file "../rag/chunks_kaggle/chunks.jsonl"
log "Done"