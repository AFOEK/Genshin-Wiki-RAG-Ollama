#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOCK="/tmp/genshin_rag_pipeline.lock"
if [ "${CRON_MODE:-0}" = "1" ]; then
    if [ -f "$LOCK" ]; then
        LOCKED_PID=$(cat "$LOCK")
        if kill -0 "$LOCKED_PID" 2>/dev/null; then
            echo "Already running (PID $LOCKED_PID), skipping"
            exit 0
        else
            rm -f "$LOCK"
        fi
    fi
    echo $$ > "$LOCK"
    trap 'rm -f "$LOCK"' EXIT
fi

sudo renice -n -10 -p $$ 2>/dev/null || true

LOG="rag/logs/pipeline_run.log"
mkdir -p rag/logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

log "Pipeline starting (cron=${CRON_MODE:-0})"

log "Activating virtual environment"
source "$SCRIPT_DIR/../.venv/bin/activate"

log "Starting crawl, repair, audit, and FAISS migrations"
python3 rag/main.py --DB_CRAWL=True --DB_AUDIT=True --DB_REPAIR=True --FAISS_MIGRATE=True --FAISS_AUDIT=True
log "Crawling, repair, audit, and FAISS. Done"

log "Extracting chunks"
python3 kaggle_tools/extract_chunks.py
log "Done extracting chunks"

log "uploading to Kaggle"
python3 kaggle_tools/upload.py --dataset-slug "AFOEK88/genshin-rag-chunks" --dataset-title "Genshin RAG Chunks Data"
log "Done"