#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-run}"

NUKE_CONFIRM="${NUKE_CONFIRM:-0}"

DATA_ROOT="/mnt/ssd/genshin_rag/data"
EXPORT_ROOT="/mnt/ssd/genshin_rag/exports"
LOG_ROOT="rag/logs"
KAGGLE_CHUNKS_ROOT="rag/chunks_kaggle"
LOCAL_RAG_DATA_ROOT="rag/data"

safe_rm_rf() {
    local target="$1"

    if [ -z "$target" ] || [ "$target" = "/" ]; then
        echo "[SAFETY] Refusing to delete unsafe path: '$target'"
        exit 1
    fi

    if [ -e "$target" ]; then
        echo "[CLEAN] Removing: $target"
        rm -rf "$target"
    else
        echo "[CLEAN] Not found, skipping: $target"
    fi
}

clean_logs() {
    safe_rm_rf "$LOG_ROOT"
    mkdir -p "$LOG_ROOT"
}

clean_exports() {
    safe_rm_rf "$EXPORT_ROOT"
    mkdir -p "$EXPORT_ROOT"
}

nuke_all() {
    if [ "$NUKE_CONFIRM" != "1" ]; then
        echo "[SAFETY] This will delete DB, FAISS, FTS data, TurboVec, local rag/data, Kaggle chunks, and logs."
        echo "[SAFETY] Run with:"
        echo "         NUKE_CONFIRM=1 ./run_pipeline.sh nuke"
        exit 1
    fi

    echo "[NUKE] Full rebuild requested."

    safe_rm_rf "$DATA_ROOT"
    safe_rm_rf "$LOG_ROOT"
    safe_rm_rf "$KAGGLE_CHUNKS_ROOT"
    safe_rm_rf "$LOCAL_RAG_DATA_ROOT"

    mkdir -p "$DATA_ROOT"
    mkdir -p "$LOG_ROOT"
}

export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

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

case "$MODE" in
    run)
        ;;
    nuke|clean_all)
        nuke_all
        ;;
    clean_logs)
        clean_logs
        exit 0
        ;;
    clean_exports)
        clean_exports
        exit 0
        ;;
    *)
        echo "Usage:"
        echo "  ./run_pipeline.sh"
        echo "  NUKE_CONFIRM=1 ./run_pipeline.sh nuke"
        echo "  NUKE_CONFIRM=1 ./run_pipeline.sh clean_all"
        echo "  ./run_pipeline.sh clean_logs"
        echo "  ./run_pipeline.sh clean_exports"
        exit 1
        ;;
esac

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

compress_db_snapshot() {
    local DB="/mnt/ssd/genshin_rag/data/genshin_rag.db"
    local EXPORT_DIR="/mnt/ssd/genshin_rag/exports"
    local TS
    TS="$(date '+%Y%m%d_%H%M%S')"

    local SNAPSHOT="$EXPORT_DIR/genshin_rag_${TS}.compact.db"
    local ARCHIVE_7Z="$EXPORT_DIR/genshin_rag_${TS}.db.7z"
    local ARCHIVE_TAR_ZST="$EXPORT_DIR/genshin_rag_${TS}.db.tar.zst"
    local ARCHIVE_TAR_GZ="$EXPORT_DIR/genshin_rag_${TS}.db.tar.gz"

    mkdir -p "$EXPORT_DIR"

    if [ ! -f "$DB" ]; then
        log "[EXPORT] DB not found: $DB"
        return 1
    fi

    log "[EXPORT] Creating SQLite checkpoint"
    sqlite3 "$DB" "PRAGMA wal_checkpoint(TRUNCATE);"

    log "[EXPORT] Creating compact DB snapshot: $SNAPSHOT"
    rm -f "$SNAPSHOT"
    sqlite3 "$DB" "VACUUM INTO '$SNAPSHOT';"

    log "[EXPORT] Running integrity check on snapshot"
    local CHECK
    CHECK="$(sqlite3 "$SNAPSHOT" "PRAGMA integrity_check;")"

    if [ "$CHECK" != "ok" ]; then
        log "[EXPORT] Integrity check failed: $CHECK"
        rm -f "$SNAPSHOT"
        return 1
    fi

    log "[EXPORT] Snapshot integrity OK"

    if command -v 7z >/dev/null 2>&1; then
        log "[EXPORT] Compressing with 7z: $ARCHIVE_7Z"
        nice -n 10 ionice -c2 -n7 7z a -t7z -mx=9 -mmt=on "$ARCHIVE_7Z" "$SNAPSHOT"
        log "[EXPORT] Done: $ARCHIVE_7Z"

    elif command -v zstd >/dev/null 2>&1; then
        log "[EXPORT] Compressing with tar.zst: $ARCHIVE_TAR_ZST"
        (
            cd "$EXPORT_DIR"
            nice -n 10 ionice -c2 -n7 tar --zstd -cf "$(basename "$ARCHIVE_TAR_ZST")" "$(basename "$SNAPSHOT")"
        )
        log "[EXPORT] Done: $ARCHIVE_TAR_ZST"

    else
        log "[EXPORT] Compressing with tar.gz: $ARCHIVE_TAR_GZ"
        (
            cd "$EXPORT_DIR"
            nice -n 10 ionice -c2 -n7 tar -czf "$(basename "$ARCHIVE_TAR_GZ")" "$(basename "$SNAPSHOT")"
        )
        log "[EXPORT] Done: $ARCHIVE_TAR_GZ"
    fi

    log "[EXPORT] Archive size:"
    ls -lh "$EXPORT_DIR"/genshin_rag_"$TS".db.* | tee -a "$LOG"

    rm -f "$SNAPSHOT"
    log "[EXPORT] Removed temporary snapshot: $SNAPSHOT"
}

log "Pipeline starting (cron=${CRON_MODE:-0})"

#rm -rf /mnt/ssd/genshin_rag/data/* &&  rm -rf rag/logs/* && rm -rf rag/chunks_kaggle && rm -rf rag/data/*

log "Activating virtual environment"
source .venv/bin/activate

log "Starting crawl, repair, audit, and FAISS migrations"
python3 rag/main.py --DB_CRAWL=True --DB_AUDIT=True --DB_REPAIR=True --FAISS_MIGRATE=True --FAISS_AUDIT=True --FAISS_OVERWRITE=True --TURBOVEC_MIGRATE=True --TURBOVEC_OVERWRITE=True --TURBOVEC_AUDIT=True --FTS_SYNC=True --PARENT_SYNC=True --BACKEND ollama
log "Crawling, repair, audit, FAISS, FTS5, and parent-child records builds. Done"

log "Compressing compact SQLite DB snapshot"
compress_db_snapshot || log "[EXPORT] DB compression failed — continuing"
log "DB compression step finished"

log "Extracting chunks"
python3 kaggle_tools/extract_chunks.py
log "Done extracting chunks"

log "Uploading to Kaggle"
python3 kaggle_tools/upload.py --dataset-slug "AFOEK88/genshin-rag-chunks" --dataset-title "Genshin RAG Chunks Data" || log "Failed upload to Kaggle, try again"
log "Done upload"

log "Test first local embedding"
python3 rag/test.py --question "What is Zhongli signature weapon?" --retriever hybrid --direct_top_k 20 --backend ollama || log "Test failed — continuing"
log "First testing done"

# log "Kaggle kernel pull, swap FAISS and pull embeddings"
# python3 kaggle_tools/pull.py --wait --replace-faiss --poll-interval 300 --kernel-timeout 18000 || log "Failed pull from Kaggle, train again"
# log "Done pull finish"

# log "Test second local embedding with better embeddings"
# python3 rag/test.py --question "What is Zhongli signature weapon?" --retriever hybrid --direct_top_k 20
# log "Second testing done"