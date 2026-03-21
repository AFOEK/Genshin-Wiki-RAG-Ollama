from __future__ import annotations

import json, yaml, logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag"))

from core.paths import resolve_db_path
from core.db import read_only_connect

log = logging.getLogger(__name__)

def load_cfg(path: str ="../rag/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def main():
    cfg = load_cfg()
    db_path = resolve_db_path(cfg)

    out_dir = Path("rag/chunks_kaggle")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chunks.jsonl"

    conn = read_only_connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
    """
    SELECT
        c.chunk_id, c.doc_id, c.chunk_index, c.text, d.source, d.url, d.title, d.tier, d.weight
    FROM chunks c
    JOIN docs d on d.doc_id = c.doc_id
    WHERE c.is_active=1
    ORDER BY c.chunk_id
    """
    )
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in cur:
            obj = {
                "chunk_id": int(row["chunk_id"]),
                "doc_id": int(row["doc_id"]),
                "chunk_index": int(row["chunk_index"]) if row["chunk_index"] is not None else None,
                "text": row["text"],
                "source": row["source"],
                "url": row["url"],
                "title": row["title"],
                "tier": row["tier"],
                "weight": float(row["weight"]) if row["weight"] is not None else 1.0,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
    
    conn.close()
    log.info(f"[INFO] Chunks exported {count} active chunks to {out_path}")

if __name__ == "__main__":
    main()
