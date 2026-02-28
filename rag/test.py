import math
import struct
import yaml

from core.db import connect
from core.embed import embed

def cosine(a, b):
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    return dot / (math.sqrt(na) * math.sqrt(nb) + 1e-9)

def tier_factor(tier: str) -> float:
    return 1.0 if tier == "primary" else 0.75

def source_factor(source: str) -> float:
    if source == "kqm_tcl":
        return 1.15
    
    if source == "genshin_wiki":
        return 1.05
    
    if source == "honey":
        return 1.00
    
    if source == "kqm_news":
        return 0.80

    return 1.0

def main():
    with open("rag/config.yaml") as f:
        cfg = yaml.safe_load(f)

    conn = connect(cfg["db_path"])
    cur = conn.cursor()

    query = input("Input your query:\n")

    q_bytes, dims = embed(cfg["ollama"]["base_url"], cfg["ollama"]["embedding_model"], query)
    q = struct.unpack(f"<{dims}f", q_bytes)

    cur.execute("""
    SELECT e.chunk_id, e.vector, d.source, d.tier, d.weight
    FROM embeddings e
    JOIN chunks c ON c.chunk_id = e.chunk_id
    JOIN docs d ON d.doc_id = c.doc_id
    WHERE c.is_active = 1
    """)

    scored = []
    for cid, vbytes, source, tier, weight in cur.fetchall():
        v = struct.unpack(f"<{dims}f", vbytes)
        base = cosine(q, v)
        score = base * float(weight) * tier_factor(tier) * source_factor(source)
        scored.append((score, base, cid, source, tier, weight))

    scored.sort(reverse=True)

    for rank, (score, base, cid, source, tier, weight) in enumerate(scored[:5], 1):
        cur.execute("""
            SELECT d.source, d.title, d.url, c.text
            FROM chunks c
            JOIN docs d ON d.doc_id = c.doc_id
            WHERE c.chunk_id = ?
        """, (cid,))
        s2, title, url, text = cur.fetchone()
        print(f"\n#{rank} score={score:.4f} base={base:.4f} source={s2} tier={tier} weight={weight}\nurl={url}\n")
        print(text[:800])

if __name__ == "__main__":
    main()