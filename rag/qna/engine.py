from __future__ import annotations

import logging, textwrap
from core.embed import embed
from core.paths import resolve_db_path, resolve_faiss_dir
from .utils import read_only_connect, normalize_query_vec, is_broad_question, chunk_batch, rerank_chunks, dedup_chunks
from .retrievers import FaissRetriever, SqliteEmbeddingRetriever
from .db_fetch import fetch_chunks
from .prompts import build_context, summarize_chunk_group, synthesize_final_answer
from .generators import generate

log = logging.getLogger(__name__)

def answer_question(
    cfg: dict,
    question: str,
    *,
    prefer_faiss: bool = True,
    direct_top_k: int = 12,
    broad_top_k: int = 60,
    summarize_batch_size: int = 8,
    backend: str | None =None
) -> str:
    db_path = resolve_db_path(cfg)
    faiss_dir = resolve_faiss_dir(cfg)
    conn = read_only_connect(str(db_path))

    runtime = cfg.get("runtime", {})
    provider = runtime.get("qa_provider", "ollama").strip().lower()
    if provider == "llamacpp":
        qa_timeout = str(cfg.get("llamacpp", {}).get("timeout", 300))
    else:
        qa_timeout = int(cfg["ollama"].get("timeout", 180000))

    retriever = None
    if prefer_faiss:
        try:
            retriever = FaissRetriever(faiss_dir)
            log.info("[QNA] using FAISS retriever")
        except Exception as e:
            log.warning("[QNA] FAISS unavailable, falling back to SQLite: %s", e)

    if retriever is None:
        retriever = SqliteEmbeddingRetriever(conn)
        log.info("[QNA] using SQLite brute-force retriever")

    q_blob, q_dims = embed(cfg, question, backend= backend)
    if q_dims != retriever.dims:
        raise RuntimeError(
            f"query embedding dims mismatch: query={q_dims} retriever={retriever.dims}"
        )

    q_vec = normalize_query_vec(q_blob, q_dims)

    broad = is_broad_question(question)
    top_k = broad_top_k if broad else direct_top_k

    candidate_k = max(top_k * 5, 85)

    results = retriever.search(q_vec, candidate_k)
    chunk_ids = [cid for cid, score in results]
    initial_scores = {cid: score for cid, score in results}

    chunks = fetch_chunks(conn, chunk_ids)
    chunks = dedup_chunks(chunks, initial_scores, max_per_doc=1)
    chunks = rerank_chunks(question, chunks, initial_scores)

    if not chunks:
        return "I couldn't retrieve any relevant chunks from the knowledge base."

    for row in chunks[:direct_top_k]:
        log.info(
            "[QNA] chunk_id=%s title=%s source=%s preview=%s",
            row["chunk_id"],
            row["title"],
            row["source"],
            (row["text"][:200] if row["text"] else "").replace("\n", " ")
        )

    if not broad:
        context = build_context(chunks[:direct_top_k])
        prompt = textwrap.dedent(f"""
            You are a retrieval-grounded Genshin Impact assistant.
            Answer the question using ONLY the provided context.
            If the answer is not explicitly supported by the context, say:
            "I don't have enough evidence in the retrieved context."

            Rules:
            - Use ONLY information present in the context.
            - You MAY infer reasonable conclusions directly supported by the context.
            For example: if a weapon's lore mentions a character and they share history,
            that weapon is likely associated with that character.
            - Cite chunk IDs inline like [chunk_id=123].
            - Cite source name like [source_name=xyz].
            - If the context genuinely does not contain enough information, say so briefly.
            - Do not guess facts not supported by the context.

            Question:
            {question}

            Context:
            {context}
        """).strip()
        result = generate(cfg, prompt)
        try:
            conn.close()
        except Exception:
            pass
        return result

    notes = []
    for group in chunk_batch(chunks, summarize_batch_size):
        notes.append(summarize_chunk_group(cfg, question, group))

    result = synthesize_final_answer(cfg, question, notes, qa_timeout)
    try:
        conn.close()
    except Exception:
        pass
    return result