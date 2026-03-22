from __future__ import annotations

import logging, textwrap
from core.embed import embed
from core.paths import resolve_db_path, resolve_faiss_dir
from .utils import read_only_connect, normalize_query_vec, is_broad_question, chunk_batch, rerank_chunks
from .retrievers import FaissRetriever, SqliteEmbeddingRetriever
from .db_fetch import fetch_chunks, dedupe_by_doc
from .prompts import build_context, summarize_chunk_group, synthesize_final_answer
from .generators import ollama_generate

log = logging.getLogger(__name__)

def answer_question(
    cfg: dict,
    question: str,
    *,
    prefer_faiss: bool = True,
    direct_top_k: int = 12,
    broad_top_k: int = 60,
    summarize_batch_size: int = 8,
) -> str:
    db_path = resolve_db_path(cfg)
    faiss_dir = resolve_faiss_dir(cfg)
    conn = read_only_connect(str(db_path))

    base_url = cfg["ollama"]["base_url"]
    embed_model = cfg["ollama"]["embedding_model"]
    qa_model = cfg["ollama"].get("qa_model", cfg["ollama"].get("model", "llama3.2"))

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

    q_blob, q_dims = embed(base_url, embed_model, question, keep_alive="15s")
    if q_dims != retriever.dims:
        raise RuntimeError(
            f"query embedding dims mismatch: query={q_dims} retriever={retriever.dims}"
        )

    q_vec = normalize_query_vec(q_blob, q_dims)

    broad = is_broad_question(question)
    top_k = broad_top_k if broad else direct_top_k

    results = retriever.search(q_vec, top_k)
    chunk_ids = [cid for cid, score in results]
    initial_scores = {cid: score for cid, score in results}

    chunks = fetch_chunks(conn, chunk_ids)
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
        context = build_context(chunks[:5])
        prompt = textwrap.dedent(f"""
            You are a retrieval-grounded Genshin Impact assistant.
            Answer the question using ONLY the provided context.
            If the answer is not explicitly supported by the context, say:
            "I don't have enough evidence in the retrieved context."

            Rules:
            - Do not use outside knowledge.
            - Do not guess.
            - Cite chunk IDs inline like [chunk_id=123].
            - Keep the answer concise and factual.

            Question:
            {question}

            Context:
            {context}
        """).strip()
        return ollama_generate(base_url, qa_model, prompt, keep_alive="20s")

    notes = []
    for group in chunk_batch(chunks, summarize_batch_size):
        notes.append(summarize_chunk_group(base_url, qa_model, question, group))

    return synthesize_final_answer(base_url, qa_model, question, notes)