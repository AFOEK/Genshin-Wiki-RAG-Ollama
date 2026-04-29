from __future__ import annotations

import logging, textwrap
from core.embed import embed
from core.paths import resolve_db_path, resolve_faiss_dir
from .utils import read_only_connect, normalize_query_vec, is_broad_question, chunk_batch, rerank_chunks, dedupe_chunks, detect_intent, filter_by_intent_source, rrf_fuse
from .retrievers import FaissRetriever, SqliteEmbeddingRetriever, BM25Retriever
from .db_fetch import fetch_chunks
from .prompts import build_context, summarize_chunk_group, synthesize_final_answer
from .generators import generate

log = logging.getLogger(__name__)

def answer_question(
    cfg: dict,
    question: str,
    *,
    retriever_name: str = "hybrid",
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
    log.info("[QNA] Use provider: %s", provider)
    if provider == "llamacpp":
        qa_timeout = str(cfg.get("llamacpp", {}).get("timeout", 300))
    else:
        qa_timeout = int(cfg["ollama"].get("timeout", 1800))

    retriever = None
    if retriever_name == "faiss":
        try:
            retriever = FaissRetriever(faiss_dir)
            log.info("[QNA] using FAISS retriever")
        except Exception as e:
            log.warning("[QNA] FAISS unavailable, falling back to SQLite: %s", e)
    elif retriever_name == "sqlite" or retriever_name == "sql":
        retriever = SqliteEmbeddingRetriever(conn)
        log.info("[QNA] using SQLite brute-force retriever")
    elif retriever_name == "bm25":
        retriever = BM25Retriever(conn)
        log.info("[QNA] using SQLite BM25 retriever")
    elif retriever_name == "hybrid":
        log.info("[QNA] using HYBRID retriever (FAISS + BM25)")
        faiss_ret = FaissRetriever(faiss_dir)
        bm25_ret = BM25Retriever(conn)
        q_blob, q_dims = embed(cfg, question, backend=backend)
        if q_dims != faiss_ret.dims:
            raise RuntimeError(
                f"query embedding dims mismatch: query={q_dims} retriever={faiss_ret.dims}"
            )
        q_vec = normalize_query_vec(q_blob, q_dims)
        faiss_results = faiss_ret.search(q_vec, candidate_k)
        bm25_results = bm25_ret.search(question.lower(), candidate_k)
        results = rrf_fuse(faiss_results, bm25_results, k=60)
        initial_scores = {cid: score for cid, score in results}
    else:
        raise RuntimeError(f"Unkown retriever: {retriever_name}")

    intent = detect_intent(question)
    broad = is_broad_question(question)
    top_k = broad_top_k if broad else direct_top_k

    if retriever_name == "bm25":
        results = retriever.search(question.lower(), top_k)
    else:
        q_blob, q_dims = embed(cfg, question, backend=backend)
        if q_dims != retriever.dims:
            raise RuntimeError(
                f"query embedding dims mismatch: query={q_dims} retriever={retriever.dims}"
            )
        q_vec = normalize_query_vec(q_blob, q_dims)
        results = retriever.search(q_vec, top_k)

    q_vec = normalize_query_vec(q_blob, q_dims)

    broad = is_broad_question(question)
    top_k = broad_top_k if broad else direct_top_k

    candidate_k = max(top_k * 5, 85)

    results = retriever.search(q_vec, candidate_k)
    chunk_ids = [cid for cid, score in results]
    initial_scores = {cid: score for cid, score in results}

    filtered_ids = filter_by_intent_source(conn, chunk_ids, intent, min_required=5, max_fallback=20)

    if intent in ("build", "mechanic", "lore", "biography") and len(filtered_ids) < 3:
        deep_results = retriever.search(q_vec, candidate_k * 5)
        deep_ids = [cid for cid, _ in deep_results]
        filtered_ids = filter_by_intent_source(conn, deep_ids, intent, min_required=5, max_fallback=40)
        
        deep_scores = {cid: score for cid, score in deep_results}
        id_set = set(filtered_ids)
        results = [(cid, deep_scores[cid]) for cid in filtered_ids]
        chunk_ids = [cid for cid, _ in results]
        initial_scores = {cid: score for cid, score in results}
    else:
        id_set = set(filtered_ids)
        results = [(cid, score) for cid, score in results if cid in id_set]
        chunk_ids = [cid for cid, _ in results]
        initial_scores = {cid: score for cid, score in results}

    chunks = fetch_chunks(conn, chunk_ids)
    chunks = dedupe_chunks(chunks, initial_scores, max_per_doc=3)
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