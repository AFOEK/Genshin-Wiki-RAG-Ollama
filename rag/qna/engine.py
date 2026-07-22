from __future__ import annotations

import logging
import re
import threading

from pathlib import Path

from core.embed import embed
from core.paths import resolve_db_path, resolve_faiss_dir, resolve_storage_root, resolve_splade_dir
from core.hyde import generate_hyde_document
from .utils import read_only_connect, normalize_query_vec, is_broad_question, chunk_batch, rerank_chunks, dedupe_chunks, detect_intent, filter_by_intent_source, as_bool, get_kqm_news_fetch_version_baseline, prefer_entity_seed_chunks, expected_model_from_cfg, make_intent_fts5_query, get_bm25_weights, detect_build_subtypes, extract_lookup_entity, make_retrieval_cache_key, retrieval_result_from_cache, retrieval_result_to_cache, build_weighted_rrf_signal, build_grounded_answer_prompt, merge_context_preserving_seeds, trim_chunks_to_context_budget, normalized_phrase, extract_lookup_target, normalize_model_name, resolve_lookup_entity_from_chunks, normalize_title_key
from .retrievers import FaissRetriever, SqliteEmbeddingRetriever, BM25Retriever, TurboVecRetriever, SpladeRetriever
from .retrieval_cache import RetrievalCache
from .db_fetch import fetch_chunks
from .prompts import build_context, summarize_chunk_group, synthesize_final_answer
from .generators import generate
from .cross_encoder import cross_encoder_rerank
from .context_expand import expand_context_windows
from .multi_hop import generate_bridge_queries, merge_multi_hop_results
from .parent_child import fetch_parent_context_chunks
from .query_decomposition import decompose_query, merge_decomposition_runs
from .types import RetrievalResult

log = logging.getLogger(__name__)
_RETRIEVAL_CACHE : RetrievalCache | None = None
_RETRIEVAL_CACHE_INIT_LOCK = threading.Lock()

HYBRID_FUSION_SPECS = {
    "hybrid": {"faiss", "bm25"},
    "hybrid_hyde": {"faiss", "bm25", "hyde"},
    "hybrid_turbovec": {"turbovec", "bm25"},
    "hybrid_faiss_turbovec": {"faiss", "turbovec", "bm25"},
    "hybrid_splade": {"faiss", "bm25", "splade"},
    "hybrid_hyde_turbovec": {"faiss", "turbovec", "bm25", "hyde"},
    "hybrid_splade_turbovec": {"faiss", "turbovec", "bm25", "splade"},
    "hybrid_hyde_splade_turbovec": {"faiss", "turbovec", "bm25", "splade", "hyde"},
    "hybrid_all": {"faiss", "turbovec", "bm25", "splade", "hyde"}
}

def get_retrieval_cache(cfg: dict) -> RetrievalCache | None:
    global _RETRIEVAL_CACHE
    cache_cfg = cfg.get("retrieval_cache", {}) or {}
    if not as_bool(cache_cfg.get("enabled", False)):
        return None
    
    if _RETRIEVAL_CACHE is None:
        with _RETRIEVAL_CACHE_INIT_LOCK:
            if _RETRIEVAL_CACHE is None:
                root = resolve_storage_root(cfg)
                rel = Path(str(cache_cfg.get("path", "data/cache/retrieval_cache.sqlite")))
                path = (rel if rel.is_absolute() else root / rel)
                _RETRIEVAL_CACHE = RetrievalCache(path, ttl_seconds=int(cache_cfg.get("ttl_seconds", 86400)), max_entries=int(cache_cfg.get("max_entries", 50000,)))

    return _RETRIEVAL_CACHE

def retrieve_question_context_uncached(cfg: dict, question: str, *, retriever_name: str = "hybrid", direct_top_k: int = 12, broad_top_k: int = 60, backend: str | None = None) -> RetrievalResult:
    strict_fts_query_used: str | None = None
    db_path = resolve_db_path(cfg)
    faiss_dir = resolve_faiss_dir(cfg)
    splade_dir = resolve_splade_dir(cfg)
    tv_cfg = cfg.get("turbovec", {}) or {}
    tv_raw = Path(str(tv_cfg.get("path", "data/turbovec")))
    turbovec_dir = tv_raw if tv_raw.is_absolute() else db_path.parent.parent / tv_raw
    runtime = cfg.get("runtime", {})
    provider = runtime.get("qa_provider", "ollama").strip().lower()
    log.info("[QNA] Use provider: %s", provider)

    reranker_cfg = cfg.get("reranker", {})
    reranker_mode = str(reranker_cfg.get("mode", "feature")).strip().lower()
    log.info("[QNA] Use reranker: %s", reranker_mode)

    retrieval_cfg = cfg.get("retrieval", {}) or {}
    candidate_k_cfg = int(retrieval_cfg.get("candidate_k", 85))
    deep_candidate_multiplier = int(retrieval_cfg.get("deep_candidate_multiplier", 5))
    dedup_max_per_doc = int(retrieval_cfg.get("dedup_max_per_doc", 2))

    rrf_k = int(retrieval_cfg.get("rrf_k", 60))

    intent = detect_intent(question)
    build_subtypes = (detect_build_subtypes(question) if intent == "build" else set())
    if (intent == "build" and re.search(
            r"\b("
            r"weapon|weapons|"
            r"sword|swords|"
            r"claymore|claymores|"
            r"polearm|polearms|"
            r"bow|bows|warbow|"
            r"catalyst|catalysts"
            r")\b", question, re.IGNORECASE)):
        build_subtypes.add("weapon")
    broad = is_broad_question(question)
    top_k = broad_top_k if broad else direct_top_k
    candidate_k = max(top_k, candidate_k_cfg)
    log.info("[QNA] retrieval cfg: top_k=%d candidate_k=%d deep_multiplier=%d dedup_max_per_doc=%d intent=%s subtypes=%s broad=%s", top_k, candidate_k, deep_candidate_multiplier, dedup_max_per_doc, intent, sorted(build_subtypes), broad)

    faiss_ret_cache = None
    bm25_ret_cache = None
    turbovec_ret_cache = None
    splade_ret_cache = None
    hyde_document_cache: str | None = None
    q_vec_cache: dict[tuple[str, str, int], object] = {}
    log.info("[CACHE] Initialize cache FAISS cache: %s, BM25 cache: %s, TurboVec cache: %s, HyDE cache: %s, SPLADE cache: %s", str(faiss_ret_cache), str(bm25_ret_cache), str(turbovec_ret_cache), str(hyde_document_cache), "unimplemented")

    hyde_used_for_request = False
    hyde_fallback_reason: str | None = None
    hyde_error: str | None = None
    decomposition_subqueries: list[str] = []
    multi_hop_queries: list[str] = []
    
    hyde_cfg = cfg.get("hyde", {}) or {}
    hyde_enabled = as_bool(hyde_cfg.get("enabled", False))
    hyde_mode = str(hyde_cfg.get("mode", "fallback")).strip().lower()
    if hyde_mode not in {"always", "fallback", "off", "disabled", "never"}:
        raise ValueError(f"Unsupported HyDE mode: {hyde_mode!r}")

    def get_q_vec(ret, query_text: str | None = None):
        effective_query = (query_text if query_text is not None else question).strip()
        model_key = normalize_model_name(getattr(ret, "model", "")) or "runtime"
        cache_key = (effective_query, model_key, int(ret.dims))

        if cache_key in q_vec_cache:
            return q_vec_cache[cache_key]

        q_blob, q_dims = embed(cfg, effective_query, backend=backend, mode="query",)

        if q_dims != ret.dims:
            raise RuntimeError(f"query embedding dims mismatch: query={q_dims} retriever={ret.dims}")

        query_vector = normalize_query_vec(q_blob, q_dims,)
        q_vec_cache[cache_key] = query_vector
        return query_vector

    expected_faiss_model = expected_model_from_cfg(cfg, backend=backend)
    faiss_mismatch_policy = str(retrieval_cfg.get("faiss_model_mismatch", "error")).strip().lower()
    
    def build_single_channel_results(raw_results: list[tuple[int, float]], channel: str) -> tuple[list[tuple[int, float]], dict[int, dict]]:
        scale = float(retrieval_cfg.get("rrf_scale", 10.0))
        signals: dict[int, dict] = {}

        for rank, (cid, raw_score) in enumerate(raw_results, start=1):
            cid = int(cid)

            signal = {
                "rrf_score": scale / (rrf_k + rank),
                "faiss_score": 0.0,
                "bm25_score": 0.0,
                "faiss_rank": None,
                "bm25_rank": None,
                "in_faiss": False,
                "in_bm25": False,
                "splade_score": 0.0,
                "splade_rank": None,
                "in_splade": False,
            }

            if channel == "bm25":
                signal["bm25_score"] = float(raw_score)
                signal["bm25_rank"] = rank
                signal["in_bm25"] = True
            elif channel == "splade":
                signal["splade_score"] = float(raw_score)
                signal["splade_rank"] = rank
                signal["in_splade"] = True
            else:
                signal["faiss_score"] = float(raw_score)
                signal["faiss_rank"] = rank
                signal["in_faiss"] = True

            signals[cid] = signal

        ranked = sorted(((cid, signal["rrf_score"]) for cid, signal in signals.items()), key=lambda item: item[1], reverse=True)
        return ranked, signals

    def get_faiss_ret():
        nonlocal faiss_ret_cache
        if faiss_ret_cache is None:
            faiss_ret_cache = FaissRetriever(faiss_dir, expected_model=expected_faiss_model, mismatch_policy=faiss_mismatch_policy)
        return faiss_ret_cache

    def search_hyde(k: int) -> list[tuple[int, float]]:
        nonlocal hyde_document_cache
        hyde_cfg = cfg.get("hyde", {}) or {}
        if not as_bool(hyde_cfg.get("enabled", False)):
            return []
        
        faiss_ret = get_faiss_ret()
        if hyde_document_cache is None:
            hyde_document_cache = generate_hyde_document(cfg, question)
        
        if not hyde_document_cache:
            return []
        
        hyde_blob, hyde_dims = embed(cfg, hyde_document_cache, backend=backend, mode="query")
        if hyde_dims != faiss_ret.dims:
            raise RuntimeError("HyDE embedding dimension mismatch: " f"query={hyde_dims} faiss={faiss_ret.dims}")
        
        hyde_vec = normalize_query_vec(hyde_blob, hyde_dims)
        configured_k = int(hyde_cfg.get("candidate_k", k))
        effective_k = min(k, configured_k)
        results = faiss_ret.search(hyde_vec, effective_k)
        log.info("[HYDE] retrieved candidates=%d requested_k=%d", len(results), effective_k)
        return results

    def get_bm25_ret():
        nonlocal bm25_ret_cache
        if bm25_ret_cache is None:
            bm25_ret_cache = BM25Retriever(conn)

        return bm25_ret_cache
    
    def get_splade_ret():
        nonlocal splade_ret_cache
        if splade_ret_cache is None:
            splade_cfg = (cfg.get("splade", {}) or {})
            if not as_bool(splade_cfg.get("enabled", False)):
                raise RuntimeError("[SPLADE] SPLADE is disabled")
            cache_folder_value = (splade_cfg.get("cache_folder"))
            cache_folder = None
            if cache_folder_value:
                cache_path = Path(str(cache_folder_value)).expanduser()
                if not cache_path.is_absolute():
                    cache_path = (resolve_storage_root(cfg) / cache_path)

                cache_path.mkdir(parents=True, exist_ok=True,)
                cache_folder = str(cache_path.resolve())

            active_dims_value = (splade_cfg.get("max_active_dims", 128))
            max_active_dims = (int(active_dims_value) if active_dims_value is not None else None)
            splade_ret_cache = SpladeRetriever(splade_dir, model_name=str(splade_cfg["model"]), device=str(splade_cfg.get("device", "auto")), max_length=int(splade_cfg.get("max_length", 256)), max_active_dims=max_active_dims, cache_folder=cache_folder, precision=str(splade_cfg.get("precision", "fp32").strip().lower()))
        return splade_ret_cache

    def merge_ranked_results(primary: list[tuple[int, float]], fallback: list[tuple[int, float]], k: int) -> list[tuple[int, float]]:
        merged = []
        seen = set()

        for cid, score in primary + fallback:
            cid = int(cid)

            if cid in seen:
                continue

            seen.add(cid)
            merged.append((cid, float(score)))

            if len(merged) >= k:
                break

        return merged
    
    def search_bm25(k: int, query_text: str | None = None):
        nonlocal strict_fts_query_used
        effective_query = (query_text or question).strip()
        query_intent = detect_intent(effective_query)
        query_weights = get_bm25_weights(query_intent)
        strict_query = make_intent_fts5_query(effective_query, query_intent)
        if effective_query == question:
            strict_fts_query_used = strict_query

        bm25_ret = get_bm25_ret()
        log.info("[BM25] query=%r intent=%s weights=%s", effective_query, query_intent, query_weights,)

        if not strict_query:
            return bm25_ret.search(effective_query, k, weights=query_weights)

        log.info("[BM25] strict intent query=%s", strict_query)

        strict_results = bm25_ret.search_fts(strict_query, k, weights=query_weights)
        minimum_strict = min(5, k)

        if len(strict_results) >= minimum_strict:
            return strict_results

        log.info("[BM25] strict query returned only %d rows; adding broad fallback", len(strict_results))
        broad_results = bm25_ret.search(effective_query, k, weights=query_weights)
        return merge_ranked_results(strict_results, broad_results, k)

    def get_turbovec_ret():
        nonlocal turbovec_ret_cache

        if turbovec_ret_cache is None:
            tv_cfg = cfg.get("turbovec", {}) or {}
            expected_model = expected_model_from_cfg(cfg, backend=backend, source=str(tv_cfg.get("model_source", "runtime")))
            turbovec_ret_cache = TurboVecRetriever(turbovec_dir, expected_model=expected_model, mismatch_policy=str(tv_cfg.get("model_mismatch", "error")))

        return turbovec_ret_cache

    def search_embedding_retriever(ret, k: int):
        q_vec = get_q_vec(ret)
        return ret.search(q_vec, k)
    
    def search_splade(k: int, query_text: str | None = None) -> list[tuple[int, float]]:
        return get_splade_ret().search((query_text or question).strip(), k)
    
    def search_hybrid_fusion_decomposed(name: str, k: int, *, force_hyde: bool = False, force_reason: str | None = None):
        decomp_cfg = cfg.get("query_decomposition", {}) or {}
        original_results, original_signals = search_hybrid_fusion(name, k, question, force_hyde=force_hyde, force_reason=force_reason)

        if not decomposition_subqueries:
            return original_results, original_signals

        runs = [(question, original_results, original_signals, float(decomp_cfg.get("original_weight", 1.0)))]
        subquery_k = min(k, int(decomp_cfg.get("candidate_k_per_subquery", 300)))

        for subquery in decomposition_subqueries:
            sub_results, sub_signals = search_hybrid_fusion(name, subquery_k, subquery)
            runs.append((subquery, sub_results, sub_signals, float(decomp_cfg.get("subquery_weight", 0.8))))

        merged_results, merged_signals = merge_decomposition_runs(runs, rrf_k=rrf_k, rrf_scale=float(retrieval_cfg.get("rrf_scale", 10.0)), max_total_candidates=int(decomp_cfg.get("max_total_candidates", 1800)))
        log.info("[DECOMP] retriever=%s original=%d subqueries=%d candidates=%d", name, len(original_results), len(decomposition_subqueries), len(merged_results))
        return merged_results, merged_signals

    def should_trigger_hyde(faiss_results: list[tuple[int, float]], bm25_results: list[tuple[int, float]]) -> tuple[bool, str]:
        if not hyde_enabled:
            return False, "hyde_disabled"
        
        if hyde_mode == "always":
            return True, "mode_always"
        
        if hyde_mode in {"off", "disabled", "never"}:
            return False, "mode_disabled"
        
        if not faiss_results and not bm25_results:
            return True, "both_channels_empty"

        if not faiss_results:
            return True, "faiss_empty"

        if not bm25_results:
            return True, "bm25_empty"

        top_n = max(1, int(hyde_cfg.get("fallback_top_n", 12)))
        faiss_chunk_ids = [int(chunk_id) for chunk_id, _ in faiss_results[:top_n]]
        bm25_chunk_ids = [int(chunk_id) for chunk_id, _ in bm25_results[:top_n]]
        top_chunk_ids = list(dict.fromkeys(faiss_chunk_ids + bm25_chunk_ids))
        rows = fetch_chunks(conn, top_chunk_ids)
        chunk_to_doc = {int(row["chunk_id"]): int(row["doc_id"]) for row in rows}
        faiss_doc_ids = {chunk_to_doc[chunk_id] for chunk_id in faiss_chunk_ids if chunk_id in chunk_to_doc}
        bm25_doc_ids = {chunk_to_doc[chunk_id] for chunk_id in bm25_chunk_ids if chunk_id in chunk_to_doc}
        shared_doc_ids = (faiss_doc_ids & bm25_doc_ids)
        minimum_shared_docs = max(0, int(hyde_cfg.get("fallback_min_shared_docs",1)))

        if len(shared_doc_ids) < minimum_shared_docs:
            return (True, f"low_channel_agreement: shared_docs={len(shared_doc_ids)} <{minimum_shared_docs}")

        return (False, f"normal_retrieval_sufficient: shared_docs={len(shared_doc_ids)}")
    
    def search_hybrid_fusion(name: str, k: int, query_text: str | None = None, *, force_hyde: bool = False, force_reason: str | None = None):
        nonlocal hyde_used_for_request, hyde_fallback_reason, hyde_error

        spec = HYBRID_FUSION_SPECS[name]
        effective_query = (query_text or question).strip()
        channels: dict[str, list[tuple[int, float]]] = {}
        weights: dict[str, float] = {}

        faiss_ret = None
        faiss_results: list[tuple[int, float]] = []
        bm25_results = search_bm25(k, effective_query)

        channels["bm25"] = bm25_results
        weights["bm25"] = float(retrieval_cfg.get("bm25_rrf_weight", 1.0))

        if "faiss" in spec:
            faiss_ret = get_faiss_ret()
            faiss_results = faiss_ret.search(get_q_vec(faiss_ret, effective_query), k)
            channels["faiss"] = faiss_results
            weights["faiss"] = float(retrieval_cfg.get("faiss_rrf_weight", 1.0))

        if "turbovec" in spec:
            tv_ret = get_turbovec_ret()

            if faiss_ret is not None:
                faiss_model = normalize_model_name(faiss_ret.model)
                tv_model = normalize_model_name(tv_ret.model)

                if faiss_ret.dims != tv_ret.dims:
                    raise RuntimeError(f"FAISS/TurboVec dimension mismatch: faiss={faiss_ret.dims} turbovec={tv_ret.dims}")

                if faiss_model and tv_model and faiss_model != tv_model:
                    raise RuntimeError(f"FAISS/TurboVec model mismatch: faiss={faiss_ret.model!r} turbovec={tv_ret.model!r}")

            tv_k = min(k, int(tv_cfg.get("candidate_k", k)))
            tv_results = tv_ret.search(get_q_vec(tv_ret, effective_query), tv_k)
            channels["turbovec"] = tv_results
            default_tv_weight = 0.5 if "faiss" in spec else 1.0
            weights["turbovec"] = float(tv_cfg.get("rrf_weight", default_tv_weight))

        if "splade" in spec:
            splade_cfg = cfg.get("splade", {}) or {}
            splade_k = min(k, int(splade_cfg.get("candidate_k", 300)))
            channels["splade"] = search_splade(splade_k, effective_query)
            weights["splade"] = float(splade_cfg.get("rrf_weight", 0.75))

        if "hyde" in spec and effective_query == question:
            if faiss_ret is None:
                raise RuntimeError("HyDE fusion requires FAISS")

            if force_hyde and hyde_enabled:
                use_hyde = True
                reason = force_reason or "forced_fallback"
            else:
                use_hyde, reason = should_trigger_hyde(faiss_results, bm25_results)

            hyde_results: list[tuple[int, float]] = []

            if use_hyde:
                try:
                    hyde_results = search_hyde(k)
                    hyde_used_for_request = bool(hyde_results)
                    hyde_fallback_reason = reason
                except Exception as exc:
                    hyde_error = f"{type(exc).__name__}: {exc}"
                    hyde_fallback_reason = f"{reason};hyde_failed"
                    log.warning("[HYDE] generation/search failed; continuing without HyDE: %s", hyde_error)
            elif hyde_fallback_reason is None:
                hyde_fallback_reason = reason

            channels["hyde"] = hyde_results
            weights["hyde"] = float(hyde_cfg.get("rrf_weight", 0.75))
            log.info("[HYDE] mode=%s used=%s reason=%s candidates=%d", hyde_mode, bool(hyde_results), reason, len(hyde_results))

        signals = build_weighted_rrf_signal(channels, weights=weights, rrf_k=rrf_k, rrf_scale=float(retrieval_cfg.get("rrf_scale", 10.0)))
        results = sorted(((cid, signal["rrf_score"]) for cid, signal in signals.items()), key=lambda item: item[1], reverse=True)
        counts = " ".join(f"{channel}={len(values)}" for channel, values in channels.items())
        log.info("[FUSION] retriever=%s query=%r %s fused=%d", name, effective_query, counts, len(results))
        return results, signals

    try:
        conn = read_only_connect(str(db_path))
        retriever_name = retriever_name.strip().lower()
        if retriever_name == "sql":
            retriever_name = "sqlite"

        if retriever_name in HYBRID_FUSION_SPECS:
            decomposition_subqueries = decompose_query(cfg, question, backend=backend)

        if retriever_name == "faiss":
            log.info("[QNA] using FAISS retriever")
            retriever = get_faiss_ret()
            raw_results = search_embedding_retriever(retriever, candidate_k)
            results, retrieval_signals = build_single_channel_results(raw_results, "semantic")
        elif retriever_name == "sqlite":
            log.info("[QNA] using SQLite brute-force retriever")
            retriever = SqliteEmbeddingRetriever(conn)
            raw_results = search_embedding_retriever(retriever, candidate_k)
            results, retrieval_signals = build_single_channel_results(raw_results, "semantic")
        elif retriever_name == "bm25":
            log.info("[QNA] using SQLite BM25 retriever")
            retriever = get_bm25_ret()
            raw_results = search_bm25(candidate_k)
            results, retrieval_signals = build_single_channel_results(raw_results, "bm25")
        elif retriever_name == "splade":
            log.info("[QNA] using SPLADE retriever")
            retriever = get_splade_ret()
            raw_results = search_splade(candidate_k)
            results, retrieval_signals = (build_single_channel_results(raw_results, "splade"))
        elif retriever_name == "turbovec":
            log.info("[QNA] using TurboVec retriever")
            retriever = get_turbovec_ret()
            raw_results = search_embedding_retriever(retriever, candidate_k)
            results, retrieval_signals = build_single_channel_results(raw_results, "semantic")
        elif retriever_name in HYBRID_FUSION_SPECS:
            channels = "+".join(sorted(HYBRID_FUSION_SPECS[retriever_name]))
            log.info("[QNA] using %s channels=%s", retriever_name, channels)
            retriever = None
            results, retrieval_signals = search_hybrid_fusion_decomposed(retriever_name, candidate_k)
        else:
            raise RuntimeError(f"Unknown retriever: {retriever_name}")

        chunk_ids = [cid for cid, score in results]
        rank_scores = {cid: (float(sig.get("rrf_score", 0.0)) if isinstance(sig, dict) else float(sig)) for cid, sig in retrieval_signals.items()}
        initial_scores = rank_scores
        filtered_ids = filter_by_intent_source(conn, chunk_ids, intent, min_required=5, max_fallback=20)

        if intent in ("build", "mechanic", "lore", "biography", "location") and len(filtered_ids) < 3:
            deep_k = candidate_k * deep_candidate_multiplier
            log.info("[QNA] intent filter returned too few chunks; deep search k=%d", deep_k)

            if retriever_name == "faiss":
                raw_deep_results = search_embedding_retriever(retriever, deep_k)
                deep_results, retrieval_signals = build_single_channel_results(raw_deep_results, "semantic")
            elif retriever_name == "sqlite":
                raw_deep_results = search_embedding_retriever(retriever, deep_k)
                deep_results, retrieval_signals = build_single_channel_results(raw_deep_results, "semantic")
            elif retriever_name == "bm25":
                raw_deep_results = search_bm25(deep_k)
                deep_results, retrieval_signals = build_single_channel_results(raw_deep_results, "bm25")
            elif retriever_name == "splade":
                raw_deep_results = search_splade(deep_k)
                deep_results, retrieval_signals = build_single_channel_results(raw_deep_results, "splade",)
            elif retriever_name == "turbovec":
                raw_deep_results = search_embedding_retriever(retriever, deep_k)
                deep_results, retrieval_signals = build_single_channel_results(raw_deep_results, "semantic")
            elif retriever_name in HYBRID_FUSION_SPECS:
                force_hyde = "hyde" in HYBRID_FUSION_SPECS[retriever_name]
                deep_results, retrieval_signals = search_hybrid_fusion_decomposed(retriever_name, deep_k, force_hyde=force_hyde, force_reason=f"intent_filter_too_few: count={len(filtered_ids)}")
            else:
                raise RuntimeError(f"Unknown retriever: {retriever_name}")

            deep_ids = [cid for cid, _ in deep_results]
            filtered_ids = filter_by_intent_source(conn, deep_ids, intent, min_required=5, max_fallback=40)
            deep_scores = {cid: (float(sig.get("rrf_score", 0.0)) if isinstance(sig, dict) else float(sig)) for cid, sig in retrieval_signals.items()}
            results = [(cid, deep_scores[cid]) for cid in filtered_ids if cid in deep_scores]
            chunk_ids = [cid for cid, _ in results]
            initial_scores = deep_scores

        else:
            id_set = set(filtered_ids)
            results = [(cid, score) for cid, score in results if cid in id_set]
            chunk_ids = [cid for cid, _ in results]
            initial_scores = {cid: score for cid, score in results}

        multi_hop_cfg = cfg.get("multi_hop", {}) or {}
        multi_hop_supported = retriever_name in HYBRID_FUSION_SPECS

        if multi_hop_supported and as_bool(multi_hop_cfg.get("enabled", False)) and results:
            evidence_k = max(1, int(multi_hop_cfg.get("evidence_k", 6)))
            evidence_ids = [int(cid) for cid, _ in results[:evidence_k]]
            evidence_rows = fetch_chunks(conn, evidence_ids)
            evidence_by_id = {int(row["chunk_id"]): row for row in evidence_rows}
            evidence_chunks = [evidence_by_id[cid] for cid in evidence_ids if cid in evidence_by_id]
            multi_hop_queries = generate_bridge_queries(cfg, question, evidence_chunks, prior_queries=[question, *decomposition_subqueries], backend=backend)

            if multi_hop_queries:
                hop_k = min(candidate_k, int(multi_hop_cfg.get("candidate_k_per_query", 300)))
                hop_runs = []

                for bridge_query in multi_hop_queries:
                    hop_results, hop_signals = search_hybrid_fusion(retriever_name, hop_k, bridge_query)
                    hop_runs.append((bridge_query, hop_results, hop_signals))

                results, retrieval_signals = merge_multi_hop_results(results, retrieval_signals, hop_runs, rrf_k=rrf_k, rrf_scale=float(retrieval_cfg.get("rrf_scale", 10.0)), hop_weight=float(multi_hop_cfg.get("hop_weight", 0.65)), max_total_candidates=int(multi_hop_cfg.get("max_total_candidates", 1800)))
                merged_ids = [int(cid) for cid, _ in results]
                filtered_ids = filter_by_intent_source(conn, merged_ids, intent, min_required=5, max_fallback=40)
                filtered_set = set(filtered_ids)
                results = [(cid, score) for cid, score in results if cid in filtered_set]
                chunk_ids = [cid for cid, _ in results]
                initial_scores = {cid: score for cid, score in results}
                log.info("[MULTIHOP] merged bridge_queries=%d candidates=%d", len(multi_hop_queries), len(results))
        
        chunks = fetch_chunks(conn, chunk_ids)
        lookup_entity: str | None = None
        lookup_facets: set[str] = set()
        resolved_lookup_entity: str | None = None
        lookup_resolution_score = 0.0
        ranking_question = question
        biography_entity: str | None = None

        if intent == "lookup":
            lookup_entity, lookup_facets = extract_lookup_target(question)

            if lookup_entity:
                (resolved_lookup_entity, lookup_resolution_score,) = resolve_lookup_entity_from_chunks(
                    lookup_entity,
                    chunks[:500],
                    minimum_similarity=float(
                        (cfg.get("lookup", {}) or {}).get(
                            "fuzzy_min_similarity",
                            0.84,
                        )
                    ),
                )

                if (resolved_lookup_entity and normalize_title_key(resolved_lookup_entity) != normalize_title_key(lookup_entity)):
                    ranking_question = re.sub(
                        re.escape(lookup_entity),
                        resolved_lookup_entity,
                        question,
                        count=1,
                        flags=re.IGNORECASE,
                    )

                    log.info("[LOOKUP] fuzzy entity resolution raw=%r resolved=%r score=%.3f corrected_question=%r", lookup_entity, resolved_lookup_entity, lookup_resolution_score, ranking_question,)

        elif intent == "biography":
            biography_entity = extract_lookup_entity(question)

        if (ranking_question != question and retriever_name in HYBRID_FUSION_SPECS):
            lookup_cfg = cfg.get("lookup", {}) or {}
            correction_k = min(candidate_k, int(lookup_cfg.get("correction_candidate_k", 300,)))
            correction_results, correction_signals = (search_hybrid_fusion(retriever_name, correction_k, ranking_question))
            results, retrieval_signals = merge_decomposition_runs(
                [
                    (
                        question,
                        results,
                        retrieval_signals,
                        1.0,
                    ),
                    (
                        ranking_question,
                        correction_results,
                        correction_signals,
                        float(
                            lookup_cfg.get(
                                "correction_weight",
                                0.80,
                            )
                        ),
                    ),
                ],
                rrf_k=rrf_k,
                rrf_scale=float(
                    retrieval_cfg.get(
                        "rrf_scale",
                        10.0,
                    )
                ),
                max_total_candidates=int(
                    lookup_cfg.get(
                        "max_total_candidates",
                        1800,
                    )
                ),
            )

            merged_ids = [int(chunk_id) for chunk_id, _score in results]

            filtered_ids = filter_by_intent_source(
                conn,
                merged_ids,
                intent,
                min_required=5,
                max_fallback=40,
            )

            filtered_set = set(filtered_ids)

            results = [
                (chunk_id, score)
                for chunk_id, score in results
                if chunk_id in filtered_set
            ]

            chunk_ids = [
                int(chunk_id)
                for chunk_id, _score in results
            ]

            initial_scores = {
                int(chunk_id): float(score)
                for chunk_id, score in results
            }

            chunks = fetch_chunks(
                conn,
                chunk_ids,
            )

            log.info(
                "[LOOKUP] corrected retrieval "
                "query=%r candidates=%d",
                ranking_question,
                len(results),
            )

        max_per_doc = dedup_max_per_doc

        if intent == "build":
            entity = extract_lookup_entity(question)
            if entity:
                entity_key = normalized_phrase(entity)
                matching = [row for row in chunks if entity_key in normalized_phrase(str(row.get("title") or ""))]
                if len(matching) >= 5:
                    chunks = matching

        if intent == "build":
            max_per_doc = max(dedup_max_per_doc, 6)
        elif intent == "lookup":
            max_per_doc = max(dedup_max_per_doc, 6)
        elif intent in ("biography", "location"):
            max_per_doc = max(dedup_max_per_doc, 2)

        baseline_label, baseline_ord = get_kqm_news_fetch_version_baseline(conn)
        log.info("[QNA] current version baseline from kqm_news: label=%s ord=%s", baseline_label, baseline_ord,)

        if reranker_mode in ("feature", "cross_encoder"):
            chunks = rerank_chunks(ranking_question, chunks, retrieval_signals, baseline_ord,)
            dedupe_scores = {int(row["chunk_id"]): float(row["_rerank_score"]) for row in chunks}
        else:
            dedupe_scores = initial_scores

        chunks = dedupe_chunks(chunks, dedupe_scores, max_per_doc=max_per_doc)
        
        if reranker_mode == "cross_encoder":
            chunks = cross_encoder_rerank(ranking_question, chunks, model_name=reranker_cfg.get("cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"), top_n=int(reranker_cfg.get("cross_encoder_top_n", 32)), batch_size=int(reranker_cfg.get("cross_encoder_batch_size", 8)), max_pair_text_chars=int(reranker_cfg.get("max_pair_text_chars", 1200)))
        elif reranker_mode not in ("none", "feature", "cross_encoder"):
            raise RuntimeError(f"Unknown reranker mode: {reranker_mode}")

        effective_lookup_entity = (
            resolved_lookup_entity
            or lookup_entity
        )

        exact_page_entity: str | None = None

        if (
            intent == "lookup"
            and effective_lookup_entity
            and not lookup_facets
        ):
            # Direct entity definition, such as:
            # "What is Frost Moon?"
            exact_page_entity = effective_lookup_entity

        elif (
            intent == "biography"
            and biography_entity
        ):
            # Identity question, such as:
            # "Who is Columbina?"
            exact_page_entity = biography_entity

        if exact_page_entity:
            lookup_cfg = cfg.get("lookup", {}) or {}

            exact_seed_chunks = fetch_exact_lookup_seed_chunks(
                conn,
                exact_page_entity,
                max_docs=int(
                    lookup_cfg.get(
                        "exact_title_max_docs",
                        2,
                    )
                ),
                chunks_per_doc=int(
                    lookup_cfg.get(
                        "exact_title_chunks_per_doc",
                        4,
                    )
                ),
            )

            if exact_seed_chunks:
                exact_ids = {
                    int(row["chunk_id"])
                    for row in exact_seed_chunks
                }

                chunks = (
                    exact_seed_chunks
                    + [
                        row
                        for row in chunks
                        if int(row["chunk_id"])
                        not in exact_ids
                    ]
                )

                log.info(
                    "[ENTITY] exact-title seeds "
                    "intent=%s entity=%r chunks=%d",
                    intent,
                    exact_page_entity,
                    len(exact_seed_chunks),
                )


        for row in chunks:
            row.pop("_rerank_score", None)

        hyde_enabled = as_bool(hyde_cfg.get("enabled", False))
        hyde_signal_count = sum(1 for signal in (retrieval_signals or {}).values() if isinstance(signal, dict) and signal.get("in_hyde"))
        hyde_used = hyde_signal_count > 0

        if not chunks:
            return RetrievalResult(
                question=question,
                intent=intent,
                build_subtypes=set(build_subtypes),
                broad=broad,
                candidate_chunks=[],
                selected_chunks=[],
                context="",
                retrieval_signals=retrieval_signals,
                baseline_label=baseline_label,
                baseline_ord=baseline_ord,
                strict_fts_query=strict_fts_query_used,
                diagnostics={
                    "retriever": retriever_name,
                    "candidate_k": candidate_k,
                    "top_k": top_k,
                    "candidate_chunk_ids": [],
                    "selected_chunk_ids": [],
                    "hyde_enabled": hyde_enabled,
                    "hyde_used": hyde_used,
                    "hyde_fallback_reason": hyde_fallback_reason,
                    "hyde_error": hyde_error,
                    "hyde_candidate_count": hyde_signal_count,
                    "query_decomposition_enabled": as_bool((cfg.get( "query_decomposition",{}) or {}).get("enabled", False,)),
                    "query_decomposition_used": bool(decomposition_subqueries), "decomposition_subqueries": list(decomposition_subqueries),
                    "multi_hop_enabled": as_bool((cfg.get("multi_hop", {}) or {}).get("enabled", False)),
                    "multi_hop_used": bool(multi_hop_queries),
                    "multi_hop_queries": list(multi_hop_queries),
                },
            )

        for row in chunks[:top_k]:
            log.info("[QNA] chunk_id=%s title=%s source=%s preview=%s", row["chunk_id"], row["title"], row["source"], (row["text"][:200] if row["text"] else "").replace("\n", " "))

        context_max_per_doc = (8 if intent == "lookup" else 4)
        candidate_chunks = [dict(row) for row in chunks]
        if broad:
            selected_chunks = candidate_chunks
        else:
            context_cfg = cfg.get("context_expansion", {}) or {}
            ctx_enabled = as_bool(context_cfg.get("enabled", False))
            if intent == "lookup":
                lookup_cfg = cfg.get("lookup", {}) or {}
                lookup_seed_k = max(1, min(direct_top_k, int(lookup_cfg.get("seed_k", 4,)),),)
                if lookup_facets:
                    selected_chunks = chunks[:lookup_seed_k]
                else:
                    selected_chunks = prefer_entity_seed_chunks(ranking_question, chunks, min_keep=1,)[:lookup_seed_k]
            elif intent in {"biography", "location"}:
                selected_chunks = prefer_entity_seed_chunks(question, chunks[:direct_top_k], min_keep=3,)[:direct_top_k]
            else:
                selected_chunks = chunks[:direct_top_k]

            parent_cfg = cfg.get("parent_child", {}) or {}
            parent_enabled = as_bool(parent_cfg.get("enabled", False))

            if parent_enabled:
                seed_chunks = list(selected_chunks)
                parent_max_total = max(int(parent_cfg.get("max_total_chunks", 16)), len(seed_chunks),)
                parent_chunks = fetch_parent_context_chunks(conn, seed_chunks, max_parents=int(parent_cfg.get("max_parents", 8)), max_total_chunks=parent_max_total,)
                selected_chunks = merge_context_preserving_seeds(seed_chunks, parent_chunks, max_total=parent_max_total, max_per_doc=context_max_per_doc,)
                log.info("[PARENT] expanded selected chunks to parent context chunks=%d", len(selected_chunks))

            if ctx_enabled:
                seed_chunks = list(selected_chunks)
                ctx_max_total = max(int(context_cfg.get("max_total_chunks_after_parent", context_cfg.get("max_total_chunks", 20),)), len(seed_chunks))
                expanded_chunks = expand_context_windows(conn, seed_chunks, before=int(context_cfg.get("before", 1)), after=int(context_cfg.get("after", 1)), max_total_chunks=ctx_max_total,)
                selected_chunks = merge_context_preserving_seeds(seed_chunks, expanded_chunks, max_total=ctx_max_total, max_per_doc=context_max_per_doc,)
                log.info("[CTX_EXPAND] expanded context chunks=%d before=%s after=%s", len(selected_chunks), context_cfg.get("before", 1), context_cfg.get("after", 1))

        selected_chunks = [dict(row) for row in selected_chunks]
        answer_context_cfg = cfg.get("answer_context", {}) or {}
        if intent == "lookup":
            default_max_chunks = 6
            default_max_chars = 9000
        elif intent == "build":
            default_max_chunks = 8
            default_max_chars = 14000
        else:
            default_max_chunks = 8
            default_max_chars = 12000

        selected_chunks = trim_chunks_to_context_budget(selected_chunks, max_chunks=int(answer_context_cfg.get(f"{intent}_max_chunks", default_max_chunks)), max_chars=int(answer_context_cfg.get(f"{intent}_max_chars", default_max_chars)), max_chars_per_chunk=int(answer_context_cfg.get("max_chars_per_chunk", 2200)))
        hyde_selected_count = sum(1 for row in selected_chunks if (retrieval_signals.get(int(row["chunk_id"]), {}).get("in_hyde", False)))
        hyde_used = hyde_selected_count > 0
        context = build_context(selected_chunks)
        if (intent == "version" and baseline_ord is not None):
            major_version = int(baseline_ord) // 100
            minor_version = int(baseline_ord) % 100
            numeric_version = (f"{major_version}.{minor_version}")
            baseline_header = (f"[Current version metadata]\nThe current indexed Genshin Impact version is Version {numeric_version}")

            if baseline_label:
                baseline_header += (f" ({baseline_label}).")
            else:
                baseline_header += "."

            baseline_header += ("\nVersions with expected future release dates are not the current version.")
            context = (baseline_header + "\n\n" + context)

        log.info("[CONTEXT] final chunk IDs=%s", [int(row["chunk_id"]) for row in selected_chunks])
        return RetrievalResult(
            question=question,
            intent=intent,
            build_subtypes=set(build_subtypes),
            broad=broad,
            candidate_chunks=candidate_chunks,
            selected_chunks=selected_chunks,
            context=context,
            retrieval_signals=retrieval_signals,
            baseline_label=baseline_label,
            baseline_ord=baseline_ord,
            strict_fts_query=strict_fts_query_used,
            diagnostics={
                "retriever": retriever_name,
                "candidate_k": candidate_k,
                "top_k": top_k,
                "candidate_chunk_ids": [int(row["chunk_id"]) for row in candidate_chunks],
                "selected_chunk_ids": [int(row["chunk_id"]) for row in selected_chunks],
                "hyde_enabled": hyde_enabled,
                "hyde_used": hyde_used,
                "hyde_candidate_count": hyde_signal_count,
                "hyde_selected_count": hyde_selected_count,
                "query_decomposition_enabled": as_bool((cfg.get("query_decomposition", {}) or {}).get("enabled", False)),
                "query_decomposition_used": bool(decomposition_subqueries),
                "decomposition_subqueries": list(decomposition_subqueries),
                "multi_hop_enabled": as_bool((cfg.get("multi_hop", {}) or {}).get("enabled", False)),
                "multi_hop_used": bool(multi_hop_queries),
                "multi_hop_queries": list(multi_hop_queries),
            })
    finally:
        conn.close()

def fetch_exact_lookup_seed_chunks(conn, entity: str, *, max_docs: int = 2, chunks_per_doc: int = 4) -> list[dict]:
    entity = re.sub(r"\s+", " ", entity).strip()

    if not entity:
        return []

    document_rows = conn.execute(
        """
        SELECT
            d.doc_id,
            d.source,
            d.title
        FROM docs d
        WHERE COALESCE(d.status, 1) = 1
          AND (
                LOWER(TRIM(d.title)) = LOWER(?)
             OR LOWER(d.title) LIKE LOWER(?)
             OR LOWER(d.title) LIKE LOWER(?)
          )
        ORDER BY
            CASE d.source
                WHEN 'genshin_wiki' THEN 0
                WHEN 'game8' THEN 1
                WHEN 'honey' THEN 2
                WHEN 'genshin_gg' THEN 3
                ELSE 4
            END,
            d.doc_id
        LIMIT ?
        """,
        (
            entity,
            f"{entity} |%",
            f"{entity}｜%",
            max(1, int(max_docs)),
        ),
    ).fetchall()

    if not document_rows:
        return []

    ordered_chunk_ids: list[int] = []

    for document in document_rows:
        chunk_rows = conn.execute(
            """
            SELECT c.chunk_id
            FROM chunks c
            WHERE c.doc_id = ?
              AND c.is_active = 1
            ORDER BY
                c.chunk_index,
                c.chunk_id
            LIMIT ?
            """,
            (
                int(document["doc_id"]),
                max(1, int(chunks_per_doc)),
            ),
        ).fetchall()

        ordered_chunk_ids.extend(
            int(row["chunk_id"])
            for row in chunk_rows
        )

    if not ordered_chunk_ids:
        return []

    fetched = fetch_chunks(
        conn,
        ordered_chunk_ids,
    )

    fetched_by_id = {
        int(row["chunk_id"]): dict(row)
        for row in fetched
    }

    return [
        fetched_by_id[chunk_id]
        for chunk_id in ordered_chunk_ids
        if chunk_id in fetched_by_id
    ]

def retrieve_question_context(cfg: dict, question: str, *, retriever_name: str = "hybrid", direct_top_k: int = 12, broad_top_k: int = 60, backend: str | None = None) -> RetrievalResult:
    intent = detect_intent(question)
    build_subtypes = detect_build_subtypes(question) if intent == "build" else set()
    cache = get_retrieval_cache(cfg)

    if cache is None:
        return retrieve_question_context_uncached(cfg, question, retriever_name=retriever_name, direct_top_k=direct_top_k, broad_top_k=broad_top_k, backend=backend)

    db_path = resolve_db_path(cfg)
    cache_cfg = cfg.get("retrieval_cache", {}) or {}
    cache_version = int(cache_cfg.get("version", 1))
    cache_key = make_retrieval_cache_key(question=question, retriever_name=retriever_name, backend=backend, direct_top_k=direct_top_k, intent=f"{intent}:v{cache_version}", subtypes=build_subtypes, db_path=db_path, index_meta={})

    cached = cache.get(cache_key)
    if cached is not None:
        log.info("[RETRIEVAL_CACHE] hit key=%s question=%r", cache_key[:12], question)
        return retrieval_result_from_cache(cached)

    log.info("[RETRIEVAL_CACHE] miss key=%s question=%r", cache_key[:12], question)
    result = retrieve_question_context_uncached(cfg, question, retriever_name=retriever_name, direct_top_k=direct_top_k, broad_top_k=broad_top_k, backend=backend)
    cache.set(cache_key, retrieval_result_to_cache(result))
    log.info("[RETRIEVAL_CACHE] stored key=%s chunks=%d context_chars=%d", cache_key[:12], len(result.selected_chunks), len(result.context))
    return result

def answer_question(cfg: dict, question: str, *, retriever_name: str = "hybrid", direct_top_k: int = 12, broad_top_k: int = 60, summarize_batch_size: int = 8, backend: str | None = None) -> str:
    result = retrieve_question_context(cfg, question, retriever_name=retriever_name, direct_top_k=direct_top_k, broad_top_k=broad_top_k, backend=backend)
    if not result.selected_chunks:
        return "I couldn't retrieve any relevant chunks from the knowledge base."

    if result.broad:
        runtime = cfg.get("runtime", {}) or {}
        provider = str(runtime.get("qa_provider", "ollama")).strip().lower()
        if provider == "llamacpp":
            qa_timeout = int(cfg.get("llamacpp", {}).get("timeout", 300))
        else:
            qa_timeout = int(cfg.get("ollama", {}).get("timeout", 1800))

        notes = []
        for group in chunk_batch(result.candidate_chunks, summarize_batch_size):
            notes.append(summarize_chunk_group(cfg, question, group))
        return synthesize_final_answer(cfg, question, notes, qa_timeout)
    
    answer_style_cfg = cfg.get("answer_style", {}) or {} 
    prompt = build_grounded_answer_prompt(question, result.context, intent=result.intent, build_subtypes=result.build_subtypes, max_recommendations=int(answer_style_cfg.get("max_build_recommendations", 5)))
    answer = generate(cfg, prompt).strip()

    if len(answer.split()) < 3:
        log.warning("[QNA] Generator returned suspiciously short output: %r", answer)

        if result.intent == "lookup":
            return ("I couldn't produce a reliable answer from the retrieved context.")

    return answer