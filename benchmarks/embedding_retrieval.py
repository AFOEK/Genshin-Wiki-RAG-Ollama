from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time

from pathlib import Path
from copy import deepcopy
from contextlib import closing

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rag"))

from core.embed import embed, embedding_model_name
from core.embedding_workspace import activate_hq_workspace
from core.paths import resolve_db_path, resolve_faiss_dir
from qna.retrievers import FaissRetriever
from qna.utils import normalize_query_vec
from qna.retrieval_metrics import DenseRetrievalMetrics, make_dense_metric_record

def load_benchmark(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            positive = row.get("positive") or {}
            question = (row.get("question") or metadata.get("question"))
            chunk_id = row.get("positive_chunk_id", metadata.get("positive_chunk_id", positive.get("chunk_id")))
            doc_id = row.get("positive_doc_id", metadata.get("positive_doc_id", positive.get("doc_id")))
            if not question:
                raise ValueError(f"Missing question on line {line_number}")
            if chunk_id is None or doc_id is None:
                raise ValueError(f"Missing relevance labels on line " f"{line_number}")
            rows.append({
                "id": str(row.get("id") or f"query_{line_number}"),
                "question": question,
                "positive_chunk_id": int(chunk_id),
                "positive_doc_id": int(doc_id),
            })
    if not rows:
        raise RuntimeError("Empty benchmark dataset")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate benchmark query identifiers")
    return rows

def configure_model(base_cfg: dict, model_dir: Path) -> dict:
    cfg = deepcopy(base_cfg)
    model_dir = model_dir.resolve()
    metadata_path = (model_dir / "quantum_config.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mode = metadata["adapter_mode"]
    seed = metadata["seed"]
    regime = (
        "noise"
        if metadata.get("use_ibm_fake_noise")
        else (
            "ideal"
            if metadata.get("shots") is None
            else "ideal_shots"
        )
    )
    preset = (metadata.get("module_target_preset") or "pooled")
    model_id = (f"{regime}__{mode}__{preset}" f"__seed_{seed}")

    cfg.setdefault("runtime", {}).update({
        "embedding_provider": "quantum_peft",
        "faiss_expected_model": "runtime",
        "faiss_model_mismatch": "error",
    })

    cfg.setdefault("quantum_peft", {}).update({
        "model_dir": str(model_dir),
        "model_id": model_id,
        "weights_kind": "adapter",
        "auto_prepare": True,
        "query_max_length": int(metadata.get("query_max_length", 80)),
        "document_max_length": int(metadata.get("document_max_length", 320)),
        "include_document_title": True,
    })
    cfg.setdefault("embedding_workspace", {}).update({"enabled": True, "build_turbovec": False})
    cfg.setdefault("faiss", {}).update({"index_mode": "flat_ip", "metric": "cosine"})
    cfg.setdefault("retrieval_cache", {})["enabled"] = False
    return cfg


def load_chunk_document_mapping(cfg: dict) -> dict:
    db_path = resolve_db_path(cfg)
    with closing(sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)) as conn:
        rows = conn.execute("""
            SELECT c.chunk_id, c.doc_id
            FROM chunks c
            JOIN docs d
                ON c.doc_id = d.doc_id
            WHERE c.is_active = 1
              AND COALESCE(d.status, 1) = 1
        """)
        return {int(chunk_id): int(doc_id) for chunk_id, doc_id in rows}


def validate_benchmark(benchmark: list[dict], chunk_to_doc: dict):
    for row in benchmark:
        actual_doc_id = chunk_to_doc.get(row["positive_chunk_id"])
        if actual_doc_id != row["positive_doc_id"]:
            raise RuntimeError(f"Invalid benchmark relevance label: " f"{row['id']}")


def benchmark_model(base_cfg: dict, model_dir: Path, benchmark: list[dict], output_root: Path, top_k: int = 100):
    cfg = configure_model(base_cfg, model_dir)
    model_id = embedding_model_name(cfg)
    output_dir = output_root / model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 70)
    print("Benchmarking:", model_id)
    print("=" * 70)

    # Build or reuse model-specific embeddings and FAISS.
    prepare_started = time.perf_counter()
    cfg = activate_hq_workspace(cfg)
    prepare_seconds = (time.perf_counter() - prepare_started)

    # Load the same FAISS retriever used by production.
    retriever = FaissRetriever(resolve_faiss_dir(cfg), expected_model=model_id, mismatch_policy="error")
    chunk_to_doc = load_chunk_document_mapping(cfg)
    validate_benchmark(benchmark, chunk_to_doc)
    indexed_ids = {int(cid) for cid in retriever.ids}
    if indexed_ids != set(chunk_to_doc):
        raise RuntimeError("FAISS index does not match active corpus")
    actual_k = min(top_k, retriever.index.ntotal)
    metrics = DenseRetrievalMetrics(top_k=actual_k)

    # Exclude initial model-loading time from warm latency.
    embed(cfg, "Embedding benchmark warm-up", mode="query")

    for i, row in enumerate(benchmark, 1):
        query_started = time.perf_counter()
        blob, dims = embed(cfg, row["question"], mode="query")
        query_vector = normalize_query_vec(blob, dims)
        embedding_ms = (time.perf_counter() - query_started) * 1000
        if dims != retriever.dims:
            raise RuntimeError("Query/index dimension mismatch")

        search_started = time.perf_counter()
        results = retriever.search(query_vector, actual_k)
        search_ms = (time.perf_counter() - search_started) * 1000

        retrieved_chunks = [int(chunk_id) for chunk_id, score in results]
        retrieved_docs = [chunk_to_doc[chunk_id] for chunk_id in retrieved_chunks]

        record = make_dense_metric_record(
            query_id=row["id"],
            question=row["question"],
            positive_chunk_id=(row["positive_chunk_id"]),
            positive_doc_id=(row["positive_doc_id"]),
            retrieved_chunk_ids=retrieved_chunks,
            retrieved_doc_ids=retrieved_docs,
            embedding_latency_ms=embedding_ms,
            search_latency_ms=search_ms,
        )
        metrics.add(record)
        if i % 25 == 0:
            print(f"[{model_id}] " f"{i}/{len(benchmark)} queries")

    assert len(metrics) == len(benchmark)

    metadata = {
        "model": model_id,
        "model_dir": str(model_dir),
        "corpus_chunks": retriever.index.ntotal,
        "benchmark_queries": len(benchmark),
        "workspace_prepare_seconds": prepare_seconds,
        "index_type": "flat_ip",
        "metric": "cosine",
        "database": str(resolve_db_path(cfg)),
    }

    metrics.save(output_dir, run_id="embedding_benchmark", metadata=metadata)
    metrics.print()

    result = {**metadata, **metrics.summary(), "status": "completed"}

    (output_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def discover_models(models_root: Path, seed: int = 42) -> list[Path]:
    models = []
    for config_path in sorted(models_root.rglob("quantum_config.json")):
        model_dir = config_path.parent
        meta = json.loads(config_path.read_text(encoding="utf-8"))
        if int(meta["seed"]) != seed:
            continue
        if meta.get("use_ibm_fake_noise", False):
            continue
        if not (model_dir / "adapter.safetensors").is_file():
            print("[SKIP] Missing adapter:", model_dir)
            continue
        if meta.get("reconstructed_frozen_parameters", False):
            print("[SKIP] Reconstructed adapter:", model_dir)
            continue
        models.append(model_dir)
    if not models:
        raise RuntimeError("No eligible exported models found")
    return models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "rag/config.yaml")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--limit-queries", type=int, default=0)
    parser.add_argument("--select", nargs="+", default=None)

    args = parser.parse_args()
    with args.config.open("r") as f:
        base_cfg = yaml.safe_load(f)
    benchmark = load_benchmark(args.benchmark)
    if args.limit_queries > 0:
        benchmark = benchmark[:args.limit_queries]
    models = discover_models(args.models, seed=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []

    for model_dir in models:
        meta = json.loads((model_dir / "quantum_config.json").read_text())
        if (args.select and meta["adapter_mode"] not in args.select):
            continue
        try:
            result = benchmark_model(
                base_cfg=base_cfg,
                model_dir=model_dir,
                benchmark=benchmark,
                output_root=args.output,
                top_k=args.top_k,
            )
            results.append(result)
        except Exception as exc:
            results.append({"model_dir": str(model_dir), "status": "failed", "error": repr(exc)})
            print("[FAILED]", model_dir, exc)
        pd.json_normalize(results).to_csv(args.output / "benchmark_summary.csv", index=False)

    if not results:
        raise RuntimeError("No models matched the requested selection")
    print("\nBenchmark completed.")



if __name__ == "__main__":
    main()