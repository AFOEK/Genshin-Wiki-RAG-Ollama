from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import requests
import sqlite3
import sys
import time
import yaml
import threading
import uuid

from pathlib import Path
from contextlib import ExitStack
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag"))

from utils.logging_setup import setup_logging
from qna.engine import retrieve_question_context, build_grounded_answer_prompt
from qna.generators import generate
from qna.utils import extract_entity_terms
from qna.types import RetrievalResult

REPO_ROOT = Path(__file__).resolve().parents[1]

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a retrieval-grounded Genshin Impact assistant. "
    "Answer using only the provided context. "
    "Do not invent unsupported facts. "
    "If the context is insufficient, state that there is not "
    "enough evidence."
)

REFUSAL_MARKERS = (
    "i don't have enough evidence",
    "not enough evidence",
    "cannot determine from the context",
    "context does not provide",
    "couldn't retrieve any relevant chunks",
)

NEGATIVE_VALIDATION_SCHEMA = {"type": "object", "properties": {"question_answerable": {"type": "boolean"}, "supports_reference_answer": {"type": "boolean"}, "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}, "reason": {"type": "string"}}, "required": ["question_answerable", "supports_reference_answer", "confidence", "reason"], "additionalProperties": False}

_worker_local = threading.local()
_ollama_semaphore = threading.BoundedSemaphore(2)

@dataclass
class ChunkTaskResult:
    task_index: int
    chunk_id: int

    records: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)

    retrieval_pairs: list[dict] = field(default_factory=list)
    double_negative_records: list[dict] = field(default_factory=list)
    negative_sft_records: list[dict] = field(default_factory=list)

    skipped: int = 0
    errors: list[str] = field(default_factory=list)

def is_refusal(answer: str) -> bool:
    answer_l = answer.lower()

    return any(marker in answer_l for marker in REFUSAL_MARKERS)

def make_record_id(chunk_id: int, question: str) -> str:
    question_hash = hashlib.sha256(
        question.encode("utf-8")
    ).hexdigest()[:12]

    return f"genshin_{chunk_id}_{question_hash}"

def clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_existing_sft_keys(path: Path) -> tuple[set[str], set[str]]:
    record_ids: set[str] = set()
    questions: set[str] = set()

    if not path.exists() or path.stat().st_size == 0:
        return record_ids, questions

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            record_id = str(row.get("id") or "").strip()
            if record_id:
                record_ids.add(record_id)

            metadata = row.get("metadata") or {}
            question = str(metadata.get("question") or "").strip().lower()
            question = re.sub(r"\s+", " ", question).strip()
            if question:
                questions.add(question)

    return record_ids, questions

def is_good_chunk(text: str, title: str, source: str) -> bool:
    t = (text or "").lower()
    title_l = (title or "").lower()

    if len(t) < 350:
        return False
    
    bad_markers = [
        "create your free account",
        "what can you do as a free member",
        "article watchlist",
        "game bookmarks",
        "comment rating",
        "all message boards",
        "friend requests",
        "post details",
        "data:image",
        "comments section",
    ]

    if any(b in t for b in bad_markers):
        return False

    bad_titles = [
        "comment",
        "message board",
        "gallery",
        "media",
        "change history",
    ]

    if any(b in title_l for b in bad_titles):
        return False

    return True

def load_cfg(path: str | None) -> dict:
    if not path:
        return {}

    p = Path(path)

    if not p.is_absolute() and not p.exists():
        p = REPO_ROOT / path

    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def expand_path(x) -> Path:
    return Path(os.path.expandvars(str(x))).expanduser()


def resolve_db_path_from_cfg(cfg: dict) -> Path:
    storage = cfg.get("storage", {}) or {}
    db_rel = cfg.get("db_path", "data/genshin_rag.db")

    candidates = []

    primary = storage.get("primary_root")
    if primary:
        candidates.append((expand_path(primary) / db_rel).resolve())

    secondary = storage.get("secondary_root")
    if secondary:
        candidates.append((expand_path(secondary) / db_rel).resolve())

    for p in candidates:
        if p.exists():
            return p

    if candidates:
        return candidates[0]

    return Path(db_rel).resolve()

def cfg_options(raw: Any, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    output: dict[str, Any] = dict(defaults or {})
    if not raw:
        return output

    if not isinstance(raw, dict):
        raise ValueError(f"Expected options dict, got {type(raw).__name__}")

    int_keys = {"num_ctx", "num_predict", "top_k", "seed"}
    float_keys = {
        "temperature",
        "top_p",
        "min_p",
        "repeat_penalty",
        "presence_penalty",
        "frequency_penalty",
        "mirostat",
        "mirostat_tau",
        "mirostat_eta",
    }

    passthrough_keys = {"stop"}
    allowed = int_keys | float_keys | passthrough_keys

    for key, value in raw.items():
        key = str(key).strip()
        if value is None:
            continue
        if key not in allowed:
            log.warning("[OPTIONS] ignoring unsupported Ollama option: %s=%r", key, value)
            continue
        if key in int_keys:
            output[key] = int(value)
        elif key in float_keys:
            output[key] = float(value)
        else:
            output[key] = value

    return output

def jitter_options(options: dict[str, Any], jitter_cfg: dict[str, Any] | None, rng: random.Random) -> dict[str, Any]:
    out = dict(options or {})

    jitter_cfg = jitter_cfg or {}
    if not cfg_bool(jitter_cfg.get("enabled"), False):
        return out

    def jfloat(name: str):
        lo_key = f"{name}_min"
        hi_key = f"{name}_max"

        if lo_key in jitter_cfg and hi_key in jitter_cfg:
            out[name] = float(rng.uniform(float(jitter_cfg[lo_key]), float(jitter_cfg[hi_key])))

    def jint(name: str):
        lo_key = f"{name}_min"
        hi_key = f"{name}_max"

        if lo_key in jitter_cfg and hi_key in jitter_cfg:
            out[name] = int(rng.randint(int(jitter_cfg[lo_key]), int(jitter_cfg[hi_key])))

    jfloat("temperature")
    jfloat("top_p")
    jfloat("min_p")
    jfloat("repeat_penalty")
    jfloat("presence_penalty")
    jfloat("frequency_penalty")
    jint("top_k")
    jint("num_predict")

    return out

def resolve_output_path(path_value: str, cfg: dict) -> Path:
    p = expand_path(path_value)

    if p.is_absolute():
        return p

    storage = cfg.get("storage", {}) or {}
    primary = storage.get("primary_root")

    if primary:
        return (expand_path(primary) / p).resolve()

    return p.resolve()


def cfg_bool(x, default: bool = False) -> bool:
    if x is None:
        return default
    return str(x).strip().lower() in ("1", "true", "yes", "y", "on")


def cfg_int(x, default: int) -> int:
    if x is None:
        return default
    return int(x)


def cfg_float(x, default: float) -> float:
    if x is None:
        return default
    return float(x)

def resolve_user_path(value: str | Path) -> Path:
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else (Path.cwd() / path).resolve()

def cfg_sources(x, default: str) -> list[str]:
    if x is None:
        x = default

    if isinstance(x, list):
        return [str(s).strip() for s in x if str(s).strip()]

    return [s.strip() for s in str(x).split(",") if s.strip()]

def get_worker_http_session() -> requests.Session:
    session = getattr(_worker_local, "http_session", None)

    if session is None:
        session = requests.Session()
        _worker_local.http_session = session

    return session

def ollama_generate(base_url: str, model: str, prompt: str, timeout: int = 300, response_format: dict | str | None = None, temperature: float = 0.2, top_p: float = 0.9, repeat_penalty: float = 1.05, min_p: float = 0.05, thinking: bool | str | None = None, options: dict[str, Any] | None = None) -> str:
    request_options: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "min_p": min_p,
        "repeat_penalty": repeat_penalty,
    }

    if options:
        request_options.update(options)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": request_options,
    }

    if thinking is not None:
        payload["think"] = thinking

    if response_format is not None:
        payload["format"] = response_format

    with _ollama_semaphore:
        response = get_worker_http_session().post(f"{base_url.rstrip('/')}/api/generate", json=payload, timeout=timeout)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise requests.HTTPError(
                f"{exc}; body={response.text[:1000]!r}; model={model!r}; url={response.url!r}"
            ) from exc

    data = response.json()
    thinking_trace = str(data.get("thinking") or "").strip()
    raw_answer = str(data.get("response") or "").strip()
    done_reason = str(data.get("done_reason") or "").strip()
    answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL | re.IGNORECASE,).strip()
    log.info(
    "[DATASET_OLLAMA] model=%s think=%r thinking_chars=%d "
    "response_chars=%d response_preview=%r "
    "prompt_tokens=%s output_tokens=%s "
    "done_reason=%r total_duration=%.2fs options=%s",
    model,
    thinking,
    len(thinking_trace),
    len(answer),
    answer[:120],
    data.get("prompt_eval_count"),
    data.get("eval_count"),
    done_reason,
    float(data.get("total_duration", 0)) / 1_000_000_000,
    request_options)
    if not answer:
        if thinking_trace or "<think>" in raw_answer.lower() or done_reason == "length":
            raise ValueError(
                f"Ollama produced thinking but no final answer: model={model!r}, "
                f"think={thinking!r}, done_reason={done_reason!r}, "
                f"thinking_chars={len(thinking_trace)}, raw_response_chars={len(raw_answer)}"
            )

        raise ValueError("Ollama returned an empty response")
    return answer

def stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)

def process_source_row(task_index: int, row: dict, *, cfg: dict, settings: dict[str, Any]) -> ChunkTaskResult:
    chunk_id = int(row["chunk_id"])
    result = ChunkTaskResult(task_index=task_index, chunk_id=chunk_id)
    doc_id = int(row["doc_id"])
    source = str(row.get("source") or "")
    url = str(row.get("url") or "")
    title = str(row.get("title") or "")
    text = clean_text(str(row.get("text") or ""))
    # uid7 = uuid.uuid7()
    # uid4 = uuid.uuid4()
    # rng = random.Random(int(uid7.hex[:5], 16) + settings["seed"] + chunk_id + int(uid4.hex[:-5], 16))
    draft_rng = random.Random(stable_seed(settings["seed"], chunk_id, "draft"))
    answer_rng = random.Random(stable_seed(settings["seed"], chunk_id, "answer"))
    validator_rng = random.Random(stable_seed(settings["seed"], chunk_id, "validator"))

    if not is_good_chunk(text, title, source):
        result.skipped += 1
        result.rejected.append(
            {
                "reason": "source_chunk_not_useful",
                "positive_chunk_id": chunk_id,
                "positive_doc_id": doc_id,
                "source": source,
                "title": title,
            }
        )
        return result

    prompt = make_prompt(title=title, source=source, url=url, text=text[: settings["max_chars"]], n=settings["qa_per_chunk"])

    try:
        draft_options = jitter_options(
            settings.get("draft_options") or {},
            settings.get("draft_jitter") or {},
            draft_rng
        )
        raw = ollama_generate(settings["ollama_url"], settings["draft_model"], prompt, timeout=settings["request_timeout"], thinking=settings["draft_think"], options=draft_options)
        items = extract_json_array(raw)

    except Exception as exc:
        result.skipped += 1
        result.rejected.append(
            {
                "reason": "draft_generation_failed",
                "positive_chunk_id": chunk_id,
                "positive_doc_id": doc_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return result

    if not items:
        log.warning("[DRAFT_EMPTY] Retrying chunk_id=%d source=%s title=%r raw=%r", chunk_id, source, title, raw[:120])
        retry_prompt = (
        prompt + """
    Retry instruction:
    The source passed programmatic content filtering.
    If it contains even one explicit factual statement, return exactly one
    useful and directly supported question-reference_answer pair.

    Do not return [] merely because the text is fragmentary, contains tables,
    or lacks a complete article introduction.

    Return [] only if there is genuinely no factual statement in the source.
    """)
        retry_options = dict(draft_options)
        retry_options.update(
            {
                "temperature": 0.0,
                "top_p": 0.9,
                "top_k": 20,
                "min_p": 0.0,
                "repeat_penalty": 1.0,
                "num_predict": 512,
            }
        )

        try:
            retry_raw = ollama_generate(
                settings["ollama_url"],
                settings["draft_model"],
                retry_prompt,
                timeout=settings["request_timeout"],
                thinking=False,
                options=retry_options,
            )

            items = extract_json_array(retry_raw)

        except Exception as exc:
            log.warning("[DRAFT_EMPTY] retry failed chunk_id=%d error=%s: %s", chunk_id, type(exc).__name__, exc)
        
        if not items:
            result.skipped += 1
            result.rejected.append(
                {
                    "reason": "draft_generator_returned_empty_array",
                    "positive_chunk_id": chunk_id,
                    "positive_doc_id": doc_id,
                }
            )
            return result

    positive = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "text": text,
        "source": source,
        "url": url,
        "title": title,
    }

    for item in items[:settings["qa_per_chunk"]]:
        if not isinstance(item, dict):
            result.skipped += 1
            result.rejected.append(
                {
                    "reason": "draft_item_not_object",
                    "positive_chunk_id": chunk_id,
                    "positive_doc_id": doc_id,
                    "draft_item": item,
                }
            )
            continue

        question = str(item.get("question", "")).strip()
        reference_answer = str(item.get("reference_answer") or item.get("answer") or "").strip()

        if not question or not reference_answer:
            result.skipped += 1
            result.rejected.append(
                {
                    "reason": "missing_question_or_reference_answer",
                    "positive_chunk_id": chunk_id,
                    "positive_doc_id": doc_id,
                    "draft_item": item,
                }
            )
            continue

        try:
            answer_options = jitter_options(
                settings.get("answer_options") or {},
                settings.get("answer_jitter") or {},
                answer_rng,
            )
            record, rejection, retrieval = process_generated_pair(
                cfg,
                question=question,
                reference_answer=reference_answer,
                positive=positive,
                retriever_name=settings["retriever_name"],
                direct_top_k=settings["direct_top_k"],
                backend=settings["backend"],
                require_positive_document=settings["require_positive_document"],
                answer_model=settings["answer_model"],
                answer_think=settings["answer_think"],
                answer_options=answer_options
            )

        except Exception as exc:
            result.skipped += 1
            result.rejected.append(
                {
                    "reason": "processing_exception",
                    "question": question,
                    "reference_answer": reference_answer,
                    "positive_chunk_id": chunk_id,
                    "positive_doc_id": doc_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue

        if record is None:
            result.skipped += 1

            result.rejected.append(
                rejection
                or {
                    "reason": "unknown_pair_rejection",
                    "question": question,
                    "positive_chunk_id": chunk_id,
                    "positive_doc_id": doc_id,
                }
            )
            continue

        result.records.append(record)

        if settings["make_negatives"]:
            try:
                validator_options = jitter_options(
                    settings.get("validator_options") or {},
                    settings.get("validator_jitter") or {},
                    validator_rng,
                )
                hard_candidates = get_hard_negative_candidates(
                    retrieval,
                    positive,
                    pool_size=settings["negative_pool_size"],
                )

                hard_negatives = select_validated_negatives(
                    hard_candidates,
                    question=question,
                    reference_answer=reference_answer,
                    count=settings["hard_negative_count"],
                    ollama_url=settings["ollama_url"],
                    validator_model=settings["validator_model"],
                    min_confidence=settings[
                        "negative_min_confidence"
                    ],
                    timeout=settings["request_timeout"],
                    think=settings["validator_think"],
                    options=validator_options,
                )

                hard_doc_ids = {
                    int(candidate["doc_id"])
                    for candidate in hard_negatives
                }

                negative_conn = sqlite3.connect(
                    f"file:{settings['db_path']}?mode=ro",
                    uri=True,
                )
                negative_conn.row_factory = sqlite3.Row

                try:
                    easy_candidates = find_easy_negative_candidates(
                        negative_conn,
                        positive=positive,
                        question=question,
                        count=max(
                            settings["easy_negative_count"] * 4,
                            8,
                        ),
                        excluded_doc_ids=hard_doc_ids,
                        seed=settings["seed"],
                    )
                finally:
                    negative_conn.close()

                easy_negatives = select_validated_negatives(
                    easy_candidates,
                    question=question,
                    reference_answer=reference_answer,
                    count=settings["easy_negative_count"],
                    ollama_url=settings["ollama_url"],
                    validator_model=settings["validator_model"],
                    min_confidence=settings[
                        "negative_min_confidence"
                    ],
                    timeout= settings["request_timeout"],
                    think=settings["validator_think"],
                    options=validator_options,
                )

                if hard_negatives or easy_negatives:
                    pair_record = make_retrieval_pair_record(
                        base_id=record["id"],
                        question=question,
                        positive=positive,
                        hard_negatives=hard_negatives,
                        easy_negatives=easy_negatives,
                        max_chars=settings["negative_text_chars"],
                    )
                    result.retrieval_pairs.append(pair_record)

                double_record = make_double_negative_record(
                    base_id=record["id"],
                    question=question,
                    positive=positive,
                    hard_negatives=hard_negatives,
                    easy_negatives=easy_negatives,
                    max_chars=settings["negative_text_chars"],
                )

                if double_record is not None:
                    result.double_negative_records.append(
                        double_record
                    )

                selected_negatives = (
                    hard_negatives + easy_negatives
                )[: settings["negative_sft_per_positive"]]

                for negative in selected_negatives:
                    negative_type = (
                        "hard"
                        if int(negative["doc_id"]) in hard_doc_ids
                        else "easy"
                    )

                    negative_sft = make_negative_sft_record(
                        base_id=record["id"],
                        question=question,
                        negative=negative,
                        negative_type=negative_type,
                        max_chars=settings["negative_text_chars"],
                    )

                    result.negative_sft_records.append(
                        negative_sft
                    )
            except Exception as exc:
                log.warning(
                "[NEGATIVE] generation failed "
                "chunk_id=%d question=%r err=%s: %s", chunk_id, question, type(exc).__name__, exc)

                result.rejected.append(
                    {
                        "reason": "negative_generation_failed",
                        "question": question,
                        "positive_chunk_id": chunk_id,
                        "positive_doc_id": doc_id,
                        "origin_record_id": record["id"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

    if settings["sleep"] > 0:
        time.sleep(settings["sleep"])

    return result

def run_bounded_workers(rows: list[dict], *, workers: int, max_inflight: int, cfg: dict, settings: dict[str, Any]):
    if workers <= 0:
        raise ValueError(
            f"workers must be positive, got {workers}"
        )

    if max_inflight < workers:
        max_inflight = workers

    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="dataset",
    ) as executor:
        pending: dict[Future, int] = {}
        next_submit = 0

        while next_submit < len(rows) or pending:
            while (
                next_submit < len(rows)
                and len(pending) < max_inflight
            ):
                task_index = next_submit

                future = executor.submit(
                    process_source_row,
                    task_index,
                    rows[task_index],
                    cfg=cfg,
                    settings=settings,
                )

                pending[future] = task_index
                next_submit += 1

            completed, _ = wait(
                pending,
                return_when=FIRST_COMPLETED,
            )

            for future in completed:
                task_index = pending.pop(future)

                try:
                    result = future.result()

                except Exception as exc:
                    result = ChunkTaskResult(
                        task_index=task_index,
                        chunk_id=int(
                            rows[task_index]["chunk_id"]
                        ),
                        skipped=1,
                        rejected=[
                            {
                                "reason": "worker_crashed",
                                "task_index": task_index,
                                "positive_chunk_id": int(
                                    rows[task_index]["chunk_id"]
                                ),
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        ],
                    )

                yield result

def extract_json_array(text: str) -> list[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array found in model output")

    data = json.loads(text[start:end + 1])

    if not isinstance(data, list):
        raise ValueError("Model output is not a JSON array")
    return data

def find_document_rank(chunks: list[dict], doc_id: int) -> int | None:
    for rank, chunk in enumerate(chunks, start=1):
        if int(chunk["doc_id"]) == int(doc_id):
            return rank
    return None

def compact_chunk(row: dict | sqlite3.Row, max_chars: int = 1400) -> dict:
    data = dict(row)

    return {
        "chunk_id": int(data["chunk_id"]),
        "doc_id": int(data["doc_id"]),
        "source": data.get("source"),
        "title": data.get("title"),
        "url": data.get("url"),
        "text": clean_text(data.get("text") or "")[:max_chars],
    }

def extract_json_object(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start < 0 or end <= start:
        raise ValueError("No JSON object found")

    data = json.loads(text[start:end + 1])

    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")

    return data

def validate_negative_candidate(*, question: str, reference_answer: str, candidate: dict, ollama_url: str, validator_model: str, min_confidence: float, timeout: int, think: bool | str | None, options: dict[str, Any] | None = None) -> tuple[bool, dict]:
    candidate_context = f"Title: {candidate.get('title')}\nSource: {candidate.get('source')}\nURL: {candidate.get('url')}\n\nText:\n{clean_text(candidate.get('text') or '')[:1800]}"
    prompt = f"Determine whether the candidate passage answers the question or supports the reference answer. Return only the required JSON object.\n\nQuestion:\n{question}\n\nReference answer:\n{reference_answer}\n\nCandidate passage:\n{candidate_context}"
    last_error: Exception | None = None
    for attempt in range(5):
        raw = ollama_generate(ollama_url, validator_model, prompt, timeout=timeout, response_format=NEGATIVE_VALIDATION_SCHEMA, temperature=0.0, thinking=think, options=options)
        try:
            result = json.loads(raw)
            break
        except json.JSONDecodeError as exc:
            last_error = exc
            log.warning("[NEGATIVE] malformed validator JSON attempt=%d/5 chunk_id=%s raw=%r", attempt + 1, candidate.get("chunk_id"), raw[:500])
    else:
        raise ValueError(f"Validator returned invalid JSON after 2 attempts: {last_error}")
    confidence = float(result["confidence"])
    question_answerable = bool(result["question_answerable"])
    supports_reference = bool(result["supports_reference_answer"])
    return not question_answerable and not supports_reference and confidence >= min_confidence, result

def get_hard_negative_candidates(retrieval: RetrievalResult, positive: dict, *, pool_size: int) -> list[dict]:
    positive_doc_id = int(positive["doc_id"])

    output: list[dict] = []
    seen_docs = {positive_doc_id}

    for candidate in retrieval.candidate_chunks:
        candidate = dict(candidate)
        doc_id = int(candidate["doc_id"])

        if doc_id in seen_docs:
            continue

        text = clean_text(candidate.get("text") or "")
        title = candidate.get("title") or ""
        source = candidate.get("source") or ""

        if not is_good_chunk(text, title, source):
            continue

        seen_docs.add(doc_id)
        output.append(candidate)

        if len(output) >= pool_size:
            break

    return output

def select_validated_negatives(candidates: list[dict], *, question: str, reference_answer: str, count: int, ollama_url: str, validator_model: str, min_confidence: float, timeout: int, think: bool | str | None, options: dict[str, Any] | None = None,) -> list[dict]:
    accepted: list[dict] = []

    for candidate in candidates:
        try:
            valid, validation = validate_negative_candidate(
                question=question,
                reference_answer=reference_answer,
                candidate=candidate,
                ollama_url=ollama_url,
                validator_model=validator_model,
                min_confidence=min_confidence,
                timeout=timeout,
                think=think,
                options=options
            )
        except Exception as exc:
            log.warning(
                "[NEGATIVE] validation failed chunk_id=%s: %s",
                candidate.get("chunk_id"),
                exc,
            )
            continue

        if not valid:
            continue

        candidate = dict(candidate)
        candidate["negative_validation"] = validation
        accepted.append(candidate)

        if len(accepted) >= count:
            break

    return accepted

def find_easy_negative_candidates(conn: sqlite3.Connection, *, positive: dict, question: str, count: int, excluded_doc_ids: set[int], seed: int, max_attempts: int = 100) -> list[dict]:
    cur = conn.cursor()
    cur.execute("SELECT MAX(chunk_id) FROM chunks")
    max_chunk_id = int(cur.fetchone()[0] or 0)

    if max_chunk_id <= 0:
        return []

    entity_terms = [
        term.lower()
        for term in extract_entity_terms(question)
        if len(term) >= 3
    ]

    rng = random.Random(
        seed + int(positive["chunk_id"])
    )

    output: list[dict] = []
    seen_docs = {
        int(positive["doc_id"]),
        *excluded_doc_ids,
    }

    for _ in range(max_attempts):
        if len(output) >= count:
            break

        start_id = rng.randint(1, max_chunk_id)

        cur.execute(
            """
            SELECT
                c.chunk_id,
                c.doc_id,
                c.text,
                d.source,
                d.url,
                d.title
            FROM chunks c
            JOIN docs d ON d.doc_id = c.doc_id
            WHERE c.chunk_id >= ?
              AND c.is_active = 1
              AND COALESCE(d.status, 1) = 1
            ORDER BY c.chunk_id
            LIMIT 16
            """,
            (start_id,),
        )

        for row in cur.fetchall():
            candidate = dict(row)
            doc_id = int(candidate["doc_id"])

            if doc_id in seen_docs:
                continue

            text = clean_text(candidate.get("text") or "")
            title = candidate.get("title") or ""
            combined = f"{title}\n{text}".lower()

            if len(text) < 350:
                continue

            if any(term in combined for term in entity_terms):
                continue

            if not is_good_chunk(
                text,
                title,
                candidate.get("source") or "",
            ):
                continue

            seen_docs.add(doc_id)
            output.append(candidate)

            if len(output) >= count:
                break

    return output

def make_prompt(title: str, source: str, url: str, text: str, n: int) -> str:
    return f"""
You are generating candidate supervised fine-tuning examples
for a retrieval-grounded Genshin Impact assistant.

Use ONLY the source context below.

Generate {n} high-quality question and reference-answer pairs.

Return format:
- Return valid JSON only.
- Return a JSON array.
- Each item must have exactly:
  - question
  - reference_answer

Hard rules:
- The reference_answer must be directly supported by the source context.
- Do not use external knowledge.
- Do not ask questions requiring information absent from the source.
- Do not generate questions about page metadata, comments, navigation, ads, membership prompts, unrelated links, or boilerplate.
- The source has already passed a programmatic usefulness filter.
- Even if the text is fragmentary, extract the clearest explicit factual claim that can support a useful question.
- Do not return an empty array merely because the source is imperfectly formatted.
- Do not generate questions where the answer is only identical to the page title.
- Do not generate vague questions like "What is this page about?"
- Keep reference answers concise but complete.

Retrieval-friendly question rules:
- Every question must include the main entity name from the Source title when applicable.
- Prefer questions that a real player would ask.
- Prefer gameplay, build, material, location, quest, lore, event, enemy, item-use, or mechanic questions.
- Avoid alternate-language-name questions unless the source context is clearly a language table.
- Avoid asking about Japanese, Korean, Spanish, French, Thai, German, Portuguese, Chinese, or pronunciation unless the source is specifically about names/translations.
- Avoid questions about lists of unrelated names unless the source clearly explains the relationship between them.
- Avoid questions whose answer depends only on one isolated title line.

Quality rules:
- Each question should be answerable from the source context alone.
- Each reference_answer should use the same entity names and terminology found in the source.
- Do not over-explain.
- Do not invent recommendations, rankings, locations, versions, materials, or mechanics.
- If only one good question can be made, return only one item.
- Return [] only when the source contains no explicit factual claim from which any directly supported question can be formed.

Good example:
[
  {{
    "question": "What materials are needed to ascend Zhongli?",
    "reference_answer": "Zhongli's ascension materials include ... "
  }}
]

Bad examples:
[
  {{
    "question": "What is the official English name?",
    "reference_answer": "The official English name is Brightwood Room Doors."
  }},
  {{
    "question": "What are the Japanese and Korean names?",
    "reference_answer": "..."
  }},
  {{
    "question": "What is this page about?",
    "reference_answer": "..."
  }}
]

Source title:
{title}

Source name:
{source}

Source URL:
{url}

Source context:
{text}
""".strip()

def fetch_chunks(conn: sqlite3.Connection, *, sources: list[str], limit: int, min_chars: int, max_chars: int, seed: int) -> list[sqlite3.Row]:
    source_placeholders = ",".join("?" for _ in sources)

    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT
            c.chunk_id,
            c.doc_id,
            c.text,
            d.source,
            d.url,
            d.title
        FROM chunks c
        JOIN docs d ON d.doc_id = c.doc_id
        WHERE c.is_active = 1
        AND COALESCE(d.status, 1) = 1
        AND d.source IN ({source_placeholders})
        AND LENGTH(c.text) BETWEEN ? AND ?
        ORDER BY (
            (c.chunk_id * 1103515245 + ?) % 2147483647
        )
        LIMIT ?
        """,
        (
            *sources,
            min_chars,
            max_chars,
            seed,
            limit,
        ),
    )

    return cur.fetchall()

def process_generated_pair(cfg: dict, *, question: str, reference_answer: str, positive: dict, retriever_name: str, direct_top_k: int, backend: str | None, require_positive_document: bool, answer_model: str = "llama3.2:3b", answer_think: bool | str | None = None, answer_options: dict[str, Any] | None = None) -> tuple[dict | None, dict | None, RetrievalResult]:
    retrieval = retrieve_question_context(cfg, question, retriever_name=retriever_name, direct_top_k=direct_top_k, backend=backend)
    candidate_doc_rank = find_document_rank(retrieval.candidate_chunks, int(positive["doc_id"]))
    selected_doc_rank = find_document_rank(retrieval.selected_chunks, int(positive["doc_id"]))

    if (require_positive_document and selected_doc_rank is None):
        return None, {
            "reason": "positive_document_not_retrieved",
            "question": question,
            "reference_answer": reference_answer,
            "positive_chunk_id": int(
                positive["chunk_id"]
            ),
            "positive_doc_id": int(
                positive["doc_id"]
            ),
            "candidate_doc_rank": candidate_doc_rank,
            "selected_chunk_ids": retrieval.diagnostics.get(
                "selected_chunk_ids",
                [],
            ),
        }, retrieval

    if not retrieval.context.strip():
        return None, {
            "reason": "empty_retrieved_context",
            "question": question,
            "positive_chunk_id": int(
                positive["chunk_id"]
            ),
            "positive_doc_id": int(
                positive["doc_id"]
            ),
        }, retrieval

    answer_style_cfg = cfg.get("answer_style", {}) or {}
    answer_prompt = build_grounded_answer_prompt(question, retrieval.context, intent=retrieval.intent, build_subtypes=retrieval.build_subtypes, max_recommendations=int(answer_style_cfg.get("max_build_recommendations", 5)))
    final_answer = str(generate(cfg, answer_prompt, model_override=answer_model, think_override=answer_think, options_override=answer_options)).strip()

    if not final_answer:
        return None, {
            "reason": "empty_generated_answer",
            "question": question,
            "positive_chunk_id": int(
                positive["chunk_id"]
            ),
        }, retrieval

    if is_refusal(final_answer):
        return None, {
            "reason": "generator_refused",
            "question": question,
            "reference_answer": reference_answer,
            "final_answer": final_answer,
            "positive_chunk_id": int(
                positive["chunk_id"]
            ),
            "selected_chunk_ids": retrieval.diagnostics.get(
                "selected_chunk_ids",
                [],
            ),
        }, retrieval

    intent = retrieval.intent
    subtypes = sorted(retrieval.build_subtypes)
    entity_terms = extract_entity_terms(question)
    record_id = make_record_id(int(positive["chunk_id"]), question)

    user_content = (
        f"Question:\n{question}\n\n"
        f"Context:\n{retrieval.context}"
    )

    record = {
        "id": record_id,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            },
            {
                "role": "assistant",
                "content": final_answer,
            },
        ],
        "metadata": {
            "type": "retrieval_grounded_sft",
            "question": question,
            "reference_answer": reference_answer,
            "source": positive["source"],
            "url": positive["url"],
            "title": positive["title"],
            "positive_chunk_id": int(
                positive["chunk_id"]
            ),
            "positive_doc_id": int(
                positive["doc_id"]
            ),
            "intent": intent,
            "build_subtypes": subtypes,
            "entity_terms": entity_terms,
            "retriever": retriever_name,
            "candidate_doc_rank": candidate_doc_rank,
            "selected_doc_rank": selected_doc_rank,
            "retrieved_chunk_ids": (
                retrieval.diagnostics.get(
                    "selected_chunk_ids",
                    [],
                )
            ),
            "strict_fts_query": (
                retrieval.strict_fts_query
            ),
            "retrieval_validated": selected_doc_rank is not None,
            "answer_support_validated": False,
            "human_verified": False,
        },
    }

    return record, None, retrieval

def make_negative_sft_record(*, base_id: str, question: str, negative: dict, negative_type: str, max_chars: int) -> dict:
    context = (
        f"[chunk_id={negative['chunk_id']}] "
        f"[source_name={negative.get('source')}]\n"
        f"Title: {negative.get('title')}\n"
        f"URL: {negative.get('url')}\n\n"
        f"{clean_text(negative.get('text') or '')[:max_chars]}"
    )

    return {
        "id": (
            f"{base_id}_negative_"
            f"{negative_type}_{negative['chunk_id']}"
        ),
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Context:\n{context}"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "I don't have enough evidence in the "
                    "provided context to answer that."
                ),
            },
        ],
        "metadata": {
            "type": "negative_answerability",
            "negative_type": negative_type,
            "origin_record_id": base_id,
            "negative_chunk_id": int(
                negative["chunk_id"]
            ),
            "negative_doc_id": int(
                negative["doc_id"]
            ),
            "source": negative.get("source"),
            "title": negative.get("title"),
            "url": negative.get("url"),
            "retrieval_validated": True,
            "human_verified": False,
        },
    }

def make_retrieval_pair_record(*, base_id: str, question: str, positive: dict, hard_negatives: list[dict], easy_negatives: list[dict], max_chars: int) -> dict:
    return {
        "id": f"{base_id}_retrieval",
        "origin_record_id": base_id,
        "query": question,
        "positive": compact_chunk(
            positive,
            max_chars,
        ),
        "hard_negatives": [
            compact_chunk(row, max_chars)
            for row in hard_negatives
        ],
        "easy_negatives": [
            compact_chunk(row, max_chars)
            for row in easy_negatives
        ],
    }

def make_double_negative_record(*, base_id: str, question: str, positive: dict, hard_negatives: list[dict], easy_negatives: list[dict], max_chars: int) -> dict | None:
    negative_items: list[tuple[str, dict]] = []

    if hard_negatives:
        negative_items.append(
            ("hard", hard_negatives[0])
        )

    if easy_negatives:
        negative_items.append(
            ("easy", easy_negatives[0])
        )

    if len(negative_items) < 2 and len(hard_negatives) > 1:
        negative_items.append(
            ("hard", hard_negatives[1])
        )

    if len(negative_items) < 2:
        return None

    first_type, first = negative_items[0]
    second_type, second = negative_items[1]

    return {
        "id": f"{base_id}_double_negative",
        "origin_record_id": base_id,
        "query": question,
        "positive": compact_chunk(
            positive,
            max_chars,
        ),
        "negative_1": {
            "type": first_type,
            **compact_chunk(first, max_chars),
        },
        "negative_2": {
            "type": second_type,
            **compact_chunk(second, max_chars),
        },
    }

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--config", default="rag/config.yaml")
    ap.add_argument("--db", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ollama-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--qa-per-chunk", type=int, default=None)
    ap.add_argument("--min-chars", type=int, default=None)
    ap.add_argument("--max-chars", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sources", default=None)
    ap.add_argument("--sleep", type=float, default=None)
    ap.add_argument("--make-negatives", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--retrieval-pairs-out", default=None)
    ap.add_argument("--double-negative-out", default=None)
    ap.add_argument("--sft-negative-out", default=None)
    ap.add_argument("--hard-negatives", type=int, default=None)
    ap.add_argument("--easy-negatives", type=int, default=None)
    ap.add_argument("--negative-pool-size", type=int, default=None)
    ap.add_argument("--negative-text-chars", type=int, default=None)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--append", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--flush-every", type=int, default=None)

    args = ap.parse_args()

    cfg = load_cfg(args.config)
    setup_logging(
        cfg.get("logging", {}).get("file"),
        cfg.get("logging", {}).get("level", "INFO")
    )
    ds_cfg = cfg.get("dataset_creation", {}) or {}
    retriever_name = str(ds_cfg.get("retriever", "hybrid")).strip().lower()
    backend = str(args.backend or ds_cfg.get("backend") or cfg.get("runtime", {}).get("embedding_provider", "ollama")).strip().lower()
    append_outputs = cfg_bool(args.append, cfg_bool(ds_cfg.get("append", True)))
    flush_every = cfg_int(args.flush_every, cfg_int(ds_cfg.get("flush_every", 25), 25))
    direct_top_k = cfg_int(ds_cfg.get("direct_top_k"), 10)
    require_positive_document = cfg_bool(ds_cfg.get("require_positive_document"), True)
    ollama_cfg = cfg.get("ollama", {}) or {}

    args.ollama_url = args.ollama_url or ds_cfg.get("ollama_url") or ollama_cfg.get("base_url", "http://localhost:11434")
    default_model = str(args.model or ds_cfg.get("model") or ollama_cfg.get("qa_model") or "llama3.2:3b").strip()
    
    args.limit = cfg_int(args.limit, cfg_int(ds_cfg.get("limit"), 1000))
    args.qa_per_chunk = cfg_int(args.qa_per_chunk, cfg_int(ds_cfg.get("qa_per_chunk"), 2))
    args.min_chars = cfg_int(args.min_chars, cfg_int(ds_cfg.get("min_chars"), 500))
    args.max_chars = cfg_int(args.max_chars, cfg_int(ds_cfg.get("max_chars"), 2500))
    args.seed = cfg_int(args.seed, cfg_int(ds_cfg.get("seed"), 1337))
    args.sleep = cfg_float(args.sleep, cfg_float(ds_cfg.get("sleep"), 0.0))

    make_negatives = cfg_bool(args.make_negatives, cfg_bool(ds_cfg.get("make_negatives"), False))
    hard_negative_count = cfg_int(args.hard_negatives, cfg_int(ds_cfg.get("hard_negatives"), 2))
    easy_negative_count = cfg_int(args.easy_negatives, cfg_int(ds_cfg.get("easy_negatives"), 2))
    negative_pool_size = cfg_int(args.negative_pool_size, cfg_int(ds_cfg.get("negative_pool_size"), 20))
    negative_text_chars = cfg_int(args.negative_text_chars, cfg_int(ds_cfg.get("negative_text_chars"), 1400))
    draft_model = str(ds_cfg.get("draft_model") or default_model).strip()
    answer_model = str(ds_cfg.get("answer_model") or default_model).strip()
    validator_model = str(ds_cfg.get("validator_model") or answer_model).strip()
    negative_min_confidence = cfg_float(ds_cfg.get("negative_validation_confidence"),0.80)
    negative_sft_per_positive = cfg_int(ds_cfg.get("negative_sft_per_positive"),1)
    workers = cfg_int(ds_cfg.get("workers"), 2)
    max_inflight = cfg_int(ds_cfg.get("max_inflight"), workers * 2)
    preserve_output_order = cfg_bool(ds_cfg.get("preserve_output_order"), True)
    request_timeout = cfg_int(ds_cfg.get("request_timeout"), 600)

    draft_think = ds_cfg.get("draft_think", False)
    answer_think = ds_cfg.get("answer_think", False)
    validator_think = ds_cfg.get("validator_think", False)

    draft_options = cfg_options(
        ds_cfg.get("draft_options"),
        {
            "temperature": 0.25,
            "top_p": 0.9,
            "top_k": 50,
            "min_p": 0.05,
            "repeat_penalty": 1.05,
            "num_predict": 1024,
            "num_ctx": (ollama_cfg.get("qa_num_ctx", 16384))
        }
    )

    answer_options = cfg_options(
        ds_cfg.get("answer_options"),
        {
            "temperature": 0.0,
            "top_p": 0.9,
            "top_k": 40,
            "min_p": 0.05,
            "repeat_penalty": 1.08,
            "num_predict": 1024,
            "num_ctx": (ollama_cfg.get("qa_num_ctx", 16384))
        },
    )

    validator_options = cfg_options(
        ds_cfg.get("validator_options"),
        {
            "temperature": 0.0,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.05,
            "repeat_penalty": 1.0,
            "num_predict": 1024,
            "num_ctx": (ollama_cfg.get("qa_num_ctx", 16384))
        },
    )

    draft_jitter = dict(ds_cfg.get("draft_jitter") or {})
    answer_jitter = dict(ds_cfg.get("answer_jitter") or {})
    validator_jitter = dict(ds_cfg.get("validator_jitter") or {})

    db_path = Path(args.db or ds_cfg.get("db_path") or resolve_db_path_from_cfg(cfg)).expanduser()
    default_filename = str(ds_cfg.get("sft_out", ds_cfg.get("lora_out", "genshin_rag_sft_candidates.jsonl")))

    if args.out:
        out_arg = resolve_user_path(args.out)
        if out_arg.suffix:
            out_path = out_arg
            out_dir = out_path.parent
        else:
            out_dir = out_arg
            out_path = out_dir / default_filename
    else:
        configured_out_dir = ds_cfg.get("out_dir")
        if configured_out_dir:
            out_dir = resolve_user_path(configured_out_dir)
        else:
            out_dir = db_path.parent / "training"
        out_path = out_dir / default_filename

    rejected_path = out_dir / str(ds_cfg.get("rejected_out", "genshin_rejected.jsonl"))
    retrieval_pairs_path = out_dir / str(ds_cfg.get("retrieval_pairs_out", "genshin_retrieval_pairs.jsonl"))
    double_negative_path = out_dir / str(ds_cfg.get("double_negative_out", "genshin_double_negative.jsonl"))
    sft_negative_path = out_dir / str(ds_cfg.get("sft_negative_out", "genshin_sft_negative_answerability.jsonl"))

    retrieval_pairs_arg = getattr(args, "retrieval_pairs_out", None)
    double_negative_arg = getattr(args, "double_negative_out", None)
    sft_negative_arg = getattr(args, "sft_negative_out", None)

    if retrieval_pairs_arg:
        retrieval_pairs_path = resolve_user_path(retrieval_pairs_arg)

    if double_negative_arg:
        double_negative_path = resolve_user_path(double_negative_arg)

    if sft_negative_arg:
        sft_negative_path = resolve_user_path(sft_negative_arg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    retrieval_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    double_negative_path.parent.mkdir(parents=True, exist_ok=True)
    sft_negative_path.parent.mkdir(parents=True, exist_ok=True)

    sources = cfg_sources(args.sources, ds_cfg.get("sources", "genshin_wiki, kqm_tcl, kqm_news, honey, genshin_gg, game8"))

    worker_settings = {
        "db_path": str(db_path),
        "ollama_url": args.ollama_url,
        "draft_model": draft_model,
        "answer_model": answer_model,
        "request_timeout": request_timeout,
        "qa_per_chunk": args.qa_per_chunk,
        "max_chars": args.max_chars,
        "sleep": args.sleep,
        "retriever_name": retriever_name,
        "direct_top_k": direct_top_k,
        "backend": backend,
        "require_positive_document": require_positive_document,

        "draft_think": draft_think,
        "answer_think": answer_think,
        "validator_think": validator_think,

        "draft_options": draft_options,
        "answer_options": answer_options,
        "validator_options": validator_options,

        "draft_jitter": draft_jitter,
        "answer_jitter": answer_jitter,
        "validator_jitter": validator_jitter,

        "make_negatives": make_negatives,
        "hard_negative_count": hard_negative_count,
        "easy_negative_count": easy_negative_count,
        "negative_pool_size": negative_pool_size,
        "negative_text_chars": negative_text_chars,
        "validator_model": validator_model,
        "negative_min_confidence": negative_min_confidence,
        "negative_sft_per_positive": negative_sft_per_positive,
        "seed": args.seed,
    }

    if not sources:
        raise RuntimeError("[LORA_DATASET] No dataset sources configured.")

    log.info(f"[LORA_CONFIG] db={db_path}")
    log.info(f"[LORA_CONFIG] out={out_path}")
    log.info("[LORA_CONFIG] draft_model=%s answer_model=%s validator_model=%s", draft_model, answer_model, validator_model)
    log.info(f"[LORA_CONFIG] sources={sources}")
    log.info("[LORA_CONFIG] append=%s flush_every=%d out=%s", append_outputs, flush_every, out_path)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    try:
        source_rows = fetch_chunks(
            conn,
            sources=sources,
            limit=args.limit,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            seed=args.seed,
        )
    finally:
        conn.close()

    rows = [dict(row)for row in source_rows]
    log.info(f"[LORA_DATASET] Loaded candidate chunks: {len(rows)}")

    with ExitStack() as stack:
        file_mode = "a" if append_outputs else "w"
        buffer_size = int(ds_cfg.get("file_buffer_size", 1024 * 1024))

        output_f = stack.enter_context(out_path.open(file_mode, encoding="utf-8", buffering=buffer_size))
        rejected_f = stack.enter_context(rejected_path.open(file_mode, encoding="utf-8", buffering=buffer_size))

        pair_f = None
        double_negative_f = None
        negative_sft_f = None

        if make_negatives:
            pair_f = stack.enter_context(retrieval_pairs_path.open(file_mode, encoding="utf-8", buffering=buffer_size))
            double_negative_f = stack.enter_context(double_negative_path.open(file_mode, encoding="utf-8", buffering=buffer_size))
            negative_sft_f = stack.enter_context(sft_negative_path.open(file_mode, encoding="utf-8", buffering=buffer_size))
        
        completed_tasks = 0
        written = 0
        skipped = 0

        written_pairs = 0
        written_double_negatives = 0
        written_negative_sft = 0

        seen_record_ids, seen_questions = load_existing_sft_keys(out_path) if append_outputs else (set(), set())

        result_buffer: dict[int, ChunkTaskResult] = {}
        next_write_index = 0

        if retriever_name in {
            "faiss",
            "hybrid",
            "turbovec",
            "hybrid_turbovec",
            "hybrid_all",
            "hybrid_faiss_turbovec",
        }:
            log.info("[DATASET] Warming retrieval index")

            try:
                retrieve_question_context(
                    cfg,
                    "Who is Venti?",
                    retriever_name=retriever_name,
                    direct_top_k=50,
                    backend=backend,
                )
            except Exception as exc:
                log.warning("[DATASET] Retrieval warm-up failed: %s", exc)

        for result in run_bounded_workers(rows, workers=workers, max_inflight=max_inflight, cfg=cfg, settings=worker_settings):
            if preserve_output_order:
                result_buffer[result.task_index] = result
                ready_results: list[ChunkTaskResult] = []

                while next_write_index in result_buffer:
                    ready_results.append(result_buffer.pop(next_write_index))
                    next_write_index += 1
            else:
                ready_results = [result]

            for ready in ready_results:
                accepted_record_ids: set[str] = set()
                completed_tasks += 1
                skipped += ready.skipped

                for rejection in ready.rejected:
                    rejected_f.write(
                        json.dumps(
                            rejection,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                for record in ready.records:
                    record_id = str(
                        record.get("id") or ""
                    ).strip()

                    metadata = record.get("metadata") or {}
                    question = str(
                        metadata.get("question") or ""
                    ).strip()

                    question_key = re.sub(
                        r"\s+",
                        " ",
                        question.lower(),
                    ).strip()

                    if not record_id:
                        skipped += 1
                        rejected_f.write(
                            json.dumps(
                                {
                                    "reason": "record_missing_id",
                                    "task_index": ready.task_index,
                                    "positive_chunk_id": ready.chunk_id,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        continue

                    if record_id in seen_record_ids:
                        skipped += 1
                        continue

                    if (
                        question_key
                        and question_key in seen_questions
                    ):
                        skipped += 1
                        continue

                    output_f.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    accepted_record_ids.add(record_id)
                    seen_record_ids.add(record_id)

                    if question_key:
                        seen_questions.add(question_key)

                    written += 1

                if pair_f is not None:
                    for pair_record in ready.retrieval_pairs:
                        origin_id = pair_record.get(
                            "origin_record_id"
                        )

                        if origin_id not in accepted_record_ids:
                            continue

                        pair_f.write(
                            json.dumps(
                                pair_record,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        written_pairs += 1

                if double_negative_f is not None:
                    for double_record in ready.double_negative_records:
                        origin_id = double_record.get(
                            "origin_record_id"
                        )

                        if origin_id not in accepted_record_ids:
                            continue

                        double_negative_f.write(
                            json.dumps(
                                double_record,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        written_double_negatives += 1

                if negative_sft_f is not None:
                    for negative_record in ready.negative_sft_records:
                        origin_id = (
                            negative_record
                            .get("metadata", {})
                            .get("origin_record_id")
                        )

                        if origin_id not in accepted_record_ids:
                            continue

                        negative_sft_f.write(
                            json.dumps(
                                negative_record,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        written_negative_sft += 1

                if completed_tasks % flush_every == 0 or completed_tasks == len(rows):
                    log.info(
                        "[DATASET] tasks=%d/%d written=%d "
                        "skipped=%d retrieval_pairs=%d "
                        "double_negatives=%d negative_sft=%d",
                        completed_tasks,
                        len(rows),
                        written,
                        skipped,
                        written_pairs,
                        written_double_negatives,
                        written_negative_sft,
                    )

                    output_f.flush()
                    rejected_f.flush()

                    if pair_f is not None:
                        pair_f.flush()

                    if double_negative_f is not None:
                        double_negative_f.flush()

                    if negative_sft_f is not None:
                        negative_sft_f.flush()

    log.info(f"[LORA_DATASET] Done. written={written} skipped={skipped} out={out_path}")

if __name__ == "__main__":
    main()