from __future__ import annotations

import argparse, json, random, re, sqlite3, time, requests, os, yaml, sys, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rag"))

from utils.logging_setup import setup_logging

REPO_ROOT = Path(__file__).resolve().parents[1]
RAG_DIR   = REPO_ROOT / "rag"

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a Genshin Impact assistant. "
    "Answer only from the provided source context. "
    "Do not invent facts."
)

def clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

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

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "who", "what", "when", "where", "why", "how",
    "i", "you", "he", "she", "it", "they", "we",
    "of", "to", "in", "on", "for", "from", "with", "and", "or",
    "does", "do", "did", "can", "could", "would", "should",
    "come", "comes", "about", "tell", "me",
}

def tokenize(s: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_']+", (s or "").lower())) - STOPWORDS

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


def cfg_sources(x, default: str) -> list[str]:
    if x is None:
        x = default

    if isinstance(x, list):
        return [str(s).strip() for s in x if str(s).strip()]

    return [s.strip() for s in str(x).split(",") if s.strip()]

def make_fts5_query(user_query: str) -> str:
    raw_tokens = re.findall(r"[A-Za-z0-9_']+", user_query.lower())

    tokens = []
    for t in raw_tokens:
        t = t.strip("'")
        if not t or t in STOPWORDS or len(t) < 2:
            continue
        t = t.replace('"', '""')
        tokens.append(t)

    if not tokens:
        return ""

    parts = []

    if 1 <= len(tokens) <= 4:
        parts.append(f'"{" ".join(tokens)}"')

    parts.extend(f'"{t}"' for t in tokens)

    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)

    return " OR ".join(out)


def detect_intent(question: str) -> str:
    q = question.lower()

    if any(x in q for x in ["weapon", "artifact", "build", "team", "recommended", "best", "signature", "bis"]):
        return "build"

    if any(x in q for x in ["who is", "who was", "born", "birthday", "family", "occupation", "come from", "comes from"]):
        return "biography"

    if any(x in q for x in ["where is", "location", "where can i find", "map", "region", "area"]):
        return "location"

    if any(x in q for x in ["skill", "burst", "talent", "constellation", "passive", "ability", "how does"]):
        return "mechanic"

    if any(x in q for x in ["lore", "story", "history", "archon", "god", "what happened"]):
        return "lore"

    return "general"


def extract_entity(question: str, positive_title: str = "") -> str:
    q = question.strip()

    patterns = [
        r"\bwho\s+is\s+([A-Za-z][A-Za-z' -]{1,50})",
        r"\bwho\s+was\s+([A-Za-z][A-Za-z' -]{1,50})",
        r"\bwhat\s+is\s+([A-Za-z][A-Za-z' -]{1,50})\s+(?:recommended|best|signature|weapon|build|artifact)",
        r"\bbest\s+(?:weapon|artifact|build|team)\s+for\s+([A-Za-z][A-Za-z' -]{1,50})",
        r"\brecommended\s+(?:weapon|artifact|build|team)\s+for\s+([A-Za-z][A-Za-z' -]{1,50})",
        r"\b([A-Z][A-Za-z' -]{1,50})\s+(?:recommended|best|signature)\s+(?:weapon|artifact|build|team)",
    ]

    for pat in patterns:
        m = re.search(pat, q, re.I)
        if not m:
            continue

        ent = m.group(1)
        ent = re.split(r"\?|,|\.|\band\b|\bwhere\b|\bwhat\b", ent, flags=re.I)[0]
        ent = ent.strip().lower()
        ent = re.sub(r"'s\b", "", ent).strip()
        ent = re.sub(
            r"\b(recommended|best|signature|weapon|weapons|build|artifact|artifacts)$",
            "",
            ent,
        ).strip()

        if ent:
            return ent

    title = positive_title or ""
    title = re.sub(r"\|.*$", "", title)
    title = re.sub(r"\s*-\s*Genshin.*$", "", title, flags=re.I)
    title = title.replace("Genshin Impact", "")
    title = title.replace("Build", "")
    return title.strip().lower()


def compact_chunk(row, max_chars: int = 1200) -> dict:
    return {
        "chunk_id": int(row["chunk_id"]),
        "doc_id": int(row["doc_id"]),
        "source": row["source"],
        "title": row["title"],
        "url": row["url"],
        "text": clean_text(row["text"] or "")[:max_chars],
    }


def score_hard_negative(question: str, entity: str, intent: str, positive, candidate) -> float:
    q_terms = tokenize(question)

    title = (candidate["title"] or "").lower()
    text = (candidate["text"] or "").lower()
    source = candidate["source"] or ""

    pos_title = (positive["title"] or "").lower()

    score = 0.0
    score += 0.05 * len(q_terms & tokenize(text[:2500]))
    score += 0.15 * len(q_terms & tokenize(title))

    if entity and entity in title:
        score += 0.75
    if entity and entity in text[:2500]:
        score += 0.35

    if intent == "build":
        if "build" in title:
            score += 0.40
        if entity and entity in text[:2500] and entity not in title:
            score += 0.80
        if "weapon" in text[:2500] or "artifact" in text[:2500]:
            score += 0.20

    if intent == "biography":
        bad_bio_titles = [
            "avatar", "stella fortuna", "namecard", "good fortune",
            "kitsune dreaming", "quest item", "normal attack",
            "constellation", "fan art contest",
        ]
        if entity and entity in title and any(x in title for x in bad_bio_titles):
            score += 1.00
        if "friendship level 10" in text[:1500] or "how to obtain" in text[:1500]:
            score += 0.50

    if int(candidate["doc_id"]) == int(positive["doc_id"]):
        score -= 10.0

    if title == pos_title:
        score -= 5.0

    if source != positive["source"]:
        score += 0.10

    return score


def find_hard_negatives(conn: sqlite3.Connection, question: str, positive, *, n: int, pool_size: int) -> list[dict]:
    fts_query = make_fts5_query(question)
    if not fts_query:
        return []

    intent = detect_intent(question)
    entity = extract_entity(question, positive["title"] or "")

    cur = conn.cursor()

    try:
        cur.execute(
            """
            SELECT
                f.chunk_id,
                f.doc_id,
                f.source,
                d.url,
                f.title,
                f.text,
                -bm25(chunks_fts) AS bm25_score
            FROM chunks_fts f
            JOIN docs d ON d.doc_id = f.doc_id
            WHERE chunks_fts MATCH ?
              AND f.chunk_id != ?
              AND f.doc_id != ?
              AND COALESCE(d.status, 1) = 1
            ORDER BY bm25(chunks_fts)
            LIMIT ?
            """,
            (
                fts_query,
                int(positive["chunk_id"]),
                int(positive["doc_id"]),
                int(pool_size),
            ),
        )
        rows = [dict(r) for r in cur.fetchall()]
    except sqlite3.OperationalError:
        return []

    scored = []
    seen_docs = set()

    for row in rows:
        doc_id = int(row["doc_id"])
        if doc_id in seen_docs:
            continue

        s = score_hard_negative(question, entity, intent, positive, row)
        if s <= 0:
            continue

        row["negative_score"] = s
        scored.append(row)
        seen_docs.add(doc_id)

    scored.sort(key=lambda r: float(r["negative_score"]), reverse=True)
    return scored[:n]


def find_easy_negatives(conn: sqlite3.Connection, positive, question: str, *, n: int, max_attempts: int = 20) -> list[dict]:
    entity = extract_entity(question, positive["title"] or "")

    cur = conn.cursor()
    cur.execute("SELECT MAX(chunk_id) FROM chunks")
    max_chunk_id = int(cur.fetchone()[0] or 0)

    if max_chunk_id <= 0:
        return []

    rng = random.Random(1337 + int(positive["chunk_id"]))
    out = []
    seen_docs = {int(positive["doc_id"])}

    for _ in range(max_attempts):
        if len(out) >= n:
            break

        start_id = rng.randint(1, max_chunk_id)

        cur.execute(
            """
            SELECT
                c.chunk_id,
                c.doc_id,
                c.chunk_index,
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
            LIMIT 20
            """,
            (start_id,),
        )

        for row in cur.fetchall():
            row = dict(row)
            doc_id = int(row["doc_id"])
            if doc_id in seen_docs:
                continue

            title = (row["title"] or "").lower()
            text = (row["text"] or "").lower()

            if entity and (entity in title or entity in text[:1500]):
                continue

            if len(text.strip()) < 300:
                continue

            seen_docs.add(doc_id)
            out.append(row)

            if len(out) >= n:
                break

    return out

def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type IN ('table', 'view')
          AND name = ?
        LIMIT 1
        """,
        (table_name,),
    )
    return cur.fetchone() is not None


def get_count(conn: sqlite3.Connection, sql: str) -> int:
    cur = conn.cursor()
    cur.execute(sql)
    return int(cur.fetchone()[0] or 0)


def preflight_for_negatives(conn: sqlite3.Connection) -> None:
    if not table_exists(conn, "chunks_fts"):
        raise RuntimeError(
            "chunks_fts table does not exist. Run FTS sync before using --make-negatives."
        )

    fts_rows = get_count(conn, "SELECT COUNT(*) FROM chunks_fts")
    active_chunks = get_count(conn, "SELECT COUNT(*) FROM chunks WHERE is_active=1")

    log.info(f"[PREFLIGHT] active_chunks={active_chunks} fts_rows={fts_rows}")

    if active_chunks <= 0:
        raise RuntimeError("No active chunks found. Cannot create training dataset.")

    if fts_rows <= 0:
        raise RuntimeError(
            "chunks_fts is empty. Cannot mine hard negatives. "
            "Run FTS sync before using --make-negatives."
        )

def make_negative_sft_record(base_id: str, question: str, negative: dict) -> dict:
    context = (
        f"Title: {negative.get('title')}\n"
        f"Source: {negative.get('source')}\n"
        f"URL: {negative.get('url')}\n\n"
        f"Text:\n{clean_text(negative.get('text') or '')[:1600]}"
    )

    return {
        "id": f"{base_id}_neg_sft_{negative['chunk_id']}",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a retrieval-grounded Genshin Impact assistant. "
                    "Answer only from the provided context. "
                    "If the context does not support the answer, say you do not have enough evidence."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
            {
                "role": "assistant",
                "content": "I don't have enough evidence in the provided context to answer that.",
            },
        ],
        "metadata": {
            "type": "negative_answerability",
            "negative_chunk_id": int(negative["chunk_id"]),
            "negative_doc_id": int(negative["doc_id"]),
            "source": negative.get("source"),
            "url": negative.get("url"),
            "title": negative.get("title"),
            "verified": False,
        },
    }

def ollama_generate(base_url: str, model: str, prompt: str, timeout: int = 300) -> str:
    r = requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
            },
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()


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


def make_prompt(title: str, source: str, url: str, text: str, n: int) -> str:
    return f"""
You are generating supervised fine-tuning examples for a Genshin Impact assistant.

Use ONLY the source context below.

Generate {n} high-quality question-answer pairs.

Rules:
- Return valid JSON only.
- Return a JSON array.
- Each item must have: question, answer.
- Questions should sound like real user questions.
- Answers must be directly supported by the context.
- If the context is mostly navigation, comments, membership text, or not useful, return [].
- Do not create questions that require facts outside the context.
- Do not mention "the context says" unless necessary.
- Keep answers concise but complete.

Source:
title: {title}
source: {source}
url: {url}

Context:
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
        """,
        (*sources, min_chars, max_chars),
    )

    rows = cur.fetchall()

    rng = random.Random(seed)
    rng.shuffle(rows)

    return rows[:limit]


def main() -> None:
    setup_logging(
        cfg.get("logging", {}).get("file"),
        cfg.get("logging", {}).get("level", "INFO")
    )
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
    ap.add_argument("--sft-negative-out", default=None)
    ap.add_argument("--hard-negatives", type=int, default=None)
    ap.add_argument("--easy-negatives", type=int, default=None)
    ap.add_argument("--negative-pool-size", type=int, default=None)
    ap.add_argument("--negative-text-chars", type=int, default=None)

    args = ap.parse_args()

    cfg = load_cfg(args.config)
    ds_cfg = cfg.get("dataset_creation", {}) or {}
    ollama_cfg = cfg.get("ollama", {}) or {}

    db_path = Path(args.db or ds_cfg.get("db_path") or resolve_db_path_from_cfg(cfg))

    out_dir = resolve_output_path(ds_cfg.get("out_dir", "data/training"), cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = (resolve_output_path(args.out, cfg) if args.out else out_dir / ds_cfg.get("lora_out", "genshin_lora_candidates.jsonl"))
    retrieval_pairs_path = (resolve_output_path(args.retrieval_pairs_out, cfg) if args.retrieval_pairs_out else out_dir / ds_cfg.get("retrieval_pairs_out", "genshin_retrieval_pairs.jsonl"))
    sft_negative_path = (resolve_output_path(args.sft_negative_out, cfg) if args.sft_negative_out else out_dir / ds_cfg.get("sft_negative_out", "genshin_sft_negative_answerability.jsonl"))
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    retrieval_pairs_path.parent.mkdir(parents=True, exist_ok=True)
    sft_negative_path.parent.mkdir(parents=True, exist_ok=True)

    args.ollama_url = args.ollama_url or ds_cfg.get("ollama_url") or ollama_cfg.get("base_url", "http://localhost:11434")
    args.model = args.model or ds_cfg.get("model") or ollama_cfg.get("qa_model", "llama3.2:3b")

    args.limit = cfg_int(args.limit, cfg_int(ds_cfg.get("limit"), 1000))
    args.qa_per_chunk = cfg_int(args.qa_per_chunk, cfg_int(ds_cfg.get("qa_per_chunk"), 2))
    args.min_chars = cfg_int(args.min_chars, cfg_int(ds_cfg.get("min_chars"), 500))
    args.max_chars = cfg_int(args.max_chars, cfg_int(ds_cfg.get("max_chars"), 2500))
    args.seed = cfg_int(args.seed, cfg_int(ds_cfg.get("seed"), 1337))
    args.sleep = cfg_float(args.sleep, cfg_float(ds_cfg.get("sleep"), 0.0))

    args.make_negatives = (args.make_negatives if args.make_negatives is not None else cfg_bool(ds_cfg.get("make_negatives"), False))

    args.hard_negatives = cfg_int(args.hard_negatives, cfg_int(ds_cfg.get("hard_negatives"), 3))
    args.easy_negatives = cfg_int(args.easy_negatives, cfg_int(ds_cfg.get("easy_negatives"), 2))
    args.negative_pool_size = cfg_int(args.negative_pool_size, cfg_int(ds_cfg.get("negative_pool_size"), 120))
    args.negative_text_chars = cfg_int(args.negative_text_chars, cfg_int(ds_cfg.get("negative_text_chars"), 1200))

    if not sources:
        raise RuntimeError("[LORA_DATASET] No dataset sources configured.")

    sources = cfg_sources(args.sources, ds_cfg.get("sources", "genshin_wiki,kqm_tcl,kqm_news,honey,genshin_gg"))

    log.info(f"[LORA_CONFIG] db={db_path}")
    log.info(f"[LORA_CONFIG] out={out_path}")
    log.info(f"[LORA_CONFIG] model={args.model}")
    log.info(f"[LORA_CONFIG] sources={sources}")
    log.info(f"[LORA_CONFIG] make_negatives={args.make_negatives}")
    log.info(f"[LORA_CONFIG] retrieval_pairs_out={retrieval_pairs_path}")
    log.info(f"[LORA_CONFIG] sft_negative_out={sft_negative_path}")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    if args.make_negatives:
        preflight_for_negatives(conn)

    pair_f = None
    sft_neg_f = None

    written = 0
    skipped = 0
    written_neg_pairs = 0
    written_sft_neg = 0

    try:
        rows = fetch_chunks(
            conn,
            sources=sources,
            limit=args.limit,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            seed=args.seed,
        )

        log.info(f"[LORA_DATASET] Loaded candidate chunks: {len(rows)}")

        if args.make_negatives:
            pair_f = retrieval_pairs_path.open("w", encoding="utf-8")
            sft_neg_f = sft_negative_path.open("w", encoding="utf-8")

        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                chunk_id = int(row["chunk_id"])
                doc_id = int(row["doc_id"])
                source = row["source"]
                url = row["url"]
                title = row["title"] or ""
                text = clean_text(row["text"] or "")

                if not is_good_chunk(text, title, source):
                    skipped += 1
                    continue

                prompt = make_prompt(title=title, source=source, url=url, text=text[: args.max_chars], n=args.qa_per_chunk)

                try:
                    raw = ollama_generate(args.ollama_url, args.model, prompt)
                    items = extract_json_array(raw)
                except Exception as e:
                    log.warning(f"[LORA_DATASET] failed chunk_id={chunk_id} err={type(e).__name__}: {e}")
                    skipped += 1
                    continue

                for i, item in enumerate(items):
                    question = str(item.get("question", "")).strip()
                    answer = str(item.get("answer", "")).strip()

                    if not question or not answer:
                        continue

                    rec = {
                        "id": f"genshin_{chunk_id}_{i}",
                        "messages": [
                            {
                                "role": "system",
                                "content": SYSTEM_PROMPT,
                            },
                            {
                                "role": "user",
                                "content": question,
                            },
                            {
                                "role": "assistant",
                                "content": answer,
                            },
                        ],
                        "metadata": {
                            "source": source,
                            "url": url,
                            "title": title,
                            "chunk_id": chunk_id,
                            "doc_id": doc_id,
                            "verified": False,
                        },
                    }

                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1

                    if args.make_negatives and pair_f and sft_neg_f:
                        positive = {
                            "chunk_id": chunk_id,
                            "doc_id": doc_id,
                            "text": text,
                            "source": source,
                            "url": url,
                            "title": title,
                        }

                        hard_negs = find_hard_negatives(
                            conn,
                            question,
                            positive,
                            n=args.hard_negatives,
                            pool_size=args.negative_pool_size,
                        )

                        easy_negs = find_easy_negatives(
                            conn,
                            positive,
                            question,
                            n=args.easy_negatives,
                        )

                        pair_record = {
                            "id": f"genshin_{chunk_id}_{i}_retrieval",
                            "query": question,
                            "intent": detect_intent(question),
                            "entity": extract_entity(question, title),
                            "positive": compact_chunk(
                                positive,
                                args.negative_text_chars,
                            ),
                            "hard_negatives": [
                                compact_chunk(x, args.negative_text_chars)
                                for x in hard_negs
                            ],
                            "easy_negatives": [
                                compact_chunk(x, args.negative_text_chars)
                                for x in easy_negs
                            ],
                            "metadata": {
                                "origin_record_id": rec["id"],
                                "positive_chunk_id": chunk_id,
                                "verified": False,
                            },
                        }

                        pair_f.write(json.dumps(pair_record, ensure_ascii=False) + "\n")
                        written_neg_pairs += 1

                        for hn in hard_negs[:2]:
                            neg_sft = make_negative_sft_record(
                                rec["id"],
                                question,
                                hn,
                            )
                            sft_neg_f.write(json.dumps(neg_sft, ensure_ascii=False) + "\n")
                            written_sft_neg += 1

                if written % 100 == 0 and written > 0:
                    log.info(
                        f"[LORA_DATASET] written={written} skipped={skipped} "
                        f"retrieval_pairs={written_neg_pairs} "
                        f"sft_negatives={written_sft_neg}"
                    )

                if args.sleep > 0:
                    time.sleep(args.sleep)

    finally:
        if pair_f:
            pair_f.close()
        if sft_neg_f:
            sft_neg_f.close()
        conn.close()

    log.info(f"[LORA_DATASET] Done. written={written} skipped={skipped} out={out_path}")

    if args.make_negatives:
        log.info(f"[LORA_DATASET] Retrieval pairs written={written_neg_pairs} out={retrieval_pairs_path}")
        log.info(f"[LORA_DATASET] SFT negatives written={written_sft_neg} out={sft_negative_path}")

if __name__ == "__main__":
    main()