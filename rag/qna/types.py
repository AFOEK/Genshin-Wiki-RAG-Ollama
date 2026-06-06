from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass
class RetrievalResult:
    question: str
    intent: str
    build_subtypes: set[str]
    broad: bool

    candidate_chunks: list[dict]
    selected_chunks: list[dict]
    context: str

    retrieval_signals: dict[int, Any]

    baseline_label: str | None = None
    baseline_ord: int | None = None
    strict_fts_query: str | None = None

    diagnostics: dict[str, Any] = field(default_factory=dict)