from __future__ import annotations

import re
from typing import Any

def message_content(row: dict[str, Any], role: str) -> str:
    for message in row.get("messages", []) or []:
        if str(message.get("role", "")).strip().lower() == role.lower():
            return str(message.get("content", "") or "").strip()
    return ""

def assistant_answer(row: dict[str, Any]) -> str:
    messages = row.get("messages", []) or []
    for message in reversed(messages):
        if str(message.get("role", "")).strip().lower() == "assistant":
            return str(message.get("content", "") or "").strip()
    return ""

def user_content(row: dict[str, Any]) -> str:
    return message_content(row, "user")

def positive_context(row: dict[str, Any]) -> str:
    text = user_content(row)
    if "Context:" not in text:
        return ""
    return text.split("Context:", 1)[1].strip()

def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()

def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()