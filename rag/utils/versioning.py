from __future__ import annotations

import re

VERSION_RE = re.compile(r"\b(?:version\s*)?v?(\d{1,2})\.(\d{1,2})\b", re.IGNORECASE)
LUNA_RE = re.compile(r"\bluna\s+([ivx]+|\d+)\b", re.IGNORECASE)

ROMAN = {
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
    "vi": 6,
    "vii": 7,
    "viii": 8,
    "ix": 9,
    "x": 10
}

def extract_numeric_versions(s: str) -> list[tuple[str, int]]:
    out = []
    for m in VERSION_RE.finditer(s or ""):
        major = int(m.group(1))
        minor = int(m.group(2))

        if major < 1 or major > 20:
            continue
        if minor < 0 or minor > 20:
            continue

        label = f"{major}.{minor}"
        out.append((label, version_ord_from_major_minor(major, minor)))

def extract_luna_versions(s: str, aliases: dict[str, int]) -> list[tuple[str, int]]:
    out = []
    for m in LUNA_RE.finditer(s or ""):
        raw = f"luna {m.group(1)}"
        key = normalize_version_alias_key(raw)

        val = m.group(1).lower()
        if val in ROMAN:
            key2 = normalize_version_alias_key(f"luna {ROMAN[val]}")
        else:
            key2 = key

        if key in aliases:
            out.append((key, aliases[key]))
        elif key2 in aliases:
            out.append((key2, aliases[key2]))

    return out

def version_ord_from_major_minor(major: int, minor: int) -> int:
    return major * 100 + minor

def normalize_version_alias_key(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def extract_version_signal(title: str, text: str = "", cfg: dict | None = None, source: str | None = None) -> tuple[str | None, int | None]:
    cfg = cfg or {}
    versioning_cfg = cfg.get("versioning", {}) or {}
    source = (source or "").strip().lower()
    aliases = {
        normalize_version_alias_key(k): int(v) for k, v in (versioning_cfg.get("aliases", {}) or {}).items()
    }
    title = title or ""
    text = text or ""

    title_candidates = []
    title_candidates.extend(extract_luna_versions(title, aliases))
    title_candidates.extend(extract_numeric_versions(title))

    if title_candidates:
        title_candidates.sort(key=lambda x: x[1], reverse=True)
        return title_candidates[0]

    early_text = text[:800]
    early_candidates = []
    early_candidates.extend(extract_luna_versions(early_text, aliases))
    early_candidates.extend(extract_numeric_versions(early_text))

    if early_candidates:
        early_candidates.sort(key=lambda x: x[1], reverse=True)
        return early_candidates[0]

    body_text = text[:3000]
    body_candidates = []
    body_candidates.extend(extract_luna_versions(body_text, aliases))
    body_candidates.extend(extract_numeric_versions(body_text))

    if not body_candidates:
        return None, None

    body_candidates.sort(key=lambda x: x[1], reverse=True)

    if source == "kqm_news":
        return body_candidates[0]

    unique_versions = sorted(set(v for _, v in body_candidates), reverse=True)

    if len(unique_versions) == 1:
        return body_candidates[0]

    return None, None