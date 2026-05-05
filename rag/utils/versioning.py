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

def version_ord_from_major_minor(major: int, minor: int) -> int:
    return major * 100 + minor

def normalize_version_alias_key(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def extract_version_signal(title: str, text: str = "", cfg: dict | None = None) -> tuple[str | None, int | None]:
    cfg = cfg or {}
    versioning_cfg = cfg.get("versioning", {}) or {}
    aliases = {
        normalize_version_alias_key(k): int(v) for k, v in (versioning_cfg.get("aliases", {}) or {}).items()
    }
    haystack = f"{title}\n{text[:3000]}"

    for m in LUNA_RE.finditer(haystack):
        raw = f"luna {m.group(1)}"
        key = normalize_version_alias_key(raw)

        val = m.group(1).lower()
        if val in ROMAN:
            key2 = f"luna {ROMAN[val]}"
            key2 = normalize_version_alias_key(key2)
        else:
            key2 = key

        if key in aliases:
            return key, aliases[key]
        if key2 in aliases:
            return key2, aliases[key2]
        
    candidates: list[tuple[str, int]] = []
    for m in VERSION_RE.finditer(haystack):
        major = int(m.group(1))
        minor = int(m.group(2))

        if major < 1 or major > 20:
            continue
        if minor < 0 or minor > 20:
            continue

        label = f"{major}.{minor}"
        candidates.append((label, version_ord_from_major_minor(major, minor)))

    if not candidates:
        return None, None
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0]