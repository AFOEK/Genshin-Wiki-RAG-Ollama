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