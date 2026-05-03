from __future__ import annotations

import re

VERSION_RE = re.compile(r"\b(?:version\s*)?v?(\d{1,2})\.(\d{1,2})\b", re.IGNORECASE)
LUNA_RE = re.compile(r"\bluna\s+([ivx]+|\d+)\b", re.IGNORECASE)

ROMAN = {}