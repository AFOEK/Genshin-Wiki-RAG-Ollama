from __future__ import annotations

import re
import logging
import unicodedata

log = logging.getLogger(__name__)

def entity_key(name: str) -> str:
    value = unicodedata.normalize("NFKC", name)
    value = value.casefold().strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s'-]", " ", value)
    return value