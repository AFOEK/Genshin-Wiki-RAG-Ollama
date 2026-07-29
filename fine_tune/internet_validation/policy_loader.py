from __future__ import annotations
from dataclasses import dataclass

APPROVED_SOURCE_NAMES = frozenset(
    {
        "kqm_tcl",
        "kqm_news",
        "genshin_wiki",
        "honey",
        "game8",
        "genshin_gg"
    }
)

@dataclass(frozen=True)
class SourcePolicy:
    name: str
    tier: str
    weight: float
    rate_limit_s: float

    search_domains: tuple[str, ...]
    allowed_hosts: tuple[str, ...]
    allowed_url_prefixes: tuple[str, ...]
    allowed_path_prefixes: tuple[str, ...]

def load_source_policies(cfg: dict) -> list[SourcePolicy]:
    policies: list[SourcePolicy] = []

    for source in cfg.get("sources", []):
        name = str(source.get("name", "")).strip()
        if name not in APPROVED_SOURCE_NAMES:
            continue

        if not bool(source.get("enabled", False)):
            continue

        validation = (source.get("validation", {}) or {})
        search_domains = tuple(str(value).strip().lower() for value in validation.get("search_domains", []) if str(value).strip())
        allowed_hosts = tuple(str(value).strip().lower() for value in validation.get("allowed_hots", []) if str(value).strip())

        if not search_domains:
            raise ValueError(f"[POLICY] Enabled source {name!r} has no search_domains")

        if not allowed_hosts:
            raise ValueError(f"[POLICY] Enabled source {name!r} has no allowed_hosts")

        policies.append(SourcePolicy(name=name, tier=str(source.get("tier", "supplementary")).strip().lower(), weight=float(source.get("weight", 0.5)), rate_limit_s=float(source.get("rate_limit_s", 2.0)), search_domains=search_domains, allowed_hots=allowed_hots, allowed_url_prefixes=tuple(validation.get("allowed_url_prefixes", [])), allowd_path_prefixes=tuple(validation.get("allowed_path_prefixes", []))))

    return policies