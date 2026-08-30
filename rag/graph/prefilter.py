from __future__ import annotations

import re
from dataclasses import dataclass

@dataclass(frozen=True)
class GraphFilterResult:
    eligible: bool
    score: int
    groups: tuple[str, ...]
    hits: tuple[str, ...]


RELATION_SIGNAL_PATTERNS = {
    "family": (
        (3, r"\b(?:parent|father|mother|son|daughter|brother|sister|sibling|spouse|husband|wife|ancestor|descendant|relative|guardian)\b"),
        (3, r"\b(?:adopted by|child of|parent of|sibling of|married to)\b"),
    ),

    "social": (
        (3, r"\b(?:friend of|ally of|enemy of|rival of|companion of|lover of|partner of)\b"),
        (2, r"\b(?:friend|ally|rival|companion|lover|betrayed|trusts|respects|fears)\b"),
    ),

    "teaching": (
        (3, r"\b(?:mentor of|teacher of|student of|disciple of|master of|apprentice of)\b"),
        (2, r"\b(?:mentor|teacher|student|disciple|apprentice)\b"),
    ),

    "organization": (
        (3, r"\b(?:member of|leader of|founder of|affiliated with|works for|worked for|employed by|serves under|served under|subordinate of|commands|commanded by|appointed by)\b"),
        (2, r"\b(?:member|leader|founder|commander|captain|general|chief|director|envoy|retainer)\b"),
    ),

    "creation": (
        (3, r"\b(?:created by|created|made by|invented by|invented|forged by|forged|built by|built|discovered by|discovered)\b"),
        (3, r"\b(?:owned by|owns|wielded by|wields|carried by|carries)\b"),
    ),

    "location": (
        (3, r"\b(?:located in|located at|resides in|lives in|born in|died in|originates from|native of|part of|contained in|contains|borders|governs|governed by|rules|travels to|visited)\b"),
        (2, r"\b(?:resident|birthplace|homeland|hometown)\b"),
    ),

    "combat": (
        (3, r"\b(?:fought|fought against|defeated|defeated by|killed|killed by|attacked|defended|captured|imprisoned|escaped from|sealed by|sealed|destroyed by|destroyed|opposes|opposed)\b"),
        (1, r"\b(?:battle|war|conflict)\b"),
    ),

    "religion": (
        (3, r"\b(?:worships|worshipped by|worshipped|worshiped by|worshiped)\b"),
        (1, r"\b(?:god|goddess|deity|archon|divine)\b"),
    ),

    "element": (
        (3, r"\b(?:has|possesses|possessed)\s+(?:a|an|the)?\s*(?:vision|delusion|gnosis)\b"),
        (3, r"\b(?:uses|wields|commands|controls)\s+(?:the\s+)?(?:pyro|hydro|anemo|electro|dendro|cryo|geo)\b"),
        (2, r"\b(?:resonates with|associated with)\s+(?:pyro|hydro|anemo|electro|dendro|cryo|geo)\b"),
    ),

    "event": (
        (3, r"\b(?:participated in|involved in|appears in|appeared in|featured in|takes place in|occurred in|occurred during)\b"),
        (1, r"\b(?:event|festival|incident)\b"),
    ),

    "chronology": (
        (3, r"\b(?:successor of|predecessor of|succeeded by|preceded by|resulted in|caused by|causes|triggered by|triggers)\b"),
        (2, r"\b(?:successor|predecessor|succeeded|preceded)\b"),
    ),

    "quest": (
        (3, r"\b(?:starts quest|unlocks|unlocked by|required for|rewarded by|rewards|appears in)\b"),
        (2, r"\b(?:quest giver|quest reward)\b"),
    ),

    "items": (
        (3, r"\b(?:dropped by|obtained from|purchased from|sold by|crafted from|crafts into|used for|required by|material for)\b"),
        (3, r"\b(?:used to ascend|upgrades talent|ascension material for|talent material for)\b"),
    ),

    "identity": (
        (3, r"\b(?:known as|also known as|identity of|form of|incarnation of|creation of|has the title|holds the title)\b"),
    ),

    "generic": (
        (2, r"\b(?:associated with|related to|connected to)\b"),
    ),
}


_COMPILED = {group: tuple((score, re.compile(pattern, re.IGNORECASE)) for score, pattern in patterns) for group, patterns in RELATION_SIGNAL_PATTERNS.items}


def cheap_graph_filter(text: str, *, title: str = "", min_score: int = 2) -> GraphFilterResult:
    haystack=f"{title}\n{text}"
    total_score=0
    groups=[]
    hits=[]

    for group, patterns in _COMPILED.items():
        group_score=0
        group_hit=None

        for score, pattern in patterns:
            match=pattern.search(haystack)
            if match and score > group_score:
                group_score=score
                group_hit=match.group(0)

        if group_score:
            total_score+=group_score
            groups.append(group)

            if group_hit:
                hits.append(group_hit)

    return GraphFilterResult(eligible=total_score >= min_score, score=total_score, groups=tuple(groups), hits=tuple(hits),)