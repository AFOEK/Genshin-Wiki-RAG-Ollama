import re

RE_REF = re.compile(r"\[\d+\]")
RE_MULTI_WS = re.compile(r"[ \t]+\n|\n[ \t]+")
RE_CATEGORIES = re.compile(r"(?im)^categories?:.*$")
RE_TEMPLATE = re.compile(r"(?s)\{\{.*?\}\}")
RE_FILE = re.compile(r"(?im)^file:.*$")
RE_NAV = re.compile(r"(?im)^(navigation|see also|external links)\s*$")

def clean_fandom_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = s.replace("\u200b", "")

    s = RE_TEMPLATE.sub("", s)
    s = RE_REF.sub("", s)
    s = RE_CATEGORIES.sub("", s)
    s = RE_FILE.sub("", s)
    s = RE_MULTI_WS.sub("\n", s)

    s = re.sub(r"(?m)^[-=_]{3,}\s*$", "", s)
    return s.strip()