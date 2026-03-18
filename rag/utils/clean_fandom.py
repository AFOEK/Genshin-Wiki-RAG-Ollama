import re

RE_REF = re.compile(r"\[\d+\]")
RE_MULTI_WS = re.compile(r"[ \t]+\n|\n[ \t]+")
RE_CATEGORIES = re.compile(r"(?im)^categories?:.*$")
RE_TEMPLATE = re.compile(r"(?s)\{\{.*?\}\}")
RE_FILE = re.compile(r"(?im)^file:.*$")
RE_NAV = re.compile(r"(?im)^(navigation|see also|external links|gallery|media|languages?)\s*$")

RE_WIKIA_URL = re.compile(r"https?://static\.wikia\.nocookie\.net\S+", re.I)
RE_DATA_URI = re.compile(r"data:image/[a-zA-Z+]+;base64,[A-Za-z0-9+/=]+", re.I)
RE_MEDIA_LINK = re.compile(r"\[?Media:[^\]\n]+\]?", re.I)
RE_REVISION_URL = re.compile(r"https?://\S+/revision/latest\S*", re.I)
RE_CDN_URL = re.compile(r"https?://\S+\.(png|jpg|jpeg|gif|webp|svg|mp4|webm|mp3|wav|ogg)(\?\S*)?", re.I)
RE_BARE_BASE64 = re.compile(r"[A-Za-z0-9+/]{60,}={0,2}")

RE_EMPTY_TABLE_CELL = re.compile(r"\|\s*\|")
RE_BROKEN_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
RE_MD_LINK_ONLY = re.compile(r"\[[^\]]+\]\([^)]*\)")
RE_HTML_TAG = re.compile(r"<[^>]+>")
RE_LANG_MENU = re.compile(r"(?i)\b(Español|Français|Bahasa Indonesia|Русский язык|Português|한국어|日本語|ภาษาไทย|Tiếng Việt|Deutsch|Italiano)\b")
RE_ICONY_LINE = re.compile(r"(?i)(sprites?|gallery|icons?|costumes?|profile|voice-overs?)")
RE_TABLE_NOISE = re.compile(r"(?m)^\s*\|.*\|\s*$")

def clean_fandom_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = s.replace("\u200b", "")
    s = RE_TEMPLATE.sub("", s)
    s = RE_REF.sub("", s)
    s = RE_CATEGORIES.sub("", s)
    s = RE_FILE.sub("", s)
    s = RE_WIKIA_URL.sub("", s)
    s = RE_REVISION_URL.sub("", s)
    s = RE_CDN_URL.sub("", s)
    s = RE_DATA_URI.sub("", s)
    s = RE_MEDIA_LINK.sub("", s)
    s = RE_BROKEN_MD_IMAGE.sub("", s)
    s = RE_BARE_BASE64.sub("", s)
    s = RE_HTML_TAG.sub("", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
    s = RE_EMPTY_TABLE_CELL.sub("|", s)
    s = RE_MULTI_WS.sub("\n", s)
    s = re.sub(r"(?m)^[-=_]{3,}\s*$", "", s)

    lines = s.splitlines()
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            clean_lines.append("")
            continue
        if RE_NAV.match(stripped):
            continue
        if RE_LANG_MENU.search(stripped):
            continue
        if RE_ICONY_LINE.search(stripped) and len(stripped) < 80:
            continue
        # skip table-ish noise lines
        if RE_TABLE_NOISE.match(stripped) and len(stripped) < 120:
            continue
        # If >60% of line chars are URL-ish special chars, discard
        url_chars = sum(1 for c in stripped if c in "%?=&/:.#_+|[]()")
        if len(stripped) > 30 and url_chars / len(stripped) > 0.6:
            continue
        clean_lines.append(stripped)
    s = "\n".join(clean_lines)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()