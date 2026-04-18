import re, unicodedata

SENTENCE_END = re.compile(r'(?<=[.!?。！？])\s+')

def normalize(text: str) -> str:
    text  = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text(text: str, size=2400, overlap=300) -> list[str]:
    text = text.strip()
    if not text:
        return []
    
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + size)
        
        if j < n:
            search_start = max(i, j - 200)
            snippet = text[search_start:j]
            last_sentence = None
            for m in SENTENCE_END.finditer(snippet):
                last_sentence = m
            if last_sentence:
                j = search_start + last_sentence.end()
        
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)

        if j >= n:
            break

        i = max(0, j - overlap)

        if i >= j:
            i = j

    return chunks