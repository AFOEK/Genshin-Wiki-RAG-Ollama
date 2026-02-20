import zstandard as zstd

def zstd_compress_text(s: str, level: int = 6) -> bytes:
    c = zstd.ZstdCompressor(level=level)
    return c.compress(s.encode("utf-8"))

def zstd_decompress_text(b: bytes) -> str:
    d = zstd.ZstdDecompressor()
    return d.decompress(b).decode("utf-8")