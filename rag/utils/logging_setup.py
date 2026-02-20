import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging(log_path: str | None = None, level: str = "INFO"):
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    root.addHandler(stream)

    if log_path:
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        fileh = RotatingFileHandler(log_path, maxBytes=25_000_000, backupCount=3)
        fileh.setFormatter(fmt)
        root.addHandler(fileh)