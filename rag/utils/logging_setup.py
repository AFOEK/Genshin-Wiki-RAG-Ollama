import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

class ColorFormatter(logging.Formatter):
    RESET = "\033[0m"
    COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITCAL: "\033[38;5;88m"
    }

    EXCEPTION_COLOR = "\033[38;5;5;88m"

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if record.exc_info:
            color = self.EXCEPTION_COLOR
        else:
            color = self.COLORS.get(record.levelno, "")
        
        if not color:
            return message
        
        return f"{color}{message}{self.RESET}"

def setup_logging(log_path: str | None = None, level: str = "INFO"):
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in root.handlers[:]:
        handler.close()
        root.removeHandler(handler)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(ColorFormatter(fmt))
    root.addHandler(stream)

    if log_path:
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        file = RotatingFileHandler(log_path, maxBytes=25_000_000, backupCount=3, encoding="utf-8")
        file.setFormatter(logging.Formatter(fmt))
        root.addHandler(file)