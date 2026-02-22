import logging
import sys
from typing import Optional

_DEFAULT_FMT = "[%(levelname)s] %(asctime)s | %(name)s | %(message)s"

def setup_logging(
    level: int = logging.INFO,
    *,
    fmt: str = _DEFAULT_FMT,
    datefmt: str = "%H:%M:%S",
    stream=None,
    force: bool = False,
) -> None:
    """
    - force=True will remove existing handlers (useful if you want to reconfigure).
    """
    root = logging.getLogger()
    root.setLevel(level)

    if stream is None:
        stream = sys.stdout

    if force:
        for h in list(root.handlers):
            root.removeHandler(h)

    # Avoid duplicate handlers in Colab (rerun cells)
    if any(isinstance(h, logging.StreamHandler) for h in root.handlers) and not force:
        return

    h = logging.StreamHandler(stream)
    h.setLevel(level)
    h.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    root.addHandler(h)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name if name else "my_lib")