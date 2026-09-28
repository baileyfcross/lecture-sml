"""Application logging setup."""

import logging
from pathlib import Path


def configure_logging(*, level: int = logging.INFO, log_file: Path | None = None) -> None:
    """Configure concise console logging and optional file logging."""

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
