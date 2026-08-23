import logging
from typing import Any


def configure_logging(level: int = logging.INFO) -> None:
    """Configure concise process-wide logging for the demonstration service."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Emit a structured-enough log line without adding a logging dependency."""

    details = " ".join(f"{key}={value!r}" for key, value in sorted(fields.items()))
    logger.info("%s %s", event, details)
