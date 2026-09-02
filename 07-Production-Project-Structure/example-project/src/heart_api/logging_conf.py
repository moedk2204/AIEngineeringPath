"""Structured logging. In production you want machine-readable lines, not print()."""
import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'
    ))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
