"""Logging estructurado: una línea `evento key=valor ...` fácil de leer y de parsear."""

from __future__ import annotations

import json
import logging
from typing import Any


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (dict, list, tuple, set)):
        try:
            return json.dumps(v, default=str, separators=(",", ":"))
        except (TypeError, ValueError):
            return str(v)
    return str(v)


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emite `event k1=v1 k2=v2` y añade `fields` en `extra` para formateadores JSON."""
    if not logger.isEnabledFor(level):
        return
    body = " ".join(f"{k}={_fmt(v)}" for k, v in fields.items())
    logger.log(level, "%s %s", event, body, extra={"event": event, "fields": fields})
