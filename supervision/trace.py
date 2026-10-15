"""Append-only trace / audit log (in memory, optionally mirrored to JSONL)."""
from __future__ import annotations

import json
import time

_REDACT = {"api_key", "authorization", "token", "secret", "password"}


def _redact(o):
    if isinstance(o, dict):
        return {k: ("[REDACTED]" if k.lower() in _REDACT else _redact(v)) for k, v in o.items()}
    if isinstance(o, list):
        return [_redact(x) for x in o]
    return o


class Tracer:
    def __init__(self, path: str | None = None, clock=time.time):
        self.path = path
        self.clock = clock
        self.events: list = []
        self.run_id: str | None = None

    def emit(self, etype: str, **data) -> dict:
        ev = {"ts": round(self.clock(), 3), "run_id": self.run_id, "type": etype, **_redact(data)}
        self.events.append(ev)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, default=str) + "\n")
        return ev

    def of_type(self, etype: str) -> list:
        return [e for e in self.events if e["type"] == etype]
