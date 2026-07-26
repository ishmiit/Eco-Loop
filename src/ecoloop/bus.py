"""Communication bus.

One class, two jobs:

* in-process fan-out to subscribers (the agent thread, KPI accumulator);
* durable newline-delimited JSON on disk (``events.jsonl``), which the web
  server tails to drive the dashboard over SSE and which doubles as the
  submission's raw data export.

Writing through a file rather than a socket means the simulation subprocess and
the dashboard process share no state and cannot deadlock each other — if the
dashboard dies, the simulation keeps running, and vice versa.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterable

EventHandler = Callable[[dict[str, Any]], None]

# Event kinds emitted by the system.
TELEMETRY = "telemetry"
DECISION = "decision"
TOOL_CALL = "tool_call"
LOG = "log"
STATUS = "status"
KPI = "kpi"
ECM = "ecm"


class EventBus:
    def __init__(self, path: Path | None = None, history: int = 4096) -> None:
        self.path = path
        self._fh = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", buffering=1, encoding="utf-8")
        self._lock = threading.Lock()
        self._subs: list[EventHandler] = []
        self._history: deque[dict[str, Any]] = deque(maxlen=history)
        self._seq = 0

    def subscribe(self, handler: EventHandler) -> None:
        with self._lock:
            self._subs.append(handler)

    def publish(self, kind: str, payload: dict[str, Any] | None = None, **kw: Any) -> dict[str, Any]:
        data = dict(payload or {})
        data.update(kw)
        with self._lock:
            self._seq += 1
            event = {"seq": self._seq, "kind": kind, "wall": round(time.time(), 3), **data}
            self._history.append(event)
            if self._fh is not None:
                try:
                    self._fh.write(json.dumps(event, default=str) + "\n")
                except (ValueError, OSError):  # closed file — never kill the sim
                    pass
            subs = list(self._subs)
        for handler in subs:
            try:
                handler(event)
            except Exception:  # a broken subscriber must not stop the loop
                pass
        return event

    def history(self, kinds: Iterable[str] | None = None) -> list[dict[str, Any]]:
        want = set(kinds) if kinds else None
        with self._lock:
            return [e for e in self._history if want is None or e["kind"] in want]

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.flush()
                    self._fh.close()
                finally:
                    self._fh = None

    def __enter__(self) -> "EventBus":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_events(path: Path, kinds: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Read a finished run's event log. Tolerates a truncated final line."""
    want = set(kinds) if kinds else None
    out: list[dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if want is None or event.get("kind") in want:
                out.append(event)
    return out
