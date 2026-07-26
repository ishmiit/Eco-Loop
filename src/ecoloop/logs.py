"""Turning EnergyPlus's firehose into something an LLM can read.

A three-day run emits a few thousand log lines; a severe input error can emit
tens of thousands of near-identical ones. Feeding that to a 3B model is both
useless and unaffordable, so every message goes through this digest:

1. **classify** — Fatal / Severe / Warning / Info by prefix.
2. **fingerprint** — strip zone names, numbers, file paths and timestamps, so
   "Zone OFFICE temperature 34.12 C" and "Zone PROD_HALL temperature 31.88 C"
   collapse to one entry with ``count: 2`` and two examples.
3. **rank and cap** — Fatal first, then Severe, then Warning by frequency;
   ``digest()`` renders under a caller-supplied character budget.

``LogDigest.for_llm()`` is what the agent's ``read_simulation_log`` tool
returns: worst-first, deduplicated, hard-capped. A 40 000-line failure becomes
~600 characters that still name the offending object.
"""

from __future__ import annotations

import re
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

FATAL, SEVERE, WARNING, INFO = "fatal", "severe", "warning", "info"
_LEVEL_ORDER = {FATAL: 0, SEVERE: 1, WARNING: 2, INFO: 3}

_NUMBER = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")
_QUOTED = re.compile(r"[\"']([^\"']{1,80})[\"']")
_PATHISH = re.compile(r"(?:/[\w.\-]+){2,}")
_WS = re.compile(r"\s+")

# EnergyPlus prefixes, longest first so "** Severe  **" wins over "**".
_PREFIXES: tuple[tuple[str, str], ...] = (
    ("** Fatal  **", FATAL),
    ("** Severe  **", SEVERE),
    ("** Warning **", WARNING),
    ("**   ~~~   **", INFO),
    ("*************", INFO),
    ("** Fatal **", FATAL),
    ("** Severe **", SEVERE),
)


def classify(line: str) -> tuple[str, str]:
    """Return ``(level, message)`` for one raw log line."""
    text = line.rstrip()
    stripped = text.strip()
    for prefix, level in _PREFIXES:
        if stripped.startswith(prefix):
            return level, stripped[len(prefix):].strip()
    lowered = stripped.lower()
    if lowered.startswith("fatal"):
        return FATAL, stripped
    if lowered.startswith("severe"):
        return SEVERE, stripped
    if lowered.startswith("warning"):
        return WARNING, stripped
    return INFO, stripped


def fingerprint(message: str) -> str:
    """Collapse a message to its shape so repeats can be counted."""
    text = _QUOTED.sub("<name>", message)
    text = _PATHISH.sub("<path>", text)
    text = _NUMBER.sub("#", text)
    text = _WS.sub(" ", text).strip()
    return text[:180]


@dataclass
class LogEntry:
    level: str
    fingerprint: str
    count: int = 0
    first_example: str = ""
    last_example: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "count": self.count,
            "pattern": self.fingerprint,
            "example": self.first_example,
        }


@dataclass
class LogDigest:
    """Thread-safe: EnergyPlus calls the message callback from its own thread."""

    max_tail: int = 400
    entries: dict[str, LogEntry] = field(default_factory=dict)
    tail: deque[tuple[str, str]] = field(default_factory=lambda: deque(maxlen=400))
    counts: dict[str, int] = field(default_factory=lambda: {FATAL: 0, SEVERE: 0, WARNING: 0, INFO: 0})
    total: int = 0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self.tail = deque(maxlen=self.max_tail)

    # -- ingestion ----------------------------------------------------------
    def add_line(self, line: str) -> tuple[str, str]:
        level, message = classify(line)
        if not message:
            return level, message
        key = f"{level}|{fingerprint(message)}"
        with self._lock:
            self.total += 1
            self.counts[level] = self.counts.get(level, 0) + 1
            entry = self.entries.get(key)
            if entry is None:
                entry = LogEntry(level=level, fingerprint=fingerprint(message), first_example=message)
                self.entries[key] = entry
            entry.count += 1
            entry.last_example = message
            self.tail.append((level, message))
        return level, message

    def add_text(self, text: str) -> None:
        for line in text.splitlines():
            if line.strip():
                self.add_line(line)

    def add_file(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            return
        with p.open(encoding="latin-1", errors="replace") as fh:
            for line in fh:
                if line.strip():
                    self.add_line(line)

    # -- reporting ----------------------------------------------------------
    @property
    def has_fatal(self) -> bool:
        return self.counts.get(FATAL, 0) > 0

    @property
    def severe_count(self) -> int:
        return self.counts.get(SEVERE, 0) + self.counts.get(FATAL, 0)

    @property
    def warning_count(self) -> int:
        return self.counts.get(WARNING, 0)

    def ranked(self, levels: Iterable[str] | None = None) -> list[LogEntry]:
        want = {lv.lower() for lv in levels} if levels else None
        with self._lock:
            items = [e for e in self.entries.values() if want is None or e.level in want]
        return sorted(items, key=lambda e: (_LEVEL_ORDER.get(e.level, 9), -e.count))

    def tail_lines(self, n: int = 20, levels: Iterable[str] | None = None) -> list[str]:
        want = {lv.lower() for lv in levels} if levels else None
        with self._lock:
            rows = [f"[{lv}] {msg}" for lv, msg in self.tail if want is None or lv in want]
        return rows[-n:]

    def search(self, pattern: str, limit: int = 10) -> list[str]:
        """Regex search over the deduplicated entries (falls back to substring
        match when the caller sends an invalid regex — an LLM often will)."""
        try:
            rx = re.compile(pattern, re.IGNORECASE)
            match = lambda s: rx.search(s) is not None  # noqa: E731
        except re.error:
            # A model will send an unbalanced bracket sooner or later. Searching
            # for the broken pattern *literally* finds nothing and looks like a
            # clean log, so strip the metacharacters that made it invalid and
            # fall back to a substring match on what is left.
            needle = pattern.rstrip("[]{}()\\|+*?^$.").lower()
            match = (lambda s: needle in s.lower()) if needle else (lambda s: False)  # noqa: E731
        out: list[str] = []
        for entry in self.ranked():
            if match(entry.first_example) or match(entry.fingerprint):
                out.append(f"[{entry.level} x{entry.count}] {entry.first_example}")
                if len(out) >= limit:
                    break
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "total_messages": self.total,
            "unique_patterns": len(self.entries),
            "fatal": self.counts.get(FATAL, 0),
            "severe": self.counts.get(SEVERE, 0),
            "warning": self.counts.get(WARNING, 0),
            "info": self.counts.get(INFO, 0),
        }

    def for_llm(self, max_chars: int = 1200, levels: Iterable[str] = (FATAL, SEVERE, WARNING)) -> str:
        """Worst-first, deduplicated, hard-capped rendering."""
        s = self.summary()
        head = (
            f"EnergyPlus log: {s['total_messages']} messages, "
            f"{s['unique_patterns']} distinct; "
            f"fatal={s['fatal']} severe={s['severe']} warning={s['warning']}"
        )
        lines = [head]
        used = len(head)
        for entry in self.ranked(levels):
            text = entry.first_example
            if len(text) > 220:
                text = text[:217] + "..."
            row = f"- [{entry.level} x{entry.count}] {text}"
            if used + len(row) + 1 > max_chars:
                lines.append("- ... (truncated; use search_simulation_log for specifics)")
                break
            lines.append(row)
            used += len(row) + 1
        if len(lines) == 1:
            lines.append("- no warnings or errors")
        return "\n".join(lines)

    def to_dict(self, top: int = 25) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "entries": [e.to_dict() for e in self.ranked()[:top]],
        }
