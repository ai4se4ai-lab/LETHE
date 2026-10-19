"""Crash-safe append-only JSONL event log."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from .schemas import SCHEMA_VERSION


class EventLog:
    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seq = self._recover_sequence()

    def _recover_sequence(self) -> int:
        if not self.path.exists():
            return 0
        seq = 0
        for line in self.path.read_text(errors="replace").splitlines():
            try:
                seq = max(seq, int(json.loads(line).get("seq", 0)))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return seq

    def append(self, event: str, **payload: Any) -> dict[str, Any]:
        with self._lock:
            self._seq += 1
            record = {
                "schema_version": SCHEMA_VERSION,
                "ts": time.time(),
                "run_id": self.run_id,
                "seq": self._seq,
                "event": event,
                **payload,
            }
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            return record
