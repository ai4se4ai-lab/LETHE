"""Core shared types, event logging, budget accounting and subprocess wrapper.

Everything that crosses a file boundary in LETHE is a pydantic schema with a
``schema_version`` so run artifacts can be validated and diffed later.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def run_proc(cmd: list[str], cwd: Path | None = None, timeout_s: int = 600,
             env: dict[str, str] | None = None, input_bytes: bytes | None = None,
             timeout: int | None = None) -> tuple[int, str, str]:
    """Run a subprocess with a hard timeout, capture output. Returns (rc, stdout, stderr)."""
    if timeout is not None:
        timeout_s = timeout
    try:
        p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, input=input_bytes,
                           capture_output=True, timeout=timeout_s, env=env)
        return p.returncode, p.stdout.decode(errors="replace"), p.stderr.decode(errors="replace")
    except subprocess.TimeoutExpired:
        return -1, "", f"TIMEOUT after {timeout_s}s"


# ---------------------------------------------------------------------------
# Program model
# ---------------------------------------------------------------------------

class Function(BaseModel):
    id: str                      # "path::Class.method" or "path::func"
    file: str
    name: str
    class_name: str | None = None
    lineno: int
    end_lineno: int
    calls: list[str] = Field(default_factory=list)   # resolved or best-effort callee ids


class Statement(BaseModel):
    id: str                      # "path:LINE"
    file: str
    lineno: int
    end_lineno: int
    text: str
    func_id: str | None = None
    kind: str = "stmt"


class ProgramModel(BaseModel):
    schema_version: str = SCHEMA_VERSION
    language: str = "python"
    root: str = "."
    source_hash: str = ""
    components: list[str] = Field(default_factory=list)
    functions: list[Function] = Field(default_factory=list)
    statements: list[Statement] = Field(default_factory=list)
    unresolved_calls: list[str] = Field(default_factory=list)

    def index(self) -> None:  # attach lookup dicts as plain attributes (not models)
        self._f = {f.id: f for f in self.functions}          # type: ignore[attr-defined]
        self._s = {s.id: s for s in self.statements}         # type: ignore[attr-defined]
        self._f_by_file = {}                                  # type: ignore[attr-defined]
        for f in self.functions:
            self._f_by_file.setdefault(f.file, []).append(f)  # type: ignore[attr-defined]
        self._s_by_func = {}                                  # type: ignore[attr-defined]
        for s in self.statements:
            if s.func_id:
                self._s_by_func.setdefault(s.func_id, []).append(s)  # type: ignore[attr-defined]

    @property
    def func(self) -> dict[str, Function]:
        return getattr(self, "_f", {})

    @property
    def stmt(self) -> dict[str, Statement]:
        return getattr(self, "_s", {})

    def functions_of(self, file: str) -> list[Function]:
        return getattr(self, "_f_by_file", {}).get(file, [])

    def statements_of(self, func_id: str) -> list[Statement]:
        return getattr(self, "_s_by_func", {}).get(func_id, [])

    def resolve_element(self, element: str) -> tuple[Literal["function", "statement"], Any]:
        if element in self.stmt:
            return "statement", self.stmt[element]
        if element in self.func:
            return "function", self.func[element]
        return "unknown", None

    def statements_in_function(self, func_id: str) -> list[str]:
        return [s.id for s in self.statements_of(func_id)]

    def callers(self, func_id: str) -> list[str]:
        out = []
        for f in self.functions:
            for c in f.calls:
                if c == func_id or c.endswith("::" + f.name) and func_id.endswith(f.name):
                    if f.id != func_id:
                        out.append(f.id)
                        break
        return sorted(set(out))

    def callees(self, func_id: str) -> list[str]:
        f = self.func.get(func_id)
        return sorted(set(f.calls)) if f else []

    def enclosing_function(self, file: str, lineno: int) -> Function | None:
        for f in self.functions_of(file):
            if f.lineno <= lineno <= f.end_lineno:
                return f
        return None
