"""LLM agent runtime: OpenAI-compatible client (Ollama/any compatible server),
workspace tools and the per-turn agent loop (plan §5.8).

Tools: read_file, list_dir, grep, query_model (model queries), write_candidate,
write_patch, end_turn. The agent has no shell, no test execution, no network:
the judge is the only source of truth for what the program does.
"""
from __future__ import annotations

import fnmatch
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_BASE = os.environ.get("LETHE_LLM_BASE", "http://localhost:11434/v1")
DEFAULT_MODEL = os.environ.get("LETHE_LLM_MODEL", "qwen3.8:27b")

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file inside the workspace. path is workspace-relative (repo/..., model/..., .context/...). Optionally a line range.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "List files under a workspace-relative directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "grep",
        "description": "Regex search inside workspace files. Returns up to 40 matching 'path:line:text' lines.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "optional subdirectory"}},
            "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "query_model",
        "description": "Query the program model. kind is one of: functions (functions in a file), statements_of (function id), callers_of (function id), callees_of (function id), covered_by (statement id).",
        "parameters": {"type": "object", "properties": {
            "kind": {"type": "string"},
            "arg": {"type": "string"}},
            "required": ["kind", "arg"]}}},
    {"type": "function", "function": {
        "name": "write_candidate",
        "description": "Write your final ranked suspicion list for this turn (overwrites candidate.json). Elements are 'path::Class.method' or 'path:LINE'. ranks start at 1 without gaps, at most 10 hypotheses, at most one per element.",
        "parameters": {"type": "object", "properties": {
            "hypotheses": {"type": "array", "items": {"type": "object", "properties": {
                "rank": {"type": "integer"},
                "element": {"type": "string"},
                "granularity": {"type": "string", "enum": ["method", "statement"]},
                "rationale": {"type": "string", "description": "max 500 chars, evidence-based"}},
                "required": ["rank", "element", "granularity"]}},
            "notes": {"type": "string"}},
            "required": ["hypotheses"]}}},
    {"type": "function", "function": {
        "name": "write_patch",
        "description": "Optionally write a unified diff (patch.diff) that you believe repairs the fault at your top hypothesis. This is evidence checked by the judge; it does not replace the ranking.",
        "parameters": {"type": "object", "properties": {
            "diff": {"type": "string"}}, "required": ["diff"]}}},
    {"type": "function", "function": {
        "name": "end_turn",
        "description": "End the turn. Call after candidate.json is final, with a one-line summary of what you changed.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string"}}, "required": ["summary"]}}},
]


def llm_chat(base: str, model: str, messages: list[dict], tools: list[dict] | None = None,
             temperature: float = 0.0, max_tokens: int = 2048, timeout: int = 600) -> dict:
    """One chat completion. Returns {message, usage, latency, model, finish}."""
    import json as _json
    payload = {"model": model, "messages": messages, "temperature": temperature,
               "max_tokens": max_tokens, "stream": False}
    if tools:
        payload["tools"] = tools
    body = _json.dumps(payload).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer ollama"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = _json.loads(r.read().decode())
    latency = time.time() - t0
    choice = data["choices"][0]
    msg = choice.get("message", {})
    return {"message": msg, "tool_calls": msg.get("tool_calls") or [],
            "usage": data.get("usage", {}), "latency": latency,
            "model": data.get("model", model),
            "finish": choice.get("finish_reason", "")}


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

@dataclass
class ToolBox:
    """Read/execute tools bound to a workspace directory. The agent may only
    write candidate.json, patch.diff and notes/ (plan §5.7.1)."""
    ws: Path
    model_view: Any = None          # dict rendered by harness

    def _safe(self, path: str) -> Path:
        p = (self.ws / path.lstrip("/")).resolve()
        if not str(p).startswith(str(self.ws.resolve())):
            raise PermissionError(f"path escapes workspace: {path}")
        return p

    def read_file(self, path: str, start_line: int | None = None,
                  end_line: int | None = None) -> str:
        p = self._safe(path)
        if not p.exists() or not p.is_file():
            return f"(no such file: {path})"
        lines = p.read_text(errors="replace").splitlines()
        s = (start_line or 1) - 1
        e = end_line or min(len(lines), s + 300)
        out = []
        for i, l in enumerate(lines[s:e], start=s + 1):
            out.append(f"{i:5d} | {l}")
        if e < len(lines):
            out.append(f"... ({len(lines) - e} more lines; use start_line/end_line)")
        return "\n".join(out)

    def list_dir(self, path: str = ".") -> str:
        p = self._safe(path)
        if not p.exists():
            return f"(no such dir: {path})"
        out = []
        for c in sorted(p.iterdir()):
            rel = c.relative_to(self.ws).as_posix()
            out.append((rel + "/") if c.is_dir() else rel)
        return "\n".join(out[:200])

    def grep(self, pattern: str, path: str | None = None) -> str:
        root = self._safe(path or ".")
        if not root.exists():
            return f"(no such dir: {path})"
        rx = re.compile(pattern)
        hits: list[str] = []
        files = [root] if root.is_file() else sorted(root.rglob("*"))
        for f in files:
            if not f.is_file():
                continue
            if any(part in {".git", "node_modules", "__pycache__"} for part in f.parts):
                continue
            if f.suffix not in (".py", ".md", ".json", ".diff", ".txt", ".cfg", ".ini", ".toml"):
                continue
            try:
                lines = f.read_text(errors="replace").splitlines()
            except OSError:
                continue
            for i, l in enumerate(lines, 1):
                if rx.search(l):
                    rel = f.relative_to(self.ws).as_posix()
                    hits.append(f"{rel}:{i}: {l.strip()[:160]}")
                    if len(hits) >= 40:
                        return "\n".join(hits) + "\n(truncated)"
        return "\n".join(hits) if hits else "(no matches)"

    def query_model(self, kind: str, arg: str) -> str:
        import json as _json
        mv = self.model_view
        if mv is None:
            return "(no model available)"
        if isinstance(arg, str) and arg.startswith("repo/"):
            arg = arg[len("repo/"):]
        if kind == "functions":
            rows = [f for f in mv["functions"]
                    if f.get("file") == arg or arg in f["id"] or arg in f.get("file", "")]
            return _json.dumps(rows[:100], indent=1)
        if kind == "statements_of":
            rows = [s for s in mv["statements"] if s["func_id"] == arg]
            return _json.dumps(rows[:200], indent=1)
        if kind == "callers_of":
            rows = [f["id"] for f in mv["functions"] if any(c == arg for c in f["calls"])]
            return _json.dumps(rows, indent=1)
        if kind == "callees_of":
            f = next((x for x in mv["functions"] if x["id"] == arg), None)
            return _json.dumps(f["calls"] if f else mv.get("unresolved", [])[:50], indent=1)
        if kind == "covered_by":
            rows = mv["covered_by"].get(arg, [])
            return _json.dumps(rows, indent=1)
        return f"(unknown kind: {kind})"

    def write_candidate(self, hypotheses: list[dict], notes: str | None = None) -> str:
        norm = []
        for h in hypotheses:
            el = str(h.get("element", ""))
            if el.startswith("repo/"):
                el = el[len("repo/"):]
            nh = dict(h)
            nh["element"] = el
            if not nh.get("granularity"):
                nh["granularity"] = "method" if "::" in el else "statement"
            norm.append(nh)
        (self.ws / "candidate.json").write_text(_json_dumps({"hypotheses": norm, "notes": notes}))
        return f"wrote candidate.json with {len(norm)} hypotheses"

    def write_patch(self, diff: str) -> str:
        (self.ws / "patch.diff").write_text(diff)
        return "wrote patch.diff"


def _json_dumps(obj) -> str:
    import json as _json
    return _json.dumps(obj, indent=1)


# ---------------------------------------------------------------------------
# Prompts (plan Appendix A, adapted; describe task and loop only)
# ---------------------------------------------------------------------------

def task_md(problem_statement: str, failing_tests: list[str]) -> str:
    tests = "\n".join(f"- `{t}`" for t in failing_tests)
    return f"""# Task

Localize a fault in the program under `repo/`. The project has one known
defect: the tests listed below FAIL on this version of the code and are the
only observable evidence of the fault. Your job is to produce the ranked list
of program elements most likely to contain the fault, in `candidate.json`.

## Defect report (from the project maintainers)

{problem_statement.strip()}

## Failing tests (all currently fail; on a repaired version they must pass)

{tests}

An element is a method (`repo/.../file.py::Class.method`) or a statement
(`repo/.../file.py:LINE`). Valid ids are shown in `model/` and via the
`query_model` tool. Use at most 10 hypotheses, ranked, at most one per element.

A separate judge scores every candidate you leave: it checks your hypotheses
by attempting single-edit repairs at the suspected locations and by running
the tests. You do not run the program yourself: you have read and query tools
only. Improve `candidate.json`, optionally write `patch.diff` for your top
hypothesis, then end the turn. Work inside the workspace only.
"""


def first_turn(cfg_name: str, round_no: int, own_best: dict | None,
               peer_lines: list[str], n_hyp_max: int) -> str:
    own = ("none yet" if own_best is None
           else f"{own_best['id']} at score {own_best['score']}")
    s = f"""Round {round_no}. Your current best candidate: {own}."""
    if peer_lines:
        s += "\n" + "\n".join(peer_lines)
    s += f"""

.context/ holds what was prepared for this turn: the failing tests, spectrum
scores, the judge's feedback on your earlier candidates, and (for a team)
your peers' best candidates. Read it before changing anything.
Update candidate.json (write_candidate), optionally patch.diff, then end_turn
with a one-line summary of what you changed. Do not try to evaluate the
change yourself: another process scores your candidate after you stop and
invokes you again with the result.

IMPORTANT: end the turn only AFTER calling write_candidate — leave at least 3
ranked hypotheses in candidate.json. An empty candidate scores zero.
"""
    return s


def later_turn(round_no: int, own_best: dict | None, peer_lines: list[str]) -> str:
    own = ("unchanged" if own_best is None else f"{own_best['id']} at score {own_best['score']}")
    s = f"The judge's results are in .context/feedback/. Your best candidate so far: {own}."
    if peer_lines:
        s += "\n" + "\n".join(peer_lines)
    s += ("\nImprove candidate.json and end the session. Another process scores it,"
          " invokes you again with the result.")
    return s
