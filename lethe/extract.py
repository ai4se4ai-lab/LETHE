"""Extract a program model from a Python project using ``ast``.

Rules (plan §5.2):
- one statement per AST statement node inside a function body (docstrings,
  comments and module-level statements are excluded);
- each statement is attributed to the innermost function that contains it;
- call edges by best-effort name resolution (unresolved recorded, not guessed);
- statement identity is ``file:line`` (first line of the statement).
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from .core import Function, ProgramModel, Statement

SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "build", ".tox", ".eggs"}
TEST_NAME_DIRS = {"tests", "test"}


def _is_test_path(relp: Path) -> bool:
    parts = relp.parts
    if len(parts) > 1 and any(p in TEST_NAME_DIRS for p in parts[:-1]):
        return True
    s = relp.as_posix()
    return (s.startswith("test") or "/test_" in s or s.endswith("_test.py")
            or s.startswith("conftest") or "/conftest" in s)


def _walk_functions(node: ast.AST, class_name: str | None, rf: str, out: list[Function]) -> None:
    """Register every def (method, top-level, nested) with file-scoped ids."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            for m in child.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append(Function(
                        id=f"{rf}::{child.name}.{m.name}", file=rf, name=m.name,
                        class_name=child.name, lineno=m.lineno,
                        end_lineno=getattr(m, "end_lineno", m.lineno)))
                    _walk_functions(m, child.name, rf, out)
            _walk_functions(child, class_name, rf, out)  # nested defs outside methods
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(Function(
                id=f"{rf}::{class_name + '.' if class_name else ''}{child.name}", file=rf,
                name=child.name, class_name=class_name, lineno=child.lineno,
                end_lineno=getattr(child, "end_lineno", child.lineno)))
            _walk_functions(child, class_name, rf, out)
        else:
            _walk_functions(child, class_name, rf, out)


def extract(root: Path) -> ProgramModel:
    root = root.resolve()
    pm = ProgramModel(root=root.as_posix())
    calls_raw: list[tuple[str, str]] = []
    unresolved: set[str] = set()
    source_digest = hashlib.sha256()

    for p in sorted(root.rglob("*.py")):
        try:
            relp = p.relative_to(root)
        except ValueError:
            continue
        if _is_test_path(relp) or p.name in ("__init__.py", "setup.py", "conftest.py"):
            continue
        try:
            src = p.read_text(errors="replace")
            tree = ast.parse(src)
        except SyntaxError:
            continue
        rf = relp.as_posix()
        source_digest.update(rf.encode())
        source_digest.update(b"\0")
        source_digest.update(src.encode())
        source_digest.update(b"\0")
        pm.components.append(rf)

        file_funcs: list[Function] = []
        _walk_functions(tree, None, rf, file_funcs)
        pm.functions.extend(file_funcs)

        # ---- statements: every stmt inside the innermost function ----------
        def innermost(s: ast.AST) -> Function | None:
            best = None
            for fn in file_funcs:
                if fn.lineno <= s.lineno <= fn.end_lineno:
                    if best is None or (fn.end_lineno - fn.lineno) < (best.end_lineno - best.lineno):
                        best = fn
            return best

        for s in ast.walk(tree):
            if not isinstance(s, ast.stmt) or isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str):
                continue  # docstring
            fn = innermost(s)
            if fn is None:
                continue
            text = (ast.get_source_segment(src, s) or "").replace("\n", " ")[:200]
            pm.statements.append(Statement(
                id=f"{rf}:{s.lineno}", file=rf, lineno=s.lineno,
                end_lineno=getattr(s, "end_lineno", s.lineno),
                text=text, func_id=fn.id, kind=type(s).__name__))

        # ---- calls for each function ---------------------------------------
        for fn in file_funcs:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and node.name == fn.name and node.lineno == fn.lineno \
                        and node.end_lineno == fn.end_lineno:
                    for c in ast.walk(node):
                        if isinstance(c, ast.Call):
                            t = c.func
                            if isinstance(t, ast.Name):
                                calls_raw.append((fn.id, t.id))
                            elif isinstance(t, ast.Attribute):
                                calls_raw.append((fn.id, t.attr))
                    break

    by_name: dict[str, list[str]] = {}
    for fn in pm.functions:
        by_name.setdefault(fn.name, []).append(fn.id)
    by_id = {f.id: f for f in pm.functions}
    for fid, cname in calls_raw:
        if fid not in by_id:
            continue
        if cname not in by_name:
            unresolved.add(f"{fid} -> {cname}")
            continue
        for target in by_name[cname]:
            if target != fid and target not in by_id[fid].calls:
                by_id[fid].calls.append(target)

    pm.unresolved_calls = sorted(unresolved)
    pm.source_hash = source_digest.hexdigest()
    pm.components = sorted(set(pm.components))
    seen: set[str] = set()
    pm.statements = [s for s in sorted(pm.statements, key=lambda s: (s.file, s.lineno))
                     if not (s.id in seen or seen.add(s.id))]
    seenf: set[str] = set()
    pm.functions = [f for f in pm.functions if not (f.id in seenf or seenf.add(f.id))]
    pm.index()
    return pm
