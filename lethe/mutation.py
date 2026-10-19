"""Single-edit mutation engine for the judge (Python targets).

Every mutant is exactly one small AST edit inside the suspected element
(relational/arithmetic/logical operator swap, constant shift, unary flip,
statement deletion). Mutant sites are ordered by the spectrum score of their
line (deterministic), capped at ``max_mutants``; truncation is flagged so a
truncated probe can never be used for exoneration (plan §5.5).

Regeneration uses ``ast.unparse`` on the whole file: comment/formatting layout
changes but semantics are preserved. Mutants that fail to reparse are counted
as compile failures, never as evidence either way.
"""
from __future__ import annotations

import ast
import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path

from .core import ProgramModel

OPSET_VERSION = "v1"

_COMP = {ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Lt: ast.LtE, ast.LtE: ast.Lt,
         ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.In: ast.NotIn, ast.NotIn: ast.In,
         ast.Is: ast.IsNot, ast.IsNot: ast.Is}
_BIN = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult,
        ast.FloorDiv: ast.Mod, ast.Mod: ast.FloorDiv, ast.BitXor: ast.BitAnd,
        ast.BitAnd: ast.BitOr}


@dataclass
class Mutant:
    mutation_id: str
    op: str
    file: str          # repo-relative path
    line: int
    text: str          # full mutated file text


def _parent_slot(tree: ast.AST, target: ast.AST):
    for parent in ast.walk(tree):
        for field in parent._fields:
            val = getattr(parent, field)
            if isinstance(val, list):
                for i, v in enumerate(val):
                    if v is target:
                        return parent, field, i
            elif val is target:
                return parent, field, None
    return None


def _copy_mapped(tree: ast.AST) -> tuple[ast.AST, dict[int, ast.AST]]:
    """Deep-copy the tree and map each original node id to its copy by walking
    both trees in the same deterministic structural order."""
    m: dict[int, ast.AST] = {}
    t = copy.deepcopy(tree)

    def walk(a: ast.AST, c: ast.AST) -> None:
        m[id(a)] = c
        for name in a._fields:
            va = getattr(a, name)
            vc = getattr(c, name)
            if isinstance(va, list):
                for xa, xc in zip(va, vc):
                    if isinstance(xa, ast.AST):
                        walk(xa, xc)
            elif isinstance(va, ast.AST):
                walk(va, vc)

    walk(tree, t)
    return t, m


def _splice(t: ast.AST, target: ast.AST, repl: ast.AST) -> None:
    if target is None:
        raise ValueError("target is None")
    for parent in ast.walk(t):
        for field in parent._fields:
            val = getattr(parent, field)
            if isinstance(val, list):
                for i, v in enumerate(val):
                    if v is target:
                        val[i] = repl
                        return
            elif val is target:
                setattr(parent, field, repl)
                return
    raise ValueError("node not found in copy")


def _mutate_node(node: ast.AST, op_name: str) -> ast.AST:
    if op_name.startswith("cmp:"):
        repl = copy.deepcopy(node)
        repl.ops[0] = _COMP[type(node.ops[0])]()
        return repl
    if op_name.startswith("binop:"):
        repl = copy.deepcopy(node)
        repl.op = _BIN[type(node.op)]
        return repl
    if op_name.startswith("boolop:"):
        repl = copy.deepcopy(node)
        repl.op = ast.Or if type(node.op) is ast.And else ast.And
        return repl
    if op_name.startswith("unary:"):
        # flip  ->  drop the wrapper;  -x  ->  x (sign flip)
        return node.operand
    if op_name in ("const+1", "const-1"):
        repl = copy.deepcopy(node)
        repl.value = node.value + (1 if op_name == "const+1" else -1)
        return repl
    raise ValueError(op_name)


def generate_mutants(root: Path, rel: str, pm: ProgramModel, element: str, max_mutants: int,
                     spectrum: dict[str, float] | None = None) -> tuple[list[Mutant], bool, int]:
    """Return (mutants, truncated, compile_failures) for one element.

    ``root`` is the project checkout and ``rel`` the repo-relative file path
    of the element. ``spectrum`` maps statement ids to scores and orders sites.
    """
    kind, obj = pm.resolve_element(element)
    if kind == "unknown":
        return [], False, 0
    elem_start, elem_end = obj.lineno, obj.end_lineno
    src_path = root / rel
    try:
        src = src_path.read_text(errors="replace")
        tree = ast.parse(src)
    except (SyntaxError, OSError):
        return [], False, 0

    def skey(x: ast.AST) -> float:
        if not spectrum:
            return 0.0
        return spectrum.get(f"{rel}:{getattr(x, 'lineno', 0)}", 0.0)

    sites: list[tuple[float, int, str, str]] = []  # (-spectrum, line, order, op)
    def_node_exclude = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for n in ast.walk(tree):
        line = getattr(n, "lineno", 0) or 0
        if not (elem_start <= line <= elem_end):
            continue
        if isinstance(n, ast.Compare) and n.ops and type(n.ops[0]) in _COMP:
            sites.append((-skey(n), line, 0, "cmp"))
        elif isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            sites.append((-skey(n), line, 1, "binop"))
        elif isinstance(n, ast.BoolOp) and type(n.op) in (ast.And, ast.Or):
            sites.append((-skey(n), line, 2, "boolop"))
        elif isinstance(n, ast.UnaryOp) and type(n.op) in (ast.Not, ast.USub):
            sites.append((-skey(n), line, 3, "unary"))
        elif isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) \
                and not isinstance(n.value, bool):
            sites.append((-skey(n), line, 4, "const+1"))
            sites.append((-skey(n), line, 5, "const-1"))

    # statement deletion: first real statement of the element (never a def)
    stmt_node = None
    cands = [n for n in ast.walk(tree) if isinstance(n, ast.stmt) and not isinstance(n, def_node_exclude)
             and elem_start - 1 < (getattr(n, "lineno", 0) or 0) <= elem_end]
    if cands:
        stmt_node = min(cands, key=lambda n: getattr(n, "lineno", 0))
    if stmt_node is not None and (kind == "statement" or True):
        sites.append((0.0, getattr(stmt_node, "lineno", 0), 9, "del_stmt"))

    sites.sort(key=lambda t: (t[0], t[1], t[2]))
    file_hash = hashlib.sha256(src.encode()).hexdigest()[:16]
    truncated = False
    out: list[Mutant] = []
    compile_failures = 0
    ord = 0
    for negscore, line, order, op in sites:
        if len(out) >= max_mutants:
            truncated = True
            break
        variants = _variants_on_line(tree, op, line)
        for op_name, node in variants:
            if len(out) >= max_mutants:
                truncated = True
                break
            text = _realize(tree, node, op_name)
            if text is None:
                compile_failures += 1
                continue
            mid = hashlib.sha256(
                f"{file_hash}|{line}|{op_name}|{ord}|{OPSET_VERSION}".encode()).hexdigest()[:24]
            out.append(Mutant(mutation_id=mid, op=op_name, file=rel, line=line, text=text))
            ord += 1
    return out, truncated, compile_failures


def _variants_on_line(tree: ast.AST, op: str, line: int) -> list[tuple[str, ast.AST]]:
    res: list[tuple[str, ast.AST]] = []
    skip = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    for n in ast.walk(tree):
        ln = getattr(n, "lineno", 0) or 0
        if ln != line or isinstance(n, skip):
            continue
        if op == "cmp" and isinstance(n, ast.Compare) and n.ops and type(n.ops[0]) in _COMP:
            res.append(("cmp:" + type(n.ops[0]).__name__.lower(), n))
        elif op == "binop" and isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            res.append(("binop:" + type(n.op).__name__.lower(), n))
        elif op == "boolop" and isinstance(n, ast.BoolOp) and type(n.op) in (ast.And, ast.Or):
            res.append(("boolop:" + type(n.op).__name__.lower(), n))
        elif op == "unary" and isinstance(n, ast.UnaryOp) and type(n.op) in (ast.Not, ast.USub):
            res.append(("unary:" + type(n.op).__name__.lower(), n))
        elif op in ("const+1", "const-1") and isinstance(n, ast.Constant) \
                and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            res.append((op, n))
        elif op == "del_stmt" and isinstance(n, ast.stmt):
            res.append(("del_stmt", n))
    return res


def _realize(tree: ast.AST, node: ast.AST, op_name: str) -> str | None:
    """Apply one edit to a deep copy of the tree and return parsed text."""
    try:
        t, m = _copy_mapped(tree)
        mnode = m.get(id(node))
        if mnode is None:
            return None
        if op_name == "del_stmt":
            slot = _parent_slot(tree, node)
            if slot is None:
                return None
            parent, field, idx = slot
            cparent = m[id(parent)]
            val = getattr(cparent, field)
            if isinstance(val, list):
                if len(val) == 1:
                    val[0] = ast.Pass()
                else:
                    del val[idx]
            else:
                return None
            ast.fix_missing_locations(t)
            text = ast.unparse(t)
            ast.parse(text)
            return text
        repl = _mutate_node(node, op_name)
        if isinstance(repl, ast.stmt):
            repl = ast.copy_location(repl, node)
        repl = ast.fix_missing_locations(repl)
        _splice(t, mnode, repl)
        ast.fix_missing_locations(t)
        text = ast.unparse(t)
        ast.parse(text)
        return text
    except Exception:
        return None
