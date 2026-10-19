#!/usr/bin/env python3
"""hpc/scripts/manifest.py — record what a run used (stdlib only; called by submit.py on every submission).

Writes hpc/logs/<RUN_ID>/run_manifest.json (and a copy in $WORK/runs/<RUN_ID>/): run id, git commit and branch,
dirty files and a hash of the uncommitted diff, content hashes of CODE_HASH_DIRS, the frozen configuration,
host, and the list of submission times. Two result sets are comparable when their hashes match.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hpclib as H  # noqa: E402


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(H.REPO), *args], capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    if not root.exists():
        return "missing"
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".pyc") and p.stat().st_size < 50_000_000:
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def main() -> int:
    run_id = sys.argv[1]
    c = H.cluster(run_id)
    lr = H.log_root(run_id)
    out = lr / "run_manifest.json"
    prev = json.loads(out.read_text()) if out.exists() else {}
    dirs = (c.get("CODE_HASH_DIRS") or c.get("SRC_DIRS") or "src").split()
    exp_path = lr / "config" / "experiments.resolved.json"
    man = {
        "run_id": run_id,
        "created": prev.get("created") or time.strftime("%F %T %Z"),
        "submissions": prev.get("submissions", []) + [time.strftime("%F %T")],
        "submit_host": platform.node(),
        "git": {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": git("status", "--porcelain").splitlines()[:200],
                "diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest()[:16]},
        "content_sha256": {d: tree_hash(H.REPO / d) for d in dirs + ["hpc/config"]},
        "cluster_env": {k: v for k, v in c.items() if "TOKEN" not in k and "KEY" not in k and "PASS" not in k},
        "experiments_sha256": hashlib.sha256(exp_path.read_bytes()).hexdigest()[:16] if exp_path.exists() else "",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(man, indent=1))
    rd = H.run_dir(run_id, c)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "run_manifest.json").write_text(out.read_text())
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
