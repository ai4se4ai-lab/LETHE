#!/usr/bin/env python3
"""hpc/scripts/audit.py — live tabular audit of a run: one row per stage x cell (array task).

    python3 hpc/scripts/audit.py [RUN_ID]                 snapshot (default: newest run in hpc/logs/)
    python3 hpc/scripts/audit.py RUN_ID --watch 120       redraw every 120 s (minimum 60); stops when nothing is queued/running
    python3 hpc/scripts/audit.py RUN_ID --only failed     only failed / cancelled / timed-out rows (also: running, pending)
    python3 hpc/scripts/audit.py RUN_ID --steps           add one line per step under every row
    python3 hpc/scripts/audit.py RUN_ID --no-color

Columns: STAGE[cell]  CELL (matrix params)  STATE  NODE  RUN/LIMIT  WAIT (queue reason)  STEPS ok/total
         PROGRESS (records written / expected, from the step's "progress" spec)  CURRENT or FAILED step  DETAIL
Below the table: every failed step with its reason, the last lines of its log and the log path.

Sources (read-only; ONE squeue and ONE sacct call per snapshot — safe on a login node):
  logs/<RUN_ID>/jobs.tsv, config/experiments.resolved.json, steps.tsv (ledger), steps/<stage>/<cell>/*.running|*.log,
  $RESULTS (progress files). Every snapshot is saved without colour as logs/<RUN_ID>/audit.txt and audit.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hpclib as H  # noqa: E402

COL = {"OK": "32", "FAILED": "31", "TIMEOUT": "31", "CANCELLED": "31", "OUT_OF_MEMORY": "31", "NODE_FAIL": "31",
       "RUNNING": "1;33", "PENDING": "90", "PARTIAL": "38;5;208", "REQUEUED": "38;5;208", "NOT_SUBMITTED": "90",
       "head": "1;37", "dim": "90", "stage": "1;36", "bar": "32", "warn": "38;5;208"}
SYM = {"OK": "✔", "FAILED": "✘", "TIMEOUT": "⏱", "CANCELLED": "⊘", "OUT_OF_MEMORY": "✘", "NODE_FAIL": "✘", "RUNNING": "▶",
       "PENDING": "○", "PARTIAL": "◐", "REQUEUED": "↻", "NOT_SUBMITTED": "–"}
USE_COLOR = False


def c(kind: str, s: str) -> str:
    return f"\033[{COL.get(kind, '0')}m{s}\033[0m" if USE_COLOR else s


def hms(sec: float | None) -> str:
    if sec is None:
        return "-"
    sec = int(sec)
    d, r = divmod(sec, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    return f"{d}d{h:02d}h" if d else f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def slurm(ids: list[str]) -> dict:
    """{'123' or '123_4': {state, reason, node, elapsed, limit, exit}} from one squeue + one sacct call."""
    out: dict = {}
    if not ids or not shutil.which("squeue"):
        return out
    for line in sh(["sacct", "-n", "-P", "-X", "-j", ",".join(ids),
                    "--format=JobID,State,Elapsed,Timelimit,NodeList,ExitCode"]).splitlines():
        f = line.split("|")
        if len(f) >= 6:
            out[f[0]] = {"state": f[1].split()[0], "elapsed": f[2], "limit": f[3], "node": f[4], "exit": f[5], "reason": ""}
    for line in sh(["squeue", "-h", "-r", "-j", ",".join(ids), "-o", "%i|%T|%r|%N|%M|%l"]).splitlines():
        f = line.split("|")
        if len(f) >= 6:
            out[f[0]] = {"state": f[1], "reason": f[2] if f[1] == "PENDING" else "", "node": f[3] or "-",
                         "elapsed": f[4], "limit": f[5], "exit": ""}
    return out


def secs(t: str) -> float | None:
    if not t or t in ("INVALID", "UNLIMITED", "Partition_Limit"):
        return None
    d = 0
    if "-" in t:
        dd, t = t.split("-", 1)
        d = int(dd)
    p = [int(x) for x in t.split(":")]
    while len(p) < 3:
        p.insert(0, 0)
    return d * 86400 + p[0] * 3600 + p[1] * 60 + p[2]


def tail(p: Path, n: int = 3) -> list[str]:
    if not p.exists():
        return []
    lines = [x.rstrip() for x in p.read_text(errors="replace").splitlines() if x.strip()]
    err = [x for x in lines if re.search(r"error|exception|traceback|failed|killed|refusing|missing", x, re.I)]
    return [x[:160] for x in (err[-n:] if err else lines[-n:])]


def assemble(run_id: str) -> dict:
    lr = H.log_root(run_id)
    exp = json.loads((lr / "config" / "experiments.resolved.json").read_text())
    c = H.cluster(run_id)
    results = H.results_dir(run_id, c)
    jobs = {}
    for r in H.read_tsv(lr / "jobs.tsv"):
        jobs[r[1]] = {"id": r[0], "n": int(r[2]), "deps": r[3] if len(r) > 3 else ""}   # latest submission per stage
    sl = slurm([j["id"] for j in jobs.values()])
    ledger = H.latest_steps(run_id)
    man = {}
    if (lr / "run_manifest.json").exists():
        man = json.loads((lr / "run_manifest.json").read_text())
    rows = []
    now = time.time()
    for st in exp["stages"]:
        j = jobs.get(st["name"])
        for i, cell in enumerate(st["cells"]):
            ctx = H.context(exp, c, run_id, st["name"], cell, i)
            ctx["RESULTS"] = str(results)
            key = ctx["CELL"]
            info = {}
            if j:
                info = sl.get(f"{j['id']}_{i}") if j["n"] > 1 else sl.get(j["id"])
                if info is None and j["n"] > 1:   # still-pending arrays may be listed as 123_[0-7%2]
                    info = next((v for k, v in sl.items() if k.startswith(j["id"] + "_[")), None)
                info = info or {}
            steps, ok, failed_steps, current = [], 0, [], None
            for s in st["steps"]:
                led = ledger.get((st["name"], key, s["name"]))
                run_m = H.log_root(run_id) / "steps" / st["name"] / key / f"{s['name']}.running"
                if run_m.exists():
                    try:
                        started = json.loads(run_m.read_text()).get("start", now)
                    except (json.JSONDecodeError, OSError):
                        started = now
                    status, sec = "RUNNING", now - started
                    current = s["name"]
                elif led:
                    status, sec = ("OK" if led["rc"] == "0" else "FAILED"), float(led.get("seconds") or 0)
                else:
                    status, sec = "PENDING", None
                prog = None
                if s.get("progress"):
                    try:
                        f = results / H.fill(s["progress"]["file"], ctx)
                        exp_n = H.fill_number(s["progress"].get("expected", ""), ctx)
                        prog = (H.count_records(f), exp_n)
                    except KeyError:
                        prog = None
                steps.append({"step": s["name"], "status": status, "seconds": sec, "rc": led.get("rc") if led else None,
                              "note": led.get("note", "") if led else "", "progress": prog,
                              "log": str(H.log_root(run_id) / "steps" / st["name"] / key / f"{s['name']}.log")})
                ok += status == "OK"
                if status == "FAILED":
                    failed_steps.append(steps[-1])
            st_slurm = info.get("state", "")
            if current:   # a step is executing right now (also covers runs outside Slurm)
                state = "RUNNING"
            elif not j:
                state = "NOT_SUBMITTED" if not any(x["status"] != "PENDING" for x in steps) else ("OK" if ok == len(steps) else "PARTIAL")
            elif st_slurm in ("RUNNING", "COMPLETING"):
                state = "RUNNING"
            elif st_slurm in ("PENDING", "REQUEUED", "CONFIGURING", "") and ok < len(steps):
                state = "PENDING" if st_slurm != "REQUEUED" else "REQUEUED"
            elif st_slurm == "COMPLETED":
                state = "OK" if not failed_steps else "PARTIAL"
            elif st_slurm.startswith("CANCELLED"):
                state = "CANCELLED"
            elif st_slurm in ("TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"):
                state = st_slurm
            elif st_slurm == "FAILED":
                state = "PARTIAL" if ok else "FAILED"
            else:
                state = "OK" if ok == len(steps) else (st_slurm or "PENDING")
            progs = [x["progress"] for x in steps if x["progress"] and x["progress"][1]]
            done_n = sum(p[0] for p in progs)
            exp_n = sum(p[1] for p in progs)
            rows.append({"stage": st["name"], "cell_id": i, "cell": H.cell_label(cell), "job": (f"{j['id']}_{i}" if j and j["n"] > 1 else (j or {}).get("id", "")),
                         "state": state, "slurm_state": st_slurm, "reason": info.get("reason", ""), "node": info.get("node", "-"),
                         "elapsed": secs(info.get("elapsed", "")), "limit": secs(info.get("limit", "")),
                         "steps_ok": ok, "steps_total": len(steps), "current": current,
                         "failed": [x["step"] for x in failed_steps],
                         "progress": [done_n, exp_n] if exp_n else None, "steps": steps,
                         "gpu": st.get("resources", {}).get("gpus") or st.get("resources", {}).get("gpus_per_node") or "cpu"})
    restarts = [r for r in H.read_tsv(lr / "jobs_finished.tsv") if len(r) > 5 and r[5] == "requeued"]
    return {"run_id": run_id, "time": time.strftime("%F %T"), "smoke": exp.get("is_smoke"), "project": exp.get("project"),
            "commit": (man.get("git") or {}).get("commit", "")[:10], "results": str(results), "rows": rows,
            "requeues": len(restarts)}


def bar(p, width=10) -> str:
    if not p:
        return "-"
    done, total = p
    frac = min(done / total, 1.0) if total else 0
    full = int(round(frac * width))
    return c("bar", "█" * full) + c("dim", "░" * (width - full)) + f" {100 * frac:3.0f}% {done}/{int(total)}"


def render(a: dict, only: str, show_steps: bool) -> str:
    rows = a["rows"]
    if only == "failed":
        rows = [r for r in rows if r["state"] in ("FAILED", "PARTIAL", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL") or r["failed"]]
    elif only in ("running", "pending"):
        rows = [r for r in rows if r["state"].lower() == only]
    counts: dict = {}
    for r in a["rows"]:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    st_ok = sum(r["steps_ok"] for r in a["rows"])
    st_all = sum(r["steps_total"] for r in a["rows"])
    st_fail = sum(len(r["failed"]) for r in a["rows"])
    pr = [r["progress"] for r in a["rows"] if r["progress"]]
    w = min(shutil.get_terminal_size((160, 40)).columns, 200)
    L = [c("head", "═" * w),
         c("head", f" {a['project']} — run audit {a['run_id']}{'  [SMOKE]' if a['smoke'] else ''}   {a['time']}   commit {a['commit'] or '?'}"),
         f" results {a['results']}",
         c("head", "═" * w),
         " cells  " + "   ".join(f"{c(k, SYM.get(k, '?') + ' ' + k)} {v}" for k, v in sorted(counts.items())) + f"   (total {len(a['rows'])})",
         f" steps  {st_ok}/{st_all} ok, {st_fail} failed     requeues {a['requeues']}"
         + (f"     records {bar([sum(p[0] for p in pr), sum(p[1] for p in pr)], 20)}" if pr else ""), ""]
    hdr = ["", "STAGE[cell]", "CELL", "STATE", "GPU", "NODE", "RUN / LIMIT", "WAIT", "STEPS", "PROGRESS", "CURRENT / FAILED", "DETAIL"]
    table = []
    for r in rows:
        cur = r["current"] or (",".join(r["failed"]) if r["failed"] else "-")
        detail = "-"
        if r["state"] == "RUNNING" and r["current"]:
            s = next(x for x in r["steps"] if x["step"] == r["current"])
            detail = f"step {hms(s['seconds'])}"
        elif r["failed"]:
            s = next(x for x in r["steps"] if x["step"] == r["failed"][-1])
            detail = (s["note"] or f"rc={s['rc']}")[:40]
        elif r["state"] == "OK":
            detail = "finished"
        table.append([SYM.get(r["state"], "?"), f"{r['stage']}[{r['cell_id']}]", r["cell"][:34], r["state"], str(r["gpu"])[:14],
                      (r["node"] or "-")[:12], f"{hms(r['elapsed'])} / {hms(r['limit'])}" if r["elapsed"] is not None else "-",
                      (r["reason"] or "-")[:16], f"{r['steps_ok']}/{r['steps_total']}", r["progress"], cur[:26], detail])
    plain = lambda v: (bar(v) if isinstance(v, list) or v is None else str(v))  # noqa: E731
    strip = lambda s: re.sub(r"\033\[[0-9;]*m", "", s)  # noqa: E731
    widths = [max([len(hdr[k])] + [len(strip(plain(t[k]) if k == 9 else str(t[k]))) for t in table]) for k in range(len(hdr))]
    L.append("  ".join(c("head", h.ljust(widths[k])) for k, h in enumerate(hdr)))
    L.append(c("dim", "  ".join("─" * x for x in widths)))
    last_stage = None
    for t, r in zip(table, rows):
        if r["stage"] != last_stage and last_stage is not None:
            L.append("")
        last_stage = r["stage"]
        cells = []
        for k, v in enumerate(t):
            s = plain(v) if k == 9 else str(v)
            pad = " " * (widths[k] - len(strip(s)))
            if k in (0, 3):
                s = c(r["state"], s)
            elif k == 1:
                s = c("stage", s)
            cells.append(s + pad)
        L.append("  ".join(cells))
        if show_steps:
            for s in r["steps"]:
                L.append(c("dim", f"      └ {s['step']:<26} {s['status']:<8} {hms(s['seconds']):>8}  "
                                  f"{(bar(list(s['progress'])) if s['progress'] and s['progress'][1] else '')}  {s['note'][:60]}"))
    fails = [(r, s) for r in rows for s in r["steps"] if s["status"] == "FAILED"]
    if fails:
        L += ["", c("FAILED", f" Failures ({len(fails)})")]
        for r, s in fails:
            L.append(c("FAILED", f"  ✘ {r['stage']}[{r['cell_id']}] {s['step']}  rc={s['rc']}  {s['note']}"))
            for line in tail(Path(s["log"])):
                L.append(c("dim", f"      │ {line}"))
            L.append(c("dim", f"      log: {s['log']}"))
    L.append("")
    return "\n".join(L)


def latest_run() -> str:
    logs = H.HPC / "logs"
    runs = sorted((p for p in logs.glob("*") if (p / "jobs.tsv").exists()), key=lambda p: p.stat().st_mtime) if logs.exists() else []
    return runs[-1].name if runs else ""


def main() -> int:
    global USE_COLOR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id", nargs="?", default="")
    ap.add_argument("--watch", type=int, nargs="?", const=60, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--steps", action="store_true")
    ap.add_argument("--no-color", action="store_true")
    a = ap.parse_args()
    run_id = a.run_id or latest_run()
    if not run_id or not (H.log_root(run_id) / "config" / "experiments.resolved.json").exists():
        print("no run found (submit one with hpc/slurm/submit.py)")
        return 1
    USE_COLOR = sys.stdout.isatty() and not a.no_color and not os.environ.get("NO_COLOR")
    period = max(a.watch, 60) if a.watch else 0
    while True:
        data = assemble(run_id)
        text = render(data, a.only, a.steps)
        lr = H.log_root(run_id)
        (lr / "audit.txt").write_text(re.sub(r"\033\[[0-9;]*m", "", render(data, "", True)))
        (lr / "audit.json").write_text(json.dumps(data, indent=1, default=str))
        if period:
            print("\033[2J\033[H", end="")
        print(text)
        active = any(r["state"] in ("RUNNING", "PENDING", "REQUEUED") for r in data["rows"])
        if not period or not active:
            if period:
                print("nothing queued or running any more; leaving watch mode")
            return 0
        print(c("dim", f" refreshing every {period} s (Ctrl-C leaves the view; jobs keep running)"))
        try:
            time.sleep(period)   # login-side viewer only; never used inside jobs
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
