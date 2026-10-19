#!/usr/bin/env python3
"""hpc/scripts/run_stage.py — run the steps of one stage for one cell, inside a Slurm job.

For every step of the stage (experiments.json), in order:
  * skip it if its .done marker exists (unless "always": true)            -> resumable after requeue
  * check its inputs: every glob in "needs" (relative to $RESULTS) must match -> stages are connected by data
  * run "cmd" (bash, placeholders filled) or a built-in ("check_gpu", "collect");
    "launcher": "srun" runs it on all tasks of a multi-node allocation
  * retry it up to "retries" times
  * check its outputs: every glob in "produces" must match afterwards
  * write its full output to hpc/logs/<RUN_ID>/steps/<stage>/<cell>/<step>.log, a row to steps.tsv (the ledger),
    a .running marker while it runs (read by the live audit), and a .done marker on success
A failed step is recorded and the stage continues, unless the step is "critical" (then the stage stops).
Exit code: 0 if every step succeeded, else 1.

Refuses to run outside a Slurm job (no computation on login nodes); --local is for tiny tests of the runner itself.

    python hpc/scripts/run_stage.py --stage main --cell 3            (what stage.sh runs)
    python hpc/scripts/run_stage.py --stage main --list [--run-id R] (print the steps of every cell; no execution)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hpclib as H  # noqa: E402

JOB = os.environ.get("SLURM_JOB_ID", "0")
TASK = os.environ.get("SLURM_ARRAY_TASK_ID", "")
RESTART = os.environ.get("SLURM_RESTART_COUNT", "0")


def matches(results: Path, pattern: str) -> list[str]:
    return glob.glob(str(results / pattern), recursive=True)


def builtin(name: str, ctx: dict, log) -> int:
    if name == "check_gpu":
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                              "--format=csv"], capture_output=True, text=True)
        log(out.stdout.strip() or out.stderr.strip())
        dest = Path(ctx["RESULTS"]) / "setup"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"gpu_{socket.gethostname()}.txt").write_text(out.stdout)
        return out.returncode
    if name == "check_nodes":
        # every node of the allocation: has a GPU, sees the shared results dir, can reach the master node
        probe = Path(ctx["RESULTS"]) / "setup" / f".probe_{ctx['JOB_ID']}"
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text(socket.gethostname())
        master = os.environ.get("MASTER_ADDR", socket.gethostname())
        script = (f'h=$(hostname); g=$(nvidia-smi -L 2>/dev/null | wc -l); '
                  f'f=$(cat {shlex.quote(str(probe))} 2>/dev/null || echo NONE); '
                  f'p=$(getent hosts {shlex.quote(master)} >/dev/null && echo ok || echo FAIL); '
                  f'echo "$h gpus=$g shared_fs_sees=$f master_resolves=$p"')
        cmd = ["srun", "--ntasks-per-node=1", f"--nodes={os.environ.get('SLURM_NNODES', '1')}", "--overlap", "bash", "-c", script] \
            if os.environ.get("SLURM_JOB_ID") else ["bash", "-c", script]
        out = subprocess.run(cmd, capture_output=True, text=True)
        log(out.stdout.strip() or out.stderr.strip())
        lines = [x for x in out.stdout.splitlines() if "gpus=" in x]
        bad = [x for x in lines if "gpus=0" in x or "NONE" in x or "FAIL" in x]
        want = int(os.environ.get("SLURM_NNODES", "1"))
        if len(lines) < want or bad:
            log(f"node check failed: {len(lines)}/{want} nodes answered, problems: {bad}")
            return 1
        log(f"all {want} node(s) have GPUs, share {ctx['RESULTS']} and resolve {master}")
        return out.returncode
    if name == "collect":
        return subprocess.run([sys.executable, str(H.HPC / "scripts" / "collect_results.py"), ctx["RUN_ID"]]).returncode
    log(f"unknown builtin {name!r}")
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--cell", type=int, default=0)
    ap.add_argument("--run-id", default=os.environ.get("RUN_ID", ""))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--local", action="store_true", help="allow running outside Slurm (tiny tests only)")
    a = ap.parse_args()
    run_id = a.run_id or "local"
    cfg = H.config_dir(run_id)
    resolved = cfg / "experiments.resolved.json"
    exp = json.loads(resolved.read_text()) if resolved.exists() else H.resolve(H.load_experiments(cfg / "experiments.json"), False)
    c = H.cluster(run_id)
    st = H.stage(exp, a.stage)

    if a.list:
        for i, cell in enumerate(st["cells"]):
            ctx = H.context(exp, c, run_id, st["name"], cell, i)
            print(f"[cell {i}] {H.cell_label(cell)}")
            for s in st["steps"]:
                print(f"   {s['name']:<24} {H.fill(s['cmd'], ctx) if s.get('cmd') else '[' + s['builtin'] + ']'}")
        return 0
    if not os.environ.get("SLURM_JOB_ID") and not a.local:
        print("refusing to run outside a Slurm job (login nodes are not for computation). "
              "Submit with hpc/slurm/submit.py, or use --local for a tiny test of the runner.", file=sys.stderr)
        return 10
    if a.cell >= len(st["cells"]):
        print(f"cell {a.cell} out of range ({len(st['cells'])} cells)", file=sys.stderr)
        return 11

    cell = st["cells"][a.cell]
    ctx = H.context(exp, c, run_id, st["name"], cell, a.cell)
    results = Path(ctx["RESULTS"])
    results.mkdir(parents=True, exist_ok=True)
    key = ctx["CELL"]
    state = H.run_dir(run_id, c) / "state" / st["name"] / key
    logdir = H.log_root(run_id) / "steps" / st["name"] / key
    state.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "cell.json").write_text(json.dumps({"cell_id": a.cell, "params": cell}))
    ledger = H.log_root(run_id) / "steps.tsv"
    if not ledger.exists():
        ledger.write_text(H.LEDGER_HEADER)

    print(f"=== stage {st['name']} cell {a.cell} [{H.cell_label(cell)}]: {len(st['steps'])} steps", flush=True)
    failed = []
    for s in st["steps"]:
        marker = state / f"{s['name']}.done"
        if marker.exists() and not s.get("always"):
            print(f"--- skip {s['name']} (done {marker.read_text().strip()})", flush=True)
            continue
        logf = logdir / f"{s['name']}.log"
        running = logdir / f"{s['name']}.running"
        start = time.time()
        running.write_text(json.dumps({"job": JOB, "task": TASK, "restart": RESTART, "start": start,
                                       "host": socket.gethostname()}))
        note, rc, cmd_text = "", 0, ""
        with logf.open("a") as fh:
            def log(msg: str, _fh=fh) -> None:
                print(f"[{time.strftime('%F %T')}] {msg}", file=_fh, flush=True)
                print(f"    {msg}", flush=True)

            fh.write(f"\n===== {time.strftime('%F %T')} job {JOB} task {TASK or '-'} restart {RESTART} "
                     f"host {socket.gethostname()} =====\n")
            print(f"--- {time.strftime('%F %T')} step {s['name']}   log: {logf}", flush=True)
            try:
                missing = [p for p in (H.fill(x, ctx) for x in s.get("needs", [])) if not matches(results, p)]
                if missing:
                    note, rc = "missing input: " + ", ".join(missing), 6
                    log(f"{note} (expected from an upstream stage under {results})")
                else:
                    env = {**os.environ, **{k: H.fill(str(v), ctx) for k, v in s.get("env", {}).items()}}
                    if s.get("cmd"):
                        cmd_text = H.fill(s["cmd"], ctx)
                        if s.get("launcher") == "srun":
                            cmd_text = f"srun --ntasks={os.environ.get('SLURM_NTASKS', '1')} bash -c {shlex.quote(cmd_text)}"
                        fh.write(f"$ {cmd_text}\n")
                        fh.flush()
                    for attempt in range(int(s.get("retries", 0)) + 1):
                        if attempt:
                            log(f"retry {attempt}/{s.get('retries')} after rc={rc}")
                        if s.get("builtin"):
                            rc = builtin(s["builtin"], ctx, log)
                        else:
                            rc = subprocess.run(["bash", "-o", "pipefail", "-c", cmd_text], stdout=fh, stderr=subprocess.STDOUT,
                                                env=env, cwd=str(H.REPO)).returncode
                        if rc == 0:
                            break
                    if rc == 0:
                        absent = [p for p in (H.fill(x, ctx) for x in s.get("produces", [])) if not matches(results, p)]
                        if absent:
                            note, rc = "declared output missing: " + ", ".join(absent), 7
                            log(note)
            except KeyError as e:
                note, rc = f"config error: {e}", 8
                log(note)
            except Exception as e:  # noqa: BLE001
                note, rc = f"{type(e).__name__}: {e}", 9
                log(note)
        end = time.time()
        running.unlink(missing_ok=True)
        with ledger.open("a") as lf:
            lf.write("\t".join([time.strftime("%F %T"), JOB, TASK, RESTART, st["name"], key, s["name"],
                                time.strftime("%F %T", time.localtime(start)), time.strftime("%F %T", time.localtime(end)),
                                f"{end - start:.0f}", str(rc), note.replace("\t", " ")[:300],
                                (cmd_text or f"[{s.get('builtin')}]").replace("\t", " ").replace("\n", " ")]) + "\n")
        print(f"    rc={rc}  {end - start:.0f}s", flush=True)
        if rc == 0:
            marker.write_text(time.strftime("%F %T") + "\n")
        else:
            failed.append(s["name"])
            if s.get("critical"):
                print(f"    critical step failed: stopping stage {st['name']}", flush=True)
                break
    print(f"=== stage {st['name']} cell {a.cell} finished; failed: {failed or 'none'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
