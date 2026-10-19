#!/usr/bin/env python3
"""hpc/slurm/submit.py — submit the experiment graph of hpc/config/experiments.json as Slurm jobs.

Runs on the LOGIN node and does only light work: validate the graph, freeze the configuration,
write the run manifest, and call sbatch (one call per stage). All computation happens in the jobs.

    python3 hpc/slurm/submit.py --dry-run              # validate + print the sbatch commands, submit nothing
    python3 hpc/slurm/submit.py --smoke                # tiny end-to-end run of EVERY stage (smoke overrides)
    python3 hpc/slurm/submit.py                        # full study (new RUN_ID)
    python3 hpc/slurm/submit.py --run-id R --only main,collect   # resubmit stages of run R (finished work is skipped)
    python3 hpc/slurm/submit.py --run-id R --only collect        # intermediate results at any time
    python3 hpc/slurm/submit.py ... --watch 120        # then stay in the live audit table (refresh >= 60 s)

Graph semantics (per stage in experiments.json):
  matrix / matrix_from   -> one job ARRAY, one task per cell (`--array=0-N%throttle`)
  depends: afterok       -> start when the upstream stage (all its tasks) succeeded; a failure cancels this stage
           afterany      -> start when the upstream stage ended in any state (analysis of partial data)
           aftercorr     -> task i starts as soon as upstream task i succeeded (cell-wise pipelining)
  resources              -> gpus | gpus_per_node, nodes, ntasks_per_node, cpus, mem, time, requeue, constraint, partition
Dependencies on a stage that is not part of this submission are dropped (its outputs must already exist:
run_stage.py checks every step's `needs` before running it).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

HPC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HPC / "scripts"))
import hpclib as H  # noqa: E402


def sbatch_cmd(st: dict, exp: dict, c: dict, run_id: str, deps: list[str], args) -> list[str]:
    r = st.get("resources", {})
    lr = H.log_root(run_id)
    n = len(st["cells"])
    cmd = ["sbatch", "--parsable", f"--account={c.get('ACCOUNT', '')}", f"--job-name={exp.get('project', 'hpc')}_{st['name']}",
           f"--time={r.get('time', '01:00:00')}", f"--cpus-per-task={r.get('cpus', 4)}", f"--mem={r.get('mem', '16G')}",
           "--open-mode=append", f"--export=ALL,HPC_DIR={HPC},RUN_ID={run_id},STAGE={st['name']}"]
    if r.get("gpus_per_node"):
        cmd.append(f"--gpus-per-node={r['gpus_per_node']}")
    elif r.get("gpus"):
        cmd.append(f"--gpus={r['gpus']}")
    if int(r.get("nodes", 1)) > 1:
        cmd += [f"--nodes={r['nodes']}", f"--ntasks-per-node={r.get('ntasks_per_node', 1)}"]
    for k in ("constraint", "partition"):
        if r.get(k):
            cmd.append(f"--{k}={r[k]}")
    if r.get("requeue"):
        cmd += ["--requeue", f"--signal=B:USR1@{c.get('REQUEUE_SIGNAL_SECS', '900')}"]
    if c.get("MAIL_TYPE", "FAIL") not in ("", "NONE"):
        cmd.append(f"--mail-type={c.get('MAIL_TYPE', 'FAIL')}")
    if n > 1:
        thr = st.get("array_throttle")
        cmd.append(f"--array=0-{n - 1}" + (f"%{thr}" if thr else ""))
        cmd += [f"--output={lr}/slurm/%x_%A_%a.out", f"--error={lr}/slurm/%x_%A_%a.err"]
    else:
        cmd += [f"--output={lr}/slurm/%x_%j.out", f"--error={lr}/slurm/%x_%j.err"]
    if deps:
        cmd += [f"--dependency={','.join(deps)}", "--kill-on-invalid-dep=yes"]
    cmd += [str(x) for x in st.get("sbatch_extra", [])]
    cmd.append(str(HPC / "slurm" / "stage.sh"))
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--only", default="", help="comma list of stages to submit (default: all)")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end test of every stage")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--account", default="")
    ap.add_argument("--watch", type=int, nargs="?", const=60, default=0, help="live audit after submitting (>= 60 s)")
    a = ap.parse_args()

    if os.environ.get("SLURM_JOB_ID") and not a.dry_run:
        print("note: submitting from inside a job; jobs inherit nothing from it except the environment", file=sys.stderr)
    run_id = a.run_id or time.strftime(("am-smoke-%Y%m%d-%H%M%S" if a.smoke else "am-%Y%m%d-%H%M%S"))
    lr = H.log_root(run_id)
    frozen = (lr / "config" / "experiments.resolved.json").exists()
    if frozen:   # continuing a run: keep its frozen config (smoke flag included)
        exp = json.loads((lr / "config" / "experiments.resolved.json").read_text())
        c = H.source_env(lr / "config" / "cluster.env")
    else:
        exp = H.resolve(H.load_experiments(HPC / "config" / "experiments.json"), a.smoke)
        c = H.source_env(HPC / "config" / "cluster.env")
    if a.account:
        c["ACCOUNT"] = a.account
    errs = H.validate(exp)
    if errs:
        print("experiments.json is not valid:\n  - " + "\n  - ".join(errs), file=sys.stderr)
        return 2
    if "CHANGEME" in c.get("ACCOUNT", "CHANGEME"):
        print("set ACCOUNT in hpc/config/cluster.env (or pass --account)", file=sys.stderr)
        if not a.dry_run:
            return 2
    if not a.dry_run and not shutil.which("sbatch"):
        print("sbatch not found: run this on the cluster's login node", file=sys.stderr)
        return 2

    only = [x for x in a.only.split(",") if x]
    for o in only:
        H.stage(exp, o)
    rd = H.run_dir(run_id, c)
    if not a.dry_run:
        for d in ("config", "slurm", "steps", "env", "gpu"):
            (lr / d).mkdir(parents=True, exist_ok=True)
        for d in ("results", "state"):
            (rd / d).mkdir(parents=True, exist_ok=True)
        if not frozen:
            shutil.copy(HPC / "config" / "cluster.env", lr / "config" / "cluster.env")
            if a.account:
                with open(lr / "config" / "cluster.env", "a") as f:
                    f.write(f'\nACCOUNT="{a.account}"   # --account at submission\n')
            shutil.copy(HPC / "config" / "experiments.json", lr / "config" / "experiments.json")
            for extra in ("requirements.txt",):
                if (HPC / "config" / extra).exists():
                    shutil.copy(HPC / "config" / extra, lr / "config" / extra)
            (lr / "config" / "experiments.resolved.json").write_text(json.dumps(exp, indent=1))
        subprocess.run([sys.executable, str(HPC / "scripts" / "manifest.py"), run_id], check=False,
                       stdout=subprocess.DEVNULL)

    print("=" * 78)
    print(f"{exp.get('project', '')}  RUN_ID {run_id}{'  (SMOKE)' if exp.get('is_smoke') else ''}{'  [dry run]' if a.dry_run else ''}")
    print(f"account {c.get('ACCOUNT')}   results {rd / 'results'}   logs {lr}")
    print("=" * 78)
    ids: dict[str, str] = {}
    for st in exp["stages"]:
        if only and st["name"] not in only:
            continue
        deps = []
        for d in st.get("depends", []):
            if d["stage"] in ids:
                deps.append(f"{d.get('type', 'afterok')}:{ids[d['stage']]}")
            else:
                print(f"  note: {st['name']} depends on {d['stage']}, which is not submitted now; its outputs must exist")
        cmd = sbatch_cmd(st, exp, c, run_id, deps, a)
        n = len(st["cells"])
        if a.dry_run:
            jid = f"<{st['name']}>"
            print(f"  {shlex.join(cmd)}")
        else:
            out = subprocess.run(cmd, capture_output=True, text=True)
            if out.returncode != 0:
                print(f"sbatch failed for stage {st['name']}: {out.stderr.strip()}", file=sys.stderr)
                return 3
            jid = out.stdout.strip().split(";")[0]
            with open(lr / "jobs.tsv", "a") as f:
                f.write(f"{jid}\t{st['name']}\t{n}\t{','.join(deps)}\t{time.strftime('%F %T')}\n")
        gpu = st.get("resources", {}).get("gpus") or st.get("resources", {}).get("gpus_per_node") or "CPU-ONLY"
        print(f"  {st['name']:<22} job {jid:<12} cells {n:<4} gpu {gpu:<16} {('after ' + ','.join(deps)) if deps else ''}")
        ids[st["name"]] = jid
    if a.dry_run:
        print("\n(dry run: nothing submitted)")
        return 0
    print(f"\nLive audit:   bash {HPC}/status.sh {run_id}            (add --watch 120 to refresh; never poll faster than 60 s)")
    print(f"Step ledger:  {lr}/steps.tsv      Job logs: {lr}/slurm/")
    if a.watch:
        os.execvp(sys.executable, [sys.executable, str(HPC / "scripts" / "audit.py"), run_id, "--watch", str(max(a.watch, 60))])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
