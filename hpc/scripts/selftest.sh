#!/bin/bash
# hpc/scripts/selftest.sh — static checks of the hpc/ layer (seconds, login-node safe, nothing is submitted):
#   shell syntax, Python compilation, experiments.json validity and GPU policy, placeholder filling of every
#   step of every cell, and the sbatch commands of the full and the smoke graph (dry runs).
set -uo pipefail
HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fail=0
ok()  { echo "  ok    $*"; }
bad() { echo "  FAIL  $*"; fail=1; }

echo "== shell syntax"
for f in "$HPC_DIR"/slurm/*.sh "$HPC_DIR"/slurm/hooks/*.sh "$HPC_DIR"/status.sh "$HPC_DIR"/scripts/*.sh; do
    [ -f "$f" ] || continue
    bash -n "$f" && ok "$(basename "$f")" || bad "$f"
done
echo "== python compile"
for f in "$HPC_DIR"/scripts/*.py "$HPC_DIR"/slurm/*.py; do
    python3 -m py_compile "$f" && ok "$(basename "$f")" || bad "$f"
done
echo "== experiments.json (graph, GPU policy, placeholders of every cell)"
python3 - "$HPC_DIR" <<'EOF' || fail=1
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / "scripts"))
import hpclib as H
rc = 0
for smoke in (False, True):
    exp = H.resolve(H.load_experiments(H.HPC / "config" / "experiments.json"), smoke)
    errs = H.validate(exp)
    for e in errs:
        print("  FAIL ", e); rc = 1
    c = H.cluster(None)
    n = 0
    for st in exp["stages"]:
        for i, cell in enumerate(st["cells"]):
            ctx = H.context(exp, c, "selftest", st["name"], cell, i)
            for s in st["steps"]:
                for field in ["cmd"] + [f"needs:{k}" for k in range(len(s.get("needs", [])))] + [f"produces:{k}" for k in range(len(s.get("produces", [])))]:
                    try:
                        if field == "cmd":
                            if s.get("cmd"): H.fill(s["cmd"], ctx)
                        else:
                            kind, k = field.split(":")
                            H.fill(s[kind][int(k)], ctx)
                        n += 1
                    except KeyError as e:
                        print(f"  FAIL  {st['name']}[{i}] {s['name']} {field}: {e}"); rc = 1
            if st.get("cpu_only"):
                print(f"  note  stage {st['name']} is cpu_only (no GPU requested)")
    cells = sum(len(s["cells"]) for s in exp["stages"])
    print(f"  {'ok   ' if rc == 0 else 'FAIL '} {'smoke' if smoke else 'full '} graph: {len(exp['stages'])} stages, {cells} cells, {n} placeholders filled")
sys.exit(rc)
EOF
echo "== sbatch dry runs"
python3 "$HPC_DIR/slurm/submit.py" --dry-run > /dev/null 2>&1 && ok "full graph" || { bad "full graph"; python3 "$HPC_DIR/slurm/submit.py" --dry-run; }
python3 "$HPC_DIR/slurm/submit.py" --dry-run --smoke > /dev/null 2>&1 && ok "smoke graph" || bad "smoke graph"
echo
[ $fail -eq 0 ] && echo "SELFTEST OK — next: python3 hpc/slurm/submit.py --smoke" || echo "SELFTEST FAILED"
exit $fail
