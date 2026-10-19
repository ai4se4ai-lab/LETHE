#!/bin/bash
# hpc/slurm/stage.sh — the ONE batch script of every stage. Resources, array range, dependencies and
# requeue flags are given on the sbatch command line by hpc/slurm/submit.py (from experiments.json),
# so this file holds only the job body. Env from submit.py: HPC_DIR, RUN_ID, STAGE.
#SBATCH --open-mode=append
set -o pipefail
# shellcheck disable=SC1091
source "$HPC_DIR/slurm/common.sh"
export CPU_ONLY
CPU_ONLY=$(python3 - "$LOG_ROOT/config/experiments.resolved.json" "$STAGE" <<'EOF'
import json, sys
s = next(x for x in json.load(open(sys.argv[1]))["stages"] if x["name"] == sys.argv[2])
print(1 if s.get("cpu_only") else 0)
EOF
)
setup_traps
print_job_header
require_gpu || exit 5
setup_python || exit 4
setup_distributed
start_gpu_monitor
run_hooks || exit 6
run_stage
