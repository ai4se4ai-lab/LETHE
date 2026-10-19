#!/bin/bash
# Live audit of a run (login-node safe: one squeue + one sacct call per refresh).
#   bash hpc/status.sh                       newest run, one snapshot
#   bash hpc/status.sh <RUN_ID> --watch 120  refresh every 120 s (minimum 60); stops when nothing is queued/running
#   bash hpc/status.sh <RUN_ID> --only failed | --steps
HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$HPC_DIR/scripts/audit.py" "$@"
