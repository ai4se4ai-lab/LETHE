#!/bin/bash
# hpc/slurm/common.sh — shared functions of every job (sourced by stage.sh).
#
# A job: loads modules, builds a venv in $SLURM_TMPDIR from the wheelhouse, verifies it holds a GPU,
# samples the GPU every GPU_MONITOR_SECS, sets up multi-node variables, runs optional service hooks
# (e.g. an inference server), runs hpc/scripts/run_stage.py for its stage and cell, and requeues itself
# (same job ID, so dependencies hold) shortly before its time limit. Finished work is skipped on restart.

: "${HPC_DIR:?HPC_DIR not set (submit with hpc/slurm/submit.py)}"
: "${RUN_ID:?RUN_ID not set (submit with hpc/slurm/submit.py)}"
REPO_DIR="$(cd "$HPC_DIR/.." && pwd)"
LOG_ROOT="$HPC_DIR/logs/$RUN_ID"
CONF_DIR="$LOG_ROOT/config"; [ -f "$CONF_DIR/cluster.env" ] || CONF_DIR="$HPC_DIR/config"

set -a
# shellcheck disable=SC1091
source "$CONF_DIR/cluster.env"
set +a

# SLURM_RESTART_COUNT is not always updated after `scontrol requeue`; ask Slurm for the real count.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    _r=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -o 'Restarts=[0-9]*' | head -1 | cut -d= -f2)
    [ -n "$_r" ] && export SLURM_RESTART_COUNT="$_r"
fi

RUN_DIR="$WORK/runs/$RUN_ID"
CELL_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_TAG="${STAGE:-job}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
REQUEUE_ID="${SLURM_ARRAY_JOB_ID:+${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"; REQUEUE_ID="${REQUEUE_ID:-${SLURM_JOB_ID:-}}"
STAGE_PID=""; GPU_MON_PID=""; SERVICE_PIDS=()

log_info()  { echo "[$(date '+%F %T')] [INFO]  $*"; }
log_warn()  { echo "[$(date '+%F %T')] [WARN]  $*"; }
log_error() { echo "[$(date '+%F %T')] [ERROR] $*" >&2; }

# ── exit / requeue ───────────────────────────────────────────────────────────
_cleanup() {
    [ -n "$STAGE_PID" ] && kill -TERM "$STAGE_PID" 2>/dev/null && wait "$STAGE_PID" 2>/dev/null
    [ -n "$GPU_MON_PID" ] && kill "$GPU_MON_PID" 2>/dev/null
    local p; for p in "${SERVICE_PIDS[@]}"; do kill "$p" 2>/dev/null; done
    STAGE_PID=""; GPU_MON_PID=""; SERVICE_PIDS=()
}
_ending() {  # _ending <status>
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "${SLURM_JOB_ID:-0}" "${STAGE:-?}" "$CELL_ID" \
        "${SLURM_RESTART_COUNT:-0}" "$1" "$(hostname)" >> "$LOG_ROOT/jobs_finished.tsv"
}
_on_exit() {
    local code=$?
    _cleanup
    if [ "$code" -eq 0 ]; then _ending ok; else _ending "FAILED($code)"; log_error "job $JOB_TAG exit code $code"; fi
    log_info "job $JOB_TAG end"
}
_on_timelimit() {   # Slurm sends USR1 to the batch shell REQUEUE_SIGNAL_SECS before the limit (--signal=B:USR1@N)
    local n="${SLURM_RESTART_COUNT:-0}"
    log_warn "time limit approaching (restart $n of max $MAX_REQUEUES)"
    _cleanup
    if [ "$n" -lt "${MAX_REQUEUES:-20}" ]; then
        _ending requeued
        log_info "requeueing $REQUEUE_ID; finished steps and records are skipped on restart"
        scontrol requeue "$REQUEUE_ID"
        trap - EXIT; exit 0
    fi
    log_error "maximum number of requeues reached; work left unfinished"; exit 99
}
setup_traps() { trap _on_exit EXIT; trap _on_timelimit USR1; }

# ── header ───────────────────────────────────────────────────────────────────
print_job_header() {
    mkdir -p "$LOG_ROOT"/{slurm,steps,env,gpu} "$RUN_DIR"/{results,state}
    log_info "================================================================"
    log_info "run $RUN_ID   stage $STAGE   cell $CELL_ID   job ${SLURM_JOB_ID:-n/a}   restart ${SLURM_RESTART_COUNT:-0}"
    log_info "nodes ${SLURM_JOB_NODELIST:-$(hostname)} (${SLURM_NNODES:-1})   cpus ${SLURM_CPUS_PER_TASK:-?}   gpus ${CUDA_VISIBLE_DEVICES:-none}"
    log_info "repo $REPO_DIR @ $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo '?')"
    log_info "results $RUN_DIR/results   logs $LOG_ROOT"
    log_info "================================================================"
}

# ── GPU guarantee ────────────────────────────────────────────────────────────
require_gpu() {
    if [ "${CPU_ONLY:-0}" = "1" ]; then log_warn "stage marked cpu_only: no GPU check"; return 0; fi
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] || ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
        log_error "no GPU visible on $(hostname) (CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}'): refusing to compute"
        return 1
    fi
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | sed 's/^/[gpu] /'
}
start_gpu_monitor() {
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv -l "${GPU_MONITOR_SECS:-60}" > "$LOG_ROOT/gpu/gpu_${JOB_TAG}_r${SLURM_RESTART_COUNT:-0}.csv" 2>/dev/null &
    GPU_MON_PID=$!
}

# ── Python (Alliance practice: venv in $SLURM_TMPDIR, packages from the wheelhouse) ─
setup_python() {
    module purge 2>/dev/null
    # shellcheck disable=SC2086
    [ -n "${MODULES:-}" ] && module load $MODULES
    local venv="${SLURM_TMPDIR:-$RUN_DIR/tmp}/venv" req="$CONF_DIR/requirements.txt"
    [ -f "$req" ] || req="$HPC_DIR/config/requirements.txt"
    if [ ! -x "$venv/bin/python" ]; then
        virtualenv --no-download -q "$venv" || return 1
        # shellcheck disable=SC1091
        source "$venv/bin/activate"
        pip install --no-index -q --upgrade pip
        local fl=(); [ -n "${EXTRA_WHEELS:-}" ] && fl=(--find-links "$EXTRA_WHEELS")
        if [ -f "$req" ] && ! pip install --no-index -q "${fl[@]}" -r "$req"; then
            log_error "pip install failed: a package is missing from the wheelhouse (put its wheel in EXTRA_WHEELS)"; return 1
        fi
    else
        # shellcheck disable=SC1091
        source "$venv/bin/activate"
    fi
    pip freeze > "$LOG_ROOT/env/pip_freeze_${JOB_TAG}.txt" 2>/dev/null
    local p="$REPO_DIR" d; for d in ${SRC_DIRS:-}; do p="$p:$REPO_DIR/$d"; done
    export PYTHONPATH="$p${PYTHONPATH:+:$PYTHONPATH}"
    export RESULTS="$RUN_DIR/results" RUN_DIR LOG_ROOT HPC_DIR REPO_DIR RUN_ID STAGE CELL_ID
    export TMPDIR="${SLURM_TMPDIR:-$RUN_DIR/tmp}" MPLBACKEND=Agg PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
    mkdir -p "$TMPDIR"
    { echo "# $JOB_TAG $(date)"; module -t list 2>&1; echo; env | grep -E '^(SLURM_|CUDA_|PYTHONPATH|RESULTS|RUN_|TMPDIR)' | sort; } \
        > "$LOG_ROOT/env/env_${JOB_TAG}.txt"
}

# ── multi-node (only when the stage asks for nodes > 1) ──────────────────────
setup_distributed() {
    export MASTER_ADDR; MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" | head -1)
    export MASTER_PORT=$(( 20000 + ${SLURM_JOB_ID:-0} % 20000 ))
    export WORLD_SIZE="${SLURM_NTASKS:-1}" NNODES="${SLURM_NNODES:-1}"
    [ "${SLURM_NNODES:-1}" -gt 1 ] && log_info "distributed: $NNODES nodes, $WORLD_SIZE tasks, master $MASTER_ADDR:$MASTER_PORT"
}

# ── optional service hooks: hpc/slurm/hooks/<stage>.sh or hooks/all.sh ──────
# A hook may start a server in the background (append its PID to SERVICE_PIDS) and must wait for
# readiness without `sleep` loops (e.g. curl --retry). It is stopped on exit and before requeue.
run_hooks() {
    local h
    for h in "$HPC_DIR/slurm/hooks/all.sh" "$HPC_DIR/slurm/hooks/${STAGE}.sh"; do
        if [ -f "$h" ]; then log_info "hook $h"; source "$h" || { log_error "hook $h failed"; return 1; }; fi
    done
}

# ── run the stage in the background so USR1 is handled immediately ───────────
run_stage() {
    python "$HPC_DIR/scripts/run_stage.py" --stage "$STAGE" --cell "$CELL_ID" "$@" &
    STAGE_PID=$!
    wait "$STAGE_PID"; local rc=$?
    STAGE_PID=""
    return $rc
}
