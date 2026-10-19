# Start a private Ollama server only for reportable local-model cells.
if [ "${LETHE_LLM_MODE:-mock}" = "mock" ] || [ "${model_tier:-mock}" = "mock" ]; then
    export LETHE_LLM_BASE="mock://scripted"
    log_info "mock model tier: no inference service required"
    return 0
fi
if [ "${LETHE_LLM_MODE}" != "ollama" ] || [ -z "${OLLAMA_PREFIX:-}" ]; then
    log_error "real model cell requires LETHE_LLM_MODE=ollama and OLLAMA_PREFIX"
    return 1
fi
_port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
export OLLAMA_HOST="127.0.0.1:$_port" OLLAMA_MODELS OLLAMA_KEEP_ALIVE=24h OLLAMA_FLASH_ATTENTION=1
export OLLAMA_NUM_PARALLEL OLLAMA_MAX_LOADED_MODELS=1
export LETHE_LLM_BASE="http://$OLLAMA_HOST/v1"
"$OLLAMA_PREFIX/bin/ollama" serve > "$LOG_ROOT/env/ollama_${JOB_TAG}_r${SLURM_RESTART_COUNT:-0}.log" 2>&1 &
SERVICE_PIDS+=($!)
curl -sf --retry 90 --retry-delay 2 --retry-connrefused --retry-all-errors -m 5 "http://$OLLAMA_HOST/api/version" >/dev/null \
    || { log_error "inference server did not become ready"; return 1; }
log_info "inference server ready at $LETHE_LLM_BASE"
