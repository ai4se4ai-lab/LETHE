# Service hooks

`stage.sh` sources `hooks/all.sh` (every stage) and then `hooks/<stage>.sh` (that stage only), if they exist, after
the venv is ready and before the steps run. Use a hook to start something the steps talk to on the **same GPU
node**, such as an inference server (Ollama, vLLM, TGI), a database, or a Ray head.

Rules:

- Start the service in the background, append its PID to `SERVICE_PIDS`, and export what the steps need (URL, port).
  `common.sh` stops it on exit and before a requeue.
- Bind to `127.0.0.1` on a free port. Never use a fixed port, because other jobs may share the node.
- Wait for readiness with a retrying client, for example `curl --retry 90 --retry-delay 2 --retry-connrefused`.
  Never use `sleep` loops in jobs.
- Write the service's log to `$LOG_ROOT/env/` so the audit trail is complete.

See `example_inference_server.sh.example` for an Ollama example. Rename it to `<stage>.sh` or `all.sh` to activate it.
