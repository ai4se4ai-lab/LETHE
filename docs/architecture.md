# LETHE architecture

LETHE separates the benchmark-facing evaluation layer from the localization tool. Code under
`lethe/` never imports `evaluation/`. The tool discovers a program model and trace model, exposes
sanitized model views to agents, validates ranked candidates, and lets a repair-based judge produce
facts. A reducer determines which candidate history and session each agent sees. Forgetting starts a
new session and search state while carrying only the configured candidate/fact memory.

Ground truth and fixed patches remain under `evaluation/`; they are used only after runs complete to
score rankings. Every persisted interface is a versioned Pydantic schema. Every expensive action is
represented in `events.jsonl` and every run has one immutable configuration snapshot.
