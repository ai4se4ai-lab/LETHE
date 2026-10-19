"""Smoke test: full harness loop with a scripted (deterministic) LLM.

The mock reads .context/suspicion.json each turn and proposes the next
unseen suspects, so the run is fully deterministic and exercises:
workspace materialization, tool loop, candidate validation, real judge
(mutation probes on a real SWE-bench bug), forgetting, search state,
events and metrics.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lethe.agent as ag                     # noqa: E402
from lethe import extract                    # noqa: E402
from lethe.harness import Harness, PRESET_CONFIGS  # noqa: E402
from evaluation import swebench as swe       # noqa: E402
from evaluation import run_grid as rg        # noqa: E402

BUG = "sympy__sympy-24909"
BUG = sys.argv[1] if len(sys.argv) > 1 else BUG
doc = json.loads((ROOT / "evaluation/data/precomputed" / f"{BUG}.json").read_text())
bugs = {b.instance_id: b for b in swe.load_dataset()}
b = bugs[BUG]

# deterministic suspect queue: spectrum order
QUEUE = [h["id"] for h in sorted(doc["suspicion_top"], key=lambda h: h["rank"])]
cursor = {"i": 0}
run_dirs: dict[str, Path] = {}


def mock_chat(base, model, messages, tools=None, temperature=0.0, max_tokens=2048, timeout=600):
    last = messages[-1]
    if last.get("role") == "user":
        ws = _latest_ws()
        top = [h["id"] for h in (json.loads((ws / ".context" / "suspicion.json").read_text()) if (ws / ".context" / "suspicion.json").exists() else doc["suspicion_top"])[:1][:1]]
        # propose the next chunk of the deterministic queue
        n = cursor["i"]
        chunk = []
        seen = set()
        while len(chunk) < 4 and n + len(chunk) < len(QUEUE):
            el = QUEUE[n + len(chunk)]
            if el not in seen:
                seen.add(el)
                chunk.append(el)
        if not chunk:
            chunk = top or [QUEUE[0]]
        cursor["i"] += len(chunk)
        def gran(el: str) -> str:
            tail = el.split("::")[-1]
            return "statement" if ("::" not in el and ":" in tail and tail.rsplit(":", 1)[-1].isdigit()) else "method"
        hyps = [{"rank": i + 1, "element": "repo/" + el, "granularity": gran(el), "rationale": "spectrum"}
                for i, el in enumerate(chunk)]
        return {"message": {"content": None},
                "tool_calls": [{"id": f"c0", "function": {"name": "write_candidate",
                                                          "arguments": json.dumps({"hypotheses": hyps})}}],
                "usage": {"prompt_tokens": 500, "completion_tokens": 200},
                "latency": 0.01, "model": "mock", "finish": "tool_calls"}
    if last.get("role") == "tool" and last.get("name") == "write_candidate":
        return {"message": {"content": None},
                "tool_calls": [{"id": "c1", "function": {"name": "end_turn", "arguments": json.dumps({"summary": "proposed next suspects"})}}],
                "usage": {"prompt_tokens": 500, "completion_tokens": 50},
                "latency": 0.01, "model": "mock", "finish": "tool_calls"}
    return {"message": {"content": "done"}, "tool_calls": [],
            "usage": {"prompt_tokens": 100, "completion_tokens": 5},
            "latency": 0.01, "model": "mock", "finish": "stop"}


def _latest_ws() -> Path:
    best = None
    for rd in list(run_dirs.values()):
        for d in rd.glob("ws/agent*/round*"):
            if best is None or str(d) > str(best):
                best = d
    return best if best is not None else Path(tempfile.gettempdir())


def main():
    tmp = Path(tempfile.mkdtemp(prefix="lethe_smoke_"))
    cfg = dataclasses.replace(PRESET_CONFIGS["lethe"], budget=8, max_tool_calls=6)
    run_dirs["lethe"] = tmp / "lethe"
    ag.llm_chat = mock_chat
    pm = extract.extract(swe.checkout_dir(b)); pm.index()
    env = rg.build_bug_env(b, doc)
    mv = {"functions": doc["model"], "statements": doc["statements"],
          "call_graph": doc["call_graph"], "covered_by": {}, "suspicion_top": doc["suspicion_top"]}
    task = ag.task_md(b.problem_statement, b.fail_to_pass)
    h = Harness(env, pm, run_dirs["lethe"], cfg, mv, task, b.fail_to_pass)
    out = h.run()
    print("SMOKE DONE", json.dumps({"score": out["best"]["score"] if out.get("best") else None,
                                   "n_cands": len(out["best"]["hypotheses"]) if out.get("best") else 0}, indent=1))
    m = rg.cell_metrics(b, doc, run_dirs["lethe"])
    print("metrics:", json.dumps({"stmt": m["stmt"], "method": m["method"]}, indent=1))
    ev = (run_dirs["lethe"] / "events.jsonl").read_text().splitlines()
    kinds = {}
    for l in ev:
        e = json.loads(l)["event"]
        kinds[e] = kinds.get(e, 0) + 1
    print("events:", kinds)


if __name__ == "__main__":
    main()
