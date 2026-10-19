"""The LETHE harness (plan §5.7).

One harness, every configuration is a reducer setting:
  - `single`        1 agent, whole budget, session continues
  - `parallel4`     4 agents, budget split, private search states, no peers
  - `team4`         4 agents, budget split, peers visible, no forgetting
  - `lethe`         4 agents, peers visible, fixed rounds with forgetting
  - `lethe_nofgt`   4 agents, peers visible, same round schedule but sessions
                    CONTINUE across boundaries (the RQ6 "continue" branch)

What survives a forget is decided by the memory policy:
  - `candidate_only`        best candidate (paper-faithful)
  - `candidate_plus_facts`  best candidate + judge-verified facts
The judge is the only producer of facts; agents never write them.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import threading
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .core import ProgramModel, run_proc, SCHEMA_VERSION
from .judge import BugEnv, JudgeResult, judge_candidate
from . import agent as ag

# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    name: str
    n_agents: int
    share_peers: bool
    forget_policy: str = "never"            # never | fixed
    forget_rounds: int = 4
    memory: str = "candidate_only"          # candidate_only | candidate_plus_facts
    budget: int = 16                        # total proposals shared by the team
    model: str = ag.DEFAULT_MODEL
    base: str = ag.DEFAULT_BASE
    temperature: float = 0.0
    max_tool_calls: int = 14
    max_tokens: int = 8192
    top_k_judge: int = 4
    max_mutants: int = 16
    weights: dict = field(default_factory=lambda: {"fix": 0.70, "progress": 0.25, "spectrum": 0.05})
    seed: int = 1
    llm_timeout_s: int = 600


def trim_session(messages: list[dict], max_chars: int = 150000) -> list[dict]:
    """Prune a continuing session: keep the first user turn (task context) and
    as many most-recent turns as fit. Stale middle turns are dropped; their
    results remain in .context/feedback so nothing is lost from the agent's
    reach. This is a context-window guard, not a scoring mechanism."""
    def size(ms: list[dict]) -> int:
        return sum(len(str(m.get("content") or "")) + 400 for m in ms)
    if not messages or size(messages) <= max_chars:
        return messages
    groups: list[list[dict]] = []
    i = 0
    while i < len(messages):
        if messages[i].get("role") == "user":
            j = i
            while j + 1 < len(messages) and messages[j + 1].get("role") != "user":
                j += 1
            groups.append(messages[i:j + 1])
            i = j + 1
        else:
            groups.append([messages[i]])
            i += 1
    if len(groups) <= 1:
        return messages
    out = list(groups[0])
    kept: list[list[dict]] = []
    used = size(out)
    for g in reversed(groups[1:]):
        if used + size(g) <= max_chars:
            kept.append(g)
            used += size(g)
        else:
            break
    kept.reverse()
    return out + [m for g in kept for m in g]


PRESET_CONFIGS: dict[str, RunConfig] = {
    "single": RunConfig(name="single", n_agents=1, share_peers=False,
                        forget_policy="never", budget=16, max_tool_calls=14),
    "parallel4": RunConfig(name="parallel4", n_agents=4, share_peers=False,
                           forget_policy="never", budget=16),
    "team4": RunConfig(name="team4", n_agents=4, share_peers=True,
                       forget_policy="never", budget=16),
    "lethe": RunConfig(name="lethe", n_agents=4, share_peers=True,
                       forget_policy="fixed", forget_rounds=4,
                       memory="candidate_plus_facts", budget=16),
    "lethe_nofgt": RunConfig(name="lethe_nofgt", n_agents=4, share_peers=True,
                             forget_policy="fixed", forget_rounds=4,
                             memory="candidate_plus_facts", budget=16),
    "lethe_candidate_only": RunConfig(name="lethe_candidate_only", n_agents=4,
                                      share_peers=True, forget_policy="fixed",
                                      forget_rounds=4, memory="candidate_only", budget=16),
    "lethe_plateau": RunConfig(name="lethe_plateau", n_agents=4, share_peers=True,
                               forget_policy="plateau", forget_rounds=8,
                               memory="candidate_plus_facts", budget=16),
}


# ---------------------------------------------------------------------------
# Event log
# ---------------------------------------------------------------------------

class EventLog:
    def __init__(self, run_dir: Path, run_id: str):
        self.path = run_dir / "events.jsonl"
        self.run_id = run_id
        self._lock = threading.Lock()
        self._seq = 0

    def log(self, event: str, **payload: Any) -> int:
        with self._lock:
            self._seq += 1
            rec = {"schema_version": SCHEMA_VERSION, "ts": time.time(),
                   "run_id": self.run_id, "seq": self._seq, "event": event}
            rec.update(payload)
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return self._seq


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    agent_id: int
    candidates: list[dict] = field(default_factory=list)   # judged candidates
    facts: dict[str, str] = field(default_factory=dict)    # element -> fact kind
    session_id: int = 0
    messages: list[dict] = field(default_factory=list)
    best: dict | None = None

    def save(self, d: Path):
        d.mkdir(parents=True, exist_ok=True)
        (d / "state.json").write_text(json.dumps({
            "agent_id": self.agent_id, "candidates": self.candidates,
            "facts": self.facts, "session_id": self.session_id,
            "messages": self.messages}, indent=1))


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class Harness:
    def __init__(self, bug_env: BugEnv, pm: ProgramModel, run_dir: Path,
                 cfg: RunConfig, model_view: dict, task_text: str,
                 failing_tests: list[str]):
        self.env = bug_env
        self.pm = pm
        self.run_dir = run_dir
        self.cfg = cfg
        self.model_view = model_view
        self.task_text = task_text
        self.failing_tests = failing_tests
        self.events = EventLog(run_dir, f"{bug_env.checkout_hash[:12]}:{cfg.name}")
        self.judge_lock = threading.Lock()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._cand_counter = 0
        self._cand_lock = threading.Lock()
        self.facts: dict[str, dict] = {}

    # -- helpers -----------------------------------------------------------

    def _next_cand_id(self) -> str:
        with self._cand_lock:
            self._cand_counter += 1
            return f"c-{self._cand_counter:04d}"

    def _normalize_element(self, el: str) -> str:
        el = el.strip()
        if el.startswith("repo/"):
            el = el[len("repo/"):]
        return el

    def _validate_candidate(self, raw: dict) -> tuple[dict, list[dict]]:
        hyps = sorted(raw.get("hypotheses", []), key=lambda h: h.get("rank", 0))[:10]
        out, dropped, seen = [], [], set()
        for i, h in enumerate(hyps, 1):
            el = self._normalize_element(str(h.get("element", "")))
            kind, obj = self.pm.resolve_element(el)
            if kind == "unknown":
                dropped.append({"element": h.get("element", ""), "reason": "unknown_element"})
                continue
            if el in seen:
                dropped.append({"element": el, "reason": "duplicate_element"})
                continue
            seen.add(el)
            gran = "statement" if kind == "statement" else "method"
            out.append({"rank": i, "element": el, "granularity": gran,
                        "rationale": str(h.get("rationale", ""))[:500]})
        # fix ranks after drops (ranks must be 1..n without gaps)
        for i, h in enumerate(out, 1):
            h["rank"] = i
        return {"hypotheses": out, "notes": raw.get("notes", "")}, dropped

    # -- workspace materialization (the reducer) ----------------------------

    def _materialize_ws(self, a: AgentState, round_no: int, peers: dict[int, dict]) -> Path:
        ws = self.run_dir / "ws" / f"agent{a.agent_id}" / f"round{round_no}"
        if ws.exists():
            shutil.rmtree(ws)
        ctx = ws / ".context"
        ctx.mkdir(parents=True)
        (ws / "TASK.md").write_text(self.task_text)
        # repo: symlink to the shared buggy checkout (read-only by policy)
        (ws / "repo").symlink_to(self.env.checkout)
        (ws / "model").mkdir()
        (ws / "model" / "functions.json").write_text(json.dumps(self.model_view["functions"], indent=1))
        (ws / "model" / "statements.json").write_text(json.dumps(self.model_view["statements"][:400], indent=1))
        (ws / "model" / "call_graph.json").write_text(json.dumps(self.model_view["call_graph"], indent=1))
        # failing tests
        (ctx / "failing_tests.md").write_text(
            "Failing tests (current evidence of the fault):\n"
            + "\n".join(f"- `{t}`" for t in self.failing_tests) + "\n\n"
            + "Assertion/traceback messages are summarized in TASK.md.\n")
        # suspicion (spectrum top)
        top = self.model_view["suspicion_top"]
        (ctx / "suspicion.json").write_text(json.dumps(top, indent=1))
        # own feedback (policy-limited: latest 4 candidates)
        fb = ctx / "feedback"
        fb.mkdir()
        recent = a.candidates[-4:]
        owned_best = a.best
        for c in recent:
            (fb / f"{c['id']}.json").write_text(json.dumps(
                {"candidate": c["hypotheses"], "feedback": c.get("feedback", {})}, indent=1))
        # peers
        if self.cfg.share_peers:
            for j, pc in peers.items():
                d = ctx / "peers" / f"branch{j}"
                d.mkdir(parents=True)
                (d / "candidate.json").write_text(json.dumps(pc["hypotheses"], indent=1))
                (d / "feedback.json").write_text(json.dumps(pc.get("feedback", {}), indent=1))
                if pc.get("patch"):
                    (d / "patch.diff").write_text(pc["patch"])
        # facts (memory policy decides what survives forgetting)
        if self.cfg.memory == "candidate_plus_facts" and self.cfg.forget_policy in ("fixed", "plateau"):
            (ctx / "facts.json").write_text(json.dumps(
                {"note": "judge-verified facts (confirmed = a single-edit repair at "
                         "this element made the failing tests pass without regressions; "
                         "exonerated = no single-edit repair found there; "
                         "inconclusive = probe truncated or no mutation sites)",
                 "facts": self.facts}, indent=1))
        # seed candidate (best of own candidates; after a forget: carried best)
        seed = a.best or {}
        (ws / "candidate.json").write_text(json.dumps(
            {"hypotheses": seed.get("hypotheses", []), "notes": "seeded"}, indent=1))
        (ws / "notes").mkdir(exist_ok=True)
        return ws

    # -- peer state ---------------------------------------------------------

    def _peers_best(self, states: dict[int, AgentState], me: int) -> dict[int, dict]:
        out = {}
        for j, s in states.items():
            if j == me or not s.candidates:
                continue
            best = max(s.candidates, key=lambda c: c.get("score", -1))
            out[j] = best
        return out

    # -- agent turn -----------------------------------------------------------

    def _peer_lines(self, peers: dict[int, dict]) -> list[str]:
        return [f"Branch {j}: best {p['id']} at {p.get('score', '?')}; see .context/peers/branch{j}/"
                for j, p in sorted(peers.items())]

    def run_turn(self, a: AgentState, round_no: int, turn_in_round: int,
                 states: dict[int, AgentState]) -> dict:
        cfg = self.cfg
        ws = self._materialize_ws(a, round_no, self._peers_best(states, a.agent_id) if cfg.share_peers else {})
        first = a.session_id == 0 or not a.messages
        if first:
            msg = ag.first_turn(cfg.name, round_no, a.best,
                                self._peer_lines(self._peers_best(states, a.agent_id) if cfg.share_peers else {}),
                                10)
        else:
            msg = ag.later_turn(round_no, a.best, self._peer_lines(self._peers_best(states, a.agent_id) if cfg.share_peers else {}))

        messages = [m for m in a.messages]
        a.messages = trim_session(a.messages)
        messages.append({"role": "user", "content": msg})
        toolbox = ag.ToolBox(ws=ws, model_view=self._ws_model_view())

        cand_file = ws / "candidate.json"
        patch_file = ws / "patch.diff"
        ended = False
        tool_calls_used = 0
        nudged = 0
        t0 = time.time()
        while not ended and tool_calls_used < cfg.max_tool_calls:
            resp = ag.llm_chat(cfg.base, cfg.model, messages, tools=ag.TOOLS,
                               temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                               timeout=cfg.llm_timeout_s)
            usage = resp.get("usage", {})
            self.events.log("llm_call", agent_id=a.agent_id, round=round_no,
                            model=resp.get("model"), finish=resp.get("finish"),
                            tokens_in=usage.get("prompt_tokens", 0),
                            tokens_out=usage.get("completion_tokens", 0),
                            latency_s=round(resp["latency"], 2),
                            cached=usage.get("prompt_tokens_details", {}).get("cached_tokens", 0))
            m = resp["message"]
            tcs = resp.get("tool_calls") or []
            if m.get("content"):
                messages.append({"role": "assistant", "content": m["content"],
                                 "tool_calls": tcs or None})
            elif tcs:
                messages.append({"role": "assistant", "content": None, "tool_calls": tcs})
            else:
                # no tools, no content (thinking truncated / empty reply):
                # nudge up to twice, then end the turn
                if nudged < 2:
                    nudged += 1
                    messages.append({"role": "assistant", "content": ""})
                    messages.append({"role": "user",
                                    "content": "(no action yet) Continue: inspect evidence or call write_candidate, then end_turn."})
                    continue
                messages.append({"role": "assistant", "content": ""})
                ended = True
                continue
            acted = False
            for tc in tcs:
                name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"] or "{}")
                ok, out = self._exec_tool(toolbox, name, args)
                tool_calls_used += 1
                self.events.log("tool_call", agent_id=a.agent_id, tool=name,
                                args_hash=hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest()[:12],
                                ok=ok, out_bytes=len(out), round=round_no)
                messages.append({"role": "tool", "tool_call_id": tc.get("id", "x"),
                                 "name": name, "content": out[:4000]})
                acted = True
                if name == "end_turn":
                    ended = True
            if not tcs and not m.get("content"):
                ended = True
        a.messages = messages

        # collect candidate (possibly empty if the agent never wrote one)
        if cand_file.exists():
            try:
                raw = json.loads(cand_file.read_text())
            except json.JSONDecodeError:
                raw = {"hypotheses": []}
        else:
            raw = {"hypotheses": []}
        cand, dropped = self._validate_candidate(raw)
        patch = patch_file.read_text() if patch_file.exists() else None
        cand_id = self._next_cand_id()
        empty = (a.best is not None and json.dumps(cand["hypotheses"], sort_keys=True)
                 == json.dumps(a.best.get("hypotheses", []), sort_keys=True))
        self.events.log("candidate_committed", agent_id=a.agent_id, candidate_id=cand_id,
                        n_hypotheses=len(cand["hypotheses"]), empty_turn=empty,
                        dropped=dropped, tool_calls=tool_calls_used,
                        seconds=round(time.time() - t0, 1))

        # judge (serialized per bug)
        t1 = time.time()
        with self.judge_lock:
            jr = judge_candidate(self.env, self.pm, cand_id, cand["hypotheses"], patch,
                                 weights=cfg.weights, top_k=cfg.top_k_judge,
                                 max_mutants=cfg.max_mutants, events=self.events)
        jr.cost["judge_seconds"] = round(time.time() - t1, 2)
        a.candidates.append({"id": cand_id, "hypotheses": cand["hypotheses"],
                             "patch": patch, "feedback": jr.model_dump(),
                             "score": jr.score, "round": round_no})
        if a.best is None or jr.score > (a.best.get("score") or -1):
            a.best = a.candidates[-1]

        # facts (judge-produced only)
        for ph in jr.per_hypothesis:
            el = ph["element"]
            if ph["probe"] == "fixed":
                self.facts[el] = {"kind": "confirmed", "candidate": cand_id, "agent": a.agent_id,
                                  "fixing_edits": ph.get("fixing_edits", [])}
            elif ph["probe"] == "inconclusive":
                self.facts.setdefault(el, {"kind": "inconclusive", "candidate": cand_id})
            else:
                self.facts[el] = {"kind": "exonerated", "candidate": cand_id,
                                  "note": "single-edit assumption"}
            self.events.log("probe", element=el, probe=ph["probe"],
                            fixed=ph["probe"] == "fixed", truncated=ph.get("truncated", False),
                            mutants_run=ph.get("mutants_run", 0), candidate=cand_id,
                            agent_id=a.agent_id, cache_hit=ph.get("cache_hit", False))
        self.events.log("judge_end", candidate=cand_id, score=jr.score,
                        components=jr.components, patch_valid=jr.patch_valid,
                        cost=jr.cost)
        # search state (git-backed, plan §5.7.2)
        self._commit_search_state(a, round_no, cand_id)
        a.save(self.run_dir / "agents" / f"agent{a.agent_id}")
        return {"cand_id": cand_id, "score": jr.score, "empty": empty,
                "tool_calls": tool_calls_used, "n_hyp": len(cand["hypotheses"])}

    def _exec_tool(self, tb: ag.ToolBox, name: str, args: dict) -> tuple[bool, str]:
        try:
            if name == "read_file":
                return True, tb.read_file(args.get("path", ""), args.get("start_line"), args.get("end_line"))
            if name == "list_dir":
                return True, tb.list_dir(args.get("path", "."))
            if name == "grep":
                return True, tb.grep(args.get("pattern", ""), args.get("path"))
            if name == "query_model":
                return True, tb.query_model(args.get("kind", ""), args.get("arg", ""))
            if name == "write_candidate":
                return True, tb.write_candidate(args.get("hypotheses", []), args.get("notes"))
            if name == "write_patch":
                return True, tb.write_patch(args.get("diff", ""))
            if name == "end_turn":
                return True, "turn ended"
            return False, f"unknown tool {name}"
        except PermissionError as e:
            return False, f"refused: {e}"
        except Exception as e:
            return False, f"tool error: {type(e).__name__}: {e}"

    def _ws_model_view(self) -> dict:
        # agent-facing view: element ids with the repo/ prefix as seen in the ws
        def pref(x: str) -> str:
            return x if x.startswith("repo/") else "repo/" + x
        return {
            "functions": [{**f, "id": pref(f["id"])} for f in self.model_view["functions"]],
            "statements": [{**s, "id": pref(s["id"]),
                            "func_id": pref(s["func_id"]) if s.get("func_id") else None}
                           for s in self.model_view["statements"]],
            "call_graph": {pref(k): [pref(c) for c in v] for k, v in self.model_view["call_graph"].items()},
            "covered_by": {pref(k): [pref(t) for t in v] for k, v in self.model_view.get("covered_by", {}).items()},
            "suspicion_top": [{"id": pref(h["id"]), **{k: v for k, v in h.items() if k != "id"}}
                              for h in self.model_view["suspicion_top"][:30]],
        }

    # -- search state (git-backed, append-only) ------------------------------

    def _commit_search_state(self, a: AgentState, round_no: int, cand_id: str) -> None:
        d = self.run_dir / "agents" / f"agent{a.agent_id}" / "search_state"
        d.mkdir(parents=True, exist_ok=True)
        st = a.candidates[-1]
        (d / f"{cand_id}.json").write_text(json.dumps(st, indent=1))
        env = {**os.environ,
               "GIT_AUTHOR_NAME": f"agent{a.agent_id}", "GIT_AUTHOR_EMAIL": "l@l",
               "GIT_COMMITTER_NAME": f"agent{a.agent_id}", "GIT_COMMITTER_EMAIL": "l@l"}
        run_proc(["git", "init", "-q"], cwd=d, env=env)
        run_proc(["git", "add", "-A"], cwd=d, env=env)
        run_proc(["git", "commit", "-q", "-m", f"score={st['score']} cand={cand_id}"], cwd=d, env=env)

    # -- forgetting ----------------------------------------------------------

    def should_forget(self, round_no: int, states: dict[int, AgentState]) -> bool:
        cfg = self.cfg
        if cfg.forget_policy == "fixed":
            return round_no < cfg.forget_rounds
        if cfg.forget_policy == "plateau":
            # adaptive (plan §5.7.6 extension): W=2 rounds, eps=1e-9. Forget when
            # the team best stopped improving over the last W rounds and we are
            # not on the final budget chunk.
            bests = self._board_history["bests"]
            if len(bests) < 2:
                return False
            if round_no >= cfg.forget_rounds:
                return False
            stagnant = (bests[-1] - bests[-2]) < 1e-9
            early = round_no * cfg.n_agents <= cfg.budget // 2
            return stagnant and early
        return False

    def forget(self, round_no: int, states: dict[int, AgentState], discard_sessions: bool = True) -> None:
        best = None
        for s in states.values():
            for c in s.candidates:
                if best is None or c["score"] > best["score"]:
                    best = c
        carried_facts = dict(self.facts) if self.cfg.memory == "candidate_plus_facts" else {}
        discarded_sessions = sum(len(s.messages) for s in states.values())
        discarded_commits = sum(len(s.candidates) for s in states.values())
        for a in states.values():
            if discard_sessions:
                a.session_id = 0
                a.messages = []
                # archive old search state (kept on disk, not visible to the agent)
                old = self.run_dir / "agents" / f"agent{a.agent_id}" / "search_state"
                arch = self.run_dir / "agents" / f"agent{a.agent_id}" / f"history_r{round_no}"
                if old.exists() and not arch.exists():
                    shutil.move(str(old), str(arch))
                if self.cfg.memory == "candidate_only":
                    a.facts = {}
        if best is not None:
            for a in states.values():
                seeded = {"id": best["id"] + f"@a{a.agent_id}", "hypotheses": best["hypotheses"],
                          "patch": best.get("patch"), "score": best.get("score", 0.0),
                          "feedback": {"note": "carried across forget (best team candidate)"}}
                a.candidates = [seeded]
                a.best = seeded
        self.facts = carried_facts
        self.events.log("forget", round=round_no, carried_candidate=best["id"] if best else None,
                        carried_facts=len(carried_facts),
                        discarded_sessions=discarded_sessions,
                        discarded_commits=discarded_commits)

    # -- main loop ------------------------------------------------------------

    def run(self) -> dict:
        cfg = self.cfg
        self._board_history = {"bests": []}
        states: dict[int, AgentState] = {i: AgentState(agent_id=i) for i in range(cfg.n_agents)}
        per_agent = cfg.budget // cfg.n_agents
        rounds = cfg.forget_rounds if cfg.forget_policy in ("fixed", "plateau") else max(1, per_agent)
        if cfg.forget_policy in ("fixed", "plateau") and rounds * cfg.n_agents != cfg.budget:
            rounds = cfg.budget // cfg.n_agents
        import json as _json
        self.events.log("run_start", config=_json.dumps(cfg.__dict__, default=str),
                        model=cfg.model, budget=cfg.budget, n_agents=cfg.n_agents,
                        forget_policy=cfg.forget_policy, rounds=rounds,
                        memory=cfg.memory, share_peers=cfg.share_peers)
        proposals_used = 0
        import concurrent.futures as cf
        for r in range(1, rounds + 1):
            self.events.log("round_start", round=r, best_team=
                            max([c["score"] for s in states.values() for c in s.candidates], default=None))
            def work(i: int) -> dict:
                return self.run_turn(states[i], r, i % per_agent + 1, states)
            with cf.ThreadPoolExecutor(max_workers=cfg.n_agents) as ex:
                results = list(ex.map(work, range(cfg.n_agents)))
            for i, res in zip(range(cfg.n_agents), results):
                self.events.log("turn_end", agent_id=i, round=r, **res)
            proposals_used += cfg.n_agents
            team_best = max([c["score"] for s in states.values() for c in s.candidates], default=0.0)
            self._board_history["bests"].append(team_best)
            self.events.log("budget_update", proposals_used=proposals_used,
                            remaining=max(0, cfg.budget - proposals_used), team_best=team_best)
            self.events.log("round_end", round=r, team_best=team_best,
                            proposals_used=proposals_used)
            if self.should_forget(r, states) and cfg.forget_policy in ("fixed", "plateau"):
                if cfg.name.endswith("nofgt"):
                    # continue branch (RQ6): same state, no discard
                    self.events.log("boundary_continue", round=r)
                else:
                    self.forget(r, states, discard_sessions=True)
        # final
        best = None
        for s in states.values():
            for c in s.candidates:
                if best is None or c["score"] > best["score"]:
                    best = c
        self.events.log("run_end", status="completed", proposals_used=proposals_used,
                        final_candidate=best["id"] if best else None,
                        final_score=best.get("score", 0.0) if best else None)
        out = self.run_dir / "result.json"
        out.write_text(_json.dumps({"best": best, "facts": self.facts}, indent=1, default=str))
        for a in states.values():
            a.save(self.run_dir / "agents" / f"agent{a.agent_id}")
        return {"best": best, "facts": self.facts, "proposals": proposals_used}
