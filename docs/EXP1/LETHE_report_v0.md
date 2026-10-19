# LETHE: Model-driven Fault Localization with Teams of Forgetting LLM Agents

**EXP1 report — development, evaluation and discussion**

- Implementation brief: `docs/EXP1/LETHE_development_plan.md`
- Code: `lethe/` (approach) · `evaluation/` (SWE-bench Lite harness, baselines, stats)
- Results: `evaluation/results/main/` (run dirs, `baselines.json`, exported CSVs)

---

## 1. Overview

Fault localization (identifying where in a program a defect lives) is classically
done by spectrum-based fault localization (SBFL): run the tests, record which
statements the failing vs. passing tests touch, and score statements with
formulas such as Ochiai or DStar. SBFL is cheap and deterministic, but its
ranking quality hinges entirely on (a) having a test suite that discriminates
the fault and (b) formula heuristics that frequently rank large shared
infrastructure (exception handlers, serializers, registries) above the actual
defective line.

**LETHE** is a model-driven fault-localization system built from teams of LLM
agents that work over three artifacts:

1. a **program model** (functions, statements, call edges — extracted
   statically from the source),
2. a **trace model** (per-test statement coverage: who executed which
   statements), and
3. a **spectrum** (Ochiai/DStar/Tarantula scores computed from the trace
   model).

Each round, several agents concurrently inspect the evidence and submit a
ranked list of hypotheses (methods or statements, most suspicious first). A
**judge** then verifies each top hypothesis empirically: it generates
single-edit mutants inside the suspected element, applies each mutant in an
isolated copy of the program, and re-runs the failing tests (`fixprobe`) plus a
passing-test guard set. Mutants that make the failing tests pass while keeping
the guard tests green are counted as *support* for the hypothesis; mutants
that are ruled out accumulate *exoneration*. The judge scores the candidate
from this evidence (fix signal, progress, spectrum prior) and feeds the
structured feedback plus the accumulated facts back to the team.

The "LETHE" in the name is the forgetting mechanism: between rounds the team's
session context is periodically dropped (on a fixed schedule, or when the
score stops improving — the *plateau* policy) and the agents rebuild their
reasoning from the artifacts alone (model + trace + spectrum + judge facts).
The hypothesis, carried by the name, is that **periodic forgetting reduces
anchoring** — the tendency of an LLM to keep re-arguing the same top suspect
after weak counter-evidence — and keeps the team exploring different regions of
the program instead of over-committing to the first plausible-looking
explanation.

This report (1) describes the implemented system, (2) reports an evaluation on
7 bugs from SWE-bench Lite against SBFL and one-shot-LLM baselines and ablated
configurations, and (3) discusses what the data says about anchoring, the
value of empirical verification, and where the approach wins and loses.

---

## 2. The approach in detail

### 2.1 System components

| Component | Module | Role |
|---|---|---|
| Core types | `lethe/core.py` | Program model, statements, functions, shared `run_proc` |
| Spectrum | `lethe/spectrum.py` | Ochiai, DStar²/³, Tarantula, Barinel, OP2 + tie-aware ranking |
| Extraction | `lethe/extract.py` | Static AST extraction → program model (functions, statements, call edges) |
| Mutation | `lethe/mutation.py` | Single-edit mutant generation (cmp/binop/boolop/unary/const±1/del-stmt) |
| Judge | `lethe/judge.py` | `fixprobe` (mutant swap-in + test execution), patch validation, candidate scoring, fact cache |
| Agent I/O | `lethe/agent.py` | LLM client (OpenAI-compatible), tool definitions, prompts |
| Harness | `lethe/harness.py` | Rounds, concurrent agent turns, workspace materialization, forgetting, event log |
| Testing | `lethe/testing.py` | Uniform selected-test runner (pytest nodes / django `runtests.py` / `-k` names) |

### 2.2 Program model

Extracted statically (AST, no execution) from the project at the buggy commit:

- **functions**: id `path/to/file.py::Class.method`, file, name, class, line
  range, and a call list (resolved where resolvable);
- **statements**: one entry per executable line inside a function (plus
  module-level exec lines), id `path/to/file.py:LINE`, with the enclosing
  function, statement kind, and one-line source text;
- **call graph**: function → callee edges (intra-file first, then name-based
  cross-file resolution for `Class.method` forms).

Test files are excluded from the candidate space (they are the oracle, not the
suspicious surface). The model is the vocabulary of the protocol: agent
hypotheses, judge probes, and metric evaluation all refer to these ids, so
there is no ambiguity about "what" is localized.

### 2.3 Trace model and spectrum

For each selected test (all fail-to-pass tests, plus a sampled set of
pass-to-pass tests to keep cost bounded), the test is executed in an isolated
process under `coverage.py` and the covered statement set is recorded. The
fault-failure slice is the union of statements executed by the failing tests.
The standard SBFL counting problem is then scored per formula; the four
formulas above are kept (Ochiai, DStar², DStar³, Tarantula) and the ranking
uses a **tie-aware** policy (equal scores share a rank interval; the mean rank
is what downstream metrics see), avoiding arbitrary ordering of large ties —
common in Ochiai when many statements have the same 0.1–0.2 score band.

### 2.4 The agent loop (one round)

1. **Workspace materialization.** Each agent gets a private workspace with:
   the program under `repo/`, a `model/` view (functions, statements, call
   graph, suspicion top-k, per-test `covered_by` for the top suspects), the
   task (failing test list + defect report), and `.context/feedback/` holding
   the judge's structured feedback on the previous candidate and the team's
   accumulated facts (memory policy `candidate_plus_facts`).
2. **Concurrent turns.** In `parallel4`/`team4`/`lethe*` configurations, the
   four agents run their turns simultaneously (thread pool). In `team4` and
   `lethe*` configurations agents also see their peers' best candidate summary
   (`share_peers`). In `single` there is one agent.
3. **Tool loop.** Each agent may call: `read_file`, `list_dir`, `grep`
   (read-only over `repo/`), `query_model` (model queries: functions in file,
   statements of a function, callers/callees, covered_by test lists), and
   `write_candidate` + `write_patch` + `end_turn`. The agent may **not**
   execute code or modify the program — empirical verification belongs to the
   judge, and this separation is what makes the loop auditable.
4. **Candidate.** The agent submits a ranked list of up to 10 hypotheses with
   rationale (methods `path::Class.method` or statements `path:LINE`).

### 2.5 The judge (fixprobe)

For each top hypothesis (up to `top_k_judge`, default 4):

1. **Mutant generation.** Inside the suspected element, generate up to
   `max_mutants` single edits: comparison-operator flips (`>`↔`>=`, `==`↔`!=`,
   `in`↔`not in`, `is`↔`is not`), binary-operator swaps (`+`↔`-`, `*`↔`/`, …),
   boolean-operator flips (`and`↔`or`), unary flips (`not` drop, sign flip),
   numeric-constant ±1, and (for statement-level hypotheses) first-statement
   deletion. The mutant site is the whole file re-parsed via `ast.unparse`;
   compile-failing mutants are counted and excluded.
2. **Probe execution.** In the isolated judge copy (a git-initialized tree so
   every probe is a checked-in revert), for each mutant: swap the file in, run
   all fail-to-pass tests plus a sample of pass-to-pass guard tests. Outcome:
   - **fixed** — a single edit inside the element makes the failing tests pass
     without breaking the guard set → strong support that the true defect is
     inside (or immediately beside) this element;
   - **not_fixed** — no single edit fixed it → weakens (does not refute) the
     hypothesis; many faults require coordinated edits, so the judge reports
     this as *exoneration under the single-edit assumption*, not as proof of
     innocence.
3. **Patch validation.** If the agent also wrote a `patch.diff`, it is
   validated (applies cleanly, compiles) and run once through the same
   fail-to-pass + guard suite; a passing patch is decisive evidence.
4. **Scoring.** Candidate score = `w_fix·fix_rank + w_prog·progress +
   w_spec·spectrum_prior` (default weights 0.70/0.25/0.05), where `fix_rank`
   reflects how many top hypotheses were *fixed* vs *exonerated* and in what
   order, `progress` measures whether the candidate improves on the
   best-so-far, and `spectrum_prior` is a small regularizer keeping the score
   from drifting off the trace evidence.
5. **Facts.** Every probe outcome is stored in a per-bug fact store
   (`candidate_plus_facts` policy) keyed by element, with the *reason class*
   (fixed / not-fixed / truncated probe / truncated budget), so later rounds
   see what was already empirically tested and can build on it instead of
   re-litigating.

The judge is cached per (element, tests, mutant budget) so that different
configurations and replicates share probe cost instead of re-buying it.

### 2.6 Forgetting (the LETHE mechanism)

Session context is the agent's running conversation. The harness exposes the
full session to the agent by default; the *forget* policies define when that
session is cleared between rounds:

- `single` — one agent, no forgetting (control).
- `parallel4` — four independent agents, no peer sharing, no forgetting
  (parallelism alone).
- `team4` — four agents with peer sharing (team alone).
- `lethe` — the full system: four agents, peer sharing, fact-based memory,
  **fixed-schedule forgetting** every `forget_rounds` (4) rounds. The session
  is dropped; the artifact set (model, trace, spectrum, judge facts, best
  candidate) becomes the entire context for the next round.
- `lethe_candidate_only` — ablation: same forgetting, but the fact store is
  not carried across the forget (only the candidate survives). Isolates the
  value of judge-generated facts.
- `lethe_nofgt` — ablation: same as `lethe` but the fixed-schedule forget is
  replaced by "no forgetting after round 1" (facts kept, no scheduled amnesia).
  Isolates the value of the forgetting itself.
- `lethe_plateau` — ablation: forgetting is triggered by a *plateau* in the
  best candidate score (no improvement for 3 consecutive rounds) instead of a
  fixed schedule; `forget_rounds=8` acts as a safety ceiling. Tests whether
  *conditional* forgetting (forget when stuck) beats *scheduled* forgetting.

The design intent: an LLM team that keeps its full conversation tends to
**anchor** — after a plausible-but-wrong top suspect the team spends its
budget re-arguing that suspect, and empirical counter-evidence (judge
exoneration) is absorbed as "the judge is probably wrong" rather than as a
reason to move. Forcing a clean slate, with the durable facts preserved,
re-grounds the team on the artifacts and makes exploration of new regions the
path of least resistance. The ablations (c) separate "having facts" from
"forgetting the session" and (d) separate "scheduled" from "conditional"
forgetting.

### 2.7 Metric and budget conventions

- Each run has a **proposals budget** (default 16 candidates across the team);
  a round ends when all agents in it have used their turns or the budget is
  exhausted. The best-scored candidate is the run's answer.
- Ranking completion for metric evaluation: the hypotheses (methods expanded
  to their statements in spectrum order) are placed first, then the spectrum
  tail follows, so every method/statement has a total order and all of
  Top-N, first-rank, MRR, AP, EXAM are well-defined.
- Metrics: **Top-1/3/5/10** (faulty element in the top N), **first rank**
  (position of the best faulty element), **MRR/RR**, **AP**, **EXAM**
  (first-fault depth / |elements|, lower better) — at **both** statement and
  method granularity. Tie-aware variants use mean rank within the tie band.

---

## 3. Evaluated systems and baselines

| id | system |
|---|---|
| `single` | one agent, no team, no facts, no forgetting |
| `parallel4` | four independent agents |
| `team4` | four agents + peer sharing + judge facts (no forgetting) |
| `lethe` | four agents + peer sharing + judge facts + **scheduled forgetting** |
| `lethe_candidate_only` | `lethe` minus carried-over judge facts |
| `lethe_nofgt` | `lethe` minus the scheduled forget |
| `lethe_plateau` | `lethe` with plateau-triggered forgetting |
| `sbfl-ochiai` | Ochiai spectrum ranking (no LLM, no judge) |
| `sbfl-dstar2` / `sbfl-dstar3` / `sbfl-tarantula` | other SBFL formulas |
| `oneshot` | one tool-less LLM call over problem statement + top spectrum suspects |

---

## 4. Experimental setup

### 4.1 Benchmark and selection

SWE-bench **Lite** (300 task instances across 12 repositories; test patch +
gold patch + fail-to-pass / pass-to-pass test sets). Seven instances were
selected for this experiment by the criteria in the plan (representative
repos, tractable test suites on local hardware, single-fault gold patches,
working local environments). Each instance is run in a dedicated git-
initialized checkout of the buggy commit with a private virtualenv, so no run
can read or write another's code.

*(Full list of selected instances and per-instance setup notes: see
`evaluation/data/selected.md` / `evaluation/data/precomputed/<instance>.json`.)*

### 4.2 Model and serving

- **LLM:** Qwen (27B class) served locally via Ollama behind its
  OpenAI-compatible endpoint; all agent turns and one-shot baseline calls use
  the same model so the compared systems differ only in harness structure.
- **Hardware:** 20 CPU cores, 121 GB RAM, 1 GPU (used by the local model
  server). Judge test-execution and mutant probes run on CPU.

### 4.3 Cost and determinism controls

- Judge probes cached per (element, tests, opset, budget) → configurations
  and replicates share probe cost; cost is reported per run, not amortized
  away.
- Agent temperature 0; tool-loop caps (max tool calls, max tokens, LLM
  timeout) enforced per turn.
- Two replicates per (bug × config) with fresh agent sessions; the judge
  cache is shared (that is shared *evidence*, not shared *decisions*).
- Event logs (`events.jsonl`) record every LLM call (tokens, latency, finish
  reason), tool call, candidate, probe outcome, and forget event — all
  exports to CSV in `evaluation/results/main/csv/`.

### 4.4 Statistics

Paired comparisons per bug (and aggregated) with McNemar (Top-N) and
Wilcoxon-signed-rank (ranks), plus Vargha–Delaney A₁₂ for stochastic
superiority; bootstrap CIs for mean differences. Holm correction across
config-vs-config families. See `evaluation/stats.py`.

---

## 5. Results

_(populated from `evaluation/results/main/` and `evaluation/results/main/csv/`)_

### 5.1 Main table — statement-level localization

| bug | sbfl-ochiai first-rank | single fr | parallel4 fr | team4 fr | lethe fr | best per bug |
|---|---|---|---|---|---|---|
| _populated after runs complete_ | | | | | | |

### 5.2 Main table — method-level localization

_populated after runs complete_

### 5.3 Top-5 hit rate (per configuration, fraction of bugs where the faulty
element is in the top 5, statement level)

_populated after runs complete_

### 5.4 Ablation deltas (lethe vs team4 / candidate_only / nofgt / plateau)

_populated after runs complete_

### 5.5 Judge behavior (probes, fixed vs exonerated, truncated probes)

_populated after runs complete_

### 5.6 Forgetting traces (when did forgetting fire, score before/after)

_populated after runs complete_

### 5.7 Cost (LLM tokens, latency, probe count, wall time) per configuration

_populated after runs complete_

---

## 6. Discussion

_(draft — will be grounded in the final numbers)_

### 6.1 What the data is expected to show

1. **SBFL is unreliable at single-fault granularity.** On these instances the
   Ochiai top of the ranking is dominated by exception-handling and
   serialization infrastructure touched by nearly every test; the true
   defective statement often sits outside the top-10. This is the motivation
   for an agent layer at all.
2. **The judge changes the shape of the answer, not just its ranking.**
   A hypothesis that is *fixed* by a single in-element edit is near-certain
   local evidence; the fact store turns "exonerated" regions into negative
   knowledge the team can plan around.
3. **Forgetting should buy exploration.** The prediction: scheduled forgetting
   (`lethe`) improves over `team4` on the harder instances (high SBFL
   first-rank) where anchoring on a shared top suspect is most damaging, and
   the effect should be larger for `lethe` > `lethe_nofgt` (facts kept, no
   forget) and `lethe` > `lethe_candidate_only` (facts dropped at forget).
4. **Plateau vs scheduled.** Plateau-triggered forgetting (`lethe_plateau`)
   should win on instances where the team was on the right track (forgetting
   too early loses momentum) and lose where the team was confidently wrong.

### 6.2 Threats to validity

- **Small n.** Seven instances is a demonstration, not a population claim;
  per-instance effects must be read individually.
- **Single local model.** The anchoring/forgetting interaction is likely
  model-dependent; results transfer to the *harness design*, not to a model
  family.
- **Single-edit mutation coverage.** The judge's `fixprobe` is a
  *sound-but-incomplete* test: faults requiring coordinated multi-line edits
   (new parameters, new branches) will be reported as "not fixed" and thus
   exonerated. The fact store labels this reason class explicitly and the
   report will quantify it.
- **Test-suite quality.** SWE-bench fail-to-pass sets are small (often 1–3
   tests); coverage evidence is accordingly thin for some instances.

### 6.3 What LETHE is and isn't claiming

LETHE does not claim to *find the fix* (though a valid `patch.diff` is
captured when the agent produces one) — it claims to *rank where the fault
is* more reliably than SBFL or a single LLM call, with an auditable evidence
trail (every probe decision is logged, reproducible from the fact store and
event log). The forgetting result is a claim about *team process design* for
LLM-based verification-heavy tasks, not about LLM capability.
