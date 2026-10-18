# LETHE

**Localizing Errors Through Hypothesis Erasure.** Forget the story. Keep the model.

LETHE is a research prototype that combines **model-driven engineering (MDE)**
with **forgetting**, an idea from *Scaling Discovery Requires Forgetting*
(Tan et al.), to localize software faults.

LLM debugging agents tend to anchor on their first suspect and keep digging
there as their session grows. LETHE counters this by representing the program
as models and remembering only **verified facts** between rounds:

1. **Extract** a program model from the code (components, functions, statements, calls)
2. **Trace** the tests into a trace model and **slice** it to the failing execution
3. **Annotate** the program with a suspicion model (spectrum scores + judge-verified facts)
4. **Probe**: a team of agents proposes hypotheses; a mutation-based judge
   checks each one by trying a single-edit repair at that location
5. **Forget**: discard every agent's session, restart fresh agents from the
   suspicion model alone

## What's in this repo

- A small buggy "shop" program with one seeded off-by-one fault
- The MDE pipeline: metamodel, code-to-model extraction, trace model, slicing,
  suspicion model, model differencing, model-to-text report
- A mutation-based judge (the "fix-probe")
- Four search configurations on the same budget: single agent, parallel agents,
  collaborating team, and team with forgetting
- `run_demo.py`, which reproduces every result in the accompanying write-up

## Example result

| Configuration | Fault line found at | Rounds |
|---|---|---|
| 1 agent, one continuous session | proposal 13 | 13 |
| 4 parallel agents | not found in 24 | 6 |
| Team, no forgetting | proposal 15 | 4 |
| Team + forgetting (LETHE) | proposal 11 | 3 |

## Important caveat

The agents in this repository are **scripted stand-ins, not real LLMs**. They
encode two behaviours reported in the paper: attraction to hints in the failure
message, and anchoring on regions already explored. The models, traces, spectrum
scores and judge probes are real and executed. The results illustrate the
mechanism and are **not** evidence about real LLM behaviour. With anchoring
switched off, forgetting gives no benefit.

## Run it

Requires Python 3.9+ and no third-party packages.

    python run_demo.py

## Status

Research prototype. Next steps: plug in real LLM agents, evaluate on Defects4J
and BugsInPy, and study the judge's failure modes (multiple faults, missing
mutation sites).
