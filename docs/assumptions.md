# Assumptions and verification log

| Date | Item | Current assumption | Verification required |
|---|---|---|---|
| 2026-10-18 | Primary production spec | `docs/EXP1/LETHE_development_plan.md` wins over the earlier worked example where they differ. | Confirm with project owner before freezing the pilot. |
| 2026-10-18 | Full LETHE memory | Carry the best candidate plus judge-produced facts; candidate-only is the paper-faithful ablation. | Freeze after pilot. |
| 2026-10-18 | Python benchmark | SWE-bench Lite remains an additional path; BugsInPy is the primary Python benchmark requested by the plan. | Verify BugsInPy toolchain versions on the target cluster. |
| 2026-10-18 | Java spectra | Provider and versions are not selected. | Compare GZoltar and FLACOCO against target Defects4J Java versions. |
| 2026-10-18 | Cluster | Slurm site, account, GPU names and wheelhouse availability are unknown. | Query the target login node before smoke submission. |
| 2026-10-18 | Model backend | OpenAI-compatible local GPU serving and Anthropic are both planned; the experimental model IDs and prices are not frozen. | Select two model tiers and record dated prices before pilot. |
