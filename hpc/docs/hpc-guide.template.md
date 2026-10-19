# How <PROJECT> runs on HPC (Slurm, GPU nodes)

<!-- Filled in by the use-hpc skill from the project's paper/design document and source code.
     Keep every claim checkable: cite document labels, file paths and stage/step names. -->

Contents: 1 Big picture · 2 Source code on HPC · 3 From the document's design to jobs · 4 Coverage matrix ·
5 Anatomy of a job · 6 Slurm and GPU techniques · 7 Auditing · 8 Reproducibility · 9 Results · 10 Operating ·
11 Limitations

## 1. The big picture
<!-- The study in one paragraph: RQs, design size (factors multiplied out), estimated GPU-hours.
     ASCII diagram of the job graph: stages, arrays (cells), dependency types, GPUs per stage. -->

## 2. How the source code is used on HPC
<!-- Table: repository path -> role in the study (imported / run as module / not used).
     Table: approach component (document section) -> source file -> stage(s) that exercise it.
     How outputs are redirected into $RESULTS (env vars / flags). What is hidden from the system under test. -->

## 3. From the document's experimental design to jobs
<!-- Table: design element (subjects, factors, seeds, roles, fixed params, sizes) -> where configured.
     Table: stage -> GPUs -> steps -> research question.
     Why the dependency graph looks like this (gates, afterany, aftercorr). Validity-preserving details. -->

## 4. Coverage matrix: every claim -> step -> output -> artefact
| # | Claim / measurement (doc ref) | Stage -> step | Raw output (under `$RESULTS`) | Collected as | Document artefact | Decision rule checked in |
|---|---|---|---|---|---|---|

<!-- Then: approach component -> experiments that exercise it. Then: explicit list of what is NOT covered and why. -->

## 5. Anatomy of a job
<!-- stage.sh -> common.sh (GPU check, venv, monitor, hooks, requeue) -> run_stage.py (contracts, ledger, markers). -->

## 6. Slurm and GPU techniques used
<!-- Table: technique -> where -> what it buys (arrays, dependencies, requeue, MIG, sizing, smoke, no polling ...).
     How to reach the GPU nodes (salloc, srun --jobid --overlap). Multi-node connectivity check if used. -->

## 7. Auditing every phase
<!-- Levels: intent, code identity, environment, gates, jobs, steps, data, resources, results — with file paths. -->

## 8. Reproducibility and integrity controls

## 9. How results are collected
<!-- $RESULTS tree; collected/ tree; mapping of collected files to the document's tables and figures. -->

## 10. Operating the study
<!-- selftest -> dry run -> smoke -> full -> audit -> resubmit -> collect. -->

## 11. Limitations and open points
<!-- Synthetic or placeholder data in the document, untested aspects, internet needs, quota, sandboxing, TODOs. -->
