# Running the experiments on the cluster (Slurm, GPU nodes)

This directory runs every experiment of the project as a graph of Slurm jobs on GPU nodes. It also collects the results as CSVs, tables, figures and logs, and gives a live tabular audit while the jobs run. The login node only validates, freezes the configuration and submits. All computation happens in jobs on GPU nodes.

The design, the coverage of the paper's experiments, and the reasoning behind it are in [docs/hpc-guide.md](docs/hpc-guide.md).

## Quick start

```bash
bash hpc/scripts/selftest.sh                    # static checks (seconds; nothing is submitted)
python3 hpc/slurm/submit.py --dry-run           # show the job graph and sbatch commands
python3 hpc/slurm/submit.py --smoke --watch 60  # tiny end-to-end run of EVERY stage on GPU nodes, live audit
python3 hpc/slurm/submit.py                     # the full study (prints RUN_ID and job IDs)
bash hpc/status.sh                              # live audit of the newest run (add --watch 120 to refresh)
```

## Layout

| Path | Purpose |
|---|---|
| `config/experiments.json` | **The experiment graph**: stages, matrices, dependencies, resources, steps, smoke sizes, results spec |
| `config/cluster.env` | account, modules, storage, GPU policy, requeue settings, variables |
| `config/requirements.txt` | Python packages for the jobs (installed from the wheelhouse into `$SLURM_TMPDIR`) |
| `slurm/submit.py` | validate → freeze config → manifest → `sbatch` per stage (arrays and dependencies) |
| `slurm/stage.sh`, `slurm/common.sh` | job body: GPU check, venv, GPU monitor, multi-node variables, hooks, requeue at the time limit |
| `slurm/hooks/` | optional per-stage services (for example an inference server) |
| `scripts/run_stage.py` | runs the steps of a stage cell: contracts (`needs` / `produces`), ledger, logs, resume markers, retries |
| `scripts/audit.py` (`status.sh`) | live table: stage × cell, state, node, time, steps, progress, failures |
| `scripts/collect_results.py` | CSVs, summary tables (md/tex), figures, pipeline records → `logs/<RUN_ID>/results/` |
| `scripts/manifest.py` | code, config and git identity of a run |
| `scripts/selftest.sh` | login-safe static checks |
| `logs/<RUN_ID>/` | everything about one run (not in git) |

## Common operations

```bash
python3 hpc/slurm/submit.py --run-id <RUN_ID> --only main,collect   # resubmit stages; finished work is skipped
python3 hpc/slurm/submit.py --run-id <RUN_ID> --only collect        # refresh the collected results at any time
python3 hpc/scripts/run_stage.py --stage main --list                # show every cell's commands
bash hpc/status.sh <RUN_ID> --only failed                           # just the problems, with log excerpts
srun --jobid <JOBID> --overlap --pty watch -n 30 nvidia-smi         # look at a running job's GPU
```

To redo a step, delete its marker `$WORK/runs/<RUN_ID>/state/<stage>/<cell>/<step>.done` (and its output if the command appends), then resubmit the stage.
