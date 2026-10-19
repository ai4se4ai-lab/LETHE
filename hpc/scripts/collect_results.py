#!/usr/bin/env python3
"""hpc/scripts/collect_results.py — turn a run's raw outputs into CSVs, tables, figures and logs, and copy them
from scratch to persistent storage. Runs inside the final stage's job (built-in step "collect"); safe to rerun.

    python hpc/scripts/collect_results.py <RUN_ID>

Driven by the "results" block of experiments.json:
  tables     [{name, glob}]                 every matching .csv/.jsonl/.json under $RESULTS, concatenated
                                            (nested JSON flattened as a.b; columns _source added)
  summaries  [{name, table, group_by, metrics: {column: mean|median|sum|count|min|max|std|rate}}]
  plots      [{name, summary|table, kind: bar|line|scatter|box, x, y, hue?, title?}]   (needs matplotlib)
  copy       globs under $RESULTS copied as they are;  archive: directories packed as <dir>.tar.gz (many small files)

Output: $RESULTS/collected/{csv,tables,figures,pipeline}/ + summary.md, then everything collected is copied to
RESULTS_DEST (default hpc/logs/<RUN_ID>/results/). Standard library only, matplotlib optional.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import shutil
import statistics
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hpclib as H  # noqa: E402


def flatten(d, prefix=""):
    out = {}
    for k, v in (d.items() if isinstance(d, dict) else []):
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, list):
            out[key] = json.dumps(v)[:500]
        else:
            out[key] = v
    return out


def read_any(p: Path) -> list[dict]:
    try:
        if p.suffix == ".jsonl":
            rows = []
            for line in p.read_text(errors="replace").splitlines():
                try:
                    rows.append(flatten(json.loads(line)))
                except json.JSONDecodeError:
                    pass
            return rows
        if p.suffix == ".json":
            d = json.loads(p.read_text())
            return [flatten(x) for x in d] if isinstance(d, list) else [flatten(d)]
        if p.suffix in (".csv", ".tsv"):
            with p.open(newline="", errors="replace") as f:
                return list(csv.DictReader(f, delimiter="\t" if p.suffix == ".tsv" else ","))
    except (OSError, json.JSONDecodeError) as e:
        print(f"  cannot read {p}: {e}")
    return []


def write_csv(rows: list[dict], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def num(v):
    if isinstance(v, bool):
        return float(v)
    if v in (None, ""):
        return None
    if isinstance(v, str) and v.lower() in ("true", "false"):
        return 1.0 if v.lower() == "true" else 0.0
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


AGG = {
    "mean": lambda xs: statistics.fmean(xs) if xs else None,
    "median": lambda xs: statistics.median(xs) if xs else None,
    "sum": lambda xs: sum(xs) if xs else 0,
    "min": lambda xs: min(xs) if xs else None,
    "max": lambda xs: max(xs) if xs else None,
    "std": lambda xs: statistics.stdev(xs) if len(xs) > 1 else None,
    "rate": lambda xs: (sum(1 for x in xs if x) / len(xs)) if xs else None,
}


def summarize(rows: list[dict], group_by: list[str], metrics: dict) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(str(r.get(g, "")) for g in group_by)].append(r)
    out = []
    for key in sorted(groups):
        rs = groups[key]
        o = dict(zip(group_by, key))
        o["n"] = len(rs)
        for col, aggs in metrics.items():
            for agg in ([aggs] if isinstance(aggs, str) else aggs):
                if agg == "count":
                    o[f"{col}_count"] = sum(1 for r in rs if r.get(col) not in (None, ""))
                else:
                    v = AGG[agg]([x for x in (num(r.get(col)) for r in rs) if x is not None])
                    o[f"{col}_{agg}"] = round(v, 6) if isinstance(v, float) else v
        out.append(o)
    return out


def fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else ("" if v is None else str(v))


def md_table(rows: list[dict], limit: int = 60) -> str:
    if not rows:
        return "_(no rows)_\n"
    cols = list(rows[0])
    s = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows[:limit]:
        s += "| " + " | ".join(fmt(r.get(c)).replace("|", "/") for c in cols) + " |\n"
    return s + (f"\n_({len(rows) - limit} more rows in the CSV)_\n" if len(rows) > limit else "")


def tex_table(rows: list[dict], caption: str, label: str) -> str:
    if not rows:
        return ""
    cols = list(rows[0])
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")  # noqa: E731
    body = "\n".join(" & ".join(esc(fmt(r.get(c))) for c in cols) + r" \\" for r in rows)
    return (f"\\begin{{table}}[t]\n\\centering\\small\n\\caption{{{esc(caption)}}}\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{{'l' * len(cols)}}}\n\\toprule\n{' & '.join(esc(c) for c in cols)} \\\\\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")


def plot(spec: dict, rows: list[dict], out: Path) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib not installed (add it to requirements.txt)"
    x, y, hue, kind = spec["x"], spec["y"], spec.get("hue"), spec.get("kind", "bar")
    fig, ax = plt.subplots(figsize=spec.get("size", (7, 4)))
    series = defaultdict(list)
    for r in rows:
        series[str(r.get(hue, "")) if hue else y].append(r)
    xs = sorted({str(r.get(x)) for r in rows})
    if kind == "bar":
        width = 0.8 / max(len(series), 1)
        for k, (name, rs) in enumerate(sorted(series.items())):
            m = {str(r.get(x)): num(r.get(y)) or 0 for r in rs}
            ax.bar([i + k * width for i in range(len(xs))], [m.get(v, 0) for v in xs], width, label=name)
        ax.set_xticks([i + width * (len(series) - 1) / 2 for i in range(len(xs))])
        ax.set_xticklabels(xs, rotation=30, ha="right")
    elif kind == "box":
        groups = defaultdict(list)
        for r in rows:
            v = num(r.get(y))
            if v is not None:
                groups[str(r.get(x))].append(v)
        ax.boxplot([groups[v] for v in xs], labels=xs)
    else:
        for name, rs in sorted(series.items()):
            pts = sorted(((num(r.get(x)) if num(r.get(x)) is not None else str(r.get(x)), num(r.get(y))) for r in rs),
                         key=lambda t: (isinstance(t[0], str), t[0]))
            (ax.plot if kind == "line" else ax.scatter)([p[0] for p in pts], [p[1] for p in pts], label=name)
    ax.set_xlabel(spec.get("xlabel", x))
    ax.set_ylabel(spec.get("ylabel", y))
    ax.set_title(spec.get("title", spec["name"]))
    if hue:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{spec['name']}.{ext}", dpi=150)
    plt.close(fig)
    return None


def pipeline_records(run_id: str, dest: Path) -> tuple[list[dict], list[dict]]:
    lr = H.log_root(run_id)
    led = H.read_tsv(lr / "steps.tsv")
    steps = [dict(zip(led[0], r)) for r in led[1:]] if led else []
    write_csv(steps, dest / "steps_all_attempts.csv")
    latest = list(H.latest_steps(run_id).values())
    write_csv(latest, dest / "steps_latest.csv")
    jobs = [dict(zip(["job", "stage", "cells", "dependency", "submitted"], r)) for r in H.read_tsv(lr / "jobs.tsv")]
    write_csv(jobs, dest / "jobs.csv")
    ends = [dict(zip(["time", "job", "stage", "cell", "restart", "status", "host"], r)) for r in H.read_tsv(lr / "jobs_finished.tsv")]
    write_csv(ends, dest / "job_endings.csv")
    gpu = []
    for f in sorted((lr / "gpu").glob("*.csv")):
        util, mem = [], []
        for r in read_any(f):
            vals = list(r.values())
            try:
                util.append(float(str(vals[2]).split()[0]))
                mem.append(float(str(vals[4]).split()[0]))
            except (IndexError, ValueError):
                pass
        if util:
            gpu.append({"file": f.name, "samples": len(util), "util_mean_%": round(statistics.fmean(util), 1),
                        "util_max_%": max(util), "mem_max_MiB": max(mem) if mem else None})
    write_csv(gpu, dest / "gpu_usage.csv")
    try:
        import audit  # same directory
        a = audit.assemble(run_id)
        comp = [{k: r[k] for k in ("stage", "cell_id", "cell", "state", "steps_ok", "steps_total", "node", "elapsed")}
                | {"failed_steps": ",".join(r["failed"]), "records_done": (r["progress"] or [None, None])[0],
                   "records_expected": (r["progress"] or [None, None])[1]} for r in a["rows"]]
    except Exception as e:  # noqa: BLE001
        print(f"  completeness from audit failed: {e}")
        comp = []
    write_csv(comp, dest / "completeness.csv")
    return comp, gpu


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    run_id = sys.argv[1]
    c = H.cluster(run_id)
    cfg = H.config_dir(run_id)
    exp = json.loads((cfg / "experiments.resolved.json").read_text())
    spec = exp.get("results", {})
    res = H.results_dir(run_id, c)
    out = res / "collected"
    if out.exists():
        shutil.rmtree(out)
    (out / "csv").mkdir(parents=True)
    L = [f"# Results — {exp.get('project', '')} run {run_id}{' (SMOKE: not for reporting)' if exp.get('is_smoke') else ''}",
         f"\nCollected {time.strftime('%F %T')} from `{res}`.\n"]

    tables = {}
    L.append("## Tables\n\n| table | files | rows | csv |\n|---|---|---|---|")
    for t in spec.get("tables", []):
        rows, files = [], sorted(glob.glob(str(res / t["glob"]), recursive=True))
        for f in files:
            for r in read_any(Path(f)):
                r["_source"] = str(Path(f).relative_to(res))
                rows.append(r)
        tables[t["name"]] = rows
        write_csv(rows, out / "csv" / f"{t['name']}.csv")
        L.append(f"| {t['name']} | {len(files)} | {len(rows)} | csv/{t['name']}.csv |")

    sums = {}
    L.append("\n## Summaries\n")
    for s in spec.get("summaries", []):
        rows = summarize(tables.get(s["table"], []), s.get("group_by", []), s.get("metrics", {}))
        sums[s["name"]] = rows
        write_csv(rows, out / "csv" / f"{s['name']}.csv")
        (out / "tables").mkdir(exist_ok=True)
        (out / "tables" / f"{s['name']}.md").write_text(md_table(rows, 10_000))
        (out / "tables" / f"{s['name']}.tex").write_text(tex_table(rows, s.get("caption", s["name"]), f"tab:{s['name']}"))
        L.append(f"### {s['name']}\n\n{s.get('description', '')}\n\n{md_table(rows)}")

    figs = []
    for p in spec.get("plots", []):
        rows = sums.get(p.get("summary")) or tables.get(p.get("table")) or []
        err = plot(p, rows, out / "figures") if rows else "no data"
        figs.append(f"- {p['name']}: " + (f"figures/{p['name']}.png / .pdf" if not err else f"skipped ({err})"))
    if figs:
        L.append("## Figures\n\n" + "\n".join(figs) + "\n")

    comp, gpu = pipeline_records(run_id, out / "pipeline")
    if comp:
        done = sum(1 for r in comp if r["state"] == "OK")
        L.append(f"## Completeness\n\n{done}/{len(comp)} stage cells finished OK.\n\n" + md_table(comp, 200))
    if gpu:
        L.append("## GPU use\n\n" + md_table(gpu, 200))
    man = H.log_root(run_id) / "run_manifest.json"
    if man.exists():
        shutil.copy(man, out / "run_manifest.json")
    (out / "summary.md").write_text("\n".join(L) + "\n")

    dest = Path(c.get("RESULTS_DEST") or H.log_root(run_id) / "results")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(out, dest, dirs_exist_ok=True)
    for pat in spec.get("copy", []):
        for f in glob.glob(str(res / pat), recursive=True):
            fp = Path(f)
            if fp.is_file():
                tgt = dest / "raw" / fp.relative_to(res)
                tgt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fp, tgt)
    for d in spec.get("archive", []):
        src = res / d
        if src.is_dir():
            tmp = dest / f"{d}.tar.gz.tmp"
            with tarfile.open(tmp, "w:gz") as tf:
                tf.add(src, arcname=d)
            tmp.rename(dest / f"{d}.tar.gz")
    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    print(f"collected {len(tables)} tables, {len(sums)} summaries, {len(figs)} figures -> {dest} ({size / 2**20:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
