"""Command-line interface for LETHE."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .config import RunConfig
from .extract import extract as extract_program

app = typer.Typer(no_args_is_help=True, help="Model-driven fault localization with forgetting.")


@app.command("extract")
def extract(project: Path, out: Path) -> None:
    """Extract a versioned Python program model."""
    model = extract_program(project)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(model.model_dump_json(indent=2) + "\n")
    typer.echo(f"wrote {out}: {len(model.functions)} methods, {len(model.statements)} statements")


@app.command("validate-config")
def validate_config(config: Path) -> None:
    """Validate and print a normalized immutable run configuration."""
    parsed = RunConfig.from_yaml(config)
    typer.echo(parsed.model_dump_json(indent=2))


@app.command("report")
def report(run: Path, out: Path | None = None) -> None:
    """Render a concise report from a completed result.json."""
    result_path = run / "result.json"
    if not result_path.exists():
        raise typer.BadParameter(f"missing {result_path}")
    data = json.loads(result_path.read_text())
    best = data.get("best") or {}
    lines = ["# LETHE localization report", "", f"Candidate: `{best.get('id', 'none')}`"]
    for hyp in sorted(best.get("hypotheses", []), key=lambda h: h.get("rank", 0)):
        lines.append(f"{hyp.get('rank')}. `{hyp.get('element')}` — {hyp.get('rationale', '')}")
    facts = data.get("facts") or {}
    if facts:
        lines.extend(["", "## Judge-verified facts", ""])
        for element, fact in sorted(facts.items()):
            lines.append(f"- `{element}`: {fact.get('kind', 'unknown')}")
    text = "\n".join(lines) + "\n"
    target = out or run / "report.md"
    target.write_text(text)
    typer.echo(f"wrote {target}")


@app.command("localize")
def localize(config: Path, project: Path, out: Path) -> None:
    """Validate a localization request before adapter-specific execution."""
    parsed = RunConfig.from_yaml(config)
    if not project.is_dir():
        raise typer.BadParameter(f"project is not a directory: {project}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(
        json.dumps({"config": parsed.model_dump(), "project": str(project.resolve())}, indent=2)
        + "\n"
    )
    typer.echo(
        "request validated and manifest written; benchmark execution is launched through "
        "evaluation runners or hpc/slurm/submit.py"
    )


if __name__ == "__main__":
    app()
