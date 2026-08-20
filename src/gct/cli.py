"""Command-line contract for all GCT stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, cast

import pandas as pd
import typer

from gct.analysis.stats import run_statistics
from gct.config import load_config
from gct.data.generate import build_dataset, validate_dataset_path
from gct.metrics.evaluate import evaluate_metrics
from gct.models.loader import runtime_report
from gct.operators.fit import fit_transport_operators
from gct.pipeline import run_pipeline
from gct.probes.hidden_coordinate import fit_hidden_coordinate_probes
from gct.reporting.report import build_report
from gct.storage.activations import evaluate_behavior_shards, extract_activation_shards
from gct.verify import verify_run

app = typer.Typer(no_args_is_help=True, help="Geometry of Conditional Truth research pipeline.")
dataset_app = typer.Typer(no_args_is_help=True)
activations_app = typer.Typer(no_args_is_help=True)
behavior_app = typer.Typer(no_args_is_help=True)
transport_app = typer.Typer(no_args_is_help=True)
probes_app = typer.Typer(no_args_is_help=True)
metrics_app = typer.Typer(no_args_is_help=True)
stats_app = typer.Typer(no_args_is_help=True)
report_app = typer.Typer(no_args_is_help=True)
inspect_app = typer.Typer(no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")
app.add_typer(activations_app, name="activations")
app.add_typer(behavior_app, name="behavior")
app.add_typer(transport_app, name="transport")
app.add_typer(probes_app, name="probes")
app.add_typer(metrics_app, name="metrics")
app.add_typer(stats_app, name="stats")
app.add_typer(report_app, name="report")
app.add_typer(inspect_app, name="inspect")

ConfigOption = Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)]


def _repo_root() -> Path:
    return Path.cwd().resolve()


def _run_path(value: str) -> Path:
    direct = Path(value)
    if direct.is_dir():
        return direct.resolve()
    candidate = _repo_root() / "runs" / value
    if candidate.is_dir():
        return candidate.resolve()
    raise typer.BadParameter(f"run not found as path or run ID: {value}")


def _print(value: object) -> None:
    typer.echo(json.dumps(value, indent=2, sort_keys=True, default=str))


@app.command()
def doctor(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/experiment_full.yaml"
    ),
) -> None:
    """Report GPU/PyTorch/BF16 compatibility with actionable failures."""
    report = runtime_report(load_config(config))
    _print(report.to_dict())
    if not report.ok:
        raise typer.Exit(1)


@dataset_app.command("build")
def dataset_build(config: ConfigOption) -> None:
    cfg = load_config(config)
    run_dir = build_dataset(cfg, _repo_root())
    _print({"run_id": cfg.run_id, "run_dir": run_dir, "status": "dataset_complete"})


@dataset_app.command("validate")
def dataset_validate(run: Annotated[str, typer.Option("--run")]) -> None:
    _print(validate_dataset_path(_run_path(run)))


@activations_app.command("extract")
def activations_extract(config: ConfigOption, resume: bool = False) -> None:
    cfg = load_config(config)
    run_dir = extract_activation_shards(cfg, _repo_root(), resume=resume)
    _print({"run_id": cfg.run_id, "run_dir": run_dir, "status": "activations_complete"})


@behavior_app.command("evaluate")
def behavior_evaluate(config: ConfigOption, resume: bool = False) -> None:
    cfg = load_config(config)
    run_dir = evaluate_behavior_shards(cfg, _repo_root(), resume=resume)
    _print({"run_id": cfg.run_id, "run_dir": run_dir, "status": "behavior_complete"})


@transport_app.command("fit")
def transport_fit(config: ConfigOption) -> None:
    cfg = load_config(config)
    fit_transport_operators(cfg, _repo_root())
    _print({"run_id": cfg.run_id, "status": "operators_complete"})


@probes_app.command("fit")
def probes_fit(config: ConfigOption) -> None:
    cfg = load_config(config)
    fit_hidden_coordinate_probes(cfg, _repo_root())
    _print({"run_id": cfg.run_id, "status": "probes_complete"})


@metrics_app.command("evaluate")
def metrics_evaluate(config: ConfigOption) -> None:
    cfg = load_config(config)
    evaluate_metrics(cfg, _repo_root())
    _print({"run_id": cfg.run_id, "status": "metrics_complete"})


@stats_app.command("run")
def stats_run(config: ConfigOption) -> None:
    cfg = load_config(config)
    run_statistics(cfg, _repo_root())
    _print({"run_id": cfg.run_id, "status": "statistics_complete"})


@report_app.command("build")
def report_build(config: ConfigOption) -> None:
    cfg = load_config(config)
    build_report(cfg, _repo_root())
    _print({"run_id": cfg.run_id, "status": "complete"})


@app.command("run")
def run_command(config: ConfigOption, resume: bool = False) -> None:
    """Execute or safely resume the deterministic end-to-end pipeline."""
    cfg = load_config(config)
    run_dir = run_pipeline(cfg, _repo_root(), resume=resume)
    _print({"run_id": cfg.run_id, "run_dir": run_dir, "status": "complete"})


@inspect_app.command("run")
def inspect_run(run: str) -> None:
    path = _run_path(run)
    _print(json.loads((path / "manifest.json").read_text(encoding="utf-8")))


@inspect_app.command("sample")
def inspect_sample(sample_id: str) -> None:
    matches: list[dict[str, object]] = []
    for path in (_repo_root() / "runs").glob("*/dataset/samples.parquet"):
        frame = pd.read_parquet(path, filters=[("sample_id", "==", sample_id)])
        if not frame.empty:
            matches.extend(cast(list[dict[str, object]], frame.to_dict(orient="records")))
    if not matches:
        raise typer.BadParameter(f"sample not found: {sample_id}")
    _print(matches)


@app.command("verify")
def verify(run_id: str) -> None:
    result = verify_run(_run_path(run_id))
    _print(result)
    if not result["valid"]:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
