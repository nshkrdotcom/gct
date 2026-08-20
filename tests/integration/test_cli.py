from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from gct.cli import app


def test_cli_help_exposes_contract() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "doctor",
        "dataset",
        "activations",
        "behavior",
        "transport",
        "probes",
        "metrics",
        "stats",
        "report",
        "inspect",
        "verify",
        "run",
    ):
        assert command in result.stdout


def test_dataset_cli_round_trip(repo_root: Path, monkeypatch: object) -> None:
    # The command resolves run_root against cwd, so isolate it from repository artifacts.
    import os
    import tempfile

    runner = CliRunner()
    with tempfile.TemporaryDirectory() as directory:
        config = repo_root / "configs" / "experiment_ci.yaml"
        previous = os.getcwd()
        os.chdir(directory)
        try:
            result = runner.invoke(app, ["dataset", "build", "--config", str(config)])
            assert result.exit_code == 0, result.output
            import json

            run_id = json.loads(result.output)["run_id"]
            validation = runner.invoke(app, ["dataset", "validate", "--run", run_id])
            assert validation.exit_code == 0, validation.output
            assert json.loads(validation.output)["valid"] is True
        finally:
            os.chdir(previous)
