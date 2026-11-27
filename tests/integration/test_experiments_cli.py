"""CLI tests for `python -m agentic_research.experiments`."""

from __future__ import annotations

import json

from agentic_research.experiments.cli import main


def test_list_and_unknown_scenario(capsys) -> None:
    assert main(["--list"]) == 0
    assert "fault-drill" in capsys.readouterr().out
    assert main(["bogus"]) == 2
    assert main([]) == 2
    assert main(["fault-drill", "--variant", "bogus"]) == 2


def test_run_exports_dossiers_and_exit_codes(tmp_path, capsys) -> None:
    assert main(["contradiction", "--export-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Position A" in out and "supervisor" in out
    files = sorted(p.name.split(".dossier.")[1] for p in tmp_path.iterdir())
    assert files == ["html", "json", "md"]
    data = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert data["status"] == "completed" and data["events"]
    assert main(["fault-drill", "--variant", "exhausted"]) == 1  # failed run -> non-zero


def test_interactive_gate_uses_input(monkeypatch, capsys) -> None:
    answers = iter(["v", "add detail", "a"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert main(["human-checkpoint", "--interactive"]) == 0
    assert "Revision 1" in capsys.readouterr().out
