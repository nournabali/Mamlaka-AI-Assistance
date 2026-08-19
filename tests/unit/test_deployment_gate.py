from __future__ import annotations

import runpy
import subprocess
from types import SimpleNamespace

from mamlaka_ai.config import PROJECT_ROOT


DEPLOY_MODULE = runpy.run_path(str(PROJECT_ROOT / "scripts" / "deploy.py"))
DEPLOY_MAIN = DEPLOY_MODULE["main"]


def test_deployment_runs_quality_gate_before_docker(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert DEPLOY_MAIN() == 0
    assert calls[0][1:] == ["scripts/pre_deploy_check.py"]
    assert calls[1] == ["docker", "compose", "up", "--build", "-d"]


def test_failed_quality_gate_stops_before_docker(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert DEPLOY_MAIN() == 1
    assert len(calls) == 1
