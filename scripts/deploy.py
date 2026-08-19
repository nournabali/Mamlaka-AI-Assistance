#!/usr/bin/env python3
"""Pass the quality gate, then deploy the Streamlit application with Docker Compose."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    environment = os.environ.copy()
    quality_gate = subprocess.run(
        [sys.executable, "scripts/pre_deploy_check.py"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
    )
    if quality_gate.returncode != 0:
        print("Deployment stopped because the quality gate failed.", file=sys.stderr)
        return quality_gate.returncode

    command = ["docker", "compose", "up", "--build", "-d"]
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
