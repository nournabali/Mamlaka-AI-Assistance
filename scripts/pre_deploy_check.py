#!/usr/bin/env python3
"""Run the complete automated quality gate required before deployment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_REPORT = PROJECT_ROOT / "artifacts" / "evaluation" / "pre-deploy.json"


def _run(command: list[str], *, env: dict[str, str] | None = None) -> int:
    print(f"\n→ {' '.join(command)}", flush=True)
    return subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="run deterministic tests only; deployment must not use this option",
    )
    args = parser.parse_args(argv)

    test_env = os.environ.copy()
    test_env["RUN_EMBEDDING_TESTS"] = "1"
    test_env["PYTHONDONTWRITEBYTECODE"] = "1"
    test_command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    if _run(test_command, env=test_env) != 0:
        print("\nPre-deployment check failed: automated tests did not pass.", file=sys.stderr)
        return 1

    if args.skip_live:
        print("\nDeterministic checks passed; live provider evaluation was skipped.")
        return 0

    EVALUATION_REPORT.parent.mkdir(parents=True, exist_ok=True)
    evaluation_command = [
        sys.executable,
        "scripts/run_evaluation.py",
        "--json-output",
        str(EVALUATION_REPORT),
    ]
    if _run(evaluation_command) != 0:
        print("\nPre-deployment check failed: live evaluation did not pass.", file=sys.stderr)
        return 1

    print("\nPre-deployment quality gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
