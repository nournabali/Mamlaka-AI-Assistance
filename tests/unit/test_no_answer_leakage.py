from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_PATHS = [
    ROOT / "src" / "mamlaka_ai",
    ROOT / "scripts" / "build_index.py",
]

# Acceptance expectations are allowed here, in a test, solely to ensure they
# never leak into runtime code. Source PDFs and generated index metadata are the
# authorised corpus and are intentionally outside RUNTIME_PATHS.
FORBIDDEN_RUNTIME_FACTS = (
    "Sara Al-Rashid",
    "Digital Strategy",
    "$2,400,000",
    "$2,600,000",
    "March 15, 2027",
    "April 1, 2027",
    "40%",
    "$50,000",
    "12 languages",
    "Finance Committee",
    "smart TV integration testing",
    "explicit consent",
    "required by law",
)


def _runtime_python_files() -> list[Path]:
    files: list[Path] = []
    for path in RUNTIME_PATHS:
        if path.is_file():
            files.append(path)
        else:
            files.extend(path.rglob("*.py"))
    return files


@pytest.mark.parametrize("path", _runtime_python_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_acceptance_answers_do_not_appear_in_runtime_source(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    leaked = [fact for fact in FORBIDDEN_RUNTIME_FACTS if fact.casefold() in source.casefold()]
    assert not leaked, f"acceptance fact(s) leaked into {path}: {leaked}"


def test_runtime_does_not_import_tests_or_evaluator() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in _runtime_python_files())
    assert "from tests" not in combined
    assert "import tests" not in combined
    assert "run_evaluation" not in combined


def test_production_container_excludes_evaluation_sources() -> None:
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert "tests" in ignored
    assert "scripts/run_evaluation.py" in ignored
    assert "tests" in ignored
