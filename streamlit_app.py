"""Streamlit Community Cloud entrypoint for the Mamlaka AI application."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Streamlit executes this file again for every interaction. ``run_module``
# re-executes the UI each time instead of returning Python's cached import.
runpy.run_module("mamlaka_ai.ui.streamlit_app", run_name="__main__")
