"""Streamlit Community Cloud entrypoint for the Mamlaka AI application."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mamlaka_ai.ui import streamlit_app  # noqa: E402, F401
