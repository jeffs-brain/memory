# SPDX-License-Identifier: Apache-2.0
"""Pytest shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

SPEC_DIR = Path(__file__).resolve().parents[3] / "spec"


@pytest.fixture(scope="session")
def spec_dir() -> Path:
    """Absolute path to the shared `spec/` directory."""
    return SPEC_DIR
