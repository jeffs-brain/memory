# SPDX-License-Identifier: Apache-2.0
"""Python SDK runner.

Launches `uv run memory serve --addr 127.0.0.1:<port>` from `sdks/py`.
Matches the Go daemon's `--addr host:port` spelling so cross-SDK ops can
pass identical flags.
TODO(eval): consider `uv run --locked` once `uv.lock` is committed to the
Python SDK to guarantee reproducible CI runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sdks.base import SdkRunner

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PyRunner(SdkRunner):
    @property
    def name(self) -> str:
        return "py"

    @property
    def workdir(self) -> Path:
        return REPO_ROOT / "sdks" / "py"

    def build_command(self, port: int) -> list[str]:
        return ["uv", "run", "memory", "serve", "--addr", f"127.0.0.1:{port}"]
