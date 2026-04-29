# SPDX-License-Identifier: Apache-2.0
"""Go SDK runner.

Launches `go run ./cmd/memory serve --addr 127.0.0.1:<port>` from `go`.
TODO(eval): swap to a prebuilt binary in CI to avoid paying the `go run`
compile cost on every nightly invocation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sdks.base import SdkRunner

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class GoRunner(SdkRunner):
    @property
    def name(self) -> str:
        return "go"

    @property
    def workdir(self) -> Path:
        return REPO_ROOT / "sdks" / "go"

    def build_command(self, port: int) -> list[str]:
        return ["go", "run", "./cmd/memory", "serve", "--addr", f"127.0.0.1:{port}"]
