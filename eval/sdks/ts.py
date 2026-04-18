# SPDX-License-Identifier: Apache-2.0
"""TypeScript SDK runner.

Launches `bun run memory serve --addr :<port>` from `sdks/ts/memory`.
TODO(eval): once the TS CLI exposes `memory serve`, validate the exact flag
spelling. `--addr :0` lets the daemon pick a port itself; we pick a free one
for deterministic polling.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sdks.base import SdkRunner

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TsRunner(SdkRunner):
    @property
    def name(self) -> str:
        return "ts"

    @property
    def workdir(self) -> Path:
        return REPO_ROOT / "sdks" / "ts" / "memory"

    def build_command(self, port: int) -> list[str]:
        return ["bun", "run", "memory", "serve", "--addr", f"127.0.0.1:{port}"]
