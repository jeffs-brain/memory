# SPDX-License-Identifier: Apache-2.0
"""TypeScript SDK runner.

Launches the compiled `memory` CLI from `sdks/ts/memory/dist/cli.js` via
`node`. Builds the dist if missing. The CLI accepts `--addr host:port`.
"""
from __future__ import annotations

import subprocess
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
        cli = self.workdir / "dist" / "cli.js"
        if not cli.exists():
            subprocess.run(
                ["bun", "run", "build"],
                cwd=self.workdir,
                check=True,
            )
        return ["node", str(cli), "serve", "--addr", f"127.0.0.1:{port}"]
