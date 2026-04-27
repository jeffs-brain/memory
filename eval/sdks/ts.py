# SPDX-License-Identifier: Apache-2.0
"""TypeScript SDK runner.

Launches the compiled `memory` CLI from `sdks/ts/memory/dist/cli.js` via
`node`. Builds the dist if missing. The CLI accepts `--addr host:port`.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from sdks.base import SdkRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
PREFERRED_NODE_MAJOR = 24


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
        return [_resolve_node_binary(), str(cli), "serve", "--addr", f"127.0.0.1:{port}"]


def _resolve_node_binary() -> str:
    override = os.environ.get("JB_EVAL_TS_NODE") or os.environ.get("JB_EVAL_NODE")
    if override:
        return override

    nvm_dir = Path(os.environ.get("NVM_DIR", str(Path.home() / ".nvm")))
    candidates: list[tuple[tuple[int, int, int], Path]] = []
    for version_dir in (nvm_dir / "versions" / "node").glob(f"v{PREFERRED_NODE_MAJOR}.*"):
        node = version_dir / "bin" / "node"
        if node.exists():
            candidates.append((_node_version(version_dir.name), node))

    if candidates:
        return str(max(candidates, key=lambda candidate: candidate[0])[1])

    return shutil.which("node") or "node"


def _node_version(name: str) -> tuple[int, int, int]:
    parts = name.removeprefix("v").split(".")
    parsed: list[int] = []
    for part in parts[:3]:
        try:
            parsed.append(int(part))
        except ValueError:
            parsed.append(0)
    while len(parsed) < 3:
        parsed.append(0)
    return (parsed[0], parsed[1], parsed[2])
