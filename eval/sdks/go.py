# SPDX-License-Identifier: Apache-2.0
"""Go SDK runner.

Launches `go run ./cmd/memory serve --addr 127.0.0.1:<port>` from `sdks/go`.
TODO(eval): swap to a prebuilt binary in CI to avoid paying the `go run`
compile cost on every nightly invocation.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from sdks.base import SdkRunner

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GO_HEALTH_TIMEOUT = 180.0
MIN_GO_VERSION = (1, 25, 0)


@dataclass
class GoRunner(SdkRunner):
    health_timeout: float = DEFAULT_GO_HEALTH_TIMEOUT
    _go_binary: str | None = field(default=None, init=False, repr=False)

    @property
    def name(self) -> str:
        return "go"

    @property
    def workdir(self) -> Path:
        return REPO_ROOT / "sdks" / "go"

    def build_command(self, port: int) -> list[str]:
        self._go_binary = _resolve_go_binary()
        return [self._go_binary, "run", "./cmd/memory", "serve", "--addr", f"127.0.0.1:{port}"]

    def process_env(self) -> dict[str, str]:
        if self._go_binary is not None and _go_version_at_least(self._go_binary, MIN_GO_VERSION):
            return {"GOTOOLCHAIN": "local"}
        return {}


def _resolve_go_binary() -> str:
    override = os.environ.get("JB_EVAL_GO") or os.environ.get("JB_EVAL_GO_BIN")
    if override:
        return override

    candidates = [
        "/opt/homebrew/bin/go",
        "/opt/homebrew/opt/go/bin/go",
        *_cached_toolchain_go_binaries(),
        shutil.which("go"),
        "/usr/local/go/bin/go",
    ]
    existing = [candidate for candidate in candidates if candidate and Path(candidate).exists()]
    for candidate in existing:
        if _go_version_at_least(candidate, MIN_GO_VERSION):
            return candidate
    return existing[0] if existing else "go"


def _cached_toolchain_go_binaries() -> list[str]:
    mod_cache = Path(os.environ.get("GOMODCACHE", str(Path.home() / "go" / "pkg" / "mod")))
    binaries = [
        toolchain / "bin" / "go"
        for toolchain in mod_cache.glob("golang.org/toolchain@v*-go*.darwin-arm64")
    ]
    return [str(binary) for binary in binaries if binary.exists()]


def _go_version_at_least(binary: str, minimum: tuple[int, int, int]) -> bool:
    version = _go_binary_version(binary)
    return version is not None and version >= minimum


def _go_binary_version(binary: str) -> tuple[int, int, int] | None:
    try:
        completed = subprocess.run(
            [binary, "version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return _parse_go_version(completed.stdout)


def _parse_go_version(value: str) -> tuple[int, int, int] | None:
    match = re.search(r"\bgo(\d+)\.(\d+)(?:\.(\d+))?", value)
    if match is None:
        return None
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3) or "0"),
    )
