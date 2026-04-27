# SPDX-License-Identifier: Apache-2.0
"""Abstract `SdkRunner` contract.

Concrete subclasses know how to launch an SDK's `memory serve` daemon on a
random port, poll `/healthz` until ready, expose the `endpoint` URL, and shut
down cleanly. Each runner is single-use: call `start()` once, `stop()` once.
"""
from __future__ import annotations

import abc
import contextlib
import os
import signal
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import httpx

HEALTH_PATH = "/healthz"
DEFAULT_HEALTH_TIMEOUT = 30.0
DEFAULT_HEALTH_INTERVAL = 0.25


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class SdkRunner(abc.ABC):
    """Base class. Subclasses implement `build_command` and `workdir`."""

    host: str = "127.0.0.1"
    port: int = 0
    health_timeout: float = DEFAULT_HEALTH_TIMEOUT
    health_interval: float = DEFAULT_HEALTH_INTERVAL
    _proc: subprocess.Popen[bytes] | None = field(default=None, init=False, repr=False)
    _resolved_port: int | None = field(default=None, init=False, repr=False)
    _stderr: BinaryIO | None = field(default=None, init=False, repr=False)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable SDK name, e.g. `ts`."""

    @property
    @abc.abstractmethod
    def workdir(self) -> Path:
        """Working directory for the subprocess. Relative paths resolve from repo root."""

    @abc.abstractmethod
    def build_command(self, port: int) -> list[str]:
        """Argv to launch `memory serve` for this SDK on the given port."""

    def process_env(self) -> dict[str, str]:
        return {}

    @property
    def endpoint(self) -> str:
        if self._resolved_port is None:
            raise RuntimeError("runner not started")
        return f"http://{self.host}:{self._resolved_port}"

    def start(self, *, port: int = 0) -> None:
        if self._proc is not None:
            raise RuntimeError("runner already started")
        resolved = port if port > 0 else pick_free_port()
        self._resolved_port = resolved
        cmd = self.build_command(resolved)
        env = os.environ.copy()
        env.update(self.process_env())
        self.health_timeout = _env_float(
            f"JB_EVAL_{self.name.upper()}_HEALTH_TIMEOUT",
            _env_float("JB_EVAL_SDK_HEALTH_TIMEOUT", self.health_timeout),
        )
        self._stderr = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(  # noqa: S603
            cmd,
            cwd=str(self.workdir) if self.workdir.exists() else None,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr,
            env=env,
        )
        try:
            self._wait_healthy()
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        if self._proc is None:
            return
        with contextlib.suppress(ProcessLookupError):
            self._proc.send_signal(signal.SIGTERM)
        try:
            self._proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                self._proc.kill()
            self._proc.wait(timeout=5.0)
        self._proc = None
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None

    def _wait_healthy(self) -> None:
        deadline = time.monotonic() + self.health_timeout
        url = f"{self.endpoint}{HEALTH_PATH}"
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"{self.name} daemon exited early with code {self._proc.returncode}; "
                    f"stderr: {self._stderr_tail()}"
                )
            try:
                resp = httpx.get(url, timeout=2.0)
                if resp.status_code == 200:
                    return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
            time.sleep(self.health_interval)
        raise TimeoutError(
            f"{self.name} daemon did not become healthy within {self.health_timeout:.1f}s; "
            f"last error: {last_err!r}; stderr: {self._stderr_tail()}"
        )

    def _stderr_tail(self, limit: int = 4096) -> str:
        if self._stderr is None:
            return ""
        self._stderr.flush()
        size = self._stderr.seek(0, os.SEEK_END)
        self._stderr.seek(max(0, size - limit), os.SEEK_SET)
        return self._stderr.read(limit).decode("utf-8", errors="replace").strip()


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default
