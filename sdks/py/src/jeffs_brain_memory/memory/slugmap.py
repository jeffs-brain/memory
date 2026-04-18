# SPDX-License-Identifier: Apache-2.0
"""Machine-local map of project path -> canonical slug."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import yaml


def _default_path() -> str:
    home = os.path.expanduser("~")
    if not home or home == "~":
        home = "."
    return str(Path(home) / ".local" / "state" / "jeffs-brain" / "slug-map.yaml")


class SlugMap:
    """Persistent path -> slug mapping."""

    def __init__(self, path: str = "") -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, str] = {}
        self._path = path or _default_path()

    @property
    def path(self) -> str:
        return self._path

    def load(self) -> None:
        with self._lock:
            try:
                raw = Path(self._path).read_bytes()
            except OSError:
                return
            try:
                parsed = yaml.safe_load(raw.decode("utf-8")) or {}
            except yaml.YAMLError:
                return
            if isinstance(parsed, dict):
                self._entries = {str(k): str(v) for k, v in parsed.items()}

    def save(self) -> None:
        with self._lock:
            self._save_locked()

    def _save_locked(self) -> None:
        target = Path(self._path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(self._entries, sort_keys=True), encoding="utf-8")

    def lookup(self, project_path: str) -> str | None:
        with self._lock:
            return self._entries.get(project_path)

    def register(self, project_path: str, slug: str) -> None:
        with self._lock:
            self._entries[project_path] = slug
            self._save_locked()
