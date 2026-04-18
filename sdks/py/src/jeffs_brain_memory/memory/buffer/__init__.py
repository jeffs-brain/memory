# SPDX-License-Identifier: Apache-2.0
"""L0 rolling observation buffer for the memory stages."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .._memstore import NotFoundError, Store
from ..paths import memory_buffer_global, memory_buffer_project


class ScopeKind(str, Enum):
    GLOBAL = "global"
    PROJECT = "project"


@dataclass(slots=True)
class Scope:
    kind: ScopeKind = ScopeKind.GLOBAL
    slug: str = ""


@dataclass(slots=True)
class Observation:
    at: datetime
    intent: str = ""
    entities: list[str] = field(default_factory=list)
    outcome: str = ""
    summary: str = ""


@dataclass(slots=True)
class Config:
    token_budget: int = 8192
    compact_threshold: int = 100
    keep_recent_percent: int = 50
    max_observation_len: int = 160


def default_config() -> Config:
    return Config()


class Buffer:
    """The L0 always-in-context rolling observation buffer."""

    def __init__(self, store: Store, scope: Scope, cfg: Config) -> None:
        if cfg.token_budget <= 0:
            cfg.token_budget = 8192
        if cfg.compact_threshold <= 0:
            cfg.compact_threshold = 100
        if cfg.keep_recent_percent <= 0:
            cfg.keep_recent_percent = 50
        if cfg.max_observation_len <= 0:
            cfg.max_observation_len = 160
        self._store = store
        self._scope = scope
        self._cfg = cfg
        self._lock = threading.Lock()

    def _path(self) -> str:
        if self._scope.kind == ScopeKind.PROJECT:
            return memory_buffer_project(self._scope.slug)
        return memory_buffer_global()

    def render(self) -> str:
        with self._lock:
            try:
                data = self._store.read(self._path())
            except NotFoundError:
                return ""
            return data.decode("utf-8")

    def append(self, obs: Observation) -> None:
        with self._lock:
            summary = obs.summary
            if len(summary) > self._cfg.max_observation_len:
                summary = summary[: self._cfg.max_observation_len]
            line = format_observation(
                obs.at, obs.intent, obs.outcome, summary, obs.entities
            )
            self._store.append(self._path(), (line + "\n").encode("utf-8"))

    def token_count(self) -> int:
        with self._lock:
            try:
                data = self._store.read(self._path())
            except NotFoundError:
                return 0
            return len(data) // 4

    def needs_compaction(self) -> bool:
        tokens = self.token_count()
        threshold = (self._cfg.token_budget * self._cfg.compact_threshold) // 100
        return tokens >= threshold

    def compact(self) -> int:
        with self._lock:
            try:
                data = self._store.read(self._path())
            except NotFoundError:
                return 0
            content = data.decode("utf-8")
            lines = content.strip().split("\n")
            if len(lines) <= 1:
                return 0
            keep_count = max((len(lines) * self._cfg.keep_recent_percent) // 100, 1)
            if keep_count >= len(lines):
                return 0
            removed = len(lines) - keep_count
            kept = lines[-keep_count:]
            new_content = "\n".join(kept) + "\n"
            self._store.write(self._path(), new_content.encode("utf-8"))
            return removed


def format_observation(
    at: datetime, intent: str, outcome: str, summary: str, entities: list[str]
) -> str:
    parts = [f"- [{at.strftime('%H:%M:%S')}]"]
    if intent:
        parts.append(f"({intent})")
    if outcome and outcome != "ok":
        parts.append(f"[{outcome}]")
    out = " ".join(parts) + " " + summary
    if entities:
        out += " {" + ", ".join(entities) + "}"
    return out


__all__ = [
    "Buffer",
    "Config",
    "Observation",
    "Scope",
    "ScopeKind",
    "default_config",
    "format_observation",
]
