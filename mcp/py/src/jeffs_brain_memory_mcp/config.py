# SPDX-License-Identifier: Apache-2.0
"""Resolve runtime configuration from environment.

Two modes:

- ``hosted``: ``JB_TOKEN`` is set, server talks to the remote platform via
  httpx. Endpoint defaults to ``https://api.jeffsbrain.com`` and can be
  overridden with ``JB_ENDPOINT``.
- ``local``: no token, server runs against ``JB_HOME`` (default
  ``$HOME/.jeffs-brain``). Uses a filesystem store plus an SQLite FTS5
  index; Ollama is used for embeddings when reachable on localhost.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


DEFAULT_ENDPOINT = "https://api.jeffsbrain.com"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


@dataclass(frozen=True, slots=True)
class LocalConfig:
    """Local filesystem-backed configuration."""

    kind: Literal["local"]
    brain_root: Path
    default_brain: str | None
    ollama_base_url: str


@dataclass(frozen=True, slots=True)
class HostedConfig:
    """Hosted API-backed configuration."""

    kind: Literal["hosted"]
    endpoint: str
    token: str
    default_brain: str | None


ConfigMode = LocalConfig | HostedConfig


def resolve_config(env: dict[str, str] | None = None) -> ConfigMode:
    """Resolve the runtime configuration from an env mapping."""
    source = dict(os.environ) if env is None else dict(env)
    default_brain = source.get("JB_BRAIN") or None
    token = source.get("JB_TOKEN")
    if token:
        return HostedConfig(
            kind="hosted",
            endpoint=source.get("JB_ENDPOINT") or DEFAULT_ENDPOINT,
            token=token,
            default_brain=default_brain,
        )
    home = source.get("JB_HOME")
    if not home:
        base = source.get("HOME") or str(Path.home())
        home = str(Path(base) / ".jeffs-brain")
    return LocalConfig(
        kind="local",
        brain_root=Path(home),
        default_brain=default_brain,
        ollama_base_url=source.get("OLLAMA_HOST") or DEFAULT_OLLAMA_BASE_URL,
    )
