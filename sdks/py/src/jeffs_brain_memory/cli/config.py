# SPDX-License-Identifier: Apache-2.0
"""CLI configuration — env var resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_HOME = Path.home() / ".jeffs-brain"


@dataclass(frozen=True, slots=True)
class CliConfig:
    home: Path
    token: str | None
    endpoint: str | None

    @property
    def is_hosted(self) -> bool:
        return self.token is not None


def load() -> CliConfig:
    """Resolve CLI config from env vars. `JB_HOME`, `JB_TOKEN`, `JB_ENDPOINT`."""
    home_env = os.environ.get("JB_HOME")
    home = Path(home_env).expanduser() if home_env else DEFAULT_HOME
    return CliConfig(
        home=home,
        token=os.environ.get("JB_TOKEN"),
        endpoint=os.environ.get("JB_ENDPOINT"),
    )
