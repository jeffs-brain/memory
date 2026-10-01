# SPDX-License-Identifier: Apache-2.0
"""Shared behaviour for commands that are planned but not built yet."""

from __future__ import annotations

from typing import NoReturn

import click


def not_implemented(command: str) -> NoReturn:
    """Fail with exit status 1 so scripts never mistake a planned command
    for a successful one."""
    raise click.ClickException(
        f"memory {command} is not implemented in the Python SDK yet; "
        "run `memory serve` and use the HTTP API"
    )
