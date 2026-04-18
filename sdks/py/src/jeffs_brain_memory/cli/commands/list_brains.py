# SPDX-License-Identifier: Apache-2.0
"""`memory list-brains` — list brains in the current store."""

from __future__ import annotations

import click


@click.command(name="list-brains")
def list_brains() -> None:
    """List brains. Stub."""
    click.echo("memory list-brains: scaffold only, not yet implemented")
