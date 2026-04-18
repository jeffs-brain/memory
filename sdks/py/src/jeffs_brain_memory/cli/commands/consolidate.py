# SPDX-License-Identifier: Apache-2.0
"""`memory consolidate` — run a consolidation pass."""

from __future__ import annotations

import click


@click.command()
@click.option("--brain", default="default")
def consolidate(brain: str) -> None:
    """Run a consolidation pass. Stub."""
    click.echo(f"memory consolidate (brain={brain}): scaffold only, not yet implemented")
