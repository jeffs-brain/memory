# SPDX-License-Identifier: Apache-2.0
"""`memory reflect` — run a reflection pass."""

from __future__ import annotations

import click


@click.command()
@click.option("--brain", default="default")
def reflect(brain: str) -> None:
    """Run a reflection pass. Stub."""
    click.echo(f"memory reflect (brain={brain}): scaffold only, not yet implemented")
