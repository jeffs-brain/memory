# SPDX-License-Identifier: Apache-2.0
"""`memory init` — initialise a brain at `$JB_HOME`."""

from __future__ import annotations

import click


@click.command()
@click.option("--brain", default="default", help="Brain id to initialise.")
def init(brain: str) -> None:
    """Initialise a brain. Stub."""
    click.echo(f"memory init {brain}: scaffold only, not yet implemented")
