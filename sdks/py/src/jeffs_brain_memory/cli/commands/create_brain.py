# SPDX-License-Identifier: Apache-2.0
"""`memory create-brain` — create a new brain."""

from __future__ import annotations

import click


@click.command(name="create-brain")
@click.argument("brain_id", required=True)
def create_brain(brain_id: str) -> None:
    """Create a new brain with id BRAIN_ID. Stub."""
    click.echo(f"memory create-brain {brain_id}: scaffold only, not yet implemented")
