# SPDX-License-Identifier: Apache-2.0
"""`memory create-brain`: create a new brain."""

from __future__ import annotations

import click

from ._planned import not_implemented


@click.command(name="create-brain", hidden=True)
@click.argument("brain_id", required=True)
def create_brain(brain_id: str) -> None:
    """Create a new brain with id BRAIN_ID. Planned; not implemented yet."""
    not_implemented("create-brain")
