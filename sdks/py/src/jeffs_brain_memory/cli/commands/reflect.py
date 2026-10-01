# SPDX-License-Identifier: Apache-2.0
"""`memory reflect`: run a reflection pass."""

from __future__ import annotations

import click

from ._planned import not_implemented


@click.command(hidden=True)
@click.option("--brain", default="default")
def reflect(brain: str) -> None:
    """Run a reflection pass. Planned; not implemented yet."""
    not_implemented("reflect")
