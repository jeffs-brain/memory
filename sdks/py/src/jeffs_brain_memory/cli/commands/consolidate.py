# SPDX-License-Identifier: Apache-2.0
"""`memory consolidate`: run a consolidation pass."""

from __future__ import annotations

import click

from ._planned import not_implemented


@click.command(hidden=True)
@click.option("--brain", default="default")
def consolidate(brain: str) -> None:
    """Run a consolidation pass. Planned; not implemented yet."""
    not_implemented("consolidate")
