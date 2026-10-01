# SPDX-License-Identifier: Apache-2.0
"""`memory search`: hybrid retrieval."""

from __future__ import annotations

import click

from ._planned import not_implemented


@click.command(hidden=True)
@click.argument("query", required=True)
@click.option("--brain", default="default")
@click.option("--limit", type=int, default=20)
def search(query: str, brain: str, limit: int) -> None:
    """Search the brain for QUERY. Planned; not implemented yet."""
    not_implemented("search")
