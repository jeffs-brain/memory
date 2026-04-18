# SPDX-License-Identifier: Apache-2.0
"""`memory search` — hybrid retrieval."""

from __future__ import annotations

import click


@click.command()
@click.argument("query", required=True)
@click.option("--brain", default="default")
@click.option("--limit", type=int, default=20)
def search(query: str, brain: str, limit: int) -> None:
    """Search the brain for QUERY. Stub."""
    click.echo(f"memory search {query!r} (brain={brain}, limit={limit}): scaffold only, not yet implemented")
