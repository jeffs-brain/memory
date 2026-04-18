# SPDX-License-Identifier: Apache-2.0
"""`memory recall` — recall stored memories."""

from __future__ import annotations

import click


@click.command()
@click.argument("query", required=True)
@click.option("--brain", default="default")
def recall(query: str, brain: str) -> None:
    """Recall memories matching QUERY. Stub."""
    click.echo(f"memory recall {query!r} (brain={brain}): scaffold only, not yet implemented")
