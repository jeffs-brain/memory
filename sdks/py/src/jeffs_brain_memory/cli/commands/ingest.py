# SPDX-License-Identifier: Apache-2.0
"""`memory ingest` — ingest files, URLs, or directories."""

from __future__ import annotations

import click


@click.command()
@click.argument("source", required=True)
@click.option("--brain", default="default")
def ingest(source: str, brain: str) -> None:
    """Ingest SOURCE (path or URL). Stub."""
    click.echo(f"memory ingest {source} (brain={brain}): scaffold only, not yet implemented")
