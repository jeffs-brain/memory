# SPDX-License-Identifier: Apache-2.0
"""`memory ingest`: ingest files, URLs, or directories."""

from __future__ import annotations

import click

from ._planned import not_implemented


@click.command(hidden=True)
@click.argument("source", required=True)
@click.option("--brain", default="default")
def ingest(source: str, brain: str) -> None:
    """Ingest SOURCE (path or URL). Planned; not implemented yet."""
    not_implemented("ingest")
