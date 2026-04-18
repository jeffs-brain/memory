# SPDX-License-Identifier: Apache-2.0
"""`memory remember` — store a memory fact."""

from __future__ import annotations

import click


@click.command()
@click.argument("content", required=True)
@click.option("--brain", default="default")
def remember(content: str, brain: str) -> None:
    """Remember CONTENT. Stub."""
    click.echo(f"memory remember {content!r} (brain={brain}): scaffold only, not yet implemented")
