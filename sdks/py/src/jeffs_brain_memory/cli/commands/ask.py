# SPDX-License-Identifier: Apache-2.0
"""`memory ask` — retrieval-augmented generation."""

from __future__ import annotations

import click


@click.command()
@click.argument("question", required=True)
@click.option("--brain", default="default")
def ask(question: str, brain: str) -> None:
    """Ask QUESTION of the brain. Stub."""
    click.echo(f"memory ask {question!r} (brain={brain}): scaffold only, not yet implemented")
