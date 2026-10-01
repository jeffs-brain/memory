# SPDX-License-Identifier: Apache-2.0
"""`memory ask`: retrieval-augmented generation."""

from __future__ import annotations

import click

from ._planned import not_implemented


@click.command(hidden=True)
@click.argument("question", required=True)
@click.option("--brain", default="default")
def ask(question: str, brain: str) -> None:
    """Ask QUESTION of the brain. Planned; not implemented yet."""
    not_implemented("ask")
