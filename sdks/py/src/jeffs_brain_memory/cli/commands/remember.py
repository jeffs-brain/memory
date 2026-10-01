# SPDX-License-Identifier: Apache-2.0
"""`memory remember`: store a memory fact."""

from __future__ import annotations

import click

from ._planned import not_implemented


@click.command(hidden=True)
@click.argument("content", required=True)
@click.option("--brain", default="default")
def remember(content: str, brain: str) -> None:
    """Remember CONTENT. Planned; not implemented yet."""
    not_implemented("remember")
