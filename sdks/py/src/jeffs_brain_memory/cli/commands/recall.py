# SPDX-License-Identifier: Apache-2.0
"""`memory recall`: recall stored memories."""

from __future__ import annotations

import click

from ._planned import not_implemented


@click.command(hidden=True)
@click.argument("query", required=True)
@click.option("--brain", default="default")
def recall(query: str, brain: str) -> None:
    """Recall memories matching QUERY. Planned; not implemented yet."""
    not_implemented("recall")
