# SPDX-License-Identifier: Apache-2.0
"""`memory list-brains`: list brains in the current store."""

from __future__ import annotations

import click

from ._planned import not_implemented


@click.command(name="list-brains", hidden=True)
def list_brains() -> None:
    """List brains. Planned; not implemented yet."""
    not_implemented("list-brains")
