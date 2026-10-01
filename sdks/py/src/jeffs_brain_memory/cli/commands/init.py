# SPDX-License-Identifier: Apache-2.0
"""`memory init`: initialise a brain at `$JB_HOME`."""

from __future__ import annotations

import click

from ._planned import not_implemented


@click.command(hidden=True)
@click.option("--brain", default="default", help="Brain id to initialise.")
def init(brain: str) -> None:
    """Initialise a brain. Planned; not implemented yet."""
    not_implemented("init")
