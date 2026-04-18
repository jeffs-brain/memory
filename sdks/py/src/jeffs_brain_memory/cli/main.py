# SPDX-License-Identifier: Apache-2.0
"""`memory` CLI entry point."""

from __future__ import annotations

import click

from .. import __version__
from . import commands


@click.group()
@click.version_option(version=__version__, package_name="jeffs-brain-memory")
def main() -> None:
    """Jeffs Brain memory CLI."""


main.add_command(commands.init.init)
main.add_command(commands.ingest.ingest)
main.add_command(commands.search.search)
main.add_command(commands.ask.ask)
main.add_command(commands.serve.serve)
main.add_command(commands.remember.remember)
main.add_command(commands.recall.recall)
main.add_command(commands.reflect.reflect)
main.add_command(commands.consolidate.consolidate)
main.add_command(commands.create_brain.create_brain)
main.add_command(commands.list_brains.list_brains)


if __name__ == "__main__":
    main()
