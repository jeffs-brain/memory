# SPDX-License-Identifier: Apache-2.0
"""Console entrypoint for ``python -m jeffs_brain_memory_mcp`` and ``memory-mcp``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from . import SERVER_NAME, SERVER_VERSION
from .server import run_stdio


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="memory-mcp",
        description=(
            "Stdio Model Context Protocol server for jeffs-brain memory. "
            "Automatically runs in local mode (FS + SQLite FTS) when "
            "JB_TOKEN is unset, or hosted mode (HTTP) when it is set."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{SERVER_NAME} {SERVER_VERSION}",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging to stderr.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(argv) if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(run_stdio())
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
