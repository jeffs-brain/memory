# SPDX-License-Identifier: Apache-2.0
"""`memory serve` — start the HTTP daemon."""

from __future__ import annotations

import click


@click.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8080, type=int, show_default=True)
def serve(host: str, port: int) -> None:
    """Start the HTTP daemon matching `spec/PROTOCOL.md`."""
    import uvicorn

    from ...http import create_app

    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")
