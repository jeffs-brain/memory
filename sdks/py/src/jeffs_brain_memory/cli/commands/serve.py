# SPDX-License-Identifier: Apache-2.0
"""`memory serve` — start the HTTP daemon."""

from __future__ import annotations

import os

import click


@click.command()
@click.option(
    "--addr",
    default=None,
    help="Address to bind (host:port). Defaults to $JB_ADDR or :8080.",
)
@click.option(
    "--root",
    default=None,
    help="JB_HOME directory (default $JB_HOME or ~/.jeffs-brain).",
)
@click.option(
    "--auth-token",
    default=None,
    help="Shared bearer token (default $JB_AUTH_TOKEN, optional).",
)
@click.option(
    "--contextualise/--no-contextualise",
    default=None,
    help="Enable live extraction contextualisation.",
)
@click.option(
    "--contextualise-cache-dir",
    default=None,
    help="Optional cache directory for live extraction contextualisation.",
)
def serve(
    addr: str | None,
    root: str | None,
    auth_token: str | None,
    contextualise: bool | None,
    contextualise_cache_dir: str | None,
) -> None:
    """Start the HTTP daemon matching `spec/PROTOCOL.md`."""
    import uvicorn

    from ...http.server import create_app

    resolved_addr = addr or os.environ.get("JB_ADDR") or ":8080"
    host, port = _parse_addr(resolved_addr)
    token = auth_token or os.environ.get("JB_AUTH_TOKEN")
    resolved_contextualise = (
        contextualise
        if contextualise is not None
        else _env_enabled(os.environ.get("JB_CONTEXTUALISE"))
    )
    app = create_app(
        root=root,
        auth_token=token,
        contextualise=resolved_contextualise,
        contextualise_cache_dir=(
            contextualise_cache_dir
            or os.environ.get("JB_CONTEXTUALISE_CACHE_DIR")
        ),
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


def _parse_addr(value: str) -> tuple[str, int]:
    """Parse a host:port string into a (host, port) tuple.

    Leading colon (`:8080`) binds all interfaces defaulted to 127.0.0.1
    for safety. IPv6 literals with brackets are tolerated.
    """
    host_part, _, port_part = value.rpartition(":")
    if not port_part:
        raise click.BadParameter(f"--addr missing port: {value!r}")
    host = host_part.strip()
    if host.startswith("["):
        host = host[1:]
    if host.endswith("]"):
        host = host[:-1]
    if not host:
        host = "127.0.0.1"
    try:
        port = int(port_part)
    except ValueError as exc:
        raise click.BadParameter(f"--addr port must be numeric: {value!r}") from exc
    return host, port


def _env_enabled(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}
