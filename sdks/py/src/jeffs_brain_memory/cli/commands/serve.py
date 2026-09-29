# SPDX-License-Identifier: Apache-2.0
"""`memory serve`: start the HTTP daemon."""

from __future__ import annotations

import os

import click

DEFAULT_ADDR = "127.0.0.1:8080"


@click.command()
@click.option(
    "--addr",
    default=None,
    help=(
        "Address to bind (host:port). Defaults to $JB_ADDR or 127.0.0.1:8080. "
        "A non-loopback host requires an auth token."
    ),
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
    "--ingest-root",
    default=None,
    help=(
        "Directory ingest/file may read server-side paths from "
        "(default $JB_INGEST_ROOT; unset disables path ingest)."
    ),
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
    ingest_root: str | None,
    contextualise: bool | None,
    contextualise_cache_dir: str | None,
) -> None:
    """Start the HTTP daemon matching `spec/PROTOCOL.md`."""
    import uvicorn

    from ...http.server import create_app

    from ...http.ingest_root import resolve_ingest_root

    resolved_addr = addr or os.environ.get("JB_ADDR") or DEFAULT_ADDR
    host, port = _parse_addr(resolved_addr)
    token = auth_token or os.environ.get("JB_AUTH_TOKEN")
    _assert_bind_allowed(host, token)
    try:
        resolved_ingest_root = resolve_ingest_root(
            ingest_root or os.environ.get("JB_INGEST_ROOT")
        )
    except OSError as exc:
        raise click.BadParameter(str(exc), param_hint="--ingest-root") from exc
    resolved_contextualise = (
        contextualise
        if contextualise is not None
        else _env_enabled(os.environ.get("JB_CONTEXTUALISE"))
    )
    app = create_app(
        root=root,
        auth_token=token,
        ingest_root=resolved_ingest_root,
        contextualise=resolved_contextualise,
        contextualise_cache_dir=(
            contextualise_cache_dir
            or os.environ.get("JB_CONTEXTUALISE_CACHE_DIR")
        ),
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


def _parse_addr(value: str) -> tuple[str, int]:
    """Parse a host:port string into a (host, port) tuple.

    An empty host (`:8080`) binds loopback, never every interface. IPv6
    literals with brackets are tolerated.
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


def _assert_bind_allowed(host: str, token: str | None) -> None:
    """Refuse a non-loopback bind without a token, per spec/PROTOCOL.md."""
    from ...http.middleware.auth import is_loopback_host

    if token or is_loopback_host(host):
        return
    raise click.UsageError(
        f"serve: refusing to listen on {host} without an auth token; set "
        "--auth-token or JB_AUTH_TOKEN, or bind to 127.0.0.1"
    )


def _env_enabled(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}
