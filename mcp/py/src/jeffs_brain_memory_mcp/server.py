# SPDX-License-Identifier: Apache-2.0
"""Stdio MCP server bootstrap.

Wires the official Python Model Context Protocol SDK to the unified
:class:`MemoryClient` and registers the eleven ``memory_*`` tools.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from pydantic import ValidationError

from . import SERVER_NAME, SERVER_VERSION
from .client import MemoryClient, ProgressEmitter, create_memory_client
from .config import resolve_config
from .tools import TOOLS, ToolDef


logger = logging.getLogger("jeffs_brain_memory_mcp.server")


def _build_registry(tools: tuple[ToolDef, ...]) -> dict[str, ToolDef]:
    registry: dict[str, ToolDef] = {}
    for tool in tools:
        if tool.name in registry:
            raise RuntimeError(f"duplicate tool registration: {tool.name}")
        registry[tool.name] = tool
    return registry


def _text_summary(name: str, payload: dict[str, Any]) -> str:
    """Human-readable summary text for the response.

    The wire contract requires a short textual body alongside the
    structured payload; we render pretty JSON so MCP clients that only
    display text still see the full shape.
    """
    try:
        return json.dumps(payload, indent=2, default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return f"{name}: <unserialisable payload>"


@dataclass(slots=True)
class ServerBundle:
    """The wired server plus its backing client. Callers own shutdown."""

    server: Server
    client: MemoryClient


def create_server(client: MemoryClient) -> Server:
    """Build a fully-wired MCP :class:`Server` bound to ``client``."""
    registry = _build_registry(TOOLS)
    server: Server = Server(SERVER_NAME, version=SERVER_VERSION)

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        listed: list[types.Tool] = []
        for tool in TOOLS:
            schema = tool.input_model.model_json_schema()
            schema.setdefault("type", "object")
            listed.append(
                types.Tool(
                    name=tool.name,
                    description=tool.description,
                    inputSchema=schema,
                )
            )
        return listed

    @server.call_tool()
    async def _call_tool(
        name: str, arguments: dict[str, Any] | None
    ) -> types.CallToolResult:
        tool = registry.get(name)
        if tool is None:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"unknown tool: {name}")],
                isError=True,
            )

        try:
            parsed = tool.input_model.model_validate(arguments or {})
        except ValidationError as exc:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=str(exc))],
                isError=True,
            )

        progress = _make_progress_emitter(server)
        try:
            payload = await tool.handler(parsed, client, progress)
        except Exception as exc:
            logger.exception("tool %s raised", name)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=str(exc))],
                isError=True,
            )

        return types.CallToolResult(
            content=[types.TextContent(type="text", text=_text_summary(name, payload))],
            structuredContent=payload,
        )

    return server


def _make_progress_emitter(server: Server) -> ProgressEmitter | None:
    """Return an emitter that pushes ``notifications/progress`` frames.

    The MCP client generates a progress token and sends it via
    ``params._meta.progressToken``. When the token is missing, we return
    ``None`` so handlers skip progress emission entirely.
    """

    async def _emit(progress: float, message: str | None = None) -> None:
        try:
            ctx = server.request_context
        except LookupError:  # pragma: no cover - guarded by MCP runtime
            return
        meta = ctx.meta
        if meta is None:
            return
        token = getattr(meta, "progressToken", None)
        if token is None:
            return
        await ctx.session.send_progress_notification(
            progress_token=token,
            progress=progress,
            message=message,
        )

    return _emit


async def run_stdio() -> None:
    """Entrypoint used by ``memory-mcp``. Runs the server over stdio."""
    cfg = resolve_config()
    client = create_memory_client(cfg)
    server = create_server(client)

    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name=SERVER_NAME,
                    server_version=SERVER_VERSION,
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    finally:
        await client.close()


__all__ = ["create_server", "run_stdio", "ServerBundle"]
