# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for the jeffs-brain memory MCP server.

Drives the server through the official Python MCP SDK in-memory
transport so we exercise the real ``initialize`` + ``tools/list`` +
``tools/call`` protocol round trip.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import pytest

from mcp import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session

from jeffs_brain_memory_mcp.client import create_memory_client
from jeffs_brain_memory_mcp.config import resolve_config
from jeffs_brain_memory_mcp.server import create_server


FIXTURES = Path(__file__).parent / "fixtures"


@asynccontextmanager
async def _session(tmp_path: Path) -> AsyncIterator[ClientSession]:
    cfg = resolve_config({"JB_HOME": str(tmp_path)})
    memory_client = create_memory_client(cfg)
    server = create_server(memory_client)
    try:
        async with create_connected_server_and_client_session(server) as session:
            yield session
    finally:
        await memory_client.close()


def _parse_json(content: list[object]) -> dict[str, object]:
    """Pull the JSON payload out of a tool response's text content."""
    first = content[0]
    text = getattr(first, "text", None)
    if not isinstance(text, str):  # pragma: no cover - schema mismatch
        raise AssertionError(f"expected text content, got {first!r}")
    parsed = json.loads(text)
    assert isinstance(parsed, dict)
    return parsed


@pytest.mark.anyio
async def test_lists_eleven_tools(tmp_path: Path) -> None:
    async with _session(tmp_path) as session:
        listed = await session.list_tools()
    names = sorted(tool.name for tool in listed.tools)
    assert names == [
        "memory_ask",
        "memory_consolidate",
        "memory_create_brain",
        "memory_extract",
        "memory_ingest_file",
        "memory_ingest_url",
        "memory_list_brains",
        "memory_recall",
        "memory_reflect",
        "memory_remember",
        "memory_search",
    ]


@pytest.mark.anyio
async def test_create_and_list_brain(tmp_path: Path) -> None:
    async with _session(tmp_path) as session:
        empty = await session.call_tool("memory_list_brains", {})
        empty_payload = _parse_json(empty.content)
        assert empty_payload == {"items": []}

        created = await session.call_tool(
            "memory_create_brain", {"name": "Scratch", "slug": "scratch"}
        )
        created_payload = _parse_json(created.content)
        assert created_payload["slug"] == "scratch"
        assert created_payload["name"] == "Scratch"
        assert created_payload["visibility"] == "private"

        listed = await session.call_tool("memory_list_brains", {})
        listed_payload = _parse_json(listed.content)
        items = listed_payload["items"]
        assert isinstance(items, list) and len(items) == 1
        assert items[0]["slug"] == "scratch"


@pytest.mark.anyio
async def test_ingest_file_then_search(tmp_path: Path) -> None:
    async with _session(tmp_path) as session:
        await session.call_tool(
            "memory_create_brain", {"name": "default", "slug": "default"}
        )

        ingest_result = await session.call_tool(
            "memory_ingest_file",
            {"path": str(FIXTURES / "hello.md"), "brain": "default"},
        )
        ingest_payload = _parse_json(ingest_result.content)
        assert ingest_payload["status"] == "completed"
        assert ingest_payload["chunk_count"] >= 1

        search_result = await session.call_tool(
            "memory_search",
            {"query": "parkrun", "brain": "default"},
        )
        search_payload = _parse_json(search_result.content)
        hits = search_payload["hits"]
        assert isinstance(hits, list)
        assert len(hits) >= 1
        combined = "\n".join(str(hit.get("content", "")) for hit in hits).lower()
        assert "parkrun" in combined


@pytest.mark.anyio
async def test_remember_then_recall(tmp_path: Path) -> None:
    async with _session(tmp_path) as session:
        await session.call_tool(
            "memory_create_brain", {"name": "default", "slug": "default"}
        )
        remembered = await session.call_tool(
            "memory_remember",
            {
                "content": "# Pizza night\n\nEveryone preferred the sourdough base.",
                "title": "Pizza night",
                "brain": "default",
                "tags": ["food", "family"],
            },
        )
        remembered_payload = _parse_json(remembered.content)
        assert remembered_payload["path"].endswith(".md")
        assert remembered_payload["chunk_count"] == 1

        recall_result = await session.call_tool(
            "memory_recall",
            {"query": "sourdough", "brain": "default"},
        )
        recall_payload = _parse_json(recall_result.content)
        chunks = recall_payload["chunks"]
        assert isinstance(chunks, list) and len(chunks) >= 1


@pytest.mark.anyio
async def test_search_and_recall_honour_local_scope_and_sort(tmp_path: Path) -> None:
    async with _session(tmp_path) as session:
        await session.call_tool(
            "memory_create_brain", {"name": "default", "slug": "default"}
        )
        await session.call_tool(
            "memory_remember",
            {
                "content": "Project release checklist for this repo.",
                "title": "Project release checklist",
                "brain": "default",
                "path": "memory/project/tenant-a/release-project.md",
            },
        )
        await session.call_tool(
            "memory_remember",
            {
                "content": "Global release checklist for every repo.",
                "title": "Global release checklist",
                "brain": "default",
                "path": "memory/global/release-global.md",
            },
        )
        await asyncio.sleep(0.02)
        await session.call_tool(
            "memory_remember",
            {
                "content": "Newest project release checklist for this repo.",
                "title": "Newest project release checklist",
                "brain": "default",
                "path": "memory/project/tenant-a/release-project-newer.md",
            },
        )

        search_result = await session.call_tool(
            "memory_search",
            {
                "query": "release checklist",
                "brain": "default",
                "scope": "project",
                "sort": "recency",
                "top_k": 5,
            },
        )
        search_payload = _parse_json(search_result.content)
        hits = search_payload["hits"]
        assert isinstance(hits, list)
        assert [hit["path"] for hit in hits] == [
            "memory/project/tenant-a/release-project-newer.md",
            "memory/project/tenant-a/release-project.md",
        ]

        recall_result = await session.call_tool(
            "memory_recall",
            {
                "query": "release checklist",
                "brain": "default",
                "scope": "project",
                "top_k": 5,
            },
        )
        recall_payload = _parse_json(recall_result.content)
        chunks = recall_payload["chunks"]
        assert isinstance(chunks, list)
        assert all(chunk["path"].startswith("memory/project/") for chunk in chunks)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
