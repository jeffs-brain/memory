# SPDX-License-Identifier: Apache-2.0
"""Every tool's advertised input schema must match ``spec/MCP-TOOLS.md``:
the same top-level fields, the same required set, and matching types and
enum values. The TypeScript and Go servers run the same check."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from jeffs_brain_memory_mcp.client import create_memory_client
from jeffs_brain_memory_mcp.config import resolve_config
from jeffs_brain_memory_mcp.server import create_server

SPEC = Path(__file__).resolve().parents[3] / "spec" / "MCP-TOOLS.md"

# Known gaps against the spec. Implementing one means deleting it here.
NOT_YET_IMPLEMENTED = {"memory_ingest_batch", "memory_ingest_directory"}
MISSING_FIELDS = {"memory_ingest_file": {"extract"}, "memory_ingest_url": {"extract"}}

_SECTION_RE = re.compile(r"^## `(memory_[a-z_]+)`(.*?)(?=^## |\Z)", re.M | re.S)
_INPUT_RE = re.compile(r"\*\*Input schema\*\*(?::\s*`\{\}`|\s*```[a-z]*\n(.*?)```)", re.S)
_FIELD_RE = re.compile(r"^ {2}([A-Za-z_]+)(\?)?:\s*(.+)$", re.M)


@dataclass(frozen=True)
class SpecField:
    name: str
    required: bool
    type: str
    enum_values: tuple[str, ...] | None


def _spec_type(raw: str) -> tuple[str, tuple[str, ...] | None]:
    decl = raw.split("#", 1)[0].strip()
    if decl.startswith("'"):
        return "string", tuple(re.findall(r"'([^']+)'", decl))
    for typ in ("string", "integer", "number", "boolean"):
        if decl.startswith(typ):
            return typ, None
    if decl.startswith("Array<"):
        return "array", None
    raise AssertionError(f"unrecognised spec type: {raw}")


def _parse_spec() -> dict[str, list[SpecField]]:
    out: dict[str, list[SpecField]] = {}
    for name, body in _SECTION_RE.findall(SPEC.read_text(encoding="utf-8")):
        block = _INPUT_RE.search(body)
        assert block is not None, f"{name}: no input schema in the spec"
        fields = []
        for field, optional, rest in _FIELD_RE.findall(block.group(1) or ""):
            typ, enum_values = _spec_type(rest)
            fields.append(SpecField(field, optional == "", typ, enum_values))
        out[name] = fields
    return out


def _types(prop: dict[str, Any]) -> list[str]:
    """Flatten ``type`` and ``anyOf`` (pydantic's form for Optional fields)
    to the non-null JSON types."""
    if "anyOf" in prop:
        return [t for option in prop["anyOf"] for t in _types(option)]
    typ = prop.get("type")
    if isinstance(typ, list):
        return [t for t in typ if t != "null"]
    return [typ] if isinstance(typ, str) and typ != "null" else []


def _enum(prop: dict[str, Any]) -> list[str]:
    if "anyOf" in prop:
        return [v for option in prop["anyOf"] for v in _enum(option)]
    return list(prop.get("enum") or [])


@pytest.mark.anyio
async def test_tool_schemas_match_spec(tmp_path: Path) -> None:
    spec = _parse_spec()
    assert len(spec) == 13
    cfg = resolve_config({"JB_HOME": str(tmp_path)})
    memory_client = create_memory_client(cfg)
    try:
        async with create_connected_server_and_client_session(
            create_server(memory_client)
        ) as session:
            listed = await session.list_tools()
    finally:
        await memory_client.close()

    names = {tool.name for tool in listed.tools}
    assert names == set(spec) - NOT_YET_IMPLEMENTED

    for tool in listed.tools:
        schema = tool.inputSchema
        missing = MISSING_FIELDS.get(tool.name, set())
        fields = [f for f in spec[tool.name] if f.name not in missing]
        assert schema.get("type") == "object", tool.name
        properties: dict[str, dict[str, Any]] = schema.get("properties") or {}
        assert sorted(properties) == sorted(f.name for f in fields), tool.name
        assert sorted(schema.get("required") or []) == sorted(
            f.name for f in fields if f.required
        ), tool.name
        for field in fields:
            prop = properties[field.name]
            accepted = {"number", "integer"} if field.type == "number" else {field.type}
            types = _types(prop)
            assert len(types) == 1 and types[0] in accepted, (tool.name, field.name, prop)
            if field.enum_values is not None:
                assert sorted(_enum(prop)) == sorted(field.enum_values), (tool.name, field.name)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
