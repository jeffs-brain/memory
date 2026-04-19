# SPDX-License-Identifier: Apache-2.0
"""Memory-stage handlers: remember, recall, extract, reflect, consolidate.

Delegates to the ported :mod:`memory` package. The Go reference shapes
are matched byte-for-byte on the wire so cross-SDK clients round-trip
without translation.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from starlette.requests import Request
from starlette.responses import Response

from ...llm.types import Role
from ...memory import (
    Consolidator,
    ExtractedMemory,
    MemoryManager,
    Message as MemMessage,
    Reflector,
    RecallWeights,
    SurfacedMemory,
    extract_from_messages,
    memory_global_topic,
    memory_project_topic,
)
from ..problem import internal_error, validation_error
from ._shared import decode_json_body, ok_json, resolve_brain


def _extract_memory_to_wire(em: ExtractedMemory) -> dict[str, Any]:
    """Serialise an :class:`ExtractedMemory` as a camelCase wire payload."""
    payload = asdict(em)
    # camelCase where Go uses it; snake_case for the rest.
    rename = {
        "index_entry": "indexEntry",
        "session_id": "sessionId",
        "observed_on": "observedOn",
        "session_date": "sessionDate",
        "context_prefix": "contextPrefix",
        "modified_override": "modifiedOverride",
    }
    for src, dst in rename.items():
        if src in payload:
            payload[dst] = payload.pop(src)
    return payload


def _wire_messages_to_memory(raw: list[Any]) -> list[MemMessage]:
    messages: list[MemMessage] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        role_raw = str(item.get("role", "")).lower()
        try:
            role = Role(role_raw) if role_raw else Role.USER
        except ValueError:
            role = Role.USER
        content = str(item.get("content", ""))
        messages.append(MemMessage(role=role, content=content))
    return messages


def _project_from_body(body: dict[str, Any]) -> str:
    value = body.get("project")
    return str(value) if isinstance(value, str) else ""


async def remember(request: Request) -> Response:
    br = await resolve_brain(request)
    if isinstance(br, Response):
        return br
    body = await decode_json_body(request, 256 * 1024)
    if isinstance(body, Response):
        return body
    note = body.get("note")
    if not isinstance(note, str) or not note.strip():
        return validation_error("note required")
    slug = body.get("slug") or ""
    scope = body.get("scope") or "global"
    tags = body.get("tags") or []
    source = body.get("source") or None

    from datetime import datetime, timezone

    if not slug:
        slug = "note-" + datetime.now(tz=timezone.utc).strftime("%Y%m%dt%H%M%Sz")

    try:
        if scope == "" or scope == "global":
            path = memory_global_topic(slug)
        elif isinstance(scope, str) and scope.startswith("project:"):
            project = scope[len("project:"):].strip()
            if not project:
                return validation_error("project slug required for project scope")
            path = memory_project_topic(project, slug)
        else:
            return validation_error(f"unknown scope: {scope}")
    except Exception as exc:  # noqa: BLE001
        return validation_error(str(exc))

    rendered = _build_remember_body(
        note,
        tags if isinstance(tags, list) else [],
        source if isinstance(source, str) else None,
    )
    try:
        await br.store.write(path, rendered.encode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))
    return ok_json({"path": path, "slug": slug}, status=201)


def _build_remember_body(
    note: str, tags: list[str], source: str | None
) -> str:
    from datetime import datetime, timezone

    now = datetime.now(tz=timezone.utc).isoformat()
    lines = ["---", 'name: "remembered"']
    if source:
        lines.append(f'source: "{source}"')
    lines.append(f"created: {now}")
    lines.append(f"modified: {now}")
    if tags:
        lines.append("tags:")
        for tag in tags:
            lines.append(f"  - {tag}")
    lines.append("---")
    lines.append("")
    lines.append(note.strip())
    lines.append("")
    return "\n".join(lines)


async def recall(request: Request) -> Response:
    br = await resolve_brain(request)
    if isinstance(br, Response):
        return br
    body = await decode_json_body(request, 64 * 1024)
    if isinstance(body, Response):
        return body
    query = body.get("query")
    if not isinstance(query, str) or not query:
        return validation_error("query required")
    project = _project_from_body(body)

    daemon = request.app.state.daemon  # type: ignore[attr-defined]
    provider = daemon.llm

    manager: MemoryManager = br.memory_manager
    weights = RecallWeights(global_weight=1.0, project_weight=1.0)
    try:
        from ...memory import recall as recall_impl

        memories = await recall_impl(
            manager,
            provider,
            "",
            project,
            query,
            None,
            weights,
        )
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))

    return ok_json(
        {
            "memories": [_surfaced_to_wire(m) for m in memories],
        }
    )


def _surfaced_to_wire(m: SurfacedMemory) -> dict[str, Any]:
    return {
        "path": m.path,
        "content": m.content,
        "linkedFrom": m.linked_from,
    }


def _extract_session_summary(messages: list[MemMessage]) -> str:
    for message in messages:
        if message.role == Role.SYSTEM and message.content.strip():
            return _truncate_one_line(message.content, 240)
    for message in messages:
        if message.role == Role.USER and message.content.strip():
            return _truncate_one_line(message.content, 240)
    return ""


def _truncate_one_line(value: str, limit: int) -> str:
    compact = " ".join(value.replace("\r\n", " ").replace("\n", " ").split())
    if limit > 0 and len(compact) > limit:
        return compact[:limit] + "..."
    return compact


async def _decorate_extracted_memories(
    contextualiser,
    messages: list[MemMessage],
    session_id: str,
    session_date: str,
    extracted: list[ExtractedMemory],
) -> list[ExtractedMemory]:
    if not extracted:
        return extracted
    summary = _extract_session_summary(messages)
    for memory in extracted:
        if session_id and not memory.session_id:
            memory.session_id = session_id
        if session_date and not memory.session_date:
            memory.session_date = session_date
        if (
            contextualiser is None
            or memory.context_prefix
            or not getattr(contextualiser, "enabled", lambda: False)()
        ):
            continue
        prefix = await contextualiser.build_prefix_async(
            session_id, summary, memory.content
        )
        if prefix:
            memory.context_prefix = prefix
    return extracted


async def extract(request: Request) -> Response:
    br = await resolve_brain(request)
    if isinstance(br, Response):
        return br
    body = await decode_json_body(request, 1 << 20)
    if isinstance(body, Response):
        return body
    messages_raw = body.get("messages")
    if not isinstance(messages_raw, list) or not messages_raw:
        return validation_error("messages required")
    project = _project_from_body(body)
    model = str(body.get("model") or "")
    session_id = str(body.get("sessionId") or "")
    session_date = str(body.get("sessionDate") or "")

    messages = _wire_messages_to_memory(messages_raw)

    daemon = request.app.state.daemon  # type: ignore[attr-defined]
    provider = daemon.llm
    try:
        results = await extract_from_messages(
            provider,
            model,
            br.memory_manager,
            project,
            messages,
            session_id=session_id,
            session_date=session_date,
        )
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))

    results = await _decorate_extracted_memories(
        daemon.contextualiser,
        messages,
        session_id,
        session_date,
        results,
    )

    return ok_json(
        {
            "memories": [_extract_memory_to_wire(em) for em in results],
        }
    )


async def reflect(request: Request) -> Response:
    br = await resolve_brain(request)
    if isinstance(br, Response):
        return br
    body = await decode_json_body(request, 1 << 20)
    if isinstance(body, Response):
        return body
    messages_raw = body.get("messages")
    if not isinstance(messages_raw, list) or not messages_raw:
        return validation_error("messages required")
    project = _project_from_body(body)
    model = str(body.get("model") or "")

    messages = _wire_messages_to_memory(messages_raw)

    daemon = request.app.state.daemon  # type: ignore[attr-defined]
    provider = daemon.llm
    reflector = Reflector(br.memory_manager)
    try:
        result = await reflector.force_reflect(provider, model, project, messages)
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))
    if result is None:
        return ok_json({"result": None})
    payload: dict[str, Any] = {
        "outcome": result.outcome,
        "summary": result.summary,
        "retryFeedback": result.retry_feedback,
        "shouldRecordEpisode": result.should_record_episode,
        "heuristics": [
            {
                "rule": h.rule,
                "context": h.context,
                "confidence": h.confidence,
                "category": h.category,
                "scope": h.scope,
                "antiPattern": h.anti_pattern,
            }
            for h in result.heuristics
        ],
    }
    return ok_json({"result": payload})


async def consolidate(request: Request) -> Response:
    br = await resolve_brain(request)
    if isinstance(br, Response):
        return br

    body: dict[str, Any] = {}
    raw = await request.body()
    if raw:
        try:
            import json as _json

            parsed = _json.loads(raw.decode("utf-8"))
            if isinstance(parsed, dict):
                body = parsed
        except Exception as exc:  # noqa: BLE001
            return validation_error(f"invalid JSON: {exc}")
    mode = (
        "quick"
        if str(body.get("mode") or "").lower() == "quick"
        else "full"
    )
    model = str(body.get("model") or "")

    daemon = request.app.state.daemon  # type: ignore[attr-defined]
    provider = daemon.llm

    cons = Consolidator(provider, model, br.memory_manager)
    try:
        if mode == "quick":
            report = await cons.run_quick()
        else:
            report = await cons.run_full()
    except Exception as exc:  # noqa: BLE001
        return internal_error(str(exc))

    return ok_json(
        {
            "mode": mode,
            "episodesReviewed": report.episodes_reviewed,
            "memoriesMerged": report.memories_merged,
            "heuristicsUpdated": report.heuristics_updated,
            "indexesRebuilt": report.indexes_rebuilt,
            "staleMemoriesFlagged": report.stale_memories_flagged,
            "insightsPromoted": report.insights_promoted,
            "durationSeconds": report.duration_seconds,
            "errors": list(report.errors),
        }
    )
