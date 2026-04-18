# SPDX-License-Identifier: Apache-2.0
"""Procedural memory detection from conversation transcripts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..llm.types import Role
from .types import Message


@dataclass(slots=True)
class ProceduralRecord:
    tier: str = ""
    name: str = ""
    task_context: str = ""
    outcome: str = ""
    observed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tool_calls: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


def detect_procedurals(messages: list[Message]) -> list[ProceduralRecord]:
    records: list[ProceduralRecord] = []
    now = datetime.now(timezone.utc)

    for s in _detect_skill_invocations(messages):
        records.append(
            ProceduralRecord(
                tier="skill",
                name=s["name"],
                task_context=s["context"],
                outcome=s["outcome"],
                observed_at=now,
                tool_calls=["skill"],
                tags=["procedural", "skill", s["name"]],
            )
        )
    for a in _detect_agent_invocations(messages):
        records.append(
            ProceduralRecord(
                tier="agent",
                name=a["name"],
                task_context=a["context"],
                outcome=a["outcome"],
                observed_at=now,
                tool_calls=["agent"],
                tags=["procedural", "agent", a["name"]],
            )
        )
    return records


def _detect_skill_invocations(messages: list[Message]) -> list[dict]:
    out: list[dict] = []
    for i, m in enumerate(messages):
        if m.role != Role.ASSISTANT:
            continue
        for tc in m.tool_calls:
            if tc.name != "skill":
                continue
            try:
                args = json.loads(tc.arguments)
            except json.JSONDecodeError:
                continue
            skill = args.get("skill", "") if isinstance(args, dict) else ""
            if not skill:
                continue
            outcome = infer_tool_call_outcome(messages, i, tc.id)
            ctx = infer_procedural_context(messages, i)
            out.append({"name": skill, "context": ctx, "outcome": outcome})
    return out


def _detect_agent_invocations(messages: list[Message]) -> list[dict]:
    out: list[dict] = []
    for i, m in enumerate(messages):
        if m.role != Role.ASSISTANT:
            continue
        for tc in m.tool_calls:
            if tc.name != "agent":
                continue
            try:
                args = json.loads(tc.arguments)
            except json.JSONDecodeError:
                continue
            agent_type = args.get("type", "") if isinstance(args, dict) else ""
            if not agent_type:
                continue
            outcome = infer_tool_call_outcome(messages, i, tc.id)
            prompt = args.get("prompt", "") if isinstance(args, dict) else ""
            ctx = prompt if prompt else ""
            if ctx and len(ctx) > 160:
                ctx = ctx[:160]
            if not ctx:
                ctx = infer_procedural_context(messages, i)
            out.append({"name": agent_type, "context": ctx, "outcome": outcome})
    return out


def infer_tool_call_outcome(
    messages: list[Message], after_index: int, tool_call_id: str
) -> str:
    for m in messages[after_index:]:
        if m.role != Role.TOOL:
            continue
        if m.tool_call_id == tool_call_id or (not tool_call_id and m.name == "skill"):
            content = m.content.lower()
            if "error" in content or "failed" in content:
                return "error"
            return "ok"
    return "partial"


def infer_procedural_context(messages: list[Message], before_index: int) -> str:
    for i in range(before_index - 1, -1, -1):
        if messages[i].role == Role.USER:
            ctx = messages[i].content
            if len(ctx) > 160:
                ctx = ctx[:160]
            return ctx
    return ""


def format_procedural_record(r: ProceduralRecord) -> str:
    parts: list[str] = ["---", f"name: {r.name}"]
    parts.append("type: procedural")
    parts.append(f"tier: {r.tier}")
    parts.append(f"outcome: {r.outcome}")
    if r.tags:
        parts.append(f"tags: [{', '.join(r.tags)}]")
    parts.append(f"observed: {r.observed_at.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    parts.append("---")
    parts.append("")
    if r.task_context:
        parts.append("## Context")
        parts.append("")
        parts.append(r.task_context)
        parts.append("")
    if r.tool_calls:
        parts.append("## Tool sequence")
        parts.append("")
        parts.append(" -> ".join(r.tool_calls))
    parts.append("")
    return "\n".join(parts)
