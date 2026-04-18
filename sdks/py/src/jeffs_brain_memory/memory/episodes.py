# SPDX-License-Identifier: Apache-2.0
"""Episodic memory recorder.

``EPISODE_SYSTEM_PROMPT`` is ported verbatim from
``sdks/go/memory/episodes.go``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from ..llm.provider import Provider
from ..llm.types import CompleteRequest
from ..llm.types import Message as LLMMessage
from ..llm.types import Role
from .types import Message

EPISODE_MAX_TOKENS = 1024
EPISODE_TEMPERATURE = 0.2
EPISODE_MIN_MESSAGES = 8

WRITE_TOOL_NAMES = {"write", "edit"}


EPISODE_SYSTEM_PROMPT = """You are summarising a coding session for episodic memory.
Produce a JSON object:
{
  "significant": true/false,
  "summary": "one paragraph of what was attempted and what happened",
  "outcome": "success|partial|failure",
  "heuristics": ["generalised learning 1", "learning 2"],
  "tags": ["tag1", "tag2"]
}
If the session was routine (simple Q&A, single file read), set significant=false.
Respond with ONLY valid JSON, no other text."""


@dataclass(slots=True)
class Episode:
    project_path: str = ""
    session_id: str = ""
    summary: str = ""
    outcome: str = ""
    heuristics: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


class EpisodeStore(Protocol):
    def create_episode(self, ep: Episode) -> None: ...


class EpisodeRecorder:
    async def maybe_record(
        self,
        provider: Provider,
        model: str,
        store: EpisodeStore | None,
        project_path: str,
        session_id: str,
        messages: list[Message],
    ) -> None:
        if not should_record_episode(messages):
            return
        user_prompt = build_episode_prompt(messages)
        try:
            resp = await provider.complete(
                CompleteRequest(
                    model=model,
                    messages=[
                        LLMMessage(role=Role.SYSTEM, content=EPISODE_SYSTEM_PROMPT),
                        LLMMessage(role=Role.USER, content=user_prompt),
                    ],
                    max_tokens=EPISODE_MAX_TOKENS,
                    temperature=EPISODE_TEMPERATURE,
                )
            )
        except Exception as e:
            raise RuntimeError(f"episode summarisation: {e}")

        result = parse_episode_result(resp.text)
        if not result.get("significant"):
            return

        ep = Episode(
            project_path=project_path,
            session_id=session_id,
            summary=str(result.get("summary", "")),
            outcome=str(result.get("outcome", "")),
            heuristics=[str(h) for h in result.get("heuristics", [])],
            tags=[str(t) for t in result.get("tags", [])],
        )
        if store is None:
            return
        store.create_episode(ep)


def should_record_episode(messages: list[Message]) -> bool:
    if len(messages) < EPISODE_MIN_MESSAGES:
        return False
    for m in messages:
        if m.role != Role.ASSISTANT:
            continue
        for tc in m.tool_calls:
            if tc.name in WRITE_TOOL_NAMES:
                return True
    return False


def build_episode_prompt(messages: list[Message]) -> str:
    parts: list[str] = ["## Session transcript\n"]
    for m in messages:
        if m.role == Role.USER:
            content = m.content
            if len(content) > 1000:
                content = content[:1000] + "\n[...truncated]"
            parts.append(f"[user]: {content}\n")
        elif m.role == Role.ASSISTANT:
            content = m.content
            if len(content) > 1000:
                content = content[:1000] + "\n[...truncated]"
            if content:
                parts.append(f"[assistant]: {content}\n")
            for tc in m.tool_calls:
                args = tc.arguments
                if len(args) > 200:
                    args = args[:200] + "..."
                parts.append(f"[tool_call {tc.name}]: {args}\n")
        elif m.role == Role.TOOL:
            content = m.content
            if len(content) > 300:
                content = content[:300] + "..."
            parts.append(f"[tool_result {m.name}]: {content}\n")
    return "\n".join(parts)


def parse_episode_result(content: str) -> dict:
    content = content.strip()
    start = content.find("{")
    if start >= 0:
        end = content.rfind("}")
        if end > start:
            content = content[start : end + 1]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


class InMemoryEpisodeStore:
    """Simple episode store for tests and tooling.

    TODO(integration): replace with a persistent implementation once the
    SDK settles on a session database.
    """

    def __init__(self) -> None:
        self.episodes: list[Episode] = []

    def create_episode(self, ep: Episode) -> None:
        self.episodes.append(ep)
