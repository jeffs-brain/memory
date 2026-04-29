# SPDX-License-Identifier: Apache-2.0
"""Situating-prefix builder for extracted memory facts.

``CONTEXTUAL_PREFIX_SYSTEM_PROMPT`` is ported verbatim from
``go/memory/contextualise.go``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..llm.provider import Provider
from ..llm.types import CompleteRequest
from ..llm.types import Message as LLMMessage
from ..llm.types import Role

CONTEXTUAL_PREFIX_MAX_TOKENS = 120
CONTEXTUAL_PREFIX_TEMPERATURE = 0.0
CONTEXTUAL_PREFIX_MARKER = "Context: "


CONTEXTUAL_PREFIX_SYSTEM_PROMPT = """You situate extracted memory facts inside their parent session so downstream retrieval carries the surrounding context.

Output ONE short paragraph, 50 to 100 tokens, British English. No em dashes. No lists, no headings, no preamble. Do not repeat the fact verbatim. Do not speculate. State only what the session header and the fact body already support.

Cover in order:
1. when the session happened (date / weekday) if known,
2. the broader topic or theme of the session,
3. how this specific fact sits within that session.

Start directly with the sentence. Do not prefix with "Context:" or any label."""


@dataclass(slots=True)
class ContextualiserConfig:
    provider: Provider | None = None
    model: str = ""
    cache_dir: str = ""
    concurrency: int = 0
    max_tokens: int = 0


class Contextualiser:
    """Builds short situating prefixes for extracted memory facts."""

    def __init__(self, cfg: ContextualiserConfig) -> None:
        self._cfg = cfg
        conc = cfg.concurrency if cfg.concurrency > 0 else 12
        if conc > 32:
            conc = 32
        self._sem = asyncio.Semaphore(conc) if cfg.provider is not None else None
        self._cache_lock = threading.Lock()
        self._cache: dict[str, str] = {}

    def enabled(self) -> bool:
        return self._cfg.provider is not None

    def model_name(self) -> str:
        return self._cfg.model if self.enabled() else ""

    def build_prefix(
        self, session_id: str, session_summary: str, fact_body: str
    ) -> str:
        """Synchronous wrapper that runs the LLM call on a dedicated loop."""
        if not self.enabled() or not fact_body:
            return ""

        key = self._cache_key(session_id, fact_body)

        with self._cache_lock:
            if key in self._cache:
                return self._cache[key]

        cached = self._read_cache(key)
        if cached is not None:
            with self._cache_lock:
                self._cache[key] = cached
            return cached

        try:
            prefix = asyncio.run(self._call_provider(session_summary, fact_body))
        except RuntimeError:
            # Already inside an event loop; fall back to a task.
            loop = asyncio.new_event_loop()
            try:
                prefix = loop.run_until_complete(
                    self._call_provider(session_summary, fact_body)
                )
            finally:
                loop.close()
        except Exception:
            return ""

        if prefix:
            with self._cache_lock:
                self._cache[key] = prefix
            try:
                self._write_cache(key, prefix)
            except Exception:
                pass
        return prefix

    async def build_prefix_async(
        self, session_id: str, session_summary: str, fact_body: str
    ) -> str:
        if not self.enabled() or not fact_body:
            return ""
        key = self._cache_key(session_id, fact_body)
        with self._cache_lock:
            if key in self._cache:
                return self._cache[key]
        cached = self._read_cache(key)
        if cached is not None:
            with self._cache_lock:
                self._cache[key] = cached
            return cached
        if self._sem is not None:
            async with self._sem:
                prefix = await self._call_provider(session_summary, fact_body)
        else:
            prefix = await self._call_provider(session_summary, fact_body)
        if prefix:
            with self._cache_lock:
                self._cache[key] = prefix
            try:
                self._write_cache(key, prefix)
            except Exception:
                pass
        return prefix

    async def _call_provider(self, session_summary: str, fact_body: str) -> str:
        provider = self._cfg.provider
        assert provider is not None
        body = fact_body[:4096]
        summary = session_summary.strip() or "(no session header supplied)"
        user = (
            "Session header:\n"
            + summary
            + "\n\nFact body:\n"
            + body
            + "\n"
        )
        max_tokens = self._cfg.max_tokens or CONTEXTUAL_PREFIX_MAX_TOKENS
        try:
            resp = await provider.complete(
                CompleteRequest(
                    model=self.model_name(),
                    messages=[
                        LLMMessage(
                            role=Role.SYSTEM, content=CONTEXTUAL_PREFIX_SYSTEM_PROMPT
                        ),
                        LLMMessage(role=Role.USER, content=user),
                    ],
                    max_tokens=max_tokens,
                    temperature=CONTEXTUAL_PREFIX_TEMPERATURE,
                )
            )
        except Exception:
            return ""
        prefix = resp.text.strip().replace("\r\n", "\n")
        if prefix.lower().startswith("context:"):
            prefix = prefix[len("context:") :].strip()
        return " ".join(prefix.split())

    def _cache_key(self, session_id: str, fact_body: str) -> str:
        h = hashlib.sha256()
        h.update(b"v1\n")
        h.update(f"model={self.model_name()}\n".encode("utf-8"))
        h.update(f"session={session_id}\n".encode("utf-8"))
        h.update(f"body={fact_body}".encode("utf-8"))
        return h.hexdigest()

    def _cache_filename(self, key: str) -> Path:
        return Path(self._cfg.cache_dir) / key[:2] / f"{key}.json"

    def _read_cache(self, key: str) -> Optional[str]:
        if not self._cfg.cache_dir:
            return None
        path = self._cache_filename(key)
        try:
            raw = path.read_bytes()
        except OSError:
            return None
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return None
        val = parsed.get("prefix")
        return str(val) if isinstance(val, str) else None

    def _write_cache(self, key: str, prefix: str) -> None:
        if not self._cfg.cache_dir:
            return
        path = self._cache_filename(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "prefix": prefix,
            "model": self.model_name(),
            "written": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        path.write_text(json.dumps(record), encoding="utf-8")


def apply_contextual_prefix(prefix: str, body: str) -> str:
    prefix = prefix.strip()
    if not prefix:
        return body
    return f"{CONTEXTUAL_PREFIX_MARKER}{prefix}\n\n{body}"


def new_contextualiser(cfg: ContextualiserConfig) -> Contextualiser | None:
    if cfg.provider is None:
        return None
    return Contextualiser(cfg)
