# SPDX-License-Identifier: Apache-2.0
"""Environment-driven provider and embedder resolution."""

from __future__ import annotations

import os
from typing import Mapping

import httpx

from .anthropic import AnthropicProvider
from .fake import FakeEmbedder, FakeProvider
from .ollama import (
    DEFAULT_BASE_URL as OLLAMA_DEFAULT_BASE_URL,
    DEFAULT_EMBED_DIMS as OLLAMA_DEFAULT_EMBED_DIMS,
    OllamaEmbedder,
    OllamaProvider,
)
from .openai import OpenAIEmbedder, OpenAIProvider
from .provider import Embedder, Provider
from .types import LLMError

ENV_PROVIDER = "JB_LLM_PROVIDER"
ENV_MODEL = "JB_LLM_MODEL"
ENV_EMBED_PROVIDER = "JB_EMBED_PROVIDER"
ENV_EMBED_MODEL = "JB_EMBED_MODEL"
ENV_EMBED_DIMENSIONS = "JB_EMBED_DIMENSIONS"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_OPENAI_BASE_URL = "OPENAI_BASE_URL"
ENV_ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
ENV_OLLAMA_HOST = "OLLAMA_HOST"


def provider_from_env(env: Mapping[str, str] | None = None) -> Provider:
    """Resolve a :class:`Provider` from environment variables.

    If ``JB_LLM_PROVIDER`` is unset, probes a local Ollama instance and falls
    back to :class:`FakeProvider` when nothing responds.
    """
    env = env if env is not None else os.environ
    raw = (env.get(ENV_PROVIDER) or "").strip().lower()
    model = (env.get(ENV_MODEL) or "").strip()
    if raw == "openai":
        key = env.get(ENV_OPENAI_API_KEY) or ""
        if not key:
            raise LLMError(
                f"llm: {ENV_PROVIDER} is set to openai but {ENV_OPENAI_API_KEY} is empty"
            )
        return OpenAIProvider(
            api_key=key,
            base_url=env.get(ENV_OPENAI_BASE_URL) or None,
            model=model or None,
        )
    if raw == "anthropic":
        key = env.get(ENV_ANTHROPIC_API_KEY) or ""
        if not key:
            raise LLMError(
                f"llm: {ENV_PROVIDER} is set to anthropic but {ENV_ANTHROPIC_API_KEY} is empty"
            )
        return AnthropicProvider(api_key=key, model=model or None)
    if raw == "ollama":
        return OllamaProvider(base_url=ollama_host_from_env(env), model=model or None)
    if raw == "fake":
        return FakeProvider(["ok"])
    if raw:
        raise LLMError(f"llm: unknown {ENV_PROVIDER}={raw!r}")
    host = ollama_host_from_env(env)
    if ollama_reachable(host):
        return OllamaProvider(base_url=host)
    return FakeProvider(["ok"])


def embedder_from_env(env: Mapping[str, str] | None = None) -> Embedder:
    """Resolve an :class:`Embedder` from environment variables."""
    env = env if env is not None else os.environ
    provider = (env.get(ENV_EMBED_PROVIDER) or "").strip().lower()
    if not provider:
        provider = (env.get(ENV_PROVIDER) or "").strip().lower()
    model = (env.get(ENV_EMBED_MODEL) or "").strip()
    if provider == "openai":
        key = env.get(ENV_OPENAI_API_KEY) or ""
        if not key:
            raise LLMError(
                f"llm: {ENV_EMBED_PROVIDER} is set to openai but {ENV_OPENAI_API_KEY} is empty"
            )
        return OpenAIEmbedder(
            api_key=key,
            base_url=env.get(ENV_OPENAI_BASE_URL) or None,
            model=model or None,
        )
    if provider == "ollama":
        return OllamaEmbedder(
            base_url=ollama_host_from_env(env),
            model=model or None,
        )
    if provider == "fake":
        return FakeEmbedder(OLLAMA_DEFAULT_EMBED_DIMS)
    if provider == "anthropic":
        raise LLMError("llm: anthropic does not expose an embedding endpoint")
    if provider:
        raise LLMError(f"llm: unknown {ENV_EMBED_PROVIDER}={provider!r}")
    host = ollama_host_from_env(env)
    if ollama_reachable(host):
        return OllamaEmbedder(base_url=host)
    return FakeEmbedder(OLLAMA_DEFAULT_EMBED_DIMS)


def ollama_host_from_env(env: Mapping[str, str]) -> str:
    """Return the Ollama base URL from env, applying the default-port fallback."""
    host = (env.get(ENV_OLLAMA_HOST) or "").strip()
    if not host:
        return OLLAMA_DEFAULT_BASE_URL
    if not (host.startswith("http://") or host.startswith("https://")):
        host = "http://" + host
    rest = host
    for prefix in ("http://", "https://"):
        if rest.startswith(prefix):
            rest = rest[len(prefix):]
            break
    # OLLAMA_HOST frequently omits the port (e.g. "0.0.0.0"). Default to 11434.
    if ":" not in rest:
        host = host + ":11434"
    return host


def ollama_reachable(base: str) -> bool:
    """Issue a quick HEAD against ``base`` to decide if Ollama is up."""
    if not base:
        return False
    try:
        resp = httpx.head(f"{base}/", timeout=0.25)
    except httpx.HTTPError:
        return False
    return resp.status_code < 500
