# SPDX-License-Identifier: Apache-2.0
"""Tests for env-driven provider and embedder resolution."""

from __future__ import annotations

import pytest

from jeffs_brain_memory.llm import (
    CompleteRequest,
    FakeEmbedder,
    FakeProvider,
    LLMError,
    Message,
    OllamaProvider,
    OpenAIProvider,
    Role,
    embedder_from_env,
    provider_from_env,
)
from jeffs_brain_memory.llm.config import ollama_host_from_env


async def test_provider_from_env_fake() -> None:
    provider = provider_from_env({"JB_LLM_PROVIDER": "fake"})
    assert isinstance(provider, FakeProvider)
    resp = await provider.complete(
        CompleteRequest(messages=[Message(role=Role.USER, content="hi")])
    )
    assert resp.text == "ok"
    await provider.close()


async def test_provider_from_env_openai_missing_key() -> None:
    with pytest.raises(LLMError):
        provider_from_env({"JB_LLM_PROVIDER": "openai"})


async def test_provider_from_env_openai_with_base_url() -> None:
    provider = provider_from_env(
        {
            "JB_LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "secret",
            "OPENAI_BASE_URL": "https://local-gemma.example.com/v1",
            "JB_LLM_MODEL": "gemma4-31b-it",
        }
    )
    assert isinstance(provider, OpenAIProvider)
    await provider.close()


async def test_provider_from_env_unknown_value() -> None:
    with pytest.raises(LLMError):
        provider_from_env({"JB_LLM_PROVIDER": "mystery"})


async def test_provider_from_env_auto_detect_fallback() -> None:
    """With Ollama unreachable we fall back to FakeProvider."""
    provider = provider_from_env({"OLLAMA_HOST": "http://127.0.0.1:1"})
    assert isinstance(provider, FakeProvider)
    resp = await provider.complete(
        CompleteRequest(messages=[Message(role=Role.USER, content="x")])
    )
    assert resp.text == "ok"
    await provider.close()


async def test_provider_from_env_auto_detect_prefers_openai_over_ollama() -> None:
    """OPENAI_API_KEY present: pick OpenAI even if Ollama is reachable."""
    provider = provider_from_env(
        {
            "OPENAI_API_KEY": "secret",
            # Point Ollama at localhost:11434 to simulate the buggy case
            # where a reachable daemon would otherwise win.
            "OLLAMA_HOST": "http://127.0.0.1:11434",
        }
    )
    assert isinstance(provider, OpenAIProvider)
    await provider.close()


async def test_provider_from_env_auto_detect_prefers_anthropic_over_ollama() -> None:
    provider = provider_from_env(
        {
            "ANTHROPIC_API_KEY": "secret",
            "OLLAMA_HOST": "http://127.0.0.1:11434",
        }
    )
    from jeffs_brain_memory.llm import AnthropicProvider

    assert isinstance(provider, AnthropicProvider)
    await provider.close()


async def test_provider_from_env_auto_detect_prefers_openai_over_anthropic() -> None:
    """When both keys are set, OpenAI wins (matches Go SDK ordering)."""
    provider = provider_from_env(
        {
            "OPENAI_API_KEY": "openai-secret",
            "ANTHROPIC_API_KEY": "anthropic-secret",
        }
    )
    assert isinstance(provider, OpenAIProvider)
    await provider.close()


async def test_provider_from_env_ollama_explicit() -> None:
    provider = provider_from_env(
        {"JB_LLM_PROVIDER": "ollama", "OLLAMA_HOST": "http://ollama.lan:11434"}
    )
    assert isinstance(provider, OllamaProvider)
    await provider.close()


async def test_embedder_from_env_fake() -> None:
    embedder = embedder_from_env({"JB_EMBED_PROVIDER": "fake"})
    assert isinstance(embedder, FakeEmbedder)
    vectors = await embedder.embed(["hi"])
    assert len(vectors) == 1
    await embedder.close()


async def test_embedder_from_env_anthropic_unsupported() -> None:
    with pytest.raises(LLMError):
        embedder_from_env({"JB_EMBED_PROVIDER": "anthropic"})


def test_ollama_host_from_env_adds_default_port() -> None:
    # Reproduces the Go fix: OLLAMA_HOST="0.0.0.0" should default to 11434.
    assert ollama_host_from_env({"OLLAMA_HOST": "0.0.0.0"}) == "http://0.0.0.0:11434"


def test_ollama_host_from_env_preserves_explicit_port() -> None:
    assert ollama_host_from_env({"OLLAMA_HOST": "http://box.lan:9999"}) == "http://box.lan:9999"


def test_ollama_host_from_env_adds_scheme_and_port() -> None:
    assert ollama_host_from_env({"OLLAMA_HOST": "box.lan"}) == "http://box.lan:11434"


def test_ollama_host_from_env_default() -> None:
    assert ollama_host_from_env({}) == "http://localhost:11434"
