# SPDX-License-Identifier: Apache-2.0
"""Abstract provider and embedder protocols."""

from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

from .types import CompleteRequest, CompleteResponse, StreamChunk


@runtime_checkable
class Provider(Protocol):
    """Chat completion surface every backend implements."""

    async def complete(self, req: CompleteRequest) -> CompleteResponse:
        """Run a non-streaming completion."""
        ...

    async def complete_stream(self, req: CompleteRequest) -> AsyncIterator[StreamChunk]:
        """Start a streaming completion.

        Awaiting it validates the request and returns an async iterator
        that yields :class:`StreamChunk` values until generation is done
        or the caller cancels.
        """
        ...

    async def close(self) -> None:
        """Release any held resources."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Embedding model surface every backend implements."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` into float vectors."""
        ...

    def dimensions(self) -> int:
        """Return the embedding dimensionality."""
        ...

    async def close(self) -> None:
        """Release any held resources."""
        ...
