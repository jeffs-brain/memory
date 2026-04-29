# SPDX-License-Identifier: Apache-2.0
"""Markdown chunker and compile helpers.

Ported from ``go/knowledge/compile.go``. This is the minimal
deterministic segmenter: headings and paragraph breaks define chunk
boundaries, short stubs are merged into their predecessor, and
oversized sections are split at sentence boundaries.

The richer two-phase wiki compile pipeline in
``jeff/apps/jeff/internal/knowledge/compile.go`` depends on the LLM
provider and wiki index — both still stubbed in the Python SDK — so
this port only covers chunking.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import Chunk, Document

__all__ = [
    "CHUNK_MIN_CHARS",
    "CHUNK_MAX_CHARS",
    "MIN_CHUNK_TOKENS",
    "MAX_CHUNK_TOKENS",
    "estimate_tokens",
    "segment_document",
]


# Character-length floor for a chunk. Shorter segments merge upward
# into the preceding chunk so the index never sees single-line stubs.
# Matches Go's ``defaultChunkMinChars``.
CHUNK_MIN_CHARS = 120

# Character-length cap. Longer sections split at sentence boundaries
# when possible. Matches Go's ``defaultChunkMaxChars``.
CHUNK_MAX_CHARS = 1800

# Token-equivalent floors and ceilings. The token estimator is the same
# ``chars / 4`` rule jeff uses when no tokeniser is wired in, so the
# char constants above translate to roughly 30..450 tokens per chunk
# and the ~512-token design target sits inside that envelope.
MIN_CHUNK_TOKENS = (CHUNK_MIN_CHARS + 3) // 4
MAX_CHUNK_TOKENS = (CHUNK_MAX_CHARS + 3) // 4


@dataclass(slots=True)
class _HeadingSection:
    """One logical section split off a document body."""

    heading: str
    text: str


def estimate_tokens(text: str) -> int:
    """Return a coarse ``chars // 4`` token approximation.

    Matches Go's ``estimateTokens``. Rounded up so a 1-character chunk
    still reports ``1`` token.
    """
    return (len(text) + 3) // 4


def segment_document(doc: Document) -> list[Chunk]:
    """Split a document body into chunks.

    Markdown headings (``# `` through ``###### ``) mark chunk
    boundaries; within a section, paragraphs above ``CHUNK_MAX_CHARS``
    are split at sentence boundaries. Stub chunks under
    ``CHUNK_MIN_CHARS`` are merged upward by :func:`_merge_small_chunks`.
    """

    if doc is None or not doc.body.strip():
        return []

    sections = _split_by_headings(doc.body)
    chunks: list[Chunk] = []
    ordinal = 0
    for sec in sections:
        for piece in _split_long(sec.text, CHUNK_MAX_CHARS):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    id=f"{doc.id}:{ordinal}",
                    document_id=doc.id,
                    path=doc.path,
                    ordinal=ordinal,
                    heading=sec.heading,
                    text=piece,
                    tokens=estimate_tokens(piece),
                )
            )
            ordinal += 1
    return _merge_small_chunks(chunks, CHUNK_MIN_CHARS)


def _split_by_headings(body: str) -> list[_HeadingSection]:
    """Walk ``body`` line-by-line and emit one section per heading block.

    Lines before the first heading form an initial section with an empty
    heading, matching the Go reference.
    """
    lines = body.split("\n")
    out: list[_HeadingSection] = []
    current = _HeadingSection(heading="", text="")
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf, current
        if not buf:
            return
        current.text = "\n".join(buf).strip()
        out.append(current)
        buf = []

    for line in lines:
        trim = line.strip()
        if trim.startswith("#") and _is_heading_line(trim):
            flush()
            current = _HeadingSection(heading=trim.lstrip("# ").strip(), text="")
            continue
        buf.append(line)
    flush()
    return out


def _is_heading_line(line: str) -> bool:
    """Report whether ``line`` is a markdown heading (1..6 ``#`` + space)."""
    i = 0
    while i < len(line) and line[i] == "#":
        i += 1
    if i == 0 or i > 6:
        return False
    return i < len(line) and line[i] == " "


def _split_long(text: str, max_chars: int) -> list[str]:
    """Slice ``text`` at sentence boundaries when it exceeds ``max_chars``.

    Falls back to hard slicing when no sentence boundary is reachable
    inside the search window. Mirrors the Go reference.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    out: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        cut = _find_sentence_cut(remaining, max_chars)
        if cut <= 0:
            cut = max_chars
        piece = remaining[:cut].strip()
        if piece:
            out.append(piece)
        remaining = remaining[cut:].strip()
    if remaining.strip():
        out.append(remaining.strip())
    return out


def _find_sentence_cut(text: str, max_chars: int) -> int:
    """Return the index of a sentence boundary inside the search window.

    The window runs from ``max(120, max_chars * 0.6)`` to ``max_chars``.
    Returns ``0`` when no boundary is reachable.
    """
    lower = int(max_chars * 0.6)
    if lower < 120:
        lower = 120
    if lower > len(text):
        return 0
    window = text[lower:max_chars]
    if (idx := window.find("\n\n")) >= 0:
        return lower + idx + 2
    for sep in (". ", "! ", "? "):
        idx = window.rfind(sep)
        if idx >= 0:
            return lower + idx + len(sep)
    return 0


def _merge_small_chunks(chunks: list[Chunk], min_chars: int) -> list[Chunk]:
    """Fold chunks shorter than ``min_chars`` into the previous chunk."""
    if not chunks:
        return chunks
    out: list[Chunk] = []
    for c in chunks:
        if out and len(c.text) < min_chars:
            last = out[-1]
            last.text = last.text + "\n\n" + c.text
            last.tokens = estimate_tokens(last.text)
            continue
        out.append(c)
    for i, c in enumerate(out):
        c.ordinal = i
    return out
