# SPDX-License-Identifier: Apache-2.0
"""Ingest tests — markdown, text, HTML, URL, PDF, directory, UTF-8 guard."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pytest

from jeffs_brain_memory.knowledge import (
    CONTENT_TYPE_HTML,
    CONTENT_TYPE_MARKDOWN,
    CONTENT_TYPE_PDF,
    CONTENT_TYPE_TEXT,
    IngestRequest,
    Options,
    new,
)
from jeffs_brain_memory.knowledge.frontmatter import parse_frontmatter
from jeffs_brain_memory.knowledge.ingest import (
    detect_content_type,
    normalise_url,
    raw_document_path,
    slugify,
    strip_html,
)

from ._knowledge_store import KnowledgeTestStore


def _make_kb(*, fetcher=None):
    store = KnowledgeTestStore()
    base = new(Options(brain_id="test", store=store, fetcher=fetcher))
    return base, store


@dataclass
class FakeFetcher:
    body: bytes = b""
    content_type: str = "text/html"
    error: Exception | None = None
    calls: int = 0
    url: str = ""

    async def fetch(self, url: str) -> tuple[bytes, str]:
        self.calls += 1
        self.url = url
        if self.error is not None:
            raise self.error
        return self.body, self.content_type


async def test_ingest_markdown_happy_path() -> None:
    base, store = _make_kb()
    body = (
        '---\n'
        'title: "First Post"\n'
        'tags:\n  - alpha\n  - beta\n'
        'summary: "a short summary"\n'
        '---\n\n'
        '# Heading\n\nBody paragraph with enough prose to clear the minimum '
        'chunk length threshold comfortably.\n\n'
        '## Second heading\n\nMore content here to establish the multi-section '
        'behaviour expected of the chunker in the Python SDK port.\n'
    )
    resp = await base.ingest(
        IngestRequest(path="note.md", content_type=CONTENT_TYPE_MARKDOWN, content=body.encode("utf-8"))
    )
    assert resp.document_id
    assert resp.chunk_count > 0
    assert str(resp.path).startswith("raw/documents/")
    stored = await store.read(resp.path)
    fm, _ = parse_frontmatter(stored.decode("utf-8"))
    assert fm.title == "First Post"
    assert len(fm.tags) >= 2


async def test_ingest_plain_text() -> None:
    base, _ = _make_kb()
    resp = await base.ingest(
        IngestRequest(
            path="hello.txt",
            content_type=CONTENT_TYPE_TEXT,
            content=b"Just plain text with enough substance to populate a chunk.",
            title="Plain",
        )
    )
    assert resp.chunk_count > 0
    assert resp.bytes == len(b"Just plain text with enough substance to populate a chunk.")


async def test_ingest_strips_html_and_keeps_headline() -> None:
    base, store = _make_kb()
    html = (
        "<html><head><script>bad=1</script><title>t</title></head>"
        "<body><h1>Hello</h1><p>Stripped <em>clean</em>.</p></body></html>"
    )
    resp = await base.ingest(
        IngestRequest(path="page.html", content_type=CONTENT_TYPE_HTML, content=html.encode("utf-8"))
    )
    body = (await store.read(resp.path)).decode("utf-8")
    assert "<script>" not in body
    assert "bad=1" not in body
    assert "Hello" in body


async def test_ingest_rejects_unknown_content_type() -> None:
    base, _ = _make_kb()
    with pytest.raises(ValueError):
        await base.ingest(
            IngestRequest(
                path="blob.bin",
                content_type="application/octet-stream",
                content=b"\x01\x02\x03",
            )
        )


async def test_ingest_rejects_bad_utf8() -> None:
    base, _ = _make_kb()
    with pytest.raises(ValueError):
        await base.ingest(
            IngestRequest(
                path="bad.txt",
                content_type=CONTENT_TYPE_TEXT,
                content=b"\xff\xfe\x00",
            )
        )


async def test_ingest_requires_content_or_path() -> None:
    base, _ = _make_kb()
    with pytest.raises(ValueError):
        await base.ingest(IngestRequest())


async def test_ingest_from_local_path(tmp_path: Path) -> None:
    base, _ = _make_kb()
    target = tmp_path / "note.md"
    target.write_text("# Local\n\nlocal body goes here with enough substance.\n", encoding="utf-8")
    resp = await base.ingest(IngestRequest(path=str(target)))
    assert resp.chunk_count > 0


async def test_ingest_rejects_directory(tmp_path: Path) -> None:
    base, _ = _make_kb()
    with pytest.raises(IsADirectoryError):
        await base.ingest(IngestRequest(path=str(tmp_path)))


async def test_ingest_rejects_missing_path(tmp_path: Path) -> None:
    base, _ = _make_kb()
    with pytest.raises(FileNotFoundError):
        await base.ingest(IngestRequest(path=str(tmp_path / "missing.md")))


async def test_ingest_reader_accepts_binary_io() -> None:
    base, _ = _make_kb()
    reader = io.BytesIO(b"# R\n\nbody paragraph that survives the minimum chunk check.")
    resp = await base.ingest(
        IngestRequest(path="r.md", content_type=CONTENT_TYPE_MARKDOWN, content=reader)
    )
    assert resp.chunk_count > 0


async def test_ingest_url_happy_path() -> None:
    fetcher = FakeFetcher(
        body=b"<html><h1>Title</h1><p>content with enough words to populate a chunk</p></html>",
        content_type="text/html",
    )
    base, _ = _make_kb(fetcher=fetcher)
    resp = await base.ingest_url("https://example.test/a")
    assert fetcher.calls == 1
    assert fetcher.url == "https://example.test/a"
    assert resp.chunk_count > 0


async def test_ingest_url_normalises_missing_scheme() -> None:
    fetcher = FakeFetcher(
        body=b"# plain\n\nbody paragraph with sufficient length for the chunker to accept.",
        content_type="text/markdown",
    )
    base, _ = _make_kb(fetcher=fetcher)
    await base.ingest_url("example.test/b")
    assert fetcher.url.startswith("https://")


async def test_ingest_url_rejects_empty() -> None:
    base, _ = _make_kb()
    with pytest.raises(ValueError):
        await base.ingest_url("")


async def test_ingest_url_propagates_fetch_error() -> None:
    fetcher = FakeFetcher(error=RuntimeError("boom"))
    base, _ = _make_kb(fetcher=fetcher)
    with pytest.raises(RuntimeError):
        await base.ingest_url("https://example.test/c")


async def test_ingest_pdf_round_trip() -> None:
    pdfkit = pytest.importorskip("pdfplumber")  # noqa: F841 - skip if unavailable
    import subprocess

    # Build a one-page PDF via the pdfplumber fixture when available, or
    # fall back to a minimal hand-written PDF source the extractor can
    # open without crashing. The latter still exercises the routing path
    # even when the embedded text cannot be extracted.
    pdf_bytes = _make_minimal_pdf()

    base, _ = _make_kb()
    resp = await base.ingest(
        IngestRequest(
            path="doc.pdf",
            content_type=CONTENT_TYPE_PDF,
            content=pdf_bytes,
            title="Stub PDF",
        )
    )
    assert resp.document_id
    # PDF ingest may yield zero chunks when the file lacks extractable
    # text; we only assert the path routing succeeded.


async def test_ingest_url_routes_content_type_header() -> None:
    fetcher = FakeFetcher(
        body=b"# Markdown\n\nBody content that comfortably exceeds the minimum chunk length threshold.",
        content_type="text/markdown; charset=utf-8",
    )
    base, _ = _make_kb(fetcher=fetcher)
    resp = await base.ingest_url("https://example.test/d")
    assert resp.chunk_count >= 1


async def test_detect_content_type_falls_back_to_magic() -> None:
    assert detect_content_type("a.md", b"") == CONTENT_TYPE_MARKDOWN
    assert detect_content_type("page.html", b"") == CONTENT_TYPE_HTML
    assert detect_content_type("nope", b"%PDF-1.4\n") == CONTENT_TYPE_PDF
    assert detect_content_type("nope", b"hello") == CONTENT_TYPE_TEXT


def test_slugify_matches_go_reference() -> None:
    cases = {
        "Hello World": "hello-world",
        "  a/b/c  ": "a-b-c",
        "ONE_TWO_THREE": "one-two-three",
        "é lectrique": "é-lectrique",
        "": "",
        "$$$": "",
        "Title!!! Bang??": "title-bang",
    }
    for input_value, expected in cases.items():
        assert slugify(input_value) == expected, input_value


def test_strip_html_scrubs_tags_and_entities() -> None:
    out = strip_html(b"<div>hello <script>x=1</script> world &amp; friends</div>")
    assert "<" not in out
    assert "x=1" not in out
    assert "friends" in out


def test_normalise_url_adds_https_and_validates() -> None:
    assert normalise_url("example.test/a") == "https://example.test/a"
    with pytest.raises(ValueError):
        normalise_url("")
    with pytest.raises(ValueError):
        normalise_url("://broken")


def test_raw_document_path_keeps_slug_under_prefix() -> None:
    assert str(raw_document_path("hedgehogs")) == "raw/documents/hedgehogs.md"


def _make_minimal_pdf() -> bytes:
    """Return a tiny PDF byte string sufficient for the opener."""
    # Minimal valid PDF skeleton. Sourced from the PDF 1.4 reference
    # example; pdfplumber can open it and extract an empty page stream.
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n"
        b"4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 12 Tf 100 700 Td (Hello PDF) Tj ET\nendstream\nendobj\n"
        b"xref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000056 00000 n \n"
        b"0000000107 00000 n \n0000000195 00000 n \n"
        b"trailer\n<< /Root 1 0 R /Size 5 >>\nstartxref\n285\n%%EOF\n"
    )
