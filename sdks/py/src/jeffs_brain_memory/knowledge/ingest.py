# SPDX-License-Identifier: Apache-2.0
"""Ingest routing — markdown, plain text, HTML, PDF, and URLs.

Ported from ``go/knowledge/ingest.go``. Every ingested document
lands under ``raw/documents/<slug>.md`` with a regenerated YAML
frontmatter header so downstream search indexes see a canonical shape.
"""

from __future__ import annotations

import hashlib
import io
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import BinaryIO, Protocol, runtime_checkable
from urllib.parse import urlparse

import httpx

from ..path import BrainPath, DocumentID
from .frontmatter import Frontmatter, parse_frontmatter
from .types import Document, IngestRequest

__all__ = [
    "CONTENT_TYPE_MARKDOWN",
    "CONTENT_TYPE_TEXT",
    "CONTENT_TYPE_HTML",
    "CONTENT_TYPE_PDF",
    "CONTENT_TYPE_JSON",
    "CONTENT_TYPE_YAML",
    "MAX_READ_BYTES",
    "DEFAULT_HTTP_TIMEOUT",
    "RAW_DOCUMENTS_PREFIX",
    "Fetcher",
    "DefaultFetcher",
    "build_document",
    "build_frontmatter_yaml",
    "detect_content_type",
    "extract_plain",
    "normalise_url",
    "raw_document_path",
    "slugify",
    "strip_html",
]


# Content-type constants for the raw formats we accept. Matches the Go
# reference so cross-SDK callers can compare against the same strings.
CONTENT_TYPE_MARKDOWN = "text/markdown"
CONTENT_TYPE_TEXT = "text/plain"
CONTENT_TYPE_HTML = "text/html"
CONTENT_TYPE_PDF = "application/pdf"
CONTENT_TYPE_JSON = "application/json"
CONTENT_TYPE_YAML = "application/x-yaml"

# Cap for any single source body. Matches jeff's conservative ceiling so
# an oversized URL or file cannot exhaust memory.
MAX_READ_BYTES = 50 * 1024 * 1024

# Per-request wall clock for :class:`DefaultFetcher`. Matches Go.
DEFAULT_HTTP_TIMEOUT = 30.0

# Logical prefix for every ingest. Mirrors the Go ``RawDocumentsPrefix``
# helper so search indexes that walk ``raw/documents/`` pick up
# everything this module writes.
RAW_DOCUMENTS_PREFIX = "raw/documents"

# Precompiled regex patterns for the HTML stripper. The script/style
# patterns run first so neither JavaScript nor stylesheet text ends up in
# the extracted body. Python supports backreferences but we keep the
# trio of patterns here to match the Go RE2 implementation line for line.
_SCRIPT_RE = re.compile(r"<script[^>]*>.*?</\s*script\s*>", re.IGNORECASE | re.DOTALL)
_STYLE_RE = re.compile(r"<style[^>]*>.*?</\s*style\s*>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>", re.DOTALL)


@runtime_checkable
class Fetcher(Protocol):
    """Abstract HTTP fetcher so :func:`ingest_url` stays testable."""

    async def fetch(self, url: str) -> tuple[bytes, str]:
        """Return the response body and content-type for ``url``."""
        ...


@dataclass(slots=True)
class DefaultFetcher:
    """Production HTTP fetcher backed by :mod:`httpx`."""

    timeout: float = DEFAULT_HTTP_TIMEOUT

    async def fetch(self, url: str) -> tuple[bytes, str]:
        headers = {
            "Accept": "text/plain, text/markdown, text/html, application/pdf",
        }
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
        if resp.status_code < 200 or resp.status_code >= 300:
            raise ValueError(f"knowledge: fetch {url}: HTTP {resp.status_code}")
        body = resp.content[:MAX_READ_BYTES]
        ctype = resp.headers.get("content-type", "")
        return body, ctype


def raw_document_path(slug: str) -> BrainPath:
    """Return the ``raw/documents/<slug>.md`` logical path."""
    if not slug:
        slug = "untitled"
    return BrainPath(f"{RAW_DOCUMENTS_PREFIX}/{slug}.md")


def detect_content_type(path: str, data: bytes) -> str:
    """Pick a content-type from extension first, magic bytes second."""
    ext = _extension(path).lower()
    ext_map = {
        ".md": CONTENT_TYPE_MARKDOWN,
        ".markdown": CONTENT_TYPE_MARKDOWN,
        ".txt": CONTENT_TYPE_TEXT,
        ".text": CONTENT_TYPE_TEXT,
        ".log": CONTENT_TYPE_TEXT,
        ".html": CONTENT_TYPE_HTML,
        ".htm": CONTENT_TYPE_HTML,
        ".pdf": CONTENT_TYPE_PDF,
        ".json": CONTENT_TYPE_JSON,
        ".yaml": CONTENT_TYPE_YAML,
        ".yml": CONTENT_TYPE_YAML,
    }
    if ext in ext_map:
        return ext_map[ext]
    if len(data) > 4 and data[:4] == b"%PDF":
        return CONTENT_TYPE_PDF
    if data.startswith(b"<") and b"<html" in data[:1024].lower():
        return CONTENT_TYPE_HTML
    if _looks_like_text(data):
        return CONTENT_TYPE_TEXT
    return "application/octet-stream"


def _looks_like_text(data: bytes) -> bool:
    """Rough text-sniffer that treats high-entropy blobs as binary."""
    if not data:
        return True
    sample = data[:512]
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def extract_plain(raw: bytes, content_type: str, extension: str) -> str:
    """Route ``raw`` to the right extractor and return plain text.

    Markdown, text, JSON, YAML round-trip as decoded UTF-8. HTML is
    stripped. PDF is decoded via ``pdfplumber``. Unknown types raise.
    """
    base = _media_type(content_type)
    if base in (
        CONTENT_TYPE_MARKDOWN,
        CONTENT_TYPE_TEXT,
        CONTENT_TYPE_JSON,
        CONTENT_TYPE_YAML,
    ):
        return _decode_utf8(raw)
    if base == CONTENT_TYPE_HTML:
        return strip_html(raw)
    if base == CONTENT_TYPE_PDF:
        return _extract_pdf(raw)
    if extension == ".pdf" or (len(raw) > 4 and raw[:4] == b"%PDF"):
        return _extract_pdf(raw)
    if base.startswith("text/"):
        return _decode_utf8(raw)
    raise ValueError(f"knowledge: unsupported content-type {content_type!r}")


def _decode_utf8(raw: bytes) -> str:
    """Decode ``raw`` as UTF-8 or raise a descriptive error."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("knowledge: content is not valid UTF-8") from exc


def strip_html(raw: bytes) -> str:
    """Drop script/style blocks, tags, and common entities from ``raw``."""
    try:
        s = raw.decode("utf-8", errors="ignore")
    except Exception:  # pragma: no cover - decode(errors=ignore) never raises
        s = raw.decode("latin-1", errors="ignore")
    s = _SCRIPT_RE.sub(" ", s)
    s = _STYLE_RE.sub(" ", s)
    s = _TAG_RE.sub(" ", s)
    entities = {
        "&nbsp;": " ",
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#39;": "'",
        "&apos;": "'",
    }
    for entity, replacement in entities.items():
        s = s.replace(entity, replacement)
    return _collapse_whitespace(s)


def _extract_pdf(raw: bytes) -> str:
    """Return the concatenated plain text of every PDF page."""
    if not raw:
        raise ValueError("knowledge: empty pdf body")
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError("knowledge: pdfplumber not installed") from exc

    pieces: list[str] = []
    try:
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages:
                try:
                    text = page.extract_text() or ""
                except Exception:  # noqa: BLE001 - PDF extractors trip on fonts
                    continue
                text = text.strip()
                if text:
                    pieces.append(text)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"knowledge: opening pdf: {exc}") from exc
    return _collapse_whitespace("\n\n".join(pieces))


def build_document(
    req: IngestRequest,
    content_type: str,
    source_label: str,
    extracted: str,
    raw: bytes,
    brain_id: str = "",
) -> Document:
    """Assemble a :class:`Document` from a request plus extracted text."""

    now = datetime.now(timezone.utc)
    title = (req.title or "").strip()
    tags = list(req.tags)

    fm, body = parse_frontmatter(extracted)
    if not title:
        title = fm.title or fm.name or _derive_title(body, source_label)
    summary = fm.summary or fm.description
    if fm.tags:
        tags.extend(fm.tags)

    slug = slugify(title)
    if not slug:
        slug = _hash_slug(raw)

    doc_id = _hash_slug((slug + ":").encode("utf-8") + raw)
    path = raw_document_path(slug)

    return Document(
        id=DocumentID(doc_id),
        brain_id=brain_id or req.brain_id,
        path=path,
        title=title,
        source=source_label,
        content_type=content_type,
        summary=summary,
        body=body.strip(),
        bytes=len(raw),
        tags=_dedupe_strings(tags),
        ingested=now,
        modified=now,
    )


def build_frontmatter_yaml(doc: Document) -> str:
    """Emit the canonical YAML frontmatter block written with each ingest."""
    lines = ["---"]
    lines.append(f'title: {_quote_yaml(doc.title)}')
    if doc.summary:
        lines.append(f'summary: {_quote_yaml(doc.summary)}')
    lines.append(f'source: {_quote_yaml(doc.source)}')
    lines.append(f'source_type: {_quote_yaml(_route_source_type(doc.content_type))}')
    if doc.ingested is not None:
        lines.append(f'ingested: {_quote_yaml(_fmt_rfc3339(doc.ingested))}')
    if doc.modified is not None:
        lines.append(f'modified: {_quote_yaml(_fmt_rfc3339(doc.modified))}')
    if doc.tags:
        lines.append("tags:")
        for tag in doc.tags:
            lines.append(f"  - {tag}")
    lines.append("---")
    return "\n".join(lines)


def normalise_url(raw: str) -> str:
    """Trim, default-to-HTTPS, and validate a URL."""
    raw = raw.strip()
    if not raw:
        raise ValueError("knowledge: empty URL")
    if "://" not in raw:
        raw = "https://" + raw
    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("knowledge: URL missing host")
    return parsed.geturl()


# --- helpers ---------------------------------------------------------------


def slugify(title: str) -> str:
    """Lowercase, hyphen-separated, 60-char cap slug.

    Matches Go's ``slugify``: Unicode letters and digits survive;
    everything else becomes a hyphen. Runs of hyphens collapse to one.
    """
    s = title.strip().lower()
    pieces: list[str] = []
    for ch in s:
        if _is_letter(ch) or ch.isdigit():
            pieces.append(ch)
        else:
            pieces.append("-")
    slug = "".join(pieces)
    # Collapse consecutive hyphens.
    while "--" in slug:
        slug = slug.replace("--", "-")
    slug = slug.strip("-")
    if len(slug) > 60:
        slug = slug[:60].rstrip("-")
    return slug


def _is_letter(ch: str) -> bool:
    """Report whether ``ch`` is a Unicode letter (matches Go's unicode.IsLetter)."""
    if not ch:
        return False
    return unicodedata.category(ch).startswith("L")


def _hash_slug(data: bytes) -> str:
    """Deterministic 12-char hex fallback slug."""
    return hashlib.sha256(data).hexdigest()[:12]


def _derive_title(body: str, fallback: str) -> str:
    """Pick a best-effort title from ``body`` when the caller gives none."""
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped:
            if len(stripped) > 120:
                stripped = stripped[:120]
            return stripped
    base = _basename(fallback)
    base = _strip_extension(base)
    return base or "untitled"


def _basename(p: str) -> str:
    """POSIX-style basename that also handles Windows separators."""
    for sep in ("/", "\\"):
        idx = p.rfind(sep)
        if idx >= 0:
            p = p[idx + 1 :]
    return p


def _strip_extension(p: str) -> str:
    """Trim the rightmost extension, if any."""
    idx = p.rfind(".")
    if idx > 0:
        return p[:idx]
    return p


def _extension(p: str) -> str:
    """Return the rightmost extension including the dot, lowercased."""
    idx = p.rfind(".")
    if idx < 0:
        return ""
    return p[idx:].lower()


def _collapse_whitespace(s: str) -> str:
    """Trim and collapse whitespace runs while preserving paragraph breaks."""
    paragraphs = s.split("\n\n")
    for i, p in enumerate(paragraphs):
        paragraphs[i] = " ".join(p.split())
    return "\n\n".join(paragraphs).strip()


def _media_type(raw: str) -> str:
    """Normalise ``text/html; charset=utf-8`` to ``text/html``."""
    base = raw.strip().lower()
    idx = base.find(";")
    if idx >= 0:
        base = base[:idx].strip()
    return base


def _dedupe_strings(items: list[str]) -> list[str]:
    """Remove blanks and duplicates while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for s in items:
        key = s.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _route_source_type(content_type: str) -> str:
    """Map a content-type onto the frontmatter source_type tag."""
    mapping = {
        CONTENT_TYPE_MARKDOWN: "markdown",
        CONTENT_TYPE_TEXT: "text",
        CONTENT_TYPE_HTML: "html",
        CONTENT_TYPE_PDF: "pdf",
        CONTENT_TYPE_JSON: "json",
        CONTENT_TYPE_YAML: "yaml",
    }
    return mapping.get(content_type, "document")


def _quote_yaml(value: str) -> str:
    """Double-quote ``value`` and escape embedded quotes/backslashes."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _fmt_rfc3339(when: datetime) -> str:
    """Format a ``datetime`` as RFC 3339 with second precision."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    # Go's time.RFC3339 uses ``2006-01-02T15:04:05Z07:00``. Python's
    # ``isoformat`` matches when we trim microseconds and swap UTC to
    # the ``Z`` shorthand for clarity.
    iso = when.replace(microsecond=0).isoformat()
    if iso.endswith("+00:00"):
        iso = iso[:-6] + "Z"
    return iso


def first_non_empty(*values: str) -> str:
    """Return the first non-blank argument or an empty string."""
    for v in values:
        if v and v.strip():
            return v
    return ""


def read_body(req: IngestRequest) -> tuple[bytes, str, str]:
    """Load the request body and return ``(bytes, content_type, source)``.

    Precedence matches Go: inline ``content`` beats ``path``; when only
    ``path`` is set the file is opened from disk. Returns
    ``content_type`` as the base media type (charset stripped).
    """

    ctype = _media_type(req.content_type)

    if req.content is not None:
        data = _read_reader(req.content)
        if not ctype:
            ctype = detect_content_type(req.path, data)
        return data, ctype, first_non_empty(req.path, "inline")

    path = (req.path or "").strip()
    if not path:
        raise ValueError("knowledge: either content or path required")
    if "://" in path:
        raise ValueError(f"knowledge: use ingest_url for {path}")

    import os

    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"knowledge: stat {abs_path}: not found")
    if os.path.isdir(abs_path):
        raise IsADirectoryError(f"knowledge: {abs_path} is a directory")
    size = os.path.getsize(abs_path)
    if size > MAX_READ_BYTES:
        raise ValueError(f"knowledge: {abs_path} exceeds {MAX_READ_BYTES} byte limit")
    with open(abs_path, "rb") as handle:
        data = handle.read()
    if not ctype:
        ctype = detect_content_type(abs_path, data)
    return data, ctype, abs_path


def _read_reader(source: bytes | BinaryIO) -> bytes:
    """Read at most :data:`MAX_READ_BYTES` from ``source``."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        data = bytes(source)
        if len(data) > MAX_READ_BYTES:
            return data[:MAX_READ_BYTES]
        return data
    if hasattr(source, "read"):
        # Accept ``str`` or ``bytes`` readers; encode text defensively.
        raw = source.read(MAX_READ_BYTES + 1)
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
        if raw is None:
            raw = b""
        if len(raw) > MAX_READ_BYTES:
            raw = raw[:MAX_READ_BYTES]
        return raw
    raise TypeError(f"knowledge: unsupported content type {type(source)!r}")


# Expose the Frontmatter type via a re-export for callers that want a
# single import surface from ``knowledge.ingest``. Keeps the Go ergonomics.
__all__.append("Frontmatter")
