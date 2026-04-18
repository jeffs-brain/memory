# SPDX-License-Identifier: Apache-2.0
"""Stopword sets for the query parser.

Loads the canonical English and Dutch stopword lists from
``spec/fixtures/stopwords/{en,nl}.json``. Mirrors the Go and TypeScript
SDK behaviour: tokens of length two or fewer are treated as stopwords
alongside the curated lists.

See ``spec/QUERY-DSL.md`` (Stopword filtering section) for the
normative rules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

__all__ = [
    "Locale",
    "is_stopword",
    "load_stopwords",
    "STOPWORDS",
]

Locale = Literal["en", "nl", "all"]

_SPEC_DIR = Path(__file__).resolve().parents[5] / "spec"


def _fixture_path(locale: Literal["en", "nl"]) -> Path:
    """Return the canonical fixture path for ``locale``.

    The spec directory lives five levels above this module:
    ``sdks/py/src/jeffs_brain_memory/search/stopwords.py`` rooted at
    the memory repo. Callers that vendor the package outside the repo
    should pass ``path`` directly to :func:`load_stopwords`.
    """
    return _SPEC_DIR / "fixtures" / "stopwords" / f"{locale}.json"


def load_stopwords(locale: Literal["en", "nl"], *, path: Path | None = None) -> frozenset[str]:
    """Load the stopword set for ``locale``.

    ``path`` overrides the default fixture location, used by tests and
    vendored deployments. A missing file raises :class:`FileNotFoundError`
    rather than silently returning an empty set: downstream retrieval
    relies on the curated list to dampen natural-language filler.
    """
    target = path if path is not None else _fixture_path(locale)
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"stopwords fixture {target} is not a JSON array")
    return frozenset(str(tok) for tok in data if tok)


def _load_all() -> frozenset[str]:
    try:
        en = load_stopwords("en")
        nl = load_stopwords("nl")
    except FileNotFoundError:
        # Fixture missing (SDK vendored away from the repo). Callers
        # must load explicitly via :func:`load_stopwords` in that case.
        return frozenset()
    return en | nl


# Module-level combined set; mirrors the Go SDK's package-init load.
STOPWORDS: frozenset[str] = _load_all()


def is_stopword(token: str, locale: Locale = "all") -> bool:
    """Return ``True`` when ``token`` should be dropped from a bare query.

    Tokens of length two or fewer are always treated as stopwords to
    match the Go and TypeScript SDKs. Set membership is case-sensitive
    because callers pass the already-lowercased surface form.
    """
    if len(token) <= 2:
        return True
    if locale == "all":
        return token in STOPWORDS
    try:
        single = load_stopwords(locale)
    except FileNotFoundError:
        return False
    return token in single
