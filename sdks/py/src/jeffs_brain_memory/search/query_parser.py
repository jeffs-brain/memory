# SPDX-License-Identifier: Apache-2.0
"""Query DSL parser and FTS5 compiler.

Ported from ``go/search/query_parser.go`` and validated against
the cross-SDK goldens at ``spec/fixtures/query-parser/cases.json``.
Behaviour is bit-for-bit compatible with the TypeScript and Go SDKs:

- Parse: NFC-normalise, strip zero-width / BOM, collapse whitespace,
  tokenise into term / phrase / prefix nodes, drop stopwords when no
  operator signal is present.
- Compile: render tokens to an FTS5 ``MATCH`` expression. Default
  operator is ``OR``; explicit operators override; ``NOT`` rewrites to
  ``AND NOT`` after the first operand; a leading ``NOT`` is dropped.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Literal

from .aliases import AliasTable
from .stopwords import is_stopword

__all__ = [
    "TokenKind",
    "Token",
    "QueryAST",
    "parse",
    "compile",
    "compile_fts",
    "expand_aliases",
    "sanitise_query",
    "FTSExpr",
    "strongest_term",
]

TokenKind = Literal["term", "phrase", "prefix"]
"""The shape of a single parsed unit."""

FTSExpr = str
"""An FTS5 ``MATCH`` expression produced by :func:`compile`."""

# Characters stripped from bare terms by the FTS5 cleaner. Mirrors
# ``ftsTermReplacer`` in the Go SDK and the ``Token stripping`` block
# of ``spec/QUERY-DSL.md``. ``*`` is stripped here too; a trailing
# ``*`` is captured as a prefix marker before cleaning runs.
_FTS_TERM_STRIP = str.maketrans(
    "",
    "",
    "*():^+\"-?!.,;/\\[]{}<>|&'$#@%=~`",
)

# Characters stripped from phrase text. The only offender is the
# double quote itself; FTS5 treats the phrase body as literal.
_FTS_PHRASE_STRIP = str.maketrans("", "", '"')

# Zero-width and BOM characters removed during normalisation.
_ZW_CODEPOINTS = frozenset(("\u200b", "\u200c", "\u200d", "\ufeff"))


@dataclass(frozen=True, slots=True)
class Token:
    """One unit produced by :func:`parse`.

    ``operator`` is non-empty only when the user wrote an explicit
    uppercase boolean immediately before the token.
    """

    kind: TokenKind
    text: str
    operator: str = ""


@dataclass(slots=True)
class QueryAST:
    """Full parse result.

    ``raw`` is the NFC + whitespace-collapsed form of the input.
    ``has_operators`` records whether the input carried any quote or
    whitespace-bounded uppercase ``AND`` / ``OR`` / ``NOT``; per spec
    that flag disables stopword filtering.
    """

    raw: str
    tokens: list[Token] = field(default_factory=list)
    has_operators: bool = False


def _normalise_input(raw: str) -> str:
    """Apply the spec pre-parse normalisation.

    Unicode NFC, strip zero-width and BOM characters, map NBSP to a
    regular space, collapse runs of whitespace, trim.
    """
    if not raw:
        return ""
    nfc = unicodedata.normalize("NFC", raw)
    buf: list[str] = []
    for ch in nfc:
        if ch in _ZW_CODEPOINTS:
            continue
        if ch == "\u00a0":
            buf.append(" ")
            continue
        buf.append(ch)
    # ``str.split`` with no argument collapses any Unicode whitespace
    # run into a single split, matching Go's ``strings.Fields``.
    return " ".join("".join(buf).split())


def _has_operator_signals(raw: str) -> bool:
    """Detect whether the query carries a quote or explicit operator.

    Matches the Go ``hasOperatorSignals`` helper: any ``"`` or any
    whitespace-bounded uppercase ``AND`` / ``OR`` / ``NOT`` disables
    stopword filtering for the whole query.
    """
    if '"' in raw:
        return True
    return any(field in {"AND", "OR", "NOT"} for field in raw.split())


def _alias_token(alt: str) -> Token | None:
    """Normalise a single alias alternative into a :class:`Token`.

    Single-word alternatives become bare terms; multi-word or
    punctuation-bearing alternatives become phrase tokens (joined with
    a single space) so the FTS5 ``porter unicode61`` tokeniser still
    matches them against stored documents. ``None`` is returned when
    the alternative collapses to nothing after trimming.
    """
    alt = alt.strip().lower()
    if not alt:
        return None
    buf: list[str] = []
    for ch in alt:
        if ch.isalpha() or ch.isdigit():
            buf.append(ch)
        else:
            buf.append(" ")
    parts = "".join(buf).split()
    if not parts:
        return None
    if len(parts) == 1:
        return Token(kind="term", text=parts[0])
    return Token(kind="phrase", text=" ".join(parts))


def parse(query: str, *, aliases: AliasTable | None = None) -> QueryAST:
    """Parse ``query`` into a :class:`QueryAST`.

    ``aliases`` is consulted inline for bare terms only: phrases and
    prefixes bypass alias expansion per ``spec/QUERY-DSL.md``.
    """
    normalised = _normalise_input(query)
    ast = QueryAST(raw=normalised)
    if not normalised:
        return ast

    ast.has_operators = _has_operator_signals(normalised)
    filter_stopwords = not ast.has_operators

    chars = list(normalised)
    i = 0
    n = len(chars)
    pending_op = ""

    while i < n:
        ch = chars[i]

        if ch.isspace():
            i += 1
            continue

        if ch == '"':
            i += 1  # consume opening quote
            start = i
            while i < n and chars[i] != '"':
                i += 1
            phrase = "".join(chars[start:i])
            if i < n:
                i += 1  # consume closing quote
            phrase = phrase.translate(_FTS_PHRASE_STRIP).strip().lower()
            if phrase:
                ast.tokens.append(Token(kind="phrase", text=phrase, operator=pending_op))
                pending_op = ""
            continue

        # Bare word: run until whitespace or quote.
        start = i
        while i < n and not chars[i].isspace() and chars[i] != '"':
            i += 1
        word = "".join(chars[start:i])
        if not word:
            continue

        if word in ("AND", "OR", "NOT"):
            pending_op = word
            continue

        is_prefix = word.endswith("*")
        if is_prefix:
            word = word[:-1]

        pre_scrub_lower = word.strip().lower()

        cleaned = word.translate(_FTS_TERM_STRIP).strip()
        if not cleaned:
            pending_op = ""
            continue

        lower = cleaned.lower()

        if filter_stopwords:
            # Lowercase AND / OR / NOT are natural-language filler in
            # bare queries.
            if lower in ("and", "or", "not"):
                pending_op = ""
                continue
            if is_stopword(lower):
                pending_op = ""
                continue

        kind: TokenKind = "prefix" if is_prefix else "term"

        if kind == "term" and aliases is not None:
            expanded = aliases.expand(pre_scrub_lower)
            if len(expanded) == 1 and pre_scrub_lower != lower:
                expanded = aliases.expand(lower)
            if len(expanded) > 1:
                emitted = 0
                seen: set[tuple[str, str]] = set()
                for alt in expanded:
                    alt_tok = _alias_token(alt)
                    if alt_tok is None:
                        continue
                    dedupe_key = (alt_tok.kind, alt_tok.text)
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    if emitted == 0:
                        alt_tok = Token(
                            kind=alt_tok.kind,
                            text=alt_tok.text,
                            operator=pending_op,
                        )
                    ast.tokens.append(alt_tok)
                    emitted += 1
                if emitted > 0:
                    pending_op = ""
                    continue

        ast.tokens.append(Token(kind=kind, text=lower, operator=pending_op))
        pending_op = ""

    return ast


def _render_token(tok: Token) -> str:
    """Produce the FTS5 surface form for ``tok``.

    Empty-text tokens return ``""`` so :func:`compile` can skip them
    cleanly.
    """
    text = tok.text.strip()
    if not text:
        return ""
    if tok.kind == "phrase":
        return f'"{text}"'
    if tok.kind == "prefix":
        return f"{text}*"
    return text


def compile(ast: QueryAST) -> FTSExpr:  # noqa: A001 - parity with TS/Go API
    """Compile ``ast.tokens`` into an FTS5 ``MATCH`` expression.

    Default connective between tokens is ``OR``. Explicit operators
    on individual tokens override the default. ``NOT`` rewrites to
    ``AND NOT`` after the first operand. A leading ``NOT`` is illegal
    in FTS5 so the operator is dropped for the first token.
    """
    parts: list[str] = []
    for tok in ast.tokens:
        piece = _render_token(tok)
        if not piece:
            continue

        if not parts:
            parts.append(piece)
            continue

        op = tok.operator or "OR"
        if op == "NOT":
            op = "AND NOT"

        parts.append(op)
        parts.append(piece)

    return " ".join(parts).strip()


def compile_fts(tokens: list[Token]) -> FTSExpr:
    """Legacy entry-point accepting a raw token slice.

    Kept for symmetry with the Go ``BuildFTS5Expr`` API used by the
    retrieval layer. Prefer :func:`compile` on a :class:`QueryAST`.
    """
    return compile(QueryAST(raw="", tokens=list(tokens)))


def expand_aliases(ast: QueryAST, aliases: AliasTable) -> QueryAST:
    """Re-expand an already-parsed AST against ``aliases``.

    Most callers pass ``aliases`` directly to :func:`parse`. This
    helper exists for retrieval pipelines that receive an AST from an
    upstream source and want to attach aliases post-hoc without
    re-parsing the raw text.
    """
    out = QueryAST(raw=ast.raw, has_operators=ast.has_operators)
    for tok in ast.tokens:
        if tok.kind != "term":
            out.tokens.append(tok)
            continue
        expanded = aliases.expand(tok.text)
        if len(expanded) <= 1:
            out.tokens.append(tok)
            continue
        emitted = 0
        seen: set[tuple[str, str]] = set()
        for alt in expanded:
            alt_tok = _alias_token(alt)
            if alt_tok is None:
                continue
            dedupe_key = (alt_tok.kind, alt_tok.text)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            if emitted == 0:
                alt_tok = Token(kind=alt_tok.kind, text=alt_tok.text, operator=tok.operator)
            out.tokens.append(alt_tok)
            emitted += 1
        if emitted == 0:
            out.tokens.append(tok)
    return out


def sanitise_query(query: str) -> FTSExpr:
    """Convert a natural-language query into a valid FTS5 expression.

    Empty input or an AST with zero surviving tokens produces a light
    fallback: strip control punctuation and join the remaining words
    with a space so the caller still sees something search-worthy.
    """
    query = query.strip()
    if not query:
        return ""
    ast = parse(query)
    if not ast.tokens:
        return _fallback_sanitise(query)
    return compile(ast)


_FALLBACK_STRIP = str.maketrans("", "", "*():^+\"?!,;$#@%=")


def _fallback_sanitise(query: str) -> FTSExpr:
    stripped = query.translate(_FALLBACK_STRIP)
    words = stripped.split()
    return " ".join(words)


def strongest_term(query: str) -> str:
    """Return the longest non-stopword token of length >= 3.

    Ties break on earliest position; the first occurrence survives.
    Used by the retry ladder rung 1 in ``spec/ALGORITHMS.md``.
    """
    sanitised = query.lower()
    buf: list[str] = []
    for ch in sanitised:
        buf.append(ch if (ch.isalnum() or ch.isspace()) else " ")
    tokens = "".join(buf).split()
    best = ""
    for tok in tokens:
        if len(tok) < 3:
            continue
        if is_stopword(tok):
            continue
        if len(tok) > len(best):
            best = tok
    return best
