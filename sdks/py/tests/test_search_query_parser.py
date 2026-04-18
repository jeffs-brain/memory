# SPDX-License-Identifier: Apache-2.0
"""Search query parser conformance tests.

Validates the Python parser against the cross-SDK goldens at
``spec/fixtures/query-parser/cases.json``. All 15 parse cases and 4
compile cases must pass; the fixture is the normative source of
truth per ``spec/QUERY-DSL.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jeffs_brain_memory.search import (
    AliasTable,
    compile,
    compile_fts,
    expand_aliases,
    parse,
    sanitise_query,
    strongest_term,
)
from jeffs_brain_memory.search.query_parser import QueryAST, Token


def _load_fixture(spec_dir: Path) -> dict:
    fixture_path = spec_dir / "fixtures" / "query-parser" / "cases.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _token_to_dict(tok: Token) -> dict[str, str]:
    out = {"kind": tok.kind, "text": tok.text}
    if tok.operator:
        out["operator"] = tok.operator
    return out


def test_fixture_counts(spec_dir: Path) -> None:
    """Expect exactly 15 parse + 4 compile cases."""
    fixture = _load_fixture(spec_dir)
    assert len(fixture["cases"]) == 15
    assert len(fixture["compileToFTS"]) == 4


@pytest.mark.parametrize("index", range(15))
def test_golden_parse(spec_dir: Path, index: int) -> None:
    """Each golden parse case round-trips through :func:`parse`."""
    fixture = _load_fixture(spec_dir)
    case = fixture["cases"][index]
    ast = parse(case["input"])
    got = {
        "raw": ast.raw,
        "hasOperators": ast.has_operators,
        "tokens": [_token_to_dict(tok) for tok in ast.tokens],
    }
    expected = {
        "raw": case["expectedAst"]["raw"],
        "hasOperators": case["expectedAst"]["hasOperators"],
        "tokens": [dict(tok) for tok in case["expectedAst"].get("tokens", [])],
    }
    assert got == expected, f"{case['name']} mismatch"


@pytest.mark.parametrize("index", range(4))
def test_golden_compile(spec_dir: Path, index: int) -> None:
    """Each golden compile case round-trips through :func:`compile`."""
    fixture = _load_fixture(spec_dir)
    case = fixture["compileToFTS"][index]
    ast = parse(case["input"])
    got = compile(ast)
    assert got == case["expectedFTS"], f"{case['name']} mismatch"


def test_empty_input_yields_empty_ast() -> None:
    ast = parse("")
    assert ast.raw == ""
    assert ast.tokens == []
    assert ast.has_operators is False


def test_whitespace_only_input_is_empty() -> None:
    ast = parse("   \t\n  ")
    assert ast.raw == ""
    assert ast.tokens == []


def test_nfc_normalises_decomposed() -> None:
    """NFC collapses the decomposed form of ``café``."""
    ast = parse("cafe\u0301")
    assert ast.raw == "caf\u00e9"
    assert [tok.text for tok in ast.tokens] == ["caf\u00e9"]


def test_zero_width_joiners_stripped() -> None:
    ast = parse("hello\u200bworld")
    assert ast.raw == "helloworld"


def test_prefix_token_keeps_wildcard_suffix() -> None:
    ast = parse("kube* cluster")
    assert [tok.kind for tok in ast.tokens] == ["prefix", "term"]
    assert compile(ast) == "kube* OR cluster"


def test_phrase_preserves_stop_words() -> None:
    ast = parse('"the quick brown fox" jumps')
    assert ast.tokens[0].kind == "phrase"
    assert ast.tokens[0].text == "the quick brown fox"
    assert ast.has_operators is True


def test_bare_stop_words_collapse_to_empty() -> None:
    ast = parse("to do list")
    assert ast.tokens == []
    assert ast.has_operators is False


def test_explicit_operators_preserve_stop_words() -> None:
    ast = parse("foo AND the OR bar NOT baz")
    kinds = [tok.text for tok in ast.tokens]
    assert kinds == ["foo", "the", "bar", "baz"]
    assert [tok.operator for tok in ast.tokens] == ["", "AND", "OR", "NOT"]


def test_leading_not_dropped_on_compile() -> None:
    """FTS5 cannot begin with NOT; compiler strips the operator."""
    ast = parse("NOT alpha bravo")
    assert compile(ast) == "alpha OR bravo"


def test_not_rewritten_to_and_not() -> None:
    ast = parse("foo AND bar NOT baz")
    assert compile(ast) == "foo AND bar AND NOT baz"


def test_punctuation_stripped_from_bare_terms() -> None:
    ast = parse("(test)")
    assert [tok.text for tok in ast.tokens] == ["test"]


def test_sanitise_query_defaults_to_or() -> None:
    assert sanitise_query("hello world") == "hello OR world"


def test_sanitise_query_empty() -> None:
    assert sanitise_query("") == ""
    assert sanitise_query("   ") == ""


def test_sanitise_query_preserves_operators() -> None:
    assert sanitise_query("lleverage AND bosch") == "lleverage AND bosch"
    assert sanitise_query("lleverage NOT bosch") == "lleverage AND NOT bosch"


def test_sanitise_query_fallback_for_all_stopwords() -> None:
    """All-stopword input should still yield something searchable."""
    assert sanitise_query("the and or") == "the and or"


def test_compile_fts_accepts_raw_token_slice() -> None:
    tokens = [Token(kind="term", text="foo"), Token(kind="term", text="bar")]
    assert compile_fts(tokens) == "foo OR bar"


def test_compile_empty_ast_returns_empty_string() -> None:
    assert compile(QueryAST(raw="")) == ""


def test_alias_expansion_splits_phrase_alternatives() -> None:
    """Hyphenated alternatives become phrase tokens."""
    table = AliasTable.from_entries(
        {
            "a-ware": ["royal-aware", "royal-a-ware", "a-ware"],
        }
    )
    ast = parse("a-ware production", aliases=table)
    kinds = {(tok.kind, tok.text) for tok in ast.tokens}
    assert ("phrase", "royal aware") in kinds
    assert ("phrase", "royal a ware") in kinds
    assert ("phrase", "a ware") in kinds
    assert ("term", "production") in kinds


def test_alias_expansion_case_insensitive() -> None:
    table = AliasTable.from_entries({"bosch": ["bosch", "robert-bosch"]})
    ast = parse("BOSCH", aliases=table)
    texts = {(tok.kind, tok.text) for tok in ast.tokens}
    assert ("term", "bosch") in texts
    assert ("phrase", "robert bosch") in texts


def test_alias_miss_leaves_token_unchanged() -> None:
    table = AliasTable.from_entries({"bosch": ["bosch", "robert-bosch"]})
    ast = parse("lleverage", aliases=table)
    assert [(tok.kind, tok.text) for tok in ast.tokens] == [("term", "lleverage")]


def test_expand_aliases_post_parse() -> None:
    table = AliasTable.from_entries({"bosch": ["bosch", "robert-bosch"]})
    ast = parse("bosch")
    expanded = expand_aliases(ast, table)
    texts = {(tok.kind, tok.text) for tok in expanded.tokens}
    assert ("term", "bosch") in texts
    assert ("phrase", "robert bosch") in texts


def test_strongest_term_picks_longest_non_stopword() -> None:
    assert strongest_term("how do I deploy kubernetes clusters") == "kubernetes"


def test_strongest_term_returns_empty_when_all_short() -> None:
    assert strongest_term("a an of") == ""


def test_strongest_term_skips_stopwords() -> None:
    assert strongest_term("the and bosch") == "bosch"


def test_phrase_only_query_marks_operator_flag() -> None:
    ast = parse('"to do list"')
    assert ast.has_operators is True
    assert ast.tokens == [Token(kind="phrase", text="to do list")]
