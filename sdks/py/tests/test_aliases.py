# SPDX-License-Identifier: Apache-2.0
"""Alias table runtime semantics tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jeffs_brain_memory.search.aliases import AliasTable, load, save


def test_empty_table_roundtrips_unknown_token() -> None:
    table = AliasTable()
    assert table.expand("unknown") == ["unknown"]
    assert len(table) == 0


def test_from_entries_lowercases_and_trims() -> None:
    table = AliasTable.from_entries(
        {
            "  BOSCH  ": ["Robert Bosch", "  bosch  ", "robert bosch"],
            "empty": [],
            "": ["nothing"],
        }
    )
    assert len(table) == 1
    assert "bosch" in table
    expansion = table.expand("bosch")
    assert expansion == ["robert bosch", "bosch"]


def test_expand_is_case_insensitive() -> None:
    table = AliasTable.from_entries({"bosch": ["bosch", "robert-bosch"]})
    assert table.expand("BOSCH") == ["bosch", "robert-bosch"]
    assert table.expand("  bosch  ") == ["bosch", "robert-bosch"]


def test_expand_returns_fresh_copy() -> None:
    """Mutating the returned list must not affect the underlying state."""
    table = AliasTable.from_entries({"bosch": ["bosch", "robert-bosch"]})
    first = table.expand("bosch")
    first.append("leaked")
    second = table.expand("bosch")
    assert "leaked" not in second


def test_expand_miss_returns_single_element_list() -> None:
    table = AliasTable.from_entries({"bosch": ["bosch"]})
    assert table.expand("lleverage") == ["lleverage"]


def test_set_upserts_and_deduplicates() -> None:
    table = AliasTable()
    table.set("foo", ["bar", "  bar  ", "baz", ""])
    assert table.expand("foo") == ["bar", "baz"]


def test_set_empty_values_removes_entry() -> None:
    table = AliasTable.from_entries({"foo": ["bar"]})
    assert "foo" in table
    table.set("foo", [])
    assert "foo" not in table


def test_load_missing_file_returns_empty_table(tmp_path: Path) -> None:
    table = load(tmp_path / "missing.json")
    assert len(table) == 0
    assert table.expand("bosch") == ["bosch"]


def test_load_empty_file_returns_empty_table(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text("   \n", encoding="utf-8")
    table = load(path)
    assert len(table) == 0


def test_load_valid_file(tmp_path: Path) -> None:
    path = tmp_path / "aliases.json"
    payload = {
        "a-ware": ["royal-aware", "royal-a-ware", "a-ware"],
        "bosch": ["bosch", "robert-bosch"],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    table = load(path)
    assert len(table) == 2
    assert table.expand("a-ware") == ["royal-aware", "royal-a-ware", "a-ware"]
    assert table.expand("bosch") == ["bosch", "robert-bosch"]


def test_load_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "wrong.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError):
        load(path)


def test_load_rejects_non_list_value(tmp_path: Path) -> None:
    path = tmp_path / "wrong.json"
    path.write_text('{"foo": "bar"}', encoding="utf-8")
    with pytest.raises(ValueError):
        load(path)


def test_save_is_stub(tmp_path: Path) -> None:
    """Persistence is reserved; :func:`save` must raise."""
    table = AliasTable.from_entries({"foo": ["bar"]})
    with pytest.raises(NotImplementedError):
        save(tmp_path / "out.json", table)


def test_aliases_not_applied_to_phrases() -> None:
    from jeffs_brain_memory.search import parse

    table = AliasTable.from_entries({"bosch": ["bosch", "robert-bosch"]})
    ast = parse('"bosch factory"', aliases=table)
    assert ast.tokens[0].kind == "phrase"
    assert ast.tokens[0].text == "bosch factory"


def test_aliases_not_applied_to_prefixes() -> None:
    from jeffs_brain_memory.search import parse

    table = AliasTable.from_entries({"bosch": ["bosch", "robert-bosch"]})
    ast = parse("bosch*", aliases=table)
    assert ast.tokens[0].kind == "prefix"
    assert ast.tokens[0].text == "bosch"


def test_alias_operator_sticks_to_first_expansion() -> None:
    """The pending operator must latch to the first emitted alternative."""
    from jeffs_brain_memory.search import parse

    table = AliasTable.from_entries({"bosch": ["bosch", "robert-bosch"]})
    ast = parse("foo AND bosch", aliases=table)
    first_with_op = next(tok for tok in ast.tokens if tok.operator == "AND")
    assert first_with_op.text == "bosch"
