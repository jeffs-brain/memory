# SPDX-License-Identifier: Apache-2.0
"""Heuristic module: filename gen, confidence, Jaccard, merge."""

from __future__ import annotations

from jeffs_brain_memory.memory import (
    Heuristic,
    apply_heuristics,
    build_heuristic_content,
    confidence_from_observations,
    has_tag,
    heuristic_filename,
    jaccard_similarity,
    memory_project_topic,
    merge_heuristic,
    significant_words,
    MemoryManager,
    MemStore,
)
from jeffs_brain_memory.memory.heuristic import extract_alternative


def test_heuristic_filename_from_category():
    h = Heuristic(rule="Always run Go tests before committing", category="testing")
    assert heuristic_filename(h) == "heuristic-testing-always-run.md"


def test_heuristic_filename_anti():
    h = Heuristic(
        rule="Using SQL string concatenation for queries",
        category="debugging",
        anti_pattern=True,
    )
    out = heuristic_filename(h)
    assert out.startswith("heuristic-anti-")
    assert out.endswith(".md")


def test_confidence_from_observations_thresholds():
    assert confidence_from_observations(1) == "low"
    assert confidence_from_observations(2) == "medium"
    assert confidence_from_observations(4) == "high"


def test_jaccard_full_overlap():
    assert jaccard_similarity(["a", "b"], ["a", "b"]) == 1.0


def test_jaccard_no_overlap():
    assert jaccard_similarity(["a"], ["b"]) == 0.0


def test_jaccard_empty_both():
    assert jaccard_similarity([], []) == 1.0


def test_significant_words_filter_stopwords():
    out = significant_words("The quick brown fox and the lazy dog")
    assert "the" not in out
    assert "and" not in out


def test_has_tag_case_insensitive():
    assert has_tag(["Heuristic", "testing"], "heuristic")
    assert not has_tag(["project"], "heuristic")


def test_build_heuristic_content_fields():
    h = Heuristic(rule="Do X", category="testing", confidence="low")
    body = build_heuristic_content(h)
    assert "type: feedback" in body
    assert "confidence: low" in body
    assert "heuristic" in body


def test_build_heuristic_anti_pattern_marker():
    h = Heuristic(
        rule="Use globals", category="architecture", confidence="high", anti_pattern=True
    )
    body = build_heuristic_content(h)
    assert "- anti-pattern" in body
    assert "Don't:" in body


def test_merge_heuristic_increments_observations():
    base = build_heuristic_content(
        Heuristic(rule="Do X", category="testing", confidence="low")
    )
    out = merge_heuristic(base, Heuristic(rule="Do X", category="testing", confidence="low"))
    assert "2 observations" in out
    assert "confidence: medium" in out


def test_extract_alternative_marker():
    assert extract_alternative("use globals instead use locals") != ""


def test_apply_heuristics_writes_new_file():
    store = MemStore()
    mem = MemoryManager(store)
    h = Heuristic(
        rule="Always test stuff", category="testing", confidence="low", scope="project"
    )
    apply_heuristics(mem, "slug", [h])
    # find the file we just wrote
    paths = [p for p in store._docs if p.startswith("memory/project/slug/heuristic")]
    assert len(paths) == 1
    body = store.read(paths[0]).decode("utf-8")
    assert "heuristic" in body
