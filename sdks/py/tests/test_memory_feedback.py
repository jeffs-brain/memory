# SPDX-License-Identifier: Apache-2.0
"""Feedback classifier."""

from __future__ import annotations

import math

from jeffs_brain_memory.memory.feedback import Classifier, Reaction


def test_positive_signal():
    c = Classifier()
    result = c.classify("perfect, thanks for that", ["memory/global/t.md"])
    assert len(result.events) == 1
    assert result.events[0].reaction == Reaction.REINFORCED
    assert result.events[0].confidence > 0


def test_negative_signal():
    c = Classifier()
    result = c.classify("that's wrong, it was updated last week", ["memory/global/t.md"])
    assert result.events[0].reaction == Reaction.CORRECTED


def test_neutral_when_no_patterns():
    c = Classifier()
    r = c.classify("can you show me the deployment logs", ["memory/global/t.md"])
    assert r.events[0].reaction == Reaction.NEUTRAL
    assert r.events[0].confidence == 0.0


def test_positive_wins_on_more_matches():
    c = Classifier()
    r = c.classify("no but perfect, that's helpful, spot on", ["memory/global/t.md"])
    assert r.events[0].reaction == Reaction.REINFORCED


def test_negative_wins_on_more_matches():
    c = Classifier()
    r = c.classify(
        "yes but that's wrong, incorrect, try again", ["memory/global/t.md"]
    )
    assert r.events[0].reaction == Reaction.CORRECTED


def test_tie_goes_to_neutral():
    c = Classifier()
    r = c.classify("yes and no", ["memory/global/t.md"])
    assert r.events[0].reaction == Reaction.NEUTRAL
    assert r.events[0].confidence == 0.2


def test_empty_input_produces_no_events():
    c = Classifier()
    r = c.classify("", ["memory/global/t.md"])
    assert r.events == []


def test_empty_paths_produces_no_events():
    c = Classifier()
    r = c.classify("perfect", [])
    assert r.events == []


def test_multiple_paths_per_event():
    c = Classifier()
    paths = [
        "memory/global/a.md",
        "memory/global/b.md",
        "memory/project/x/c.md",
    ]
    r = c.classify("great, thanks", paths)
    assert len(r.events) == 3
    assert all(ev.reaction == Reaction.REINFORCED for ev in r.events)


def test_confidence_clamped():
    c = Classifier()
    r1 = c.classify("thanks", ["m.md"])
    r2 = c.classify(
        "thanks, that's helpful, you remembered, that helps, spot on",
        ["m.md"],
    )
    assert r1.events[0].confidence < r2.events[0].confidence
    assert r2.events[0].confidence <= 1.0
    assert math.isclose(r1.events[0].confidence, 0.3, abs_tol=0.001)


def test_snippet_truncation():
    c = Classifier()
    long_input = "perfect " + "x" * 300
    r = c.classify(long_input, ["m.md"])
    assert len(r.events[0].snippet) <= 203
    assert r.events[0].snippet.endswith("...")
    assert len(r.turn_content) <= 503
