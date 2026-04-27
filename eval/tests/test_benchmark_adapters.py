# SPDX-License-Identifier: Apache-2.0
"""Benchmark adapter fixture tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.base import FetchResult
from benchmarks.locomo import LoCoMoAdapter, evidence_recall, scorer_for_name
from benchmarks.memory_agent_bench import ExactContainmentScorer, MemoryAgentBenchAdapter


LOCOMO_FIXTURE = [
    {
        "sample_id": "sample_1",
        "conversation": {
            "speaker_a": "Caroline",
            "speaker_b": "Melanie",
            "session_1_date_time": "9:00 am on 1 May, 2023",
            "session_1": [
                {
                    "speaker": "Caroline",
                    "dia_id": "D1:1",
                    "text": "I went to an LGBTQ support group yesterday.",
                },
                {
                    "speaker": "Melanie",
                    "dia_id": "D1:2",
                    "text": "I painted a sunrise in 2022.",
                },
            ],
            "session_2_date_time": "10:00 am on 2 May, 2023",
            "session_2": [
                {
                    "speaker": "Caroline",
                    "dia_id": "D2:1",
                    "text": "I researched adoption agencies.",
                },
                {
                    "speaker": "Melanie",
                    "dia_id": "D2:2",
                    "text": "No one mentioned a violin lesson.",
                },
            ],
        },
        "qa": [
            {
                "question": "When did Caroline go to the LGBTQ support group?",
                "answer": "30 April 2023",
                "evidence": ["D1:1"],
                "category": 2,
            },
            {
                "question": "What did Caroline research?",
                "answer": "Adoption agencies",
                "evidence": ["D2:1"],
                "category": 1,
            },
            {
                "question": "What instrument lesson did Melanie book?",
                "answer": None,
                "evidence": [],
                "category": 5,
            },
        ],
    }
]


def _write_fixture(tmp_path: Path) -> FetchResult:
    path = tmp_path / "locomo10.json"
    payload = json.dumps(LOCOMO_FIXTURE)
    path.write_text(payload, encoding="utf-8")
    return FetchResult(local_path=path, sha256="fixture", revision="fixture")


class TestLoCoMoAdapter:
    def test_normalises_fixture_documents_and_questions(self, tmp_path: Path) -> None:
        adapter = LoCoMoAdapter()
        benchmark = adapter.normalise(_write_fixture(tmp_path), split="qa")

        assert len(benchmark.documents) == 2
        assert benchmark.documents[0].path == "locomo/sample_1/session-1.md"
        assert "benchmark: locomo" in benchmark.documents[0].content
        assert (
            "[D1:1] **Caroline**: I went to an LGBTQ support group yesterday."
            in benchmark.documents[0].content
        )
        assert benchmark.documents[0].metadata["conversation_id"] == "sample_1"

        assert len(benchmark.questions) == 3
        assert benchmark.questions[0].id == "locomo-sample_1-1"
        assert benchmark.questions[0].gold_answers == ["30 April 2023"]
        assert benchmark.questions[0].category == "2"
        assert benchmark.questions[0].evidence_ids == ["D1:1"]
        assert benchmark.questions[0].source_id == "sample_1"

    def test_limit_and_source_filter_apply_to_questions(self, tmp_path: Path) -> None:
        adapter = LoCoMoAdapter()
        benchmark = adapter.normalise(
            _write_fixture(tmp_path),
            split="qa",
            limit=1,
            source_filter="sample_1",
        )

        assert [question.id for question in benchmark.questions] == ["locomo-sample_1-1"]

    def test_sample_ids_filter_questions(self, tmp_path: Path) -> None:
        adapter = LoCoMoAdapter()
        benchmark = adapter.normalise(
            _write_fixture(tmp_path),
            split="qa",
            sample_ids=["locomo-sample_1-2"],
        )

        assert [question.id for question in benchmark.questions] == ["locomo-sample_1-2"]

    def test_rejects_unknown_split(self, tmp_path: Path) -> None:
        adapter = LoCoMoAdapter()
        with pytest.raises(ValueError, match="unsupported LoCoMo split"):
            adapter.normalise(_write_fixture(tmp_path), split="event-summary")


class TestLoCoMoScoring:
    def test_token_f1_scores_answer_overlap(self, tmp_path: Path) -> None:
        question = LoCoMoAdapter().normalise(_write_fixture(tmp_path), split="qa").questions[1]
        result = scorer_for_name("token-f1").score(
            question=question,
            answer="The adoption agencies.",
            citations=[],
        )

        assert result.passed is True
        assert result.score == pytest.approx(1.0)

    def test_adversarial_scorer_detects_abstention(self, tmp_path: Path) -> None:
        question = LoCoMoAdapter().normalise(_write_fixture(tmp_path), split="qa").questions[2]
        result = scorer_for_name("adversarial").score(
            question=question,
            answer="I don't know. It is not mentioned in the conversation.",
            citations=[],
        )

        assert result.passed is True
        assert result.score == 1.0

    def test_evidence_recall_scans_citation_text_and_path(self, tmp_path: Path) -> None:
        question = LoCoMoAdapter().normalise(_write_fixture(tmp_path), split="qa").questions[0]
        recall = evidence_recall(
            question=question,
            citations=[
                {"path": "locomo/sample_1/session-1.md", "text": "[D1:1] support group"},
                {"path": "locomo/sample_1/session-2.md#D2:1", "text": "other chunk"},
            ],
        )

        assert recall == 1.0


def _write_mab_fixture(path: Path, rows: list[dict[str, object]]) -> FetchResult:
    path.write_text(json.dumps(rows), encoding="utf-8")
    return FetchResult(local_path=path, sha256="fixture", revision="fixture")


class TestMemoryAgentBenchAdapter:
    def test_normalises_context_documents_and_questions(self, tmp_path: Path) -> None:
        fetched = _write_mab_fixture(
            tmp_path / "Accurate_Retrieval.json",
            [
                {
                    "context": (
                        "Document 1:\nAlpha lives in Bath.\n"
                        "Document 2:\nBeta moved to York."
                    ),
                    "questions": ["Where does Alpha live?", "Where did Beta move?"],
                    "answers": [["Bath"], ["York", "the city of York"]],
                    "metadata": {
                        "source": "EventQA",
                        "qa_pair_ids": ["event-1", "event-2"],
                        "question_dates": ["2024-01-01", "2024-01-02"],
                        "question_types": ["single-hop", "temporal"],
                    },
                },
                {
                    "context": "Document 1:\nLongMemEval overlap.",
                    "questions": ["Ignored?"],
                    "answers": [["yes"]],
                    "metadata": {"source": "LongMemEval", "qa_pair_ids": ["ignored"]},
                },
            ],
        )

        benchmark = MemoryAgentBenchAdapter().normalise(fetched, split="Accurate_Retrieval")

        assert benchmark.source.sha256 == fetched.sha256
        assert [document.path for document in benchmark.documents] == [
            "memory-agent-bench/Accurate_Retrieval/eventqa-1/document-1.md",
            "memory-agent-bench/Accurate_Retrieval/eventqa-1/document-2.md",
        ]
        assert benchmark.documents[0].content == "# Document 1\n\nAlpha lives in Bath.\n"
        assert [question.id for question in benchmark.questions] == ["event-1", "event-2"]
        assert benchmark.questions[1].gold_answers == ["York", "the city of York"]
        assert benchmark.questions[1].category == "temporal"
        assert benchmark.questions[1].question_date == "2024-01-02"

    def test_normalises_haystack_sessions(self, tmp_path: Path) -> None:
        fetched = _write_mab_fixture(
            tmp_path / "Conflict_Resolution.json",
            [
                {
                    "context": None,
                    "questions": ["Where does Sam work now?"],
                    "answers": [["Leeds"]],
                    "metadata": {
                        "source": "FactConsolidation",
                        "haystack_sessions": [
                            [
                                [
                                    {
                                        "role": "user",
                                        "content": "Sam works in London.",
                                        "has_answer": False,
                                    },
                                    {
                                        "role": "user",
                                        "content": "Sam now works in Leeds.",
                                        "has_answer": True,
                                    },
                                ]
                            ]
                        ],
                        "qa_pair_ids": ["conflict-1"],
                        "question_types": ["single-hop-conflict"],
                    },
                }
            ],
        )

        benchmark = MemoryAgentBenchAdapter().normalise(fetched, split="Conflict_Resolution")

        assert len(benchmark.documents) == 1
        assert "## Turn 1: user" in benchmark.documents[0].content
        assert "Sam now works in Leeds." in benchmark.documents[0].content
        assert benchmark.questions[0].id == "conflict-1"
        assert benchmark.questions[0].metadata["source"] == "FactConsolidation"

    def test_supports_limit_sample_ids_and_source_filter(self, tmp_path: Path) -> None:
        fetched = _write_mab_fixture(
            tmp_path / "Conflict_Resolution.json",
            [
                {
                    "context": "First context",
                    "questions": ["q1", "q2"],
                    "answers": [["a1"], ["a2"]],
                    "metadata": {"source": "keep", "qa_pair_ids": ["keep-1", "keep-2"]},
                },
                {
                    "context": "Second context",
                    "questions": ["q3"],
                    "answers": [["a3"]],
                    "metadata": {"source": "drop", "qa_pair_ids": ["drop-1"]},
                },
            ],
        )

        benchmark = MemoryAgentBenchAdapter().normalise(
            fetched,
            split="Conflict_Resolution",
            limit=1,
            sample_ids=["keep-2"],
            source_filter="keep",
        )

        assert [question.id for question in benchmark.questions] == ["keep-2"]
        assert len(benchmark.documents) == 1

    def test_exact_containment_scorer_matches_any_gold_answer(self, tmp_path: Path) -> None:
        fetched = _write_mab_fixture(
            tmp_path / "Conflict_Resolution.json",
            [
                {
                    "context": "Document 1:\nSam moved to Leeds.",
                    "questions": ["Where did Sam move?"],
                    "answers": [["Leeds", "the city of Leeds"]],
                    "metadata": {"source": "FactConsolidation", "qa_pair_ids": ["q1"]},
                }
            ],
        )
        question = MemoryAgentBenchAdapter().normalise(
            fetched,
            split="Conflict_Resolution",
        ).questions[0]
        scorer = ExactContainmentScorer()

        result = scorer.score(question=question, answer="Sam moved to the city of Leeds.")
        assert result.passed is True
        assert scorer.score(question=question, answer="Sam moved to London.").score == 0.0

    def test_parquet_requires_pyarrow_when_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        parquet_path = tmp_path / "Conflict_Resolution.parquet"
        parquet_path.write_bytes(b"not really parquet")
        fetched = FetchResult(local_path=parquet_path, sha256="sha", revision="fixture")

        import builtins

        original_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "pyarrow.parquet":
                raise ImportError("blocked")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(RuntimeError, match="requires pyarrow"):
            MemoryAgentBenchAdapter().normalise(fetched, split="Conflict_Resolution")
