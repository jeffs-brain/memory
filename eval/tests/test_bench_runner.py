# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

import bench_runner

REAL_ASYNC_CLIENT = httpx.AsyncClient


@dataclass
class BenchmarkSource:
    benchmark: str
    url: str
    revision: str | None = None
    sha256: str | None = None
    licence: str | None = None


@dataclass
class FetchResult:
    local_path: Path
    sha256: str
    revision: str | None = None


@dataclass
class CorpusDocument:
    path: str
    content: str
    content_type: str = "text/markdown"
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class EvalQuestion:
    id: str
    question: str
    gold_answers: list[str]
    category: str
    question_date: str | None = None
    evidence_ids: list[str] = field(default_factory=list)
    source_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class NormalisedBenchmark:
    source: BenchmarkSource
    documents: list[CorpusDocument]
    questions: list[EvalQuestion]


@dataclass
class ScorerResult:
    score: float
    passed: bool
    evidence_recall: float | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionResult:
    id: str
    question: str
    answer: str
    score: float
    passed: bool
    latency_ms: float
    evidence_recall: float | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class CategoryScore:
    total: int
    passed: int
    score: float
    confidence_interval_95: tuple[float, float] | None = None


@dataclass
class BenchmarkManifest:
    benchmark: str
    split: str | None
    source_url: str
    source_revision: str | None
    source_sha256: str
    adapter_version: str
    sdk: str
    scenario: str
    scorer: str
    retrieval_mode: str
    top_k: int
    candidate_k: int
    rerank_top_n: int
    brain_id: str
    sample_signature: str | None
    sample_size: int
    source_filter: str | None
    judge_model: str | None
    budget_usd: float | None
    run_seed: int | None
    started_at: str
    finished_at: str


@dataclass
class BenchmarkResult:
    manifest: BenchmarkManifest
    total: int
    passed: int
    pass_rate: float
    mean_score: float
    latency_p50_ms: float
    latency_p95_ms: float
    confidence_interval_95: tuple[float, float] | None = None
    per_category: dict[str, CategoryScore] = field(default_factory=dict)
    cost_usd: float = 0.0
    questions: list[QuestionResult] = field(default_factory=list)


class FakeScorer:
    name = "token-f1"

    def score(
        self,
        *,
        question: EvalQuestion,
        answer: str,
        citations: list[dict[str, Any]],
    ) -> ScorerResult:
        expected = question.gold_answers[0]
        passed = expected in answer
        return ScorerResult(score=1.0 if passed else 0.0, passed=passed)


class FakeAdapter:
    id = "fake"
    source = BenchmarkSource(
        benchmark="fake",
        url="https://example.test/fake.json",
        revision="abc123",
        sha256="source-sha",
    )

    def __init__(self) -> None:
        self.normalise_kwargs: dict[str, Any] = {}

    def fetch(self, cache_dir: Path) -> FetchResult:
        return FetchResult(
            local_path=cache_dir / "fake.json",
            sha256="fetched-sha",
            revision="abc123",
        )

    def normalise(
        self,
        fetched: FetchResult,
        *,
        split: str | None = None,
        limit: int | None = None,
        sample_ids: list[str] | None = None,
        source_filter: str | None = None,
    ) -> NormalisedBenchmark:
        self.normalise_kwargs = {
            "split": split,
            "limit": limit,
            "sample_ids": sample_ids,
            "source_filter": source_filter,
        }
        return NormalisedBenchmark(
            source=self.source,
            documents=[
                CorpusDocument(
                    path="fake/a.md",
                    content="Alpha [e1]",
                    metadata={"conversation": "a"},
                ),
                CorpusDocument(path="fake/b.md", content="Beta [e2]"),
            ],
            questions=[
                EvalQuestion(
                    id="q1",
                    question="alpha?",
                    gold_answers=["Alpha"],
                    category="single",
                    evidence_ids=["e1"],
                ),
                EvalQuestion(
                    id="q2",
                    question="beta?",
                    gold_answers=["Beta"],
                    category="multi",
                    evidence_ids=["e2"],
                ),
            ],
        )

    def default_scorer(self) -> FakeScorer:
        return FakeScorer()

    def scorer_for(self, name: str) -> FakeScorer:
        assert name == "token-f1"
        return FakeScorer()

    def available_scorers(self) -> list[str]:
        return ["token-f1"]


@pytest.fixture
def fake_benchmark_modules(monkeypatch: pytest.MonkeyPatch) -> FakeAdapter:
    adapter = FakeAdapter()
    registry = types.ModuleType("benchmarks")
    registry.get_adapter = lambda name: adapter

    base = types.ModuleType("benchmarks.base")
    for cls in (
        BenchmarkSource,
        FetchResult,
        CorpusDocument,
        EvalQuestion,
        NormalisedBenchmark,
        ScorerResult,
        QuestionResult,
        CategoryScore,
        BenchmarkManifest,
        BenchmarkResult,
    ):
        setattr(base, cls.__name__, cls)

    stats = types.ModuleType("benchmarks.stats")
    stats.latency_percentile = lambda values, percentile: sorted(values)[0] if values else 0.0
    stats.bootstrap_ci = lambda values: (min(values), max(values)) if values else None

    scoring = types.ModuleType("benchmarks.scoring")
    scoring.TokenF1Scorer = FakeScorer

    monkeypatch.setitem(sys.modules, "benchmarks", registry)
    monkeypatch.setitem(sys.modules, "benchmarks.base", base)
    monkeypatch.setitem(sys.modules, "benchmarks.stats", stats)
    monkeypatch.setitem(sys.modules, "benchmarks.scoring", scoring)
    return adapter


def test_endpoint_run_prepares_brain_runs_concurrently_and_writes_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_benchmark_modules: FakeAdapter,
) -> None:
    sample_ids = tmp_path / "sample-ids.txt"
    sample_ids.write_text("q1\nq2\n", encoding="utf-8")
    calls: list[tuple[str, str, dict[str, Any] | None]] = []
    active_searches = 0
    max_active_searches = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active_searches, max_active_searches
        payload = json.loads(request.content.decode("utf-8")) if request.content else None
        calls.append((request.method, request.url.path, payload))
        if request.method == "DELETE" and request.url.path == "/v1/brains/bench-brain":
            return httpx.Response(404)
        if request.method == "POST" and request.url.path == "/v1/brains":
            return httpx.Response(201)
        if request.method == "POST" and request.url.path == "/v1/brains/bench-brain/ingest/file":
            return httpx.Response(200)
        if request.method == "POST" and request.url.path == "/v1/brains/bench-brain/search":
            active_searches += 1
            max_active_searches = max(max_active_searches, active_searches)
            await asyncio.sleep(0.02)
            active_searches -= 1
            query = payload["query"]
            if query == "alpha?":
                return httpx.Response(
                    200,
                    json={"chunks": [{"text": "Alpha answer [e1]", "path": "fake/a.md"}]},
                )
            return httpx.Response(
                200,
                json={"chunks": [{"text": "Beta answer [e2]", "path": "fake/b.md"}]},
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        bench_runner.httpx,
        "AsyncClient",
        lambda **kwargs: REAL_ASYNC_CLIENT(
            base_url=kwargs.get("base_url", "http://test"),
            timeout=kwargs.get("timeout"),
            transport=transport,
        ),
    )

    result = CliRunner().invoke(
        bench_runner.main,
        [
            "--benchmark",
            "fake",
            "--split",
            "qa",
            "--sdk",
            "ts",
            "--scenario",
            "search-retrieve-only",
            "--mode",
            "hybrid-rerank",
            "--top-k",
            "20",
            "--candidate-k",
            "80",
            "--rerank-top-n",
            "40",
            "--scorer",
            "token-f1",
            "--concurrency",
            "2",
            "--brain",
            "bench-brain",
            "--endpoint",
            "http://test",
            "--sample-ids-file",
            str(sample_ids),
            "--source-filter",
            "conversation-a",
            "--output",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert fake_benchmark_modules.normalise_kwargs == {
        "split": "qa",
        "limit": None,
        "sample_ids": ["q1", "q2"],
        "source_filter": "conversation-a",
    }
    assert max_active_searches == 2
    assert [call[:2] for call in calls[:4]] == [
        ("DELETE", "/v1/brains/bench-brain"),
        ("POST", "/v1/brains"),
        ("POST", "/v1/brains/bench-brain/ingest/file"),
        ("POST", "/v1/brains/bench-brain/ingest/file"),
    ]

    run_dirs = list((tmp_path / "out" / "fake" / "qa").iterdir())
    assert len(run_dirs) == 1
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text(encoding="utf-8"))
    score = json.loads((run_dirs[0] / "result-ts.json").read_text(encoding="utf-8"))
    normalised = (run_dirs[0] / "normalised.jsonl").read_text(encoding="utf-8").splitlines()

    assert manifest["brain_id"] == "bench-brain"
    assert manifest["candidate_k"] == 80
    assert manifest["rerank_top_n"] == 40
    assert manifest["sample_size"] == 2
    assert score["pass_rate"] == 1.0
    assert score["questions"][0]["evidence_recall"] == 1.0
    assert json.loads(normalised[0])["question_count"] == 2


def test_prepare_only_ingests_and_does_not_run_questions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_benchmark_modules: FakeAdapter,
) -> None:
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method == "DELETE":
            return httpx.Response(204)
        if request.method == "POST" and request.url.path == "/v1/brains":
            return httpx.Response(201)
        if request.method == "POST" and request.url.path.endswith("/ingest/file"):
            return httpx.Response(200)
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        bench_runner.httpx,
        "AsyncClient",
        lambda **kwargs: REAL_ASYNC_CLIENT(
            base_url=kwargs.get("base_url", "http://test"),
            timeout=kwargs.get("timeout"),
            transport=transport,
        ),
    )

    result = CliRunner().invoke(
        bench_runner.main,
        [
            "--benchmark",
            "fake",
            "--split",
            "qa",
            "--sdk",
            "py",
            "--endpoint",
            "http://test",
            "--prepare-only",
            "--output",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert ("POST", "/v1/brains/eval-fake-qa/search") not in calls
    run_dirs = list((tmp_path / "out" / "fake" / "qa").iterdir())
    score = json.loads((run_dirs[0] / "result-py.json").read_text(encoding="utf-8"))
    assert score["total"] == 0
    assert score["manifest"]["sample_size"] == 2
