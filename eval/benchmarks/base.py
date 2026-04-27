# SPDX-License-Identifier: Apache-2.0
"""Shared benchmark adapter models."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

MetadataValue = str | int | float | bool | None


class BenchmarkSource(BaseModel):
    benchmark: str
    url: str
    revision: str | None = None
    sha256: str | None = None
    licence: str | None = None
    adapter_version: str | None = None


class FetchResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    local_path: Path
    sha256: str
    revision: str | None = None


class CorpusDocument(BaseModel):
    path: str
    content: str
    content_type: str = "text/markdown"
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class EvalQuestion(BaseModel):
    id: str
    question: str
    gold_answers: list[str]
    category: str
    question_date: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    source_id: str | None = None
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class NormalisedBenchmark(BaseModel):
    source: BenchmarkSource
    documents: list[CorpusDocument]
    questions: list[EvalQuestion]


class ScorerResult(BaseModel):
    score: float
    passed: bool
    evidence_recall: float | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class BenchmarkScorer(Protocol):
    name: str

    def score(
        self,
        *,
        question: EvalQuestion,
        answer: str,
        citations: list[dict[str, Any]],
    ) -> ScorerResult:
        ...


class BenchmarkAdapter(Protocol):
    id: str
    source: BenchmarkSource

    def fetch(self, cache_dir: Path) -> FetchResult:
        ...

    def normalise(
        self,
        fetched: FetchResult,
        *,
        split: str | None = None,
        limit: int | None = None,
        sample_ids: list[str] | None = None,
        source_filter: str | None = None,
    ) -> NormalisedBenchmark:
        ...

    def default_scorer(self) -> BenchmarkScorer:
        ...

    def available_scorers(self) -> list[str]:
        ...


class BenchmarkManifest(BaseModel):
    benchmark: str
    split: str | None = None
    source_url: str
    source_revision: str | None = None
    source_sha256: str
    adapter_version: str
    sdk: str
    scenario: str
    scorer: str
    retrieval_mode: str
    top_k: int
    candidate_k: int = 0
    rerank_top_n: int = 0
    brain_id: str
    sample_signature: str | None = None
    sample_size: int
    source_filter: str | None = None
    judge_model: str | None = None
    budget_usd: float | None = None
    run_seed: int | None = None
    started_at: str
    finished_at: str

    def is_comparable(self, other: "BenchmarkManifest") -> bool:
        return (
            self.benchmark == other.benchmark
            and self.split == other.split
            and self.source_sha256 == other.source_sha256
            and self.sample_signature == other.sample_signature
            and self.scenario == other.scenario
            and self.scorer == other.scorer
            and self.retrieval_mode == other.retrieval_mode
            and self.top_k == other.top_k
            and self.candidate_k == other.candidate_k
            and self.rerank_top_n == other.rerank_top_n
            and self.judge_model == other.judge_model
        )


class QuestionResult(BaseModel):
    id: str
    question: str
    answer: str
    score: float
    passed: bool
    latency_ms: float
    evidence_recall: float | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class CategoryScore(BaseModel):
    total: int
    passed: int
    score: float
    confidence_interval_95: tuple[float, float] | None = None


class BenchmarkResult(BaseModel):
    manifest: BenchmarkManifest
    total: int
    passed: int
    pass_rate: float
    mean_score: float
    latency_p50_ms: float
    latency_p95_ms: float
    confidence_interval_95: tuple[float, float] | None = None
    per_category: dict[str, CategoryScore] = Field(default_factory=dict)
    cost_usd: float = 0.0
    questions: list[QuestionResult] = Field(default_factory=list)
