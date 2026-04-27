# SPDX-License-Identifier: Apache-2.0
"""BPI-Bench adapter for the Jeffs Brain eval runner."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from benchmarks.base import (
    BenchmarkSource,
    CorpusDocument,
    EvalQuestion,
    FetchResult,
    NormalisedBenchmark,
)
from benchmarks.fetch import fetch_and_verify, sha256_file
from benchmarks.scoring import BPIContainmentScorer, ExactContainmentScorer, JudgeBridgeScorer

BPI_BENCH_REVISION = "local-seed"
BPI_BENCH_LICENCE = "Apache-2.0"
BPI_BENCH_URL = "https://github.com/jeffs-brain/bpi-bench"
BPI_BENCH_RAW_BASE = (
    "https://raw.githubusercontent.com/jeffs-brain/bpi-bench/main/dataset/v1"
)
ADAPTER_VERSION = "0.1.0"
SUPPORTED_SPLITS = ("full", "smoke")
DEFAULT_SPLIT = "full"


class BpiBenchAdapter:
    id = "bpi-bench"
    source = BenchmarkSource(
        benchmark=id,
        url=BPI_BENCH_URL,
        revision=BPI_BENCH_REVISION,
        licence=BPI_BENCH_LICENCE,
        adapter_version=ADAPTER_VERSION,
    )

    def fetch(self, cache_dir: Path, *, split: str = DEFAULT_SPLIT) -> FetchResult:
        selected_split = _normalise_split(split)
        filename = _filename_for_split(selected_split)
        local_path = _local_dataset_path(filename)
        if local_path is not None:
            return FetchResult(
                local_path=local_path,
                sha256=sha256_file(local_path),
                revision=BPI_BENCH_REVISION,
            )
        return fetch_and_verify(
            url=f"{BPI_BENCH_RAW_BASE}/{filename}",
            cache_dir=cache_dir,
            benchmark=self.id,
            filename=filename,
            revision=BPI_BENCH_REVISION,
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
        selected_split = _normalise_split(split or _split_from_path(fetched.local_path))
        raw = json.loads(fetched.local_path.read_text(encoding="utf-8"))
        scenarios = raw.get("scenarios")
        if not isinstance(scenarios, list):
            raise ValueError("BPI-Bench dataset must contain a scenarios list")

        wanted_ids = set(sample_ids or [])
        documents: list[CorpusDocument] = []
        questions: list[EvalQuestion] = []

        for scenario in scenarios:
            if not isinstance(scenario, dict):
                raise ValueError("BPI-Bench scenario must be an object")
            scenario_id = _required_string(scenario, "id")
            domain = _required_string(scenario, "domain")
            if source_filter is not None and source_filter not in {scenario_id, domain}:
                continue

            rules = _list_of_dicts(scenario.get("rules"), f"{scenario_id}.rules")
            cases = _list_of_dicts(scenario.get("cases"), f"{scenario_id}.cases")
            valid_rules = [_required_string(rule, "id") for rule in rules]
            documents.extend(_rule_documents(selected_split, scenario_id, domain, rules))

            for case in cases:
                question = _question(
                    selected_split=selected_split,
                    scenario_id=scenario_id,
                    domain=domain,
                    case=case,
                    valid_rules=valid_rules,
                )
                if wanted_ids and question.id not in wanted_ids:
                    continue
                questions.append(question)
                if limit is not None and len(questions) >= limit:
                    return _normalised(self.source, fetched.sha256, documents, questions)

        return _normalised(self.source, fetched.sha256, documents, questions)

    def default_scorer(self) -> BPIContainmentScorer:
        return BPIContainmentScorer()

    def scorer_for(
        self,
        name: str,
    ) -> BPIContainmentScorer | ExactContainmentScorer | JudgeBridgeScorer:
        if name in {"bpi-containment", "bpi-deterministic"}:
            return BPIContainmentScorer()
        if name == "exact-containment":
            return ExactContainmentScorer()
        if name == "judge":
            return JudgeBridgeScorer()
        expected = ", ".join(self.available_scorers())
        raise ValueError(f"unsupported BPI-Bench scorer {name!r}; expected one of: {expected}")

    def available_scorers(self) -> list[str]:
        return ["bpi-containment", "exact-containment", "judge"]


def _normalised(
    source: BenchmarkSource,
    sha256: str,
    documents: list[CorpusDocument],
    questions: list[EvalQuestion],
) -> NormalisedBenchmark:
    return NormalisedBenchmark(
        source=source.model_copy(update={"sha256": sha256}),
        documents=documents,
        questions=questions,
    )


def _rule_documents(
    split: str,
    scenario_id: str,
    domain: str,
    rules: list[dict[str, Any]],
) -> list[CorpusDocument]:
    documents: list[CorpusDocument] = []
    for rule in rules:
        rule_id = _required_string(rule, "id")
        given_at = _required_string(rule, "given_at")
        lines = [
            f"# BPI-Bench Rule {rule_id}",
            "",
            f"Scenario: {scenario_id}",
            f"Domain: {domain}",
            f"Rule ID: {rule_id}",
            f"Given at: {given_at}",
        ]
        priority = rule.get("priority")
        supersedes = rule.get("supersedes")
        if isinstance(priority, int):
            lines.append(f"Priority: {priority}")
        if isinstance(supersedes, str) and supersedes:
            lines.append(f"Supersedes: {supersedes}")
        lines.extend(["", _required_string(rule, "instruction"), ""])
        slug = f"bpi-{_slug(scenario_id)}-{_slug(rule_id)}"
        documents.append(
            CorpusDocument(
                path=f"bpi-bench/{split}/{scenario_id}/rule-{rule_id}.md",
                content="\n".join(lines),
                metadata={
                    "benchmark": "bpi-bench",
                    "split": split,
                    "scenario_id": scenario_id,
                    "domain": domain,
                    "rule_id": rule_id,
                    "given_at": given_at,
                    "priority": priority if isinstance(priority, int) else None,
                    "supersedes": supersedes if isinstance(supersedes, str) else None,
                    "ingest_method": "remember",
                    "remember_slug": slug,
                    "remember_scope": "global",
                    "source": "bpi-bench",
                },
            )
        )
    return documents


def _question(
    *,
    selected_split: str,
    scenario_id: str,
    domain: str,
    case: dict[str, Any],
    valid_rules: list[str],
) -> EvalQuestion:
    case_id = _required_string(case, "id")
    expected_rules = _strings(case.get("expected_rules"))
    expected_action = _required_string(case, "expected_action")
    situation = _required_string(case, "situation")
    category = _required_string(case, "category")
    case_date = _required_string(case, "case_date")
    prompt = (
        "You are an operations assistant. An operator has previously configured "
        "you with business rules for this domain.\n\n"
        "A new case has arrived:\n\n"
        f"{situation}\n\n"
        "Based on the business rules you were given:\n"
        "1. Which rules by ID apply to this case? Use an empty list if none apply.\n"
        "2. What action should be taken?\n"
        "3. Brief reasoning.\n\n"
        "Respond in this JSON format:\n"
        '{"applicable_rules": ["RULE-ID"], "action": "...", "reasoning": "..."}'
    )
    return EvalQuestion(
        id=case_id,
        question=prompt,
        gold_answers=[expected_action],
        category=category,
        question_date=case_date,
        evidence_ids=expected_rules,
        source_id=scenario_id,
        metadata={
            "adapter_version": ADAPTER_VERSION,
            "benchmark": "bpi-bench",
            "split": selected_split,
            "scenario_id": scenario_id,
            "domain": domain,
            "case_date": case_date,
            "expected_rules": expected_rules,
            "valid_rules": valid_rules,
            "expected_action": expected_action,
            "action_keywords": _strings(case.get("action_keywords")),
            "expected_reasoning": _optional_string(case.get("expected_reasoning")),
        },
    )


def _local_dataset_path(filename: str) -> Path | None:
    explicit = os.environ.get("BPI_BENCH_DATASET_PATH")
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return path
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / "bpi-bench" / "dataset" / "v1" / filename,
        Path(__file__).resolve().parent / "fixtures" / _fixture_name(filename),
    ]
    return next((candidate for candidate in candidates if candidate.exists()), None)


def _filename_for_split(split: str) -> str:
    return "bpi-bench-smoke.json" if split == "smoke" else "bpi-bench.json"


def _fixture_name(filename: str) -> str:
    return "bpi_bench_smoke_fixture.json" if "smoke" in filename else "bpi_bench_fixture.json"


def _split_from_path(path: Path) -> str:
    return "smoke" if "smoke" in path.name else DEFAULT_SPLIT


def _normalise_split(split: str | None) -> str:
    value = split or DEFAULT_SPLIT
    if value not in SUPPORTED_SPLITS:
        expected = ", ".join(SUPPORTED_SPLITS)
        raise ValueError(f"unsupported BPI-Bench split {value!r}; expected one of: {expected}")
    return value


def _required_string(value: dict[str, Any], key: str) -> str:
    raw = value.get(key)
    if raw is None:
        raise ValueError(f"missing required BPI-Bench field: {key}")
    text = str(raw).strip()
    if not text:
        raise ValueError(f"empty required BPI-Bench field: {key}")
    return text


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _list_of_dicts(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{label} must contain only objects")
    return value


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return slug or "item"
