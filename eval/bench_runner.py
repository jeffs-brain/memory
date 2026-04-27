# SPDX-License-Identifier: Apache-2.0
"""Benchmark adapter CLI runner.

This runner is intentionally separate from ``runner.py``. It drives the planned
``eval.benchmarks`` adapter contract against the existing daemon HTTP surface.
"""
from __future__ import annotations

import asyncio
import base64
import dataclasses
import datetime as _dt
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
from typing import Any

import click
import httpx

from http_helpers import (
    ask_one,
    brain_path,
    build_request_spec as _build_http_request_spec,
    ingest_file_path,
)
from sdks import get_runner
from sdks.base import SdkRunner

DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_K = 0
DEFAULT_RERANK_TOP_N = 0
DEFAULT_CONCURRENCY = 4
DEFAULT_SCENARIO = "ask-basic"
DEFAULT_MODE = "auto"
DEFAULT_OUTPUT = Path("results/benchmarks")
DEFAULT_CACHE = Path("datasets/cache")
ASK_TIMEOUT_S = 120.0
SCENARIOS = ("ask-basic", "ask-augmented", "search-retrieve-only")
SEARCH_MODES = ("auto", "hybrid", "hybrid-rerank", "bm25", "semantic")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _model_dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, list):
        return [_model_dump(item) for item in obj]
    if isinstance(obj, tuple):
        return [_model_dump(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _model_dump(value) for key, value in obj.items()}
    return obj


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_model_dump(payload), indent=2, sort_keys=True), encoding="utf-8")


def _load_sample_ids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8").strip()
    if text == "":
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise click.ClickException("--sample-ids-file JSON must be a list of strings")
        return data
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def _sample_signature(questions: list[Any]) -> str | None:
    if not questions:
        return None
    ids = [_get(question, "id", "") for question in questions]
    digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    return digest


def _filter_questions(
    questions: list[Any],
    sample_ids: list[str] | None,
    limit: int | None,
) -> list[Any]:
    selected = questions
    if sample_ids is not None:
        allowed = set(sample_ids)
        selected = [question for question in selected if _get(question, "id") in allowed]
        found = {_get(question, "id") for question in selected}
        missing = [sample_id for sample_id in sample_ids if sample_id not in found]
        if missing:
            raise click.ClickException(f"Sample ids not found: {', '.join(missing)}")
    if limit is not None:
        selected = selected[:limit]
    return selected


def _question_as_item(question: Any) -> dict[str, Any]:
    return {
        "id": _get(question, "id"),
        "question": _get(question, "question"),
        "questionDate": _get(question, "question_date"),
        "question_date": _get(question, "question_date"),
        "reference_answer": (_get(question, "gold_answers") or [""])[0],
    }


def build_request_spec(
    *,
    brain: str,
    question: Any,
    top_k: int,
    mode: str,
    scenario: str,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    rerank_top_n: int = DEFAULT_RERANK_TOP_N,
) -> Any:
    return _build_http_request_spec(
        brain=brain,
        item=_question_as_item(question),
        question=str(_get(question, "question", "")),
        top_k=top_k,
        mode=mode,
        scenario=scenario,
        candidate_k=candidate_k,
        rerank_top_n=rerank_top_n,
    )


async def prepare_brain(
    *,
    endpoint: str,
    brain: str,
    documents: list[Any],
    recreate: bool = True,
) -> None:
    async with httpx.AsyncClient(base_url=endpoint, timeout=ASK_TIMEOUT_S) as client:
        if recreate:
            delete_resp = await client.delete(
                brain_path(brain),
                headers={"x-confirm-delete": "yes"},
            )
            if delete_resp.status_code not in (204, 404):
                delete_resp.raise_for_status()

        create_resp = await client.post("/v1/brains", json={"brainId": brain})
        if create_resp.status_code == 409 and recreate:
            delete_resp = await client.delete(
                brain_path(brain),
                headers={"x-confirm-delete": "yes"},
            )
            if delete_resp.status_code not in (204, 404):
                delete_resp.raise_for_status()
            create_resp = await client.post("/v1/brains", json={"brainId": brain})
        if create_resp.status_code not in (200, 201, 409):
            create_resp.raise_for_status()

        for document in documents:
            content = _get(document, "content")
            metadata = _get(document, "metadata") or {}
            payload: dict[str, Any] = {
                "path": _get(document, "path"),
                "contentType": _get(document, "content_type", "text/markdown"),
                "contentBase64": base64.b64encode(str(content).encode("utf-8")).decode("ascii"),
            }
            if metadata:
                payload["metadata"] = metadata
            ingest_resp = await client.post(ingest_file_path(brain), json=payload)
            if ingest_resp.status_code not in (200, 201, 202):
                ingest_resp.raise_for_status()


def _evidence_recall(
    question: Any,
    citations: list[dict[str, Any]],
    answer: str = "",
) -> float | None:
    evidence_ids = _get(question, "evidence_ids") or []
    if not evidence_ids:
        return None
    found: set[str] = set()
    haystacks: list[str] = []
    for citation in citations:
        for key in ("path", "text", "summary", "content", "title"):
            value = citation.get(key)
            if isinstance(value, str):
                haystacks.append(value)
    if answer:
        haystacks.append(answer)
    joined = "\n".join(haystacks)
    for evidence_id in evidence_ids:
        if str(evidence_id) in joined:
            found.add(str(evidence_id))
    return len(found) / len(evidence_ids)


async def run_questions(
    *,
    endpoint: str,
    brain: str,
    questions: list[Any],
    scorer: Any,
    scenario: str,
    mode: str,
    top_k: int,
    candidate_k: int,
    rerank_top_n: int,
    concurrency: int,
) -> list[Any]:
    base = importlib.import_module("benchmarks.base")
    result_cls = base.QuestionResult
    sem = asyncio.Semaphore(max(concurrency, 1))
    results: list[Any | None] = [None] * len(questions)

    async with httpx.AsyncClient(base_url=endpoint, timeout=ASK_TIMEOUT_S) as client:
        async def process(index: int, question: Any) -> None:
            async with sem:
                spec = build_request_spec(
                    brain=brain,
                    question=question,
                    top_k=top_k,
                    mode=mode,
                    scenario=scenario,
                    candidate_k=candidate_k,
                    rerank_top_n=rerank_top_n,
                )
                t0 = _dt.datetime.now(_dt.UTC)
                outcome = await ask_one(client, spec=spec)
                latency_ms = (_dt.datetime.now(_dt.UTC) - t0).total_seconds() * 1000.0
                if outcome.error is None:
                    score_result = scorer.score(
                        question=question,
                        answer=outcome.answer,
                        citations=outcome.citations,
                    )
                    score = float(_get(score_result, "score", 0.0))
                    passed = bool(_get(score_result, "passed", score >= 0.5))
                    detail = _get(score_result, "detail", {}) or {}
                    evidence = _get(score_result, "evidence_recall")
                    if evidence is None:
                        evidence = _evidence_recall(question, outcome.citations, outcome.answer)
                else:
                    score = 0.0
                    passed = False
                    detail = {}
                    evidence = _evidence_recall(question, outcome.citations, outcome.answer)
                results[index] = result_cls(
                    id=_get(question, "id"),
                    question=_get(question, "question"),
                    answer=outcome.answer,
                    score=score,
                    passed=passed,
                    latency_ms=latency_ms,
                    evidence_recall=evidence,
                    citations=outcome.citations,
                    error=outcome.error,
                    detail=detail,
                )

        async with asyncio.TaskGroup() as tg:
            for index, question in enumerate(questions):
                tg.create_task(process(index, question))

    return [result for result in results if result is not None]


def _latency_percentile(values: list[float], percentile: float) -> float:
    try:
        stats = importlib.import_module("benchmarks.stats")
        return float(stats.latency_percentile(values, percentile))
    except Exception:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, max(0, round((percentile / 100.0) * (len(ordered) - 1))))
        return float(ordered[idx])


def _bootstrap_ci(outcomes: list[bool]) -> tuple[float, float] | None:
    if not outcomes:
        return None
    try:
        stats = importlib.import_module("benchmarks.stats")
        return stats.bootstrap_ci(outcomes, seed=0)
    except Exception:
        return None


def _build_scorer(adapter: Any, scorer_name: str | None) -> Any:
    if scorer_name is None:
        return adapter.default_scorer()
    if hasattr(adapter, "scorer_for"):
        try:
            return adapter.scorer_for(scorer_name)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    if hasattr(adapter, "get_scorer"):
        try:
            return adapter.get_scorer(scorer_name)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

    scoring = importlib.import_module("benchmarks.scoring")
    mapping = {
        "token-f1": "TokenF1Scorer",
        "exact-containment": "ExactContainmentScorer",
        "adversarial": "AdversarialAbstentionScorer",
        "judge": "JudgeBridgeScorer",
    }
    class_name = mapping.get(scorer_name)
    if class_name is None or not hasattr(scoring, class_name):
        raise click.ClickException(f"Unknown scorer for {adapter.id}: {scorer_name}")
    return getattr(scoring, class_name)()


def _scorer_name(scorer: Any, fallback: str | None) -> str:
    value = _get(scorer, "name")
    if isinstance(value, str) and value:
        return value
    return fallback or scorer.__class__.__name__


def _source_sha(source: Any, fetched: Any) -> str:
    for obj in (fetched, source):
        value = _get(obj, "sha256")
        if isinstance(value, str) and value:
            return value
    return ""


def _fetch_adapter(adapter: Any, cache_dir: Path, split: str | None) -> Any:
    signature = inspect.signature(adapter.fetch)
    if "split" in signature.parameters and split is not None:
        return adapter.fetch(cache_dir, split=split)
    return adapter.fetch(cache_dir)


def _local_fetch_result(adapter: Any, dataset_path: Path) -> Any:
    if not dataset_path.exists():
        raise click.ClickException(f"Dataset fixture not found: {dataset_path}")
    if not dataset_path.is_file():
        raise click.ClickException(f"Dataset fixture is not a file: {dataset_path}")

    base = importlib.import_module("benchmarks.base")
    fetch_result_cls = base.FetchResult
    source = adapter.source
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    return fetch_result_cls(
        local_path=dataset_path,
        sha256=digest,
        revision=_get(source, "revision"),
    )


def _cached_fetch_result(adapter: Any, cache_dir: Path, split: str | None) -> Any:
    base = importlib.import_module("benchmarks.base")
    fetch_result_cls = base.FetchResult
    source = adapter.source
    sha256 = _get(source, "sha256")
    revision = _get(source, "revision")
    if isinstance(sha256, str) and sha256:
        cache_key = sha256[:12]
    elif isinstance(revision, str) and revision:
        cache_key = revision[:12]
    else:
        raise click.ClickException("--skip-fetch requires adapter.source.sha256 or revision")

    candidates = sorted((cache_dir / adapter.id / cache_key).glob("*"))
    if split:
        split_candidates = [
            candidate for candidate in candidates if candidate.name.startswith(split)
        ]
        if split_candidates:
            candidates = split_candidates
    files = [candidate for candidate in candidates if candidate.is_file()]
    if not files:
        raise click.ClickException(f"No cached dataset found for {adapter.id} at {cache_dir}")
    local_path = files[0]
    actual = hashlib.sha256(local_path.read_bytes()).hexdigest()
    if isinstance(sha256, str) and sha256 and actual != sha256:
        raise click.ClickException(f"Cached dataset SHA256 mismatch: {local_path}")
    return fetch_result_cls(local_path=local_path, sha256=actual, revision=revision)


def _write_normalised(path: Path, normalised: Any, questions: list[Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "kind": "summary",
                    "source": _model_dump(_get(normalised, "source")),
                    "document_count": len(_get(normalised, "documents") or []),
                    "question_count": len(questions),
                },
                sort_keys=True,
            )
            + "\n"
        )
        for question in questions:
            payload = _model_dump(question)
            if isinstance(payload, dict):
                payload = {"kind": "question", **payload}
            fh.write(json.dumps(payload, sort_keys=True) + "\n")


def _aggregate_categories(results: list[Any], questions: list[Any]) -> dict[str, Any]:
    base = importlib.import_module("benchmarks.base")
    category_cls = base.CategoryScore
    grouped: dict[str, list[float]] = {}
    grouped_outcomes: dict[str, list[bool]] = {}
    passed: dict[str, int] = {}
    questions_by_id = {_get(question, "id"): question for question in questions}
    for result in results:
        question = questions_by_id.get(_get(result, "id"))
        category = str(_get(question, "category", "uncategorised"))
        grouped.setdefault(category, []).append(float(_get(result, "score", 0.0)))
        did_pass = bool(_get(result, "passed", False))
        grouped_outcomes.setdefault(category, []).append(did_pass)
        if did_pass:
            passed[category] = passed.get(category, 0) + 1
    return {
        category: category_cls(
            total=len(scores),
            passed=passed.get(category, 0),
            score=(sum(scores) / len(scores)) if scores else 0.0,
            confidence_interval_95=_bootstrap_ci(grouped_outcomes.get(category, [])),
        )
        for category, scores in grouped.items()
    }


def build_result(
    *,
    benchmark: str,
    split: str | None,
    normalised: Any,
    fetched: Any,
    sdk: str,
    scenario: str,
    scorer_name: str,
    mode: str,
    top_k: int,
    candidate_k: int,
    rerank_top_n: int,
    brain: str,
    questions: list[Any],
    results: list[Any],
    source_filter: str | None,
    started_at: str,
    finished_at: str,
) -> Any:
    base = importlib.import_module("benchmarks.base")
    manifest_cls = base.BenchmarkManifest
    result_cls = base.BenchmarkResult
    source = _get(normalised, "source")
    scores = [float(_get(result, "score", 0.0)) for result in results]
    latencies = [float(_get(result, "latency_ms", 0.0)) for result in results]
    outcomes = [bool(_get(result, "passed", False)) for result in results]
    passed = sum(1 for result in results if bool(_get(result, "passed", False)))
    total = len(results)
    judge_model = os.environ.get("JB_EVAL_JUDGE_MODEL") if scorer_name == "judge" else None
    budget = os.environ.get("JB_EVAL_BUDGET_USD")
    manifest = manifest_cls(
        benchmark=benchmark,
        split=split,
        source_url=_get(source, "url", ""),
        source_revision=_get(source, "revision"),
        source_sha256=_source_sha(source, fetched),
        adapter_version=_get(source, "adapter_version", "1"),
        sdk=sdk,
        scenario=scenario,
        scorer=scorer_name,
        retrieval_mode=mode,
        top_k=top_k,
        candidate_k=candidate_k,
        rerank_top_n=rerank_top_n,
        brain_id=brain,
        sample_signature=_sample_signature(questions),
        sample_size=len(questions),
        source_filter=source_filter,
        judge_model=judge_model,
        budget_usd=float(budget) if budget else None,
        run_seed=None,
        started_at=started_at,
        finished_at=finished_at,
    )
    return result_cls(
        manifest=manifest,
        total=total,
        passed=passed,
        pass_rate=(passed / total) if total else 0.0,
        mean_score=(sum(scores) / total) if total else 0.0,
        latency_p50_ms=_latency_percentile(latencies, 50),
        latency_p95_ms=_latency_percentile(latencies, 95),
        confidence_interval_95=_bootstrap_ci(outcomes),
        per_category=_aggregate_categories(results, questions),
        cost_usd=0.0,
        questions=results,
    )


def write_outputs(
    *,
    output: Path,
    benchmark: str,
    split: str | None,
    sdk: str,
    result: Any,
    normalised: Any,
    questions: list[Any],
) -> Path:
    timestamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d-%H%M%S")
    target_dir = output / benchmark / (split or "default") / timestamp
    target_dir.mkdir(parents=True, exist_ok=True)
    _write_json(target_dir / "manifest.json", _get(result, "manifest"))
    _write_json(target_dir / f"result-{sdk}.json", result)
    _write_normalised(target_dir / "normalised.jsonl", normalised, questions)
    return target_dir


def run_benchmark(
    *,
    benchmark: str,
    split: str | None,
    sdk: str,
    scenario: str,
    mode: str,
    top_k: int,
    candidate_k: int,
    rerank_top_n: int,
    scorer_name: str | None,
    concurrency: int,
    brain: str,
    output: Path,
    floor: float,
    limit: int | None,
    cache_dir: Path,
    dataset_path: Path | None,
    skip_fetch: bool,
    skip_ingest: bool,
    prepare_only: bool,
    sample_ids_file: Path | None,
    endpoint: str | None,
    port: int,
    source_filter: str | None,
) -> tuple[Any, Path]:
    registry = importlib.import_module("benchmarks")
    try:
        adapter = registry.get_adapter(benchmark)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    sample_ids = _load_sample_ids(sample_ids_file)
    if dataset_path is not None and skip_fetch:
        raise click.ClickException("--dataset-path cannot be combined with --skip-fetch")
    if dataset_path is not None:
        fetched = _local_fetch_result(adapter, dataset_path)
    elif skip_fetch:
        fetched = _cached_fetch_result(adapter, cache_dir, split)
    else:
        fetched = _fetch_adapter(adapter, cache_dir, split)
    normalised = adapter.normalise(
        fetched,
        split=split,
        limit=limit,
        sample_ids=sample_ids,
        source_filter=source_filter,
    )
    questions = _filter_questions(list(_get(normalised, "questions") or []), sample_ids, limit)
    scorer = _build_scorer(adapter, scorer_name)
    resolved_scorer_name = _scorer_name(scorer, scorer_name)

    runner: SdkRunner | None = None
    resolved_endpoint = endpoint
    started_at = _dt.datetime.now(_dt.UTC)
    try:
        if resolved_endpoint is None:
            runner = get_runner(sdk)
            runner.start(port=port)
            resolved_endpoint = runner.endpoint
        if not skip_ingest:
            asyncio.run(
                prepare_brain(
                    endpoint=resolved_endpoint,
                    brain=brain,
                    documents=list(_get(normalised, "documents") or []),
                    recreate=True,
                )
            )
        if prepare_only:
            results: list[Any] = []
        else:
            results = asyncio.run(
                run_questions(
                    endpoint=resolved_endpoint,
                    brain=brain,
                    questions=questions,
                    scorer=scorer,
                    scenario=scenario,
                    mode=mode,
                    top_k=top_k,
                    candidate_k=candidate_k,
                    rerank_top_n=rerank_top_n,
                    concurrency=concurrency,
                )
            )
    finally:
        if runner is not None:
            runner.stop()

    finished_at = _dt.datetime.now(_dt.UTC)
    result = build_result(
        benchmark=benchmark,
        split=split,
        normalised=normalised,
        fetched=fetched,
        sdk=sdk,
        scenario=scenario,
        scorer_name=resolved_scorer_name,
        mode=mode,
        top_k=top_k,
        candidate_k=candidate_k,
        rerank_top_n=rerank_top_n,
        brain=brain,
        questions=questions,
        results=results,
        source_filter=source_filter,
        started_at=started_at.isoformat(),
        finished_at=finished_at.isoformat(),
    )
    target_dir = write_outputs(
        output=output,
        benchmark=benchmark,
        split=split,
        sdk=sdk,
        result=result,
        normalised=normalised,
        questions=questions,
    )
    if not prepare_only and float(_get(result, "pass_rate", 0.0)) < floor:
        pass_rate = _get(result, "pass_rate")
        raise click.ClickException(f"FAIL: pass_rate={pass_rate:.3f} < floor {floor}")
    return result, target_dir


@click.command()
@click.option("--benchmark", required=True, help="Benchmark adapter id, e.g. locomo.")
@click.option("--split", default=None, help="Adapter-specific split, e.g. qa.")
@click.option("--sdk", type=click.Choice(["ts", "go", "py"]), required=True)
@click.option(
    "--scenario",
    type=click.Choice(SCENARIOS),
    default=DEFAULT_SCENARIO,
    show_default=True,
)
@click.option("--mode", type=click.Choice(SEARCH_MODES), default=DEFAULT_MODE, show_default=True)
@click.option("--top-k", type=int, default=DEFAULT_TOP_K, show_default=True)
@click.option("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K, show_default=True)
@click.option("--rerank-top-n", type=int, default=DEFAULT_RERANK_TOP_N, show_default=True)
@click.option("--scorer", "scorer_name", default=None, help="Adapter scorer name.")
@click.option("--concurrency", type=int, default=DEFAULT_CONCURRENCY, show_default=True)
@click.option("--brain", default=None, help="Brain id. Defaults to eval-{benchmark}-{split}.")
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT,
    show_default=True,
)
@click.option("--floor", type=float, default=0.0, show_default=True)
@click.option("--limit", type=int, default=None, help="Maximum question count.")
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_CACHE,
    show_default=True,
)
@click.option(
    "--dataset-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Local dataset or fixture path. Skips adapter.fetch and records the file SHA256.",
)
@click.option(
    "--prepare-only",
    is_flag=True,
    help="Fetch, normalise, ingest, write outputs, then stop.",
)
@click.option(
    "--skip-fetch",
    is_flag=True,
    help="Use the cached dataset instead of calling adapter.fetch.",
)
@click.option("--skip-ingest", is_flag=True, help="Reuse an already populated brain.")
@click.option("--sample-ids-file", type=click.Path(path_type=Path), default=None)
@click.option(
    "--endpoint",
    default=None,
    help="Existing daemon endpoint. Skips SDK daemon spawning.",
)
@click.option("--port", type=int, default=0, help="Daemon port when spawning an SDK. 0 = random.")
@click.option("--source-filter", default=None, help="Adapter-specific source row filter.")
def main(
    benchmark: str,
    split: str | None,
    sdk: str,
    scenario: str,
    mode: str,
    top_k: int,
    candidate_k: int,
    rerank_top_n: int,
    scorer_name: str | None,
    concurrency: int,
    brain: str | None,
    output: Path,
    floor: float,
    limit: int | None,
    cache_dir: Path,
    dataset_path: Path | None,
    prepare_only: bool,
    skip_fetch: bool,
    skip_ingest: bool,
    sample_ids_file: Path | None,
    endpoint: str | None,
    port: int,
    source_filter: str | None,
) -> None:
    brain_id = brain or f"eval-{benchmark}-{split or 'default'}"
    result, target_dir = run_benchmark(
        benchmark=benchmark,
        split=split,
        sdk=sdk,
        scenario=scenario,
        mode=mode,
        top_k=top_k,
        candidate_k=candidate_k,
        rerank_top_n=rerank_top_n,
        scorer_name=scorer_name,
        concurrency=concurrency,
        brain=brain_id,
        output=output,
        floor=floor,
        limit=limit,
        cache_dir=cache_dir,
        dataset_path=dataset_path,
        skip_fetch=skip_fetch,
        skip_ingest=skip_ingest,
        prepare_only=prepare_only,
        sample_ids_file=sample_ids_file,
        endpoint=endpoint,
        port=port,
        source_filter=source_filter,
    )
    manifest = _get(result, "manifest")
    click.echo(
        f"{sdk}/{benchmark}: {_get(result, 'passed')}/{_get(result, 'total')} "
        f"pass_rate={_get(result, 'pass_rate'):.3f} "
        f"mean_score={_get(result, 'mean_score'):.3f} "
        f"brain={_get(manifest, 'brain_id')} -> {target_dir}"
    )


if __name__ == "__main__":
    main()
