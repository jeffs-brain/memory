# SPDX-License-Identifier: Apache-2.0
"""Cross-SDK eval runner entrypoint.

Spawns the chosen SDK's `memory serve` daemon, drives an eval dataset through
the shared HTTP daemon scenarios, scores with either the exact or judge
scorer, and writes one result JSON to `<output>/<YYYY-MM-DD>/<sdk>.json`.
Exits non-zero if the pass rate dips below the configured floor.

Scenarios:

    ask-basic
        POST /v1/brains/{brainId}/ask
        Body: {"question": "...", "topK": 5, "mode": "auto"}

    ask-augmented
        POST /v1/brains/{brainId}/ask
        Body: {"question": "...", "topK": 5, "mode": "auto",
               "readerMode": "augmented", "questionDate": "...?"}

    search-retrieve-only
        POST /v1/brains/{brainId}/search
        Body: {"query": "...", "topK": 5, "mode": "auto",
               "questionDate": "...?", "candidateK": 80?, "rerankTopN": 40?}

Ask scenarios consume SSE and accumulate `answer_delta` payloads into the
final answer while collecting `citation` events. The retrieve-only scenario
folds returned chunk `text`, falling back to `summary`, into a retrieval-only
answer blob and records chunk metadata as citations.
"""
from __future__ import annotations

import asyncio
import base64
import datetime as _dt
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

import click
import httpx

from scorer.exact import ExactScorer
from scorer.judge import JudgeScorer
from sdks import get_runner
from sdks.base import SdkRunner

DEFAULT_TOP_K = 5
DEFAULT_CANDIDATE_K = 0
DEFAULT_RERANK_TOP_N = 0
DEFAULT_BRAIN = "eval"
ASK_TIMEOUT_S = 120.0
DEFAULT_SCENARIO = "ask-basic"
DEFAULT_MODE = "auto"
SCENARIOS = ("ask-basic", "ask-augmented", "search-retrieve-only")
SEARCH_MODES = ("auto", "hybrid", "hybrid-rerank", "bm25", "semantic")


@dataclass
class QuestionResult:
    id: str
    question: str
    answer: str
    score: float
    passed: bool
    latency_ms: float
    citations: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


@dataclass
class EvalScore:
    sdk: str
    dataset: str
    scorer: str
    total: int
    passed: int
    pass_rate: float
    mean_score: float
    started_at: str
    finished_at: str
    scenario: str = DEFAULT_SCENARIO
    mode: str = DEFAULT_MODE
    brain: str = DEFAULT_BRAIN
    questions: list[QuestionResult] = field(default_factory=list)


def _load_dataset(path: Path, limit: int | None) -> list[dict[str, Any]]:
    if not path.exists():
        raise click.ClickException(f"Dataset not found: {path}")
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(json.loads(line))
            if limit is not None and len(items) >= limit:
                break
    return items


def _build_scorer(kind: str) -> ExactScorer | JudgeScorer:
    if kind == "exact":
        return ExactScorer()
    if kind == "judge":
        return JudgeScorer()
    raise click.ClickException(f"Unknown scorer: {kind}")


def _ask_path(brain: str) -> str:
    return f"/v1/brains/{quote(brain, safe='')}/ask"


@dataclass
class _ScenarioOutcome:
    answer: str
    citations: list[dict[str, Any]]
    error: str | None


@dataclass(frozen=True)
class _RequestSpec:
    path: str
    body: dict[str, Any]
    streaming: bool


def _search_path(brain: str) -> str:
    return f"/v1/brains/{quote(brain, safe='')}/search"


def _brain_path(brain: str) -> str:
    return f"/v1/brains/{quote(brain, safe='')}"


def _ingest_file_path(brain: str) -> str:
    return f"{_brain_path(brain)}/ingest/file"


def _question_date(item: dict[str, Any]) -> str | None:
    for key in ("questionDate", "question_date"):
        value = item.get(key)
        if isinstance(value, str) and value != "":
            return value
    return None


def _safe_doc_stem(raw: str) -> str:
    chars = [ch.lower() if ch.isalnum() else "-" for ch in raw]
    stem = "".join(chars).strip("-")
    while "--" in stem:
        stem = stem.replace("--", "-")
    return stem or "item"


async def _seed_reference_brain(*, endpoint: str, dataset: Path, brain: str) -> None:
    items = _load_dataset(dataset, limit=None)

    async with httpx.AsyncClient(base_url=endpoint, timeout=ASK_TIMEOUT_S) as client:
        create_resp = await client.post("/v1/brains", json={"brainId": brain})
        if create_resp.status_code == 409:
            delete_resp = await client.delete(
                _brain_path(brain),
                headers={"x-confirm-delete": "yes"},
            )
            if delete_resp.status_code != 204:
                delete_resp.raise_for_status()
            create_resp = await client.post("/v1/brains", json={"brainId": brain})

        if create_resp.status_code != 201:
            create_resp.raise_for_status()

        for item in items:
            question = item.get("question")
            reference_answer = item.get("reference_answer")
            if not isinstance(question, str) or question.strip() == "":
                raise click.ClickException(
                    f"Cannot seed reference brain: dataset row {item.get('id', '<missing id>')} lacks a question"
                )
            if not isinstance(reference_answer, str) or reference_answer.strip() == "":
                raise click.ClickException(
                    f"Cannot seed reference brain: dataset row {item.get('id', '<missing id>')} lacks reference_answer"
                )

            stem = _safe_doc_stem(str(item.get("id") or question))
            document = f"# {question.strip()}\n\n{reference_answer.strip()}\n"
            ingest_resp = await client.post(
                _ingest_file_path(brain),
                json={
                    "path": f"smoke/{stem}.md",
                    "contentType": "text/markdown",
                    "contentBase64": base64.b64encode(document.encode("utf-8")).decode("ascii"),
                },
            )
            if ingest_resp.status_code != 200:
                ingest_resp.raise_for_status()


def _build_request_spec(
    *,
    brain: str,
    item: dict[str, Any],
    question: str,
    top_k: int,
    mode: str,
    scenario: str,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    rerank_top_n: int = DEFAULT_RERANK_TOP_N,
) -> _RequestSpec:
    if scenario == "ask-basic":
        return _RequestSpec(
            path=_ask_path(brain),
            body={"question": question, "topK": top_k, "mode": mode},
            streaming=True,
        )
    if scenario == "ask-augmented":
        body: dict[str, Any] = {
            "question": question,
            "topK": top_k,
            "mode": mode,
            "readerMode": "augmented",
        }
        question_date = _question_date(item)
        if question_date is not None:
            body["questionDate"] = question_date
        return _RequestSpec(path=_ask_path(brain), body=body, streaming=True)
    if scenario == "search-retrieve-only":
        body: dict[str, Any] = {"query": question, "topK": top_k, "mode": mode}
        question_date = _question_date(item)
        if question_date is not None:
            body["questionDate"] = question_date
        if candidate_k > 0:
            body["candidateK"] = candidate_k
        if rerank_top_n > 0:
            body["rerankTopN"] = rerank_top_n
        return _RequestSpec(
            path=_search_path(brain),
            body=body,
            streaming=False,
        )
    raise click.ClickException(f"Unknown scenario: {scenario}")


def _parse_sse_frame(raw: str) -> tuple[str, str] | None:
    """Parse a single SSE frame into (event, data_json_string).

    Returns ``None`` when the frame carries only comments or is empty.
    """
    event = "message"
    data_lines: list[str] = []
    for line in raw.splitlines():
        if not line or line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
    if not data_lines:
        return None
    return event, "\n".join(data_lines)


def _extract_delta(payload: dict[str, Any]) -> str:
    """Pull answer text out of an answer_delta / token event payload.

    The spec names the event `answer_delta` with a `delta` string field; we
    also tolerate `token` as an event name and `text`/`content` as payload
    keys to stay robust against minor SDK dialects.
    """
    for key in ("delta", "text", "content", "token"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return ""


async def _ask_one(
    client: httpx.AsyncClient,
    *,
    spec: _RequestSpec,
) -> _ScenarioOutcome:
    answer_parts: list[str] = []
    citations: list[dict[str, Any]] = []
    error: str | None = None
    final_answer_from_done: str | None = None

    try:
        if spec.streaming:
            async with client.stream(
                "POST",
                spec.path,
                json=spec.body,
                headers={"Accept": "text/event-stream"},
            ) as resp:
                resp.raise_for_status()
                buffer = ""
                async for chunk in resp.aiter_text():
                    if chunk == "":
                        continue
                    buffer += chunk
                    while "\n\n" in buffer:
                        raw_frame, buffer = buffer.split("\n\n", 1)
                        parsed = _parse_sse_frame(raw_frame)
                        if parsed is None:
                            continue
                        event, data_str = parsed
                        try:
                            payload = json.loads(data_str) if data_str else {}
                        except json.JSONDecodeError:
                            payload = {"_raw": data_str}
                        if event in ("answer_delta", "token"):
                            answer_parts.append(_extract_delta(payload))
                        elif event == "citation":
                            citations.append(payload)
                        elif event == "done":
                            if isinstance(payload, dict):
                                candidate = payload.get("answer")
                                if isinstance(candidate, str):
                                    final_answer_from_done = candidate
                            buffer = ""
                            break
                        elif event == "error":
                            code = payload.get("code") if isinstance(payload, dict) else None
                            message = payload.get("message") if isinstance(payload, dict) else None
                            error = f"stream_error: code={code} message={message}"
                            buffer = ""
                            break
                        # `retrieve` and any unrecognised events are ignored here;
                        # retrieved chunks are orthogonal to the scored answer.
                    if final_answer_from_done is not None or error is not None:
                        break
        else:
            resp = await client.post(spec.path, json=spec.body, headers={"Accept": "application/json"})
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, dict):
                payload = {}
            chunks = payload.get("chunks")
            if isinstance(chunks, list):
                for chunk in chunks:
                    if not isinstance(chunk, dict):
                        continue
                    body = chunk.get("text")
                    if not isinstance(body, str) or body == "":
                        summary = chunk.get("summary")
                        body = summary if isinstance(summary, str) else ""
                    if body != "":
                        answer_parts.append(body)
                    citations.append(
                        {
                            key: chunk[key]
                            for key in ("chunkId", "path", "title", "score")
                            if key in chunk
                        }
                    )
    except httpx.HTTPStatusError as exc:
        try:
            body = await exc.response.aread()
            text = body.decode("utf-8", errors="replace")[:512]
        except Exception:
            text = "(body unread)"
        error = f"http_{exc.response.status_code}: {text}"
    except httpx.HTTPError as exc:
        error = f"{type(exc).__name__}: {exc}"

    if spec.streaming:
        answer = final_answer_from_done if final_answer_from_done is not None else "".join(answer_parts)
    else:
        answer = "\n\n".join(part for part in answer_parts if part != "")
    return _ScenarioOutcome(answer=answer, citations=citations, error=error)


async def _run_eval_async(
    *,
    endpoint: str,
    scenario: str,
    brain: str,
    mode: str,
    items: list[dict[str, Any]],
    scorer: ExactScorer | JudgeScorer,
    top_k: int,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    rerank_top_n: int = DEFAULT_RERANK_TOP_N,
) -> list[QuestionResult]:
    results: list[QuestionResult] = []
    async with httpx.AsyncClient(base_url=endpoint, timeout=ASK_TIMEOUT_S) as client:
        for item in items:
            qid = item["id"]
            question = item["question"]
            spec = _build_request_spec(
                brain=brain,
                item=item,
                question=question,
                top_k=top_k,
                candidate_k=candidate_k,
                rerank_top_n=rerank_top_n,
                mode=mode,
                scenario=scenario,
            )
            t0 = _dt.datetime.now(_dt.UTC)
            try:
                outcome = await _ask_one(
                    client,
                    spec=spec,
                )
            except Exception as exc:  # noqa: BLE001
                outcome = _ScenarioOutcome(
                    answer="",
                    citations=[],
                    error=f"{type(exc).__name__}: {exc}",
                )
            latency = (_dt.datetime.now(_dt.UTC) - t0).total_seconds() * 1000.0

            score_value = (
                scorer.score(item=item, answer=outcome.answer) if outcome.error is None else 0.0
            )
            results.append(
                QuestionResult(
                    id=qid,
                    question=question,
                    answer=outcome.answer,
                    score=score_value,
                    passed=score_value >= 0.5,
                    latency_ms=latency,
                    citations=outcome.citations,
                    error=outcome.error,
                )
            )
    return results


def run_eval(
    *,
    endpoint: str,
    scenario: str = DEFAULT_SCENARIO,
    mode: str = DEFAULT_MODE,
    dataset: Path,
    scorer_kind: str,
    sdk: str,
    limit: int | None,
    brain: str = DEFAULT_BRAIN,
    top_k: int = DEFAULT_TOP_K,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    rerank_top_n: int = DEFAULT_RERANK_TOP_N,
) -> EvalScore:
    """Drive the dataset against the daemon and collect scores."""
    started = _dt.datetime.now(_dt.UTC)
    items = _load_dataset(dataset, limit)
    scorer = _build_scorer(scorer_kind)
    results = asyncio.run(
        _run_eval_async(
            endpoint=endpoint,
            scenario=scenario,
            brain=brain,
            mode=mode,
            items=items,
            scorer=scorer,
            top_k=top_k,
            candidate_k=candidate_k,
            rerank_top_n=rerank_top_n,
        )
    )

    finished = _dt.datetime.now(_dt.UTC)
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    pass_rate = (passed / total) if total else 0.0
    mean_score = (sum(r.score for r in results) / total) if total else 0.0

    return EvalScore(
        sdk=sdk,
        dataset=str(dataset),
        scorer=scorer_kind,
        total=total,
        passed=passed,
        pass_rate=pass_rate,
        mean_score=mean_score,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        scenario=scenario,
        mode=mode,
        brain=brain,
        questions=results,
    )


def write_result(output_dir: Path, sdk: str, score: EvalScore) -> Path:
    date = _dt.date.today().isoformat()
    target_dir = output_dir / date
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{sdk}.json"
    payload = asdict(score)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


@click.command()
@click.option("--sdk", type=click.Choice(["ts", "go", "py"]), required=True)
@click.option(
    "--scenario",
    type=click.Choice(SCENARIOS),
    default=DEFAULT_SCENARIO,
    show_default=True,
    help="Cross-SDK daemon scenario to exercise.",
)
@click.option(
    "--mode",
    type=click.Choice(SEARCH_MODES),
    default=DEFAULT_MODE,
    show_default=True,
    help="Retrieval mode forwarded to /ask or /search.",
)
@click.option("--dataset", type=click.Path(path_type=Path), default=Path("datasets/lme.jsonl"))
@click.option("--scorer", "scorer_kind", type=click.Choice(["exact", "judge"]), default="judge")
@click.option("--limit", type=int, default=None, help="Max questions to run (None = all)")
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("results/"),
    help="Scenario-specific output root. Writes <output>/<YYYY-MM-DD>/<sdk>.json.",
)
@click.option("--port", type=int, default=0, help="0 = random")
@click.option("--floor", type=float, default=0.90, help="Minimum pass rate")
@click.option(
    "--brain",
    type=str,
    default=DEFAULT_BRAIN,
    show_default=True,
    help=(
        "Pre-ingested brainId to target via POST /v1/brains/{brain}/ask or /search. "
        "Populate with `memory ingest ./corpus --brain eval` before a run."
    ),
)
@click.option(
    "--top-k",
    type=int,
    default=DEFAULT_TOP_K,
    show_default=True,
    help="topK value forwarded on each /ask or /search call.",
)
@click.option(
    "--candidate-k",
    type=int,
    default=DEFAULT_CANDIDATE_K,
    show_default=True,
    help="candidateK value forwarded on retrieve-only /search calls. Zero defers to the daemon default.",
)
@click.option(
    "--rerank-top-n",
    type=int,
    default=DEFAULT_RERANK_TOP_N,
    show_default=True,
    help="rerankTopN value forwarded on retrieve-only /search calls. Zero defers to the daemon default.",
)
@click.option(
    "--seed-reference-brain",
    is_flag=True,
    help=(
        "Delete and recreate the target brain, then ingest one markdown document per dataset row "
        "using that row's reference_answer. Intended for offline retrieval smoke checks."
    ),
)
def main(
    sdk: str,
    scenario: str,
    mode: str,
    dataset: Path,
    scorer_kind: str,
    limit: int | None,
    output: Path,
    port: int,
    floor: float,
    brain: str,
    top_k: int,
    candidate_k: int,
    rerank_top_n: int,
    seed_reference_brain: bool,
) -> None:
    scratch_home: tempfile.TemporaryDirectory[str] | None = None
    prior_jb_home = os.environ.get("JB_HOME")
    if seed_reference_brain:
        scratch_home = tempfile.TemporaryDirectory(prefix="jeffs-brain-eval-")
        os.environ["JB_HOME"] = scratch_home.name

    runner: SdkRunner = get_runner(sdk)
    try:
        runner.start(port=port)
        if seed_reference_brain:
            asyncio.run(_seed_reference_brain(endpoint=runner.endpoint, dataset=dataset, brain=brain))
        score = run_eval(
            endpoint=runner.endpoint,
            scenario=scenario,
            mode=mode,
            dataset=dataset,
            scorer_kind=scorer_kind,
            sdk=sdk,
            limit=limit,
            brain=brain,
            top_k=top_k,
            candidate_k=candidate_k,
            rerank_top_n=rerank_top_n,
        )
        target = write_result(output, sdk, score)
        click.echo(
            f"{sdk}: {score.passed}/{score.total} "
            f"pass_rate={score.pass_rate:.3f} mean_score={score.mean_score:.3f} "
            f"-> {target}"
        )
        if score.pass_rate < floor:
            click.echo(f"FAIL: {score.pass_rate:.3f} < floor {floor}", err=True)
            raise SystemExit(1)
    finally:
        runner.stop()
        if prior_jb_home is None:
            os.environ.pop("JB_HOME", None)
        else:
            os.environ["JB_HOME"] = prior_jb_home
        if scratch_home is not None:
            scratch_home.cleanup()


if __name__ == "__main__":
    main()
