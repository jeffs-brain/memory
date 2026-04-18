# SPDX-License-Identifier: Apache-2.0
"""Cross-SDK eval runner entrypoint.

Spawns the chosen SDK's `memory serve` daemon, drives an LME-style dataset
through its HTTP ask endpoint, scores with either the exact or judge scorer,
and writes a result JSON blob. Exits non-zero if the pass rate dips below the
configured floor.

Wire shape for the ask call follows `spec/PROTOCOL.md` and the companion
`spec/MCP-TOOLS.md`:

    POST /v1/brains/{brainId}/ask
    Body: {"question": "...", "topK": 5, "mode": "hybrid"}
    Response: text/event-stream with events:
        - retrieve      (first frame, chunks used for grounding)
        - answer_delta  (streamed answer text; aliased as `token` in prose)
        - citation      (AskCitationEvent, pointing into the accumulated answer)
        - done          (terminal, with the final answer string)
        - error         (terminal, with {code, message, retryable})

The runner accumulates `answer_delta` payloads into the final answer string
and collects `citation` events alongside. A `done` event short-circuits
reading. `error` events surface through the QuestionResult.error field.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
import json
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
DEFAULT_BRAIN = "eval"
ASK_TIMEOUT_S = 120.0


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
    mode: str
    dataset: str
    scorer: str
    total: int
    passed: int
    pass_rate: float
    mean_score: float
    started_at: str
    finished_at: str
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
class _AskOutcome:
    answer: str
    citations: list[dict[str, Any]]
    error: str | None


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
    brain: str,
    question: str,
    top_k: int,
    mode: str,
) -> _AskOutcome:
    body = {"question": question, "topK": top_k, "mode": mode}
    answer_parts: list[str] = []
    citations: list[dict[str, Any]] = []
    error: str | None = None
    final_answer_from_done: str | None = None

    try:
        async with client.stream(
            "POST",
            _ask_path(brain),
            json=body,
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
    except httpx.HTTPStatusError as exc:
        try:
            body = await exc.response.aread()
            text = body.decode("utf-8", errors="replace")[:512]
        except Exception:
            text = "(body unread)"
        error = f"http_{exc.response.status_code}: {text}"
    except httpx.HTTPError as exc:
        error = f"{type(exc).__name__}: {exc}"

    answer = final_answer_from_done if final_answer_from_done is not None else "".join(answer_parts)
    return _AskOutcome(answer=answer, citations=citations, error=error)


async def _run_eval_async(
    *,
    endpoint: str,
    brain: str,
    mode: str,
    items: list[dict[str, Any]],
    scorer: ExactScorer | JudgeScorer,
    top_k: int,
) -> list[QuestionResult]:
    results: list[QuestionResult] = []
    async with httpx.AsyncClient(base_url=endpoint, timeout=ASK_TIMEOUT_S) as client:
        for item in items:
            qid = item["id"]
            question = item["question"]
            t0 = _dt.datetime.now(_dt.UTC)
            try:
                outcome = await _ask_one(
                    client,
                    brain=brain,
                    question=question,
                    top_k=top_k,
                    mode=mode,
                )
            except Exception as exc:  # noqa: BLE001
                outcome = _AskOutcome(
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
    mode: str,
    dataset: Path,
    scorer_kind: str,
    sdk: str,
    limit: int | None,
    brain: str = DEFAULT_BRAIN,
    top_k: int = DEFAULT_TOP_K,
) -> EvalScore:
    """Drive the dataset against the daemon and collect scores.

    For each question, POST to ``/v1/brains/{brain}/ask`` with
    ``{"question", "topK", "mode"}`` and consume the SSE response, folding
    ``answer_delta`` / ``token`` deltas into the final answer and collecting
    ``citation`` events. ``done`` terminates; ``error`` records the failure.
    """
    started = _dt.datetime.now(_dt.UTC)
    items = _load_dataset(dataset, limit)
    scorer = _build_scorer(scorer_kind)
    results = asyncio.run(
        _run_eval_async(
            endpoint=endpoint,
            brain=brain,
            mode=mode,
            items=items,
            scorer=scorer,
            top_k=top_k,
        )
    )

    finished = _dt.datetime.now(_dt.UTC)
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    pass_rate = (passed / total) if total else 0.0
    mean_score = (sum(r.score for r in results) / total) if total else 0.0

    return EvalScore(
        sdk=sdk,
        mode=mode,
        dataset=str(dataset),
        scorer=scorer_kind,
        total=total,
        passed=passed,
        pass_rate=pass_rate,
        mean_score=mean_score,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
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
@click.option("--mode", type=click.Choice(["direct", "agentic"]), default="direct")
@click.option("--dataset", type=click.Path(path_type=Path), default=Path("datasets/lme.jsonl"))
@click.option("--scorer", "scorer_kind", type=click.Choice(["exact", "judge"]), default="judge")
@click.option("--limit", type=int, default=None, help="Max questions to run (None = all)")
@click.option("--output", type=click.Path(path_type=Path), default=Path("results/"))
@click.option("--port", type=int, default=0, help="0 = random")
@click.option("--floor", type=float, default=0.90, help="Minimum pass rate")
@click.option(
    "--brain",
    type=str,
    default=DEFAULT_BRAIN,
    show_default=True,
    help=(
        "Pre-ingested brainId to target via POST /v1/brains/{brain}/ask. "
        "Populate with `memory ingest ./corpus --brain eval` before a run."
    ),
)
@click.option(
    "--top-k",
    type=int,
    default=DEFAULT_TOP_K,
    show_default=True,
    help="topK value forwarded on each /ask call.",
)
def main(
    sdk: str,
    mode: str,
    dataset: Path,
    scorer_kind: str,
    limit: int | None,
    output: Path,
    port: int,
    floor: float,
    brain: str,
    top_k: int,
) -> None:
    runner: SdkRunner = get_runner(sdk)
    runner.start(port=port)
    try:
        score = run_eval(
            endpoint=runner.endpoint,
            mode=mode,
            dataset=dataset,
            scorer_kind=scorer_kind,
            sdk=sdk,
            limit=limit,
            brain=brain,
            top_k=top_k,
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


if __name__ == "__main__":
    main()
