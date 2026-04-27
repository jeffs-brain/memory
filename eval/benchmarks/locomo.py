# SPDX-License-Identifier: Apache-2.0
"""LoCoMo benchmark adapter."""
from __future__ import annotations

import json
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
from benchmarks.fetch import fetch_and_verify
from benchmarks.scoring import AdversarialAbstentionScorer, JudgeBridgeScorer, TokenF1Scorer


LOCOMO_REVISION = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
LOCOMO_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/"
    f"{LOCOMO_REVISION}/data/locomo10.json"
)
LOCOMO_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
LOCOMO_LICENCE = "CC-BY-NC-4.0"
ADAPTER_VERSION = "0.1.0"

_SESSION_KEY_RE = re.compile(r"^session_(\d+)$")
_EVIDENCE_RE = re.compile(r"\[([^\]]+)\]")


class LoCoMoAdapter:
    id = "locomo"
    source = BenchmarkSource(
        benchmark="locomo",
        url=LOCOMO_URL,
        revision=LOCOMO_REVISION,
        sha256=LOCOMO_SHA256,
        licence=LOCOMO_LICENCE,
        adapter_version=ADAPTER_VERSION,
    )

    def fetch(self, cache_dir: Path) -> FetchResult:
        return fetch_and_verify(
            url=LOCOMO_URL,
            cache_dir=cache_dir,
            benchmark=self.id,
            filename="locomo10.json",
            expected_sha256=LOCOMO_SHA256,
            revision=LOCOMO_REVISION,
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
        if split not in (None, "qa"):
            raise ValueError(f"unsupported LoCoMo split: {split}")

        raw = json.loads(fetched.local_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("LoCoMo dataset must be a JSON list")

        documents: list[CorpusDocument] = []
        questions: list[EvalQuestion] = []
        wanted_ids = set(sample_ids or [])

        for conversation_idx, entry in enumerate(raw, start=1):
            if not isinstance(entry, dict):
                raise ValueError(f"LoCoMo entry {conversation_idx} must be an object")

            conversation_id = _conversation_id(entry, conversation_idx)
            if source_filter is not None and conversation_id != source_filter:
                continue

            documents.extend(_normalise_documents(entry, conversation_id))

            for qa_idx, qa in enumerate(_qas(entry), start=1):
                question = _normalise_question(
                    qa=qa,
                    conversation_id=conversation_id,
                    qa_idx=qa_idx,
                )
                if wanted_ids and question.id not in wanted_ids:
                    continue
                questions.append(question)
                if limit is not None and len(questions) >= limit:
                    return NormalisedBenchmark(
                        source=self.source.model_copy(update={"sha256": fetched.sha256}),
                        documents=documents,
                        questions=questions,
                    )

        return NormalisedBenchmark(
            source=self.source.model_copy(update={"sha256": fetched.sha256}),
            documents=documents,
            questions=questions,
        )

    def default_scorer(self) -> TokenF1Scorer:
        return TokenF1Scorer()

    def available_scorers(self) -> list[str]:
        return ["token-f1", "adversarial", "judge"]

    def scorer_for(
        self,
        name: str,
    ) -> TokenF1Scorer | AdversarialAbstentionScorer | JudgeBridgeScorer:
        return scorer_for_name(name)


def evidence_recall(
    *,
    question: EvalQuestion,
    citations: list[dict[str, Any]],
) -> float | None:
    expected = set(question.evidence_ids)
    if not expected:
        return None

    found: set[str] = set()
    for citation in citations:
        for key in ("path", "text", "summary", "content"):
            value = citation.get(key)
            if isinstance(value, str):
                found.update(_EVIDENCE_RE.findall(value))

    return len(expected & found) / len(expected)


def scorer_for_name(name: str) -> TokenF1Scorer | AdversarialAbstentionScorer | JudgeBridgeScorer:
    if name == "token-f1":
        return TokenF1Scorer()
    if name == "adversarial":
        return AdversarialAbstentionScorer()
    if name == "judge":
        return JudgeBridgeScorer()
    raise ValueError(f"unsupported LoCoMo scorer: {name}")


def _conversation_id(entry: dict[str, Any], fallback_idx: int) -> str:
    for key in ("conversation_id", "sample_id"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"conversation-{fallback_idx}"


def _normalise_documents(entry: dict[str, Any], conversation_id: str) -> list[CorpusDocument]:
    if isinstance(entry.get("session_list"), list):
        return _documents_from_session_list(entry["session_list"], conversation_id)

    conversation = entry.get("conversation")
    if isinstance(conversation, dict):
        return _documents_from_conversation(conversation, conversation_id)

    raise ValueError(f"LoCoMo conversation {conversation_id} has no supported session data")


def _documents_from_session_list(
    session_list: list[Any],
    conversation_id: str,
) -> list[CorpusDocument]:
    documents: list[CorpusDocument] = []
    for idx, session in enumerate(session_list, start=1):
        if not isinstance(session, dict):
            raise ValueError(f"LoCoMo session {conversation_id}/{idx} must be an object")

        turns = _turns_from_session_object(session)
        timestamp = _string_or_none(
            session.get("timestamp")
            or session.get("date_time")
            or session.get("session_date_time")
        )
        documents.append(_document(conversation_id, idx, timestamp, turns))
    return documents


def _documents_from_conversation(
    conversation: dict[str, Any],
    conversation_id: str,
) -> list[CorpusDocument]:
    session_items: list[tuple[int, list[Any], str | None]] = []
    for key, value in conversation.items():
        match = _SESSION_KEY_RE.match(key)
        if match is None or not isinstance(value, list):
            continue
        idx = int(match.group(1))
        session_items.append((idx, value, _string_or_none(conversation.get(f"{key}_date_time"))))

    return [
        _document(conversation_id, idx, timestamp, turns)
        for idx, turns, timestamp in sorted(session_items, key=lambda item: item[0])
    ]


def _document(
    conversation_id: str,
    session_idx: int,
    timestamp: str | None,
    turns: list[Any],
) -> CorpusDocument:
    lines = [
        "---",
        "benchmark: locomo",
        f"conversation_id: {conversation_id}",
        f"session_id: session-{session_idx}",
    ]
    if timestamp:
        lines.append(f"timestamp: {timestamp}")
    lines.extend(["---", ""])

    for turn_idx, turn in enumerate(turns, start=1):
        if not isinstance(turn, dict):
            raise ValueError(
                f"LoCoMo turn {conversation_id}/session-{session_idx}/{turn_idx} must be an object"
            )
        dia_id = _required_string(turn, "dia_id")
        speaker = _required_string(turn, "speaker")
        text = _required_string(turn, "text")
        lines.append(f"[{dia_id}] **{speaker}**: {text}")

    return CorpusDocument(
        path=f"locomo/{conversation_id}/session-{session_idx}.md",
        content="\n".join(lines).rstrip() + "\n",
        metadata={
            "benchmark": "locomo",
            "conversation_id": conversation_id,
            "session_id": f"session-{session_idx}",
            "timestamp": timestamp,
        },
    )


def _turns_from_session_object(session: dict[str, Any]) -> list[Any]:
    for key in ("turns", "dialogue", "dialogs", "messages"):
        value = session.get(key)
        if isinstance(value, list):
            return value
    raise ValueError("LoCoMo session object has no turns/dialogue/messages list")


def _qas(entry: dict[str, Any]) -> list[Any]:
    value = entry.get("qas")
    if isinstance(value, list):
        return value
    value = entry.get("qa")
    if isinstance(value, list):
        return value
    return []


def _normalise_question(
    *,
    qa: Any,
    conversation_id: str,
    qa_idx: int,
) -> EvalQuestion:
    if not isinstance(qa, dict):
        raise ValueError(f"LoCoMo QA {conversation_id}/{qa_idx} must be an object")

    gold = qa.get("answer")
    gold_answers = [] if gold is None else [str(gold)]
    evidence = qa.get("evidence")
    evidence_ids = [str(item) for item in evidence] if isinstance(evidence, list) else []
    category = str(qa.get("category", "unknown"))

    return EvalQuestion(
        id=f"locomo-{conversation_id}-{qa_idx}",
        question=_required_string(qa, "question"),
        gold_answers=gold_answers,
        category=category,
        evidence_ids=evidence_ids,
        source_id=conversation_id,
        metadata={"adapter_version": ADAPTER_VERSION},
    )


def _required_string(value: dict[str, Any], key: str) -> str:
    raw = value.get(key)
    if raw is None:
        raise ValueError(f"missing required LoCoMo field: {key}")
    text = str(raw).strip()
    if not text:
        raise ValueError(f"empty required LoCoMo field: {key}")
    return text


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
