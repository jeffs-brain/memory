# SPDX-License-Identifier: Apache-2.0
"""MemoryAgentBench dataset adapter.

The adapter keeps MemoryAgentBench-specific parsing here and imports the
shared benchmark models/scorers from the local benchmark package.
"""
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
from benchmarks.scoring import ExactContainmentScorer, JudgeBridgeScorer

DATASET_ID = "ai-hyz/MemoryAgentBench"
DATASET_REVISION = "0a3c1dfd4c434f07b5516d6f7bc510700d041577"
DATASET_LICENCE = "MIT"
ADAPTER_VERSION = "0.1.0"

SUPPORTED_SPLITS = ("Accurate_Retrieval", "Conflict_Resolution")
DEFAULT_SPLIT = "Accurate_Retrieval"
_DOCUMENT_MARKER_RE = re.compile(r"(?m)^Document\s+(\d+):\s*")


class MemoryAgentBenchAdapter:
    id = "memory-agent-bench"

    def __init__(self, *, revision: str = DATASET_REVISION) -> None:
        self.revision = revision
        self.source = BenchmarkSource(
            benchmark=self.id,
            url=f"https://huggingface.co/datasets/{DATASET_ID}",
            revision=revision,
            licence=DATASET_LICENCE,
            adapter_version=ADAPTER_VERSION,
        )

    def fetch(self, cache_dir: Path, *, split: str = DEFAULT_SPLIT) -> FetchResult:
        split = _normalise_split(split)
        return fetch_and_verify(
            url=_split_url(split, self.revision),
            cache_dir=cache_dir,
            benchmark=self.id,
            filename=f"{split}-00000-of-00001.parquet",
            revision=self.revision,
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
        rows = _read_rows(fetched.local_path)
        sample_ids_set = set(sample_ids or [])
        documents: list[CorpusDocument] = []
        questions: list[EvalQuestion] = []

        for row_index, row in enumerate(rows):
            metadata = _metadata(row)
            source = _optional_string(metadata.get("source"))
            if source_filter is not None and source != source_filter:
                continue
            if selected_split == "Accurate_Retrieval" and source == "LongMemEval":
                continue

            row_id = _row_id(metadata, row_index)
            row_questions = _strings(row.get("questions"))
            row_answers = _answers(row.get("answers"))
            question_dates = _strings(metadata.get("question_dates"))
            question_types = _strings(metadata.get("question_types"))
            qa_pair_ids = _strings(metadata.get("qa_pair_ids"))
            question_ids = _strings(metadata.get("question_ids"))

            row_question_ids = [
                _question_id(row_id, index, qa_pair_ids, question_ids)
                for index in range(len(row_questions))
            ]
            if sample_ids_set and not sample_ids_set.intersection(row_question_ids):
                continue

            documents.extend(
                _documents_for_row(
                    row=row,
                    metadata=metadata,
                    row_id=row_id,
                    split=selected_split,
                )
            )

            for index, question_text in enumerate(row_questions):
                question_id = row_question_ids[index]
                if sample_ids_set and question_id not in sample_ids_set:
                    continue
                questions.append(
                    EvalQuestion(
                        id=question_id,
                        question=question_text,
                        gold_answers=row_answers[index] if index < len(row_answers) else [],
                        category=(
                            question_types[index]
                            if index < len(question_types)
                            else selected_split
                        ),
                        source_id=row_id,
                        metadata={
                            "adapter_version": ADAPTER_VERSION,
                            "row_id": row_id,
                            "row_index": row_index,
                            "question_index": index,
                            "qa_pair_id": qa_pair_ids[index] if index < len(qa_pair_ids) else None,
                            "question_date": (
                                question_dates[index] if index < len(question_dates) else None
                            ),
                            "question_id": (
                                question_ids[index] if index < len(question_ids) else None
                            ),
                            "source": source,
                            "split": selected_split,
                        },
                    )
                )
                if limit is not None and len(questions) >= limit:
                    return _normalised(self.source, fetched.sha256, documents, questions)

        return _normalised(self.source, fetched.sha256, documents, questions)

    def default_scorer(self) -> JudgeBridgeScorer:
        return JudgeBridgeScorer()

    def scorer_for(self, name: str) -> JudgeBridgeScorer | ExactContainmentScorer:
        if name == "judge":
            return JudgeBridgeScorer()
        if name == "exact-containment":
            return ExactContainmentScorer()
        expected = ", ".join(self.available_scorers())
        raise ValueError(
            f"unsupported MemoryAgentBench scorer {name!r}; expected one of: {expected}"
        )

    def available_scorers(self) -> list[str]:
        return ["judge", "exact-containment"]


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


def _read_rows(local_path: Path) -> list[dict[str, Any]]:
    if local_path.suffix == ".json":
        data = json.loads(local_path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else list(data.get("rows", []))
    if local_path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in local_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "MemoryAgentBench parquet parsing requires pyarrow. "
            "Install the eval benchmarks extra, for example: uv sync --extra benchmarks"
        ) from exc

    table = parquet.read_table(local_path)
    return [dict(row) for row in table.to_pylist()]


def _documents_for_row(
    *,
    row: dict[str, Any],
    metadata: dict[str, Any],
    row_id: str,
    split: str,
) -> list[CorpusDocument]:
    haystack_sessions = metadata.get("haystack_sessions")
    if haystack_sessions:
        return _session_documents(haystack_sessions, row_id=row_id, split=split, metadata=metadata)

    context = row.get("context")
    if not isinstance(context, str) or not context.strip():
        return []
    return _context_documents(context, row_id=row_id, split=split, metadata=metadata)


def _session_documents(
    haystack_sessions: Any,
    *,
    row_id: str,
    split: str,
    metadata: dict[str, Any],
) -> list[CorpusDocument]:
    documents: list[CorpusDocument] = []
    for session_index, session in enumerate(haystack_sessions or []):
        turns = _flatten_turns(session)
        lines = [f"# MemoryAgentBench {split} row {row_id} session {session_index + 1}"]
        for turn_index, turn in enumerate(turns):
            role = _optional_string(turn.get("role")) or "unknown"
            content = _optional_string(turn.get("content")) or ""
            has_answer = turn.get("has_answer")
            lines.append("")
            lines.append(f"## Turn {turn_index + 1}: {role}")
            if has_answer is not None:
                lines.append(f"has_answer: {bool(has_answer)}")
            lines.append(content)
        documents.append(
            CorpusDocument(
                path=f"memory-agent-bench/{split}/{row_id}/session-{session_index + 1}.md",
                content="\n".join(lines).strip() + "\n",
                metadata=_document_metadata(metadata, row_id, split, session_index + 1),
            )
        )
    return documents


def _context_documents(
    context: str,
    *,
    row_id: str,
    split: str,
    metadata: dict[str, Any],
) -> list[CorpusDocument]:
    matches = list(_DOCUMENT_MARKER_RE.finditer(context))
    if not matches:
        return [
            CorpusDocument(
                path=f"memory-agent-bench/{split}/{row_id}/context.md",
                content=context.strip() + "\n",
                metadata=_document_metadata(metadata, row_id, split, 1),
            )
        ]

    documents: list[CorpusDocument] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(context)
        document_number = int(match.group(1))
        content = context[start:end].strip()
        if not content:
            continue
        documents.append(
            CorpusDocument(
                path=f"memory-agent-bench/{split}/{row_id}/document-{document_number}.md",
                content=f"# Document {document_number}\n\n{content}\n",
                metadata=_document_metadata(metadata, row_id, split, document_number),
            )
        )
    return documents


def _document_metadata(
    metadata: dict[str, Any],
    row_id: str,
    split: str,
    document_index: int,
) -> dict[str, str | int | float | bool | None]:
    return {
        "benchmark": "MemoryAgentBench",
        "split": split,
        "row_id": row_id,
        "document_index": document_index,
        "source": _optional_string(metadata.get("source")),
    }


def _flatten_turns(session: Any) -> list[dict[str, Any]]:
    if isinstance(session, dict):
        return [session]
    turns: list[dict[str, Any]] = []
    if isinstance(session, list):
        for item in session:
            turns.extend(_flatten_turns(item))
    return turns


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _answers(value: Any) -> list[list[str]]:
    if not isinstance(value, list):
        return []
    answers: list[list[str]] = []
    for item in value:
        if isinstance(item, str):
            answers.append([item])
        elif isinstance(item, list):
            answers.append([answer for answer in item if isinstance(answer, str)])
        else:
            answers.append([])
    return answers


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _row_id(metadata: dict[str, Any], row_index: int) -> str:
    source = _optional_string(metadata.get("source")) or "row"
    return f"{_slug(source)}-{row_index + 1}"


def _question_id(row_id: str, index: int, qa_pair_ids: list[str], question_ids: list[str]) -> str:
    if index < len(qa_pair_ids) and qa_pair_ids[index]:
        return qa_pair_ids[index]
    if index < len(question_ids) and question_ids[index]:
        return question_ids[index]
    return f"{row_id}-q{index + 1}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return slug or "row"


def _normalise_split(split: str) -> str:
    if split not in SUPPORTED_SPLITS:
        expected = ", ".join(SUPPORTED_SPLITS)
        raise ValueError(
            f"unsupported MemoryAgentBench split {split!r}; expected one of: {expected}"
        )
    return split


def _split_from_path(path: Path) -> str:
    for split in SUPPORTED_SPLITS:
        if path.name.startswith(split):
            return split
    return DEFAULT_SPLIT


def _split_url(split: str, revision: str) -> str:
    return (
        f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{revision}/data/"
        f"{split}-00000-of-00001.parquet"
    )
