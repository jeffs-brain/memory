#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from collections import defaultdict
from typing import Any


def deterministic_select(items: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count >= len(items):
        return list(items)

    indices = list(range(len(items)))
    state = seed & ((1 << 64) - 1)
    for index in range(len(indices) - 1, 0, -1):
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        swap = (state >> 33) % (index + 1)
        indices[index], indices[swap] = indices[swap], indices[index]

    return [items[indices[index]] for index in range(count)]


def load_sample_ids(sample_ids_file: str) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for line in pathlib.Path(sample_ids_file).read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value:
            continue
        if value in seen:
            raise SystemExit(f"ERROR: duplicate sample id in {sample_ids_file}: {value}")
        seen.add(value)
        ids.append(value)
    return ids


def selected_question_ids(
    questions: list[dict[str, Any]],
    sample_size: int,
    seed: int,
    sample_ids_file: str,
) -> list[str]:
    if sample_ids_file:
        return load_sample_ids(sample_ids_file)

    if sample_size <= 0 or sample_size >= len(questions):
        return [
            str(question.get("question_id", "")).strip()
            for question in questions
            if str(question.get("question_id", "")).strip()
        ]

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        by_category[str(question.get("question_type", ""))].append(question)

    total = len(questions)
    allocations: list[dict[str, Any]] = []
    allocated = 0
    for category in sorted(by_category):
        category_questions = by_category[category]
        numerator = len(category_questions) * sample_size
        allocation = min(len(category_questions), numerator // total)
        allocations.append(
            {
                "category": category,
                "questions": category_questions,
                "alloc": allocation,
                "remainder": numerator % total,
            }
        )
        allocated += allocation

    remaining = sample_size - allocated
    allocations.sort(key=lambda item: (-int(item["remainder"]), str(item["category"])))
    for item in allocations:
        if remaining == 0:
            break
        if int(item["alloc"]) >= len(item["questions"]):
            continue
        item["alloc"] = int(item["alloc"]) + 1
        remaining -= 1
    allocations.sort(key=lambda item: str(item["category"]))

    selected: list[dict[str, Any]] = []
    for item in allocations:
        selected.extend(deterministic_select(item["questions"], int(item["alloc"]), seed))

    return [
        str(question.get("question_id", "")).strip()
        for question in selected
        if str(question.get("question_id", "")).strip()
    ]


def normalise_bool(raw: str) -> bool:
    return raw.strip().lower() not in {"", "0", "false"}


def normalise_extract_heuristics(raw: str) -> str:
    value = raw.strip().lower()
    return value if value else "default"


def build_inputs(args: argparse.Namespace) -> dict[str, Any]:
    raw_dataset = pathlib.Path(args.dataset).read_bytes()
    questions = json.loads(raw_dataset.decode("utf-8"))
    sample_ids = selected_question_ids(questions, args.sample_size, args.seed, args.sample_ids_file or "")
    sample_signature = hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()
    return {
        "dataset_sha": hashlib.sha256(raw_dataset).hexdigest(),
        "sample_signature": sample_signature,
        "sample_ids": sample_ids,
        "ingest_mode": "replay",
        "extract_model": args.extract_model,
        "extract_heuristics": normalise_extract_heuristics(args.extract_heuristics),
        "extraction_pipeline": args.extraction_pipeline,
        "contextualise": normalise_bool(args.contextualise),
    }


def cache_signature(inputs: dict[str, Any]) -> str:
    encoded = json.dumps(inputs, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def command_signature(args: argparse.Namespace) -> int:
    print(cache_signature(build_inputs(args)))
    return 0


def command_inputs(args: argparse.Namespace) -> int:
    inputs = build_inputs(args)
    print(json.dumps(inputs, ensure_ascii=True, indent=2))
    return 0


def command_validate(args: argparse.Namespace) -> int:
    manifest_path = pathlib.Path(args.manifest)
    if not manifest_path.is_file():
        print(f"ERROR: cache manifest missing at {manifest_path}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_inputs = manifest.get("cache_signature_inputs")
    actual_signature = manifest.get("cache_signature")
    if not isinstance(actual_inputs, dict) or not isinstance(actual_signature, str) or not actual_signature:
        print(f"ERROR: cache manifest at {manifest_path} has no cache signature", file=sys.stderr)
        return 1

    expected_inputs = build_inputs(args)
    expected_signature = cache_signature(expected_inputs)
    for key, expected_value in expected_inputs.items():
        actual_value = actual_inputs.get(key)
        if actual_value != expected_value:
            print(
                f"ERROR: cache mismatch for {key}: got {actual_value!r}, expected {expected_value!r}",
                file=sys.stderr,
            )
            return 1

    if actual_signature != expected_signature:
        print(
            f"ERROR: cache signature mismatch: got {actual_signature}, expected {expected_signature}",
            file=sys.stderr,
        )
        return 1

    return 0


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sample-size", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--sample-ids-file", default="")
    parser.add_argument("--extract-model", required=True)
    parser.add_argument("--replay-concurrency", required=True, type=int)
    parser.add_argument("--contextualise", required=True)
    parser.add_argument("--extract-heuristics", default="")
    parser.add_argument("--extraction-pipeline", required=True, type=int)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LongMemEval replay cache key helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    signature = subparsers.add_parser("signature")
    add_common_args(signature)
    signature.set_defaults(func=command_signature)

    inputs = subparsers.add_parser("inputs")
    add_common_args(inputs)
    inputs.set_defaults(func=command_inputs)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    add_common_args(validate)
    validate.set_defaults(func=command_validate)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
