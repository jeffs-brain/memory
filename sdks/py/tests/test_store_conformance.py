# SPDX-License-Identifier: Apache-2.0
"""HTTP conformance harness — replays `spec/conformance/http-contract.json`.

Each case is a request/response pair derived from the TypeScript
reference suite. Running them against the in-process fake server keeps
the Python implementation honest about the shared wire protocol.
"""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import pytest

from ._fake_server import build_app

pytestmark = pytest.mark.asyncio

SPEC_PATH = (
    Path(__file__).resolve().parents[3]
    / "spec"
    / "conformance"
    / "http-contract.json"
)


@pytest.fixture(scope="module")
def contract() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text())


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=build_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _substitute(value: Any, brain_id: str, placeholders: dict[str, str]) -> Any:
    if isinstance(value, str):
        out = value.replace("BRAIN_ID", brain_id)
        for name, val in placeholders.items():
            if name == "BRAIN_ID":
                continue
            out = out.replace(name, val)
        return out
    if isinstance(value, list):
        return [_substitute(v, brain_id, placeholders) for v in value]
    if isinstance(value, dict):
        return {k: _substitute(v, brain_id, placeholders) for k, v in value.items()}
    return value


def _decode_body(step: dict[str, Any], placeholders: dict[str, str]) -> bytes | None:
    if "bodyBase64" in step:
        raw = step["bodyBase64"]
        raw = placeholders.get(raw, raw)
        return base64.b64decode(raw)
    if "bodyJson" in step:
        return json.dumps(step["bodyJson"]).encode()
    return None


async def _send(
    client: httpx.AsyncClient,
    step: dict[str, Any],
    brain_id: str,
    placeholders: dict[str, str],
) -> httpx.Response:
    method = step["method"]
    path = _substitute(step["path"], brain_id, placeholders)
    headers = {k.lower(): v for k, v in (step.get("headers") or {}).items()}
    body = _decode_body(step, placeholders)
    if "bodyJson" in step and "content-type" not in headers:
        headers["content-type"] = "application/json"
    return await client.request(method, path, content=body, headers=headers)


async def _run_case(
    client: httpx.AsyncClient,
    case: dict[str, Any],
    placeholders: dict[str, str],
) -> None:
    # SSE streaming over ASGITransport blocks indefinitely (in-process
    # ASGI offers no incremental body flush), so the harness skips those
    # two cases here. The SSE dispatch itself is covered by
    # test_sse_dispatch_parses_change_event in test_store_http.py.
    if case.get("request", {}).get("path", "").endswith("/events") and case["request"].get(
        "method"
    ) == "GET":
        pytest.skip("SSE event stream not supported over ASGITransport")
    if any(
        step.get("kind") in ("open-sse", "await-sse-event")
        for step in case.get("setup", [])
    ):
        pytest.skip("SSE setup steps not supported over ASGITransport")

    brain_id = f"brain-{uuid.uuid4().hex[:8]}"
    # Setup steps.
    for step in case.get("setup", []):
        resp = await _send(client, step, brain_id, placeholders)
        expected = step.get("expectedStatus")
        if expected is not None:
            assert resp.status_code == expected, (
                f"setup step {step['method']} {step['path']}: expected "
                f"{expected}, got {resp.status_code}"
            )
    request = case["request"]
    if request.get("kind") == "await-sse-event":
        pytest.skip("SSE await not supported over ASGITransport")
    response = await _send(client, request, brain_id, placeholders)
    expected = case["expectedResponse"]
    if "status" in expected:
        assert response.status_code == expected["status"], (
            f"case {case['name']!r}: expected {expected['status']} "
            f"got {response.status_code}: {response.text[:200]}"
        )
    if "bodyBase64" in expected:
        raw = placeholders.get(expected["bodyBase64"], expected["bodyBase64"])
        assert response.content == base64.b64decode(raw), case["name"]
    if "body" in expected:
        parsed = response.json()
        for k, v in expected["body"].items():
            if v == "<ISO-8601>":
                assert isinstance(parsed.get(k), str) and parsed.get(k)
                continue
            if isinstance(v, list):
                # Items list — compare paths only.
                got = [it.get("path") for it in parsed.get(k, [])]
                want = [item.get("path") for item in v]
                assert got == want, f"{case['name']}: items path mismatch"
                continue
            assert parsed.get(k) == v, f"{case['name']}: field {k}"
    if "bodyAssertions" in expected:
        parsed = response.json()
        items = parsed.get("items", [])
        for assertion in expected["bodyAssertions"]:
            kind = assertion["kind"]
            if kind == "items-include-path":
                assert any(it["path"] == assertion["path"] for it in items), (
                    f"{case['name']}: missing {assertion['path']}"
                )
            elif kind == "items-exclude-path":
                assert not any(it["path"] == assertion["path"] for it in items), (
                    f"{case['name']}: unexpected {assertion['path']}"
                )
            elif kind == "items-files-equal":
                got = sorted(it["path"] for it in items if not it.get("is_dir"))
                want = sorted(assertion["paths"])
                assert got == want, f"{case['name']}: files {got} != {want}"
            elif kind == "items-dirs-equal":
                got = sorted(it["path"] for it in items if it.get("is_dir"))
                want = sorted(assertion["paths"])
                assert got == want, f"{case['name']}: dirs {got} != {want}"
            elif kind == "json-field-equals":
                assert parsed.get(assertion["field"]) == assertion["value"]
            else:
                pytest.skip(f"unsupported assertion kind: {kind}")

    # Follow-up steps.
    for step in case.get("followUp", []):
        resp = await _send(client, step, brain_id, placeholders)
        if "expectedStatus" in step:
            assert resp.status_code == step["expectedStatus"], (
                f"{case['name']}: follow-up expected {step['expectedStatus']} got {resp.status_code}"
            )
        if "expectedBodyBase64" in step:
            raw = placeholders.get(step["expectedBodyBase64"], step["expectedBodyBase64"])
            assert resp.content == base64.b64decode(raw), case["name"]


_contract = json.loads(SPEC_PATH.read_text())
_CASES = _contract["cases"]
_PLACEHOLDERS = _contract.get("placeholders", {})


@pytest.mark.parametrize(
    "case",
    _CASES,
    ids=[c["name"] for c in _CASES],
)
async def test_conformance_case(
    client: httpx.AsyncClient, case: dict[str, Any]
) -> None:
    """Each conformance case is a standalone pytest node."""
    await _run_case(client, case, _PLACEHOLDERS)


async def test_conformance_pass_rate(
    client: httpx.AsyncClient, contract: dict[str, Any]
) -> None:
    """Aggregate check — require at least 27/29 cases to pass."""
    placeholders = contract.get("placeholders", {})
    passed = 0
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []
    for case in contract["cases"]:
        try:
            await _run_case(client, case, placeholders)
            passed += 1
        except pytest.skip.Exception as exc:
            skipped.append(f"{case['name']}: {exc}")
        except AssertionError as exc:
            failed.append((case["name"], str(exc)[:200]))
    total = len(contract["cases"])
    report = f"conformance: {passed}/{total} passed, {len(skipped)} skipped, {len(failed)} failed"
    if failed:
        details = "\n".join(f"  - {n}: {m}" for n, m in failed)
        pytest.fail(f"{report}\n{details}\nskipped: {skipped}")
    # 29 cases total; two are SSE-only and skipped under ASGITransport.
    # Demand at least 27 successful replays.
    assert passed >= 27, report
