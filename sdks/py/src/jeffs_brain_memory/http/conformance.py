# SPDX-License-Identifier: Apache-2.0
"""Replay harness for `spec/conformance/http-contract.json`.

Exposes a single entry point, :func:`replay_cases`, used by tests to
drive the daemon through every conformance case. A fresh brain is
provisioned per case so state never leaks between assertions.

Because Starlette's `TestClient` and httpx's `ASGITransport` both
buffer response bodies to EOF, the harness drives each case against a
short-lived uvicorn server on a random port so SSE frames flush on
arrival.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import httpx
import uvicorn
from starlette.applications import Starlette


@dataclass(slots=True)
class CaseResult:
    name: str
    ok: bool
    skipped: bool = False
    error: str | None = None


@dataclass(slots=True)
class ConformanceReport:
    results: list[CaseResult] = field(default_factory=list)
    skipped: int = 0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.ok and not r.skipped)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.ok)


SKIP_CASES: dict[str, str] = {
    "auth header forwarded when apiKey is set": (
        "harness asserts header forwarding rather than server behaviour"
    ),
}


def load_conformance_doc(spec_dir: Path) -> dict[str, Any]:
    fixture = spec_dir / "conformance" / "http-contract.json"
    with fixture.open("r", encoding="utf-8") as fh:
        return json.load(fh)


async def replay_cases(
    app_factory: Callable[[], Starlette],
    *,
    spec_dir: Path,
) -> ConformanceReport:
    """Replay every conformance case against a freshly built app."""
    doc = load_conformance_doc(spec_dir)
    placeholders = dict(doc.get("placeholders") or {})
    cases = list(doc.get("cases") or [])

    report = ConformanceReport()
    loop = asyncio.get_running_loop()
    for case in cases:
        name = case.get("name", "")
        if name in SKIP_CASES:
            report.skipped += 1
            report.results.append(CaseResult(name=name, ok=True, skipped=True))
            continue
        try:
            await loop.run_in_executor(
                None, _run_case_sync, app_factory, case, placeholders
            )
            report.results.append(CaseResult(name=name, ok=True))
        except AssertionError as exc:
            report.results.append(
                CaseResult(name=name, ok=False, error=str(exc))
            )
        except Exception as exc:  # noqa: BLE001
            report.results.append(
                CaseResult(name=name, ok=False, error=repr(exc))
            )
    return report


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


@contextlib.contextmanager
def _uvicorn_server(app: Starlette) -> Iterator[str]:
    port = _free_port()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        lifespan="on",
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.1)
                probe.connect(("127.0.0.1", port))
            break
        except OSError:
            time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


def _run_case_sync(
    app_factory: Callable[[], Starlette],
    case: dict[str, Any],
    placeholders: dict[str, str],
) -> None:
    brain_id = "conformance-brain"
    app = app_factory()
    with _uvicorn_server(app) as base_url:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            created = client.post("/v1/brains", json={"brainId": brain_id})
            if created.status_code not in (201, 409):
                raise AssertionError(
                    f"provisioning brain failed: {created.status_code} "
                    f"{created.text}"
                )

            subs = {k: v for k, v in placeholders.items() if k != "BRAIN_ID"}
            subs["BRAIN_ID"] = brain_id
            ordered = sorted(subs.keys(), key=len, reverse=True)

            def substitute(value: str) -> str:
                out = value
                for key in ordered:
                    out = out.replace(key, subs[key])
                return out

            sse_pool: dict[str, _SSESubscriber] = {}
            try:
                for step in case.get("setup", []) or []:
                    _run_step(client, sse_pool, step, substitute, base_url)

                request = case.get("request", {}) or {}
                expected = case.get("expectedResponse", {}) or {}
                if request.get("kind") == "await-sse-event":
                    sub = sse_pool.get(request.get("name", ""))
                    if sub is None:
                        raise AssertionError(
                            f"SSE subscriber {request.get('name')!r} not opened"
                        )
                    event_name = request.get("event", "")
                    raw, ok = sub.wait_for_event(event_name, timeout=5.0)
                    if not ok:
                        raise AssertionError(
                            f"timeout waiting for SSE event {event_name!r}"
                        )
                    _assert_sse_event(expected, raw)
                else:
                    if _is_sse_expected(expected):
                        _handle_sse_request(request, expected, substitute, base_url)
                    else:
                        resp = _do_request(client, request, substitute)
                        _assert_expected_response(expected, resp, substitute)

                for step in case.get("followUp", []) or []:
                    _run_step(client, sse_pool, step, substitute, base_url)
                for step in case.get("teardown", []) or []:
                    _run_step(client, sse_pool, step, substitute, base_url)
            finally:
                for sub in sse_pool.values():
                    sub.close()


@dataclass
class _SSESubscriber:
    url: str
    events: list[tuple[str, str]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _cond: threading.Condition = field(init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _stop: threading.Event = field(default_factory=threading.Event)
    _client: httpx.Client | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._cond = threading.Condition(self._lock)

    def start(self) -> None:
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=3.0, read=5.0, write=3.0, pool=3.0)
        )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._client is not None
        try:
            with self._client.stream(
                "GET",
                self.url,
                headers={"Accept": "text/event-stream"},
            ) as resp:
                event = ""
                data_buf: list[str] = []
                for line in resp.iter_lines():
                    if self._stop.is_set():
                        return
                    if line == "":
                        if data_buf or event:
                            payload = "\n".join(data_buf)
                            with self._cond:
                                self.events.append((event or "message", payload))
                                self._cond.notify_all()
                        event = ""
                        data_buf = []
                        continue
                    if line.startswith("event:"):
                        event = line[len("event:"):].strip()
                    elif line.startswith("data:"):
                        data_buf.append(line[len("data:"):].lstrip())
        except Exception:
            return

    def wait_for_event(self, name: str, timeout: float) -> tuple[str, bool]:
        deadline = time.monotonic() + timeout
        with self._cond:
            cursor = 0
            while True:
                while cursor < len(self.events):
                    evt, data = self.events[cursor]
                    cursor += 1
                    if evt == name:
                        return data, True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return "", False
                self._cond.wait(timeout=remaining)

    def close(self) -> None:
        self._stop.set()
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def _run_step(
    client: httpx.Client,
    pool: dict[str, _SSESubscriber],
    step: dict[str, Any],
    substitute: Callable[[str], str],
    base_url: str,
) -> None:
    kind = step.get("kind", "")
    if kind == "open-sse":
        name = step.get("name", "")
        if name in pool:
            pool[name].close()
        sub = _SSESubscriber(url=base_url + substitute(step.get("path", "")))
        sub.start()
        pool[name] = sub
        return
    if kind == "await-sse-event":
        sub = pool.get(step.get("name", ""))
        if sub is None:
            raise AssertionError(
                f"SSE subscriber {step.get('name')!r} not opened"
            )
        _, ok = sub.wait_for_event(step.get("event", ""), timeout=5.0)
        if not ok:
            raise AssertionError(
                f"timeout waiting for SSE event {step.get('event')!r}"
            )
        return
    if kind == "close-sse":
        sub = pool.pop(step.get("name", ""), None)
        if sub is not None:
            sub.close()
        return
    if kind == "":
        resp = _do_request(client, step, substitute)
        expected_status = step.get("expectedStatus", 0)
        if expected_status and resp.status_code != expected_status:
            raise AssertionError(
                f"setup step {step.get('method')} {step.get('path')}: "
                f"want status {expected_status} got {resp.status_code} "
                f"body={resp.content!r}"
            )
        expected_body = step.get("expectedBodyBase64", "")
        if expected_body:
            want = base64.b64decode(substitute(expected_body))
            if resp.content != want:
                raise AssertionError(
                    f"setup step body mismatch: want {want!r} got {resp.content!r}"
                )
        return
    raise AssertionError(f"unknown step kind {kind!r}")


def _do_request(
    client: httpx.Client,
    step: dict[str, Any],
    substitute: Callable[[str], str],
) -> httpx.Response:
    method = (step.get("method") or "GET").upper()
    target = substitute(step.get("path", ""))
    headers: dict[str, str] = {}
    for key, value in (step.get("headers") or {}).items():
        if isinstance(value, bool):
            headers[key] = "true" if value else "false"
        elif isinstance(value, (int, float)):
            headers[key] = str(value)
        elif isinstance(value, str):
            headers[key] = value
        else:
            headers[key] = str(value)
    content: bytes | None = None
    json_body: Any = None
    if step.get("bodyBase64"):
        content = base64.b64decode(substitute(step["bodyBase64"]))
        headers.setdefault("Content-Type", "application/octet-stream")
    elif step.get("bodyJson") is not None:
        json_body = step["bodyJson"]
        headers.setdefault("Content-Type", "application/json")

    if json_body is not None:
        return client.request(method, target, json=json_body, headers=headers)
    return client.request(method, target, content=content, headers=headers)


def _is_sse_expected(expected: dict[str, Any]) -> bool:
    return bool(expected.get("streamAssertions")) or (
        isinstance(expected.get("contentType"), str)
        and "text/event-stream" in expected["contentType"]
    )


def _handle_sse_request(
    request: dict[str, Any],
    expected: dict[str, Any],
    substitute: Callable[[str], str],
    base_url: str,
) -> None:
    method = (request.get("method") or "GET").upper()
    if method != "GET":
        raise AssertionError(f"SSE request expects GET, got {method!r}")
    target = substitute(request.get("path", ""))
    wanted: set[str] = set()
    for assertion in expected.get("streamAssertions") or []:
        if isinstance(assertion, dict) and assertion.get("kind") == "expect-event":
            evt = assertion.get("event")
            if isinstance(evt, str):
                wanted.add(evt)
    seen: set[str] = set()
    try:
        with httpx.stream(
            "GET",
            base_url + target,
            headers={"Accept": "text/event-stream"},
            timeout=httpx.Timeout(3.0),
        ) as resp:
            if "status" in expected and resp.status_code != int(expected["status"]):
                raise AssertionError(
                    f"want status {expected['status']} got {resp.status_code}"
                )
            deadline = time.monotonic() + 3.0
            event_name = ""
            for line in resp.iter_lines():
                if time.monotonic() > deadline:
                    break
                if line == "":
                    if event_name and event_name in wanted:
                        seen.add(event_name)
                        if seen == wanted:
                            break
                    event_name = ""
                    continue
                if line.startswith("event:"):
                    event_name = line[len("event:"):].strip()
    except httpx.ReadTimeout:
        pass
    missing = wanted - seen
    if missing:
        raise AssertionError(
            f"expected SSE events {sorted(missing)} never arrived"
        )


def _assert_expected_response(
    expected: dict[str, Any],
    resp: httpx.Response,
    substitute: Callable[[str], str],
) -> None:
    status = expected.get("status")
    if isinstance(status, (int, float)) and int(status) != resp.status_code:
        raise AssertionError(
            f"want status {int(status)} got {resp.status_code} body={resp.content!r}"
        )
    content_type = expected.get("contentType")
    if isinstance(content_type, str):
        actual_ct = resp.headers.get("content-type", "")
        if content_type not in actual_ct:
            raise AssertionError(
                f"want content-type containing {content_type!r} got {actual_ct!r}"
            )
    body_b64 = expected.get("bodyBase64")
    if isinstance(body_b64, str):
        want = base64.b64decode(substitute(body_b64))
        if resp.content != want:
            raise AssertionError(
                f"body mismatch: want {want!r} got {resp.content!r}"
            )
    body_obj = expected.get("body")
    if body_obj is not None:
        _assert_json_matches(body_obj, resp.content)
    body_assertions = expected.get("bodyAssertions")
    if isinstance(body_assertions, list):
        for assertion in body_assertions:
            if isinstance(assertion, dict):
                _run_body_assertion(assertion, resp.content)


def _assert_sse_event(expected: dict[str, Any], raw: str) -> None:
    body_assertions = expected.get("bodyAssertions")
    if isinstance(body_assertions, list):
        for assertion in body_assertions:
            if isinstance(assertion, dict):
                _run_body_assertion(assertion, raw.encode("utf-8"))


def _assert_json_matches(expected: Any, actual: bytes) -> None:
    try:
        got = json.loads(actual.decode("utf-8"))
    except Exception as exc:
        raise AssertionError(
            f"decode response JSON: {exc} body={actual!r}"
        ) from exc
    error = _compare_json(expected, got)
    if error is not None:
        raise AssertionError(
            f"JSON mismatch: {error}\nwant={expected}\ngot={got}"
        )


def _compare_json(expected: Any, actual: Any) -> str | None:
    from datetime import datetime as _dt

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return f"want object got {type(actual).__name__}"
        for key, value in expected.items():
            if key not in actual:
                return f"missing key {key!r}"
            err = _compare_json(value, actual[key])
            if err is not None:
                return f"{key!r}: {err}"
        return None
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return f"want array got {type(actual).__name__}"
        if len(actual) < len(expected):
            return f"array length {len(actual)} < expected {len(expected)}"
        for i, value in enumerate(expected):
            err = _compare_json(value, actual[i])
            if err is not None:
                return f"[{i}]: {err}"
        return None
    if isinstance(expected, str):
        if expected == "<ISO-8601>":
            if not isinstance(actual, str):
                return f"want ISO-8601 string got {type(actual).__name__}"
            try:
                _dt.fromisoformat(actual.replace("Z", "+00:00"))
            except ValueError as exc:
                return f"not ISO-8601: {exc}"
            return None
        if actual != expected:
            return f"want {expected!r} got {actual!r}"
        return None
    if isinstance(expected, bool):
        if actual != expected:
            return f"want {expected!r} got {actual!r}"
        return None
    if isinstance(expected, (int, float)):
        if not isinstance(actual, (int, float)) or float(actual) != float(expected):
            return f"want number {expected} got {actual}"
        return None
    if expected is None:
        if actual is not None:
            return f"want null got {actual!r}"
        return None
    return f"unhandled expected type {type(expected).__name__}"


def _run_body_assertion(assertion: dict[str, Any], body: bytes) -> None:
    kind = assertion.get("kind", "")
    if kind == "items-include-path":
        want = assertion.get("path", "")
        items = _extract_items(body)
        for item in items:
            if item.get("path") == want:
                return
        raise AssertionError(f"items do not include {want!r}. items={items}")
    if kind == "items-exclude-path":
        unwanted = assertion.get("path", "")
        items = _extract_items(body)
        for item in items:
            if item.get("path") == unwanted:
                raise AssertionError(
                    f"items unexpectedly include {unwanted!r}. items={items}"
                )
        return
    if kind == "items-files-equal":
        want = assertion.get("paths") or []
        items = _extract_items(body)
        got = [
            it["path"]
            for it in items
            if not it.get("is_dir") and it.get("path")
        ]
        if sorted(got) != sorted(want):
            raise AssertionError(f"items-files-equal: want {want} got {got}")
        return
    if kind == "items-dirs-equal":
        want = assertion.get("paths") or []
        items = _extract_items(body)
        got = [
            it["path"]
            for it in items
            if it.get("is_dir") and it.get("path")
        ]
        if sorted(got) != sorted(want):
            raise AssertionError(f"items-dirs-equal: want {want} got {got}")
        return
    if kind == "json-field-equals":
        field_name = assertion.get("field", "")
        want_value = assertion.get("value")
        try:
            parsed = json.loads(body.decode("utf-8"))
        except Exception as exc:
            raise AssertionError(f"decode body: {exc} body={body!r}") from exc
        if not isinstance(parsed, dict):
            raise AssertionError(
                f"json-field-equals: body is not an object: {parsed!r}"
            )
        got = parsed.get(field_name)
        if got != want_value:
            raise AssertionError(
                f"field {field_name!r}: want {want_value!r} got {got!r}"
            )
        return
    raise AssertionError(f"unknown bodyAssertion kind {kind!r}")


def _extract_items(body: bytes) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise AssertionError(f"decode list body: {exc} body={body!r}") from exc
    if isinstance(parsed, dict):
        items = parsed.get("items")
        if isinstance(items, list):
            return [i for i in items if isinstance(i, dict)]
    raise AssertionError(f"decode list body: missing items array, body={body!r}")


__all__ = [
    "CaseResult",
    "ConformanceReport",
    "load_conformance_doc",
    "replay_cases",
    "SKIP_CASES",
]
