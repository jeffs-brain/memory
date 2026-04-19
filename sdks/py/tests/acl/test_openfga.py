# SPDX-License-Identifier: Apache-2.0
"""Behavioural specs for the OpenFGA HTTP adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx
import pytest

from jeffs_brain_memory.acl import (
    ReadTuplesQuery,
    Resource,
    Subject,
    Tuple,
    WriteTuplesRequest,
)
from jeffs_brain_memory.acl_openfga import (
    OpenFgaHTTPError,
    OpenFgaOptions,
    OpenFgaRequestError,
    create_openfga_provider,
)

pytestmark = pytest.mark.asyncio


def _user(uid: str) -> Subject:
    return Subject(kind="user", id=uid)


def _brain(rid: str) -> Resource:
    return Resource(type="brain", id=rid)


@dataclass(slots=True)
class _Recorder:
    calls: list[httpx.Request] = field(default_factory=list)

    def transport(self, responder: Callable[[httpx.Request], httpx.Response]) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            self.calls.append(request)
            return responder(request)

        return httpx.MockTransport(handler)


def _client(transport: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=transport)


def _json_response(body: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        headers={"content-type": "application/json"},
        content=json.dumps(body).encode("utf-8"),
    )


# ---------- check ---------------------------------------------------------


async def test_check_sends_correctly_shaped_request() -> None:
    rec = _Recorder()
    transport = rec.transport(lambda req: _json_response({"allowed": True}))
    client = _client(transport)
    try:
        acl = create_openfga_provider(
            OpenFgaOptions(
                api_url="https://fga.example.com",
                store_id="store-1",
                model_id="model-42",
                token="secret",
                client=client,
            )
        )
        result = await acl.check(_user("alice"), "write", _brain("notes"))
        assert result.allowed is True

        assert len(rec.calls) == 1
        sent = rec.calls[0]
        assert str(sent.url) == "https://fga.example.com/stores/store-1/check"
        assert sent.method == "POST"
        assert sent.headers["authorization"] == "Bearer secret"
        assert sent.headers["content-type"].startswith("application/json")
        body = json.loads(sent.content.decode("utf-8"))
        assert body["tuple_key"] == {
            "user": "user:alice",
            "relation": "writer",
            "object": "brain:notes",
        }
        assert body["authorization_model_id"] == "model-42"
    finally:
        await client.aclose()


async def test_check_maps_disallowed_response_to_deny_with_reason() -> None:
    rec = _Recorder()
    transport = rec.transport(
        lambda req: _json_response({"allowed": False, "resolution": "no tuple"})
    )
    client = _client(transport)
    try:
        acl = create_openfga_provider(
            OpenFgaOptions(
                api_url="https://fga.example.com/",
                store_id="store-1",
                client=client,
            )
        )
        result = await acl.check(_user("alice"), "read", _brain("notes"))
        assert result.allowed is False
        assert result.reason == "no tuple"
    finally:
        await client.aclose()


async def test_check_raises_http_error_on_non_2xx() -> None:
    rec = _Recorder()
    transport = rec.transport(lambda req: httpx.Response(status_code=500, content=b"boom"))
    client = _client(transport)
    try:
        acl = create_openfga_provider(
            OpenFgaOptions(
                api_url="https://fga.example.com",
                store_id="store-1",
                client=client,
            )
        )
        with pytest.raises(OpenFgaHTTPError):
            await acl.check(_user("alice"), "read", _brain("notes"))
    finally:
        await client.aclose()


async def test_check_omits_authorization_header_when_no_token() -> None:
    rec = _Recorder()
    transport = rec.transport(lambda req: _json_response({"allowed": True}))
    client = _client(transport)
    try:
        acl = create_openfga_provider(
            OpenFgaOptions(
                api_url="https://fga.example.com",
                store_id="store-1",
                client=client,
            )
        )
        await acl.check(_user("alice"), "read", _brain("notes"))
        assert "authorization" not in {k.lower() for k in rec.calls[0].headers.keys()}
    finally:
        await client.aclose()


# ---------- write ---------------------------------------------------------


async def test_write_sends_correct_body_shape() -> None:
    rec = _Recorder()
    transport = rec.transport(lambda req: _json_response({}))
    client = _client(transport)
    try:
        acl = create_openfga_provider(
            OpenFgaOptions(
                api_url="https://fga.example.com",
                store_id="store-1",
                model_id="model-42",
                client=client,
            )
        )
        await acl.write(
            WriteTuplesRequest(
                writes=[
                    Tuple(subject=_user("alice"), relation="writer", resource=_brain("notes"))
                ],
                deletes=[
                    Tuple(subject=_user("bob"), relation="reader", resource=_brain("notes"))
                ],
            )
        )
        assert len(rec.calls) == 1
        sent = rec.calls[0]
        assert str(sent.url) == "https://fga.example.com/stores/store-1/write"
        body = json.loads(sent.content.decode("utf-8"))
        assert body["writes"]["tuple_keys"] == [
            {"user": "user:alice", "relation": "writer", "object": "brain:notes"}
        ]
        assert body["deletes"]["tuple_keys"] == [
            {"user": "user:bob", "relation": "reader", "object": "brain:notes"}
        ]
        assert body["authorization_model_id"] == "model-42"
    finally:
        await client.aclose()


async def test_write_makes_no_http_call_when_request_is_empty() -> None:
    rec = _Recorder()
    transport = rec.transport(lambda req: _json_response({}))
    client = _client(transport)
    try:
        acl = create_openfga_provider(
            OpenFgaOptions(
                api_url="https://fga.example.com",
                store_id="store-1",
                client=client,
            )
        )
        await acl.write(WriteTuplesRequest())
        assert rec.calls == []
    finally:
        await client.aclose()


async def test_network_exception_wraps_into_request_error() -> None:
    def boom(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network down")

    transport = httpx.MockTransport(boom)
    client = _client(transport)
    try:
        acl = create_openfga_provider(
            OpenFgaOptions(
                api_url="https://fga.example.com",
                store_id="store-1",
                client=client,
            )
        )
        with pytest.raises(OpenFgaRequestError) as info:
            await acl.write(
                WriteTuplesRequest(
                    writes=[
                        Tuple(subject=_user("a"), relation="reader", resource=_brain("x"))
                    ]
                )
            )
        assert "network down" in str(info.value)
    finally:
        await client.aclose()


# ---------- close --------------------------------------------------------


async def test_close_does_not_aclose_a_caller_supplied_client() -> None:
    rec = _Recorder()
    transport = rec.transport(lambda req: _json_response({"allowed": True}))
    client = _client(transport)
    try:
        acl = create_openfga_provider(
            OpenFgaOptions(
                api_url="https://fga.example.com",
                store_id="store-1",
                client=client,
            )
        )
        await acl.close()
        assert client.is_closed is False
        # The supplied client must still serve requests after the provider closes.
        await acl.check(_user("alice"), "read", _brain("notes"))
        assert len(rec.calls) == 1
    finally:
        await client.aclose()


async def test_close_acloses_an_internally_constructed_client() -> None:
    acl = create_openfga_provider(
        OpenFgaOptions(
            api_url="https://fga.example.com",
            store_id="store-1",
        )
    )
    # The adapter built its own client; introspect it for verification.
    inner_client: httpx.AsyncClient = getattr(acl, "_client")
    assert inner_client.is_closed is False
    await acl.close()
    assert inner_client.is_closed is True


async def test_close_is_idempotent_for_owned_client() -> None:
    acl = create_openfga_provider(
        OpenFgaOptions(
            api_url="https://fga.example.com",
            store_id="store-1",
        )
    )
    await acl.close()
    # A second call must not raise (the underlying client is already closed).
    await acl.close()


# ---------- read ----------------------------------------------------------


async def test_read_decodes_tuples_and_skips_junk_rows() -> None:
    rec = _Recorder()
    transport = rec.transport(
        lambda req: _json_response(
            {
                "tuples": [
                    {
                        "key": {
                            "user": "user:alice",
                            "relation": "writer",
                            "object": "brain:notes",
                        }
                    },
                    {
                        "key": {
                            "user": "junk",
                            "relation": "writer",
                            "object": "brain:notes",
                        }
                    },
                ]
            }
        )
    )
    client = _client(transport)
    try:
        acl = create_openfga_provider(
            OpenFgaOptions(
                api_url="https://fga.example.com",
                store_id="store-1",
                client=client,
            )
        )
        tuples = await acl.read(ReadTuplesQuery(resource=_brain("notes")))
        assert len(tuples) == 1
        assert tuples[0].subject == _user("alice")
        assert tuples[0].relation == "writer"
        assert tuples[0].resource == _brain("notes")
    finally:
        await client.aclose()
