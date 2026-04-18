# SPDX-License-Identifier: Apache-2.0
"""Starlette HTTP daemon smoke tests."""

from __future__ import annotations

from starlette.testclient import TestClient

from jeffs_brain_memory.http import create_app


def test_healthz() -> None:
    client = TestClient(create_app())
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_protocol_endpoint_returns_problem_json() -> None:
    client = TestClient(create_app())
    response = client.get("/v1/brains/default/documents/read?path=memory/a.md")
    assert response.status_code == 501
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["status"] == 501
    assert body["code"] == "not_implemented"


def test_documents_put_returns_problem_json() -> None:
    client = TestClient(create_app())
    response = client.put(
        "/v1/brains/default/documents?path=memory/a.md",
        content=b"hello",
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 501
    assert response.json()["code"] == "not_implemented"


def test_batch_ops_returns_problem_json() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/v1/brains/default/documents/batch-ops",
        json={"reason": "ingest", "ops": []},
    )
    assert response.status_code == 501
    assert response.json()["code"] == "not_implemented"
