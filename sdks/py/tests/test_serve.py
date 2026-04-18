# SPDX-License-Identifier: Apache-2.0
"""Starlette HTTP daemon smoke tests."""

from __future__ import annotations

from starlette.testclient import TestClient

from jeffs_brain_memory.http import create_app


def test_healthz() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"ok": True}


def test_protocol_endpoint_returns_not_found_for_missing_brain() -> None:
    with TestClient(create_app()) as client:
        response = client.get(
            "/v1/brains/default/documents/read?path=memory/a.md"
        )
        assert response.status_code == 404
        assert response.headers["content-type"].startswith("application/problem+json")
        body = response.json()
        assert body["status"] == 404
        assert body["code"] == "not_found"


def test_documents_put_returns_not_found_for_missing_brain() -> None:
    with TestClient(create_app()) as client:
        response = client.put(
            "/v1/brains/default/documents?path=memory/a.md",
            content=b"hello",
            headers={"Content-Type": "application/octet-stream"},
        )
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"


def test_batch_ops_returns_not_found_for_missing_brain() -> None:
    with TestClient(create_app()) as client:
        response = client.post(
            "/v1/brains/default/documents/batch-ops",
            json={"reason": "ingest", "ops": []},
        )
        assert response.status_code == 404
        assert response.json()["code"] == "not_found"
