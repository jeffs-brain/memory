# SPDX-License-Identifier: Apache-2.0
"""Behavioural specs for the in-process RBAC provider."""

from __future__ import annotations

import re

import pytest

from jeffs_brain_memory.acl import (
    ReadTuplesQuery,
    Resource,
    Subject,
    Tuple,
    UnsupportedRelationError,
    WriteTuplesRequest,
    create_rbac_provider,
    deny_tuple,
    grant_tuple,
    parent_tuple,
)

pytestmark = pytest.mark.asyncio


def _user(uid: str) -> Subject:
    return Subject(kind="user", id=uid)


def _workspace(rid: str) -> Resource:
    return Resource(type="workspace", id=rid)


def _brain(rid: str) -> Resource:
    return Resource(type="brain", id=rid)


def _collection(rid: str) -> Resource:
    return Resource(type="collection", id=rid)


def _document(rid: str) -> Resource:
    return Resource(type="document", id=rid)


async def _seed(tuples: list[Tuple]):
    acl = create_rbac_provider()
    await acl.write(WriteTuplesRequest(writes=tuples))
    return acl


async def test_admin_on_workspace_cascades_through_parent_chain() -> None:
    alice = _user("alice")
    ws = _workspace("acme")
    br = _brain("notes")
    col = _collection("meetings")
    doc = _document("doc-1")
    acl = await _seed(
        [
            parent_tuple(br, ws),
            parent_tuple(col, br),
            parent_tuple(doc, col),
            grant_tuple(alice, "admin", ws),
        ]
    )

    assert (await acl.check(alice, "admin", br)).allowed is True
    assert (await acl.check(alice, "write", doc)).allowed is True
    assert (await acl.check(alice, "delete", doc)).allowed is True


async def test_deny_admin_at_brain_overrides_workspace_grant() -> None:
    alice = _user("alice")
    ws = _workspace("acme")
    br = _brain("notes")
    acl = await _seed(
        [
            parent_tuple(br, ws),
            grant_tuple(alice, "admin", ws),
            deny_tuple(alice, "admin", br),
        ]
    )

    admin_check = await acl.check(alice, "admin", br)
    assert admin_check.allowed is False
    assert admin_check.reason is not None
    assert "deny:admin" in admin_check.reason

    # Write requires writer (rank 2), but the admin deny (rank 3) covers it.
    write_check = await acl.check(alice, "write", br)
    assert write_check.allowed is False


async def test_reader_cannot_write_with_insufficient_reason() -> None:
    bob = _user("bob")
    doc = _document("doc-1")
    acl = await _seed([grant_tuple(bob, "reader", doc)])

    assert (await acl.check(bob, "read", doc)).allowed is True
    write = await acl.check(bob, "write", doc)
    assert write.allowed is False
    assert write.reason is not None
    assert re.search(r"insufficient", write.reason)


async def test_unknown_subject_denied_with_no_grant_reason() -> None:
    eve = _user("eve")
    doc = _document("doc-1")
    acl = await _seed([grant_tuple(_user("alice"), "admin", doc)])

    result = await acl.check(eve, "read", doc)
    assert result.allowed is False
    assert result.reason is not None
    assert "no grant" in result.reason


async def test_unsupported_relation_rejected_on_write() -> None:
    acl = create_rbac_provider()
    bogus = Tuple(
        subject=_user("a"),
        relation="bogus",
        resource=_brain("x"),
    )
    with pytest.raises(UnsupportedRelationError, match="unsupported relation"):
        await acl.write(WriteTuplesRequest(writes=[bogus]))


async def test_read_returns_stored_tuples() -> None:
    alice = _user("alice")
    br = _brain("notes")
    acl = await _seed([grant_tuple(alice, "writer", br)])
    found = await acl.read(ReadTuplesQuery(subject=alice))
    assert len(found) == 1
    assert found[0].relation == "writer"


async def test_close_is_a_noop_and_idempotent() -> None:
    acl = create_rbac_provider()
    assert await acl.close() is None
    # Second invocation must not error.
    assert await acl.close() is None
    # The provider must still be usable after close (no resources to release).
    await acl.write(
        WriteTuplesRequest(writes=[grant_tuple(_user("alice"), "reader", _brain("x"))])
    )
    assert (await acl.check(_user("alice"), "read", _brain("x"))).allowed is True
