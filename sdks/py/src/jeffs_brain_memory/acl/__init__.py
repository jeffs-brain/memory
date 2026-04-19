# SPDX-License-Identifier: Apache-2.0
"""ACL surface for `jeffs_brain_memory`.

Public re-exports from `types`, `rbac`, and `wrap`. The OpenFGA HTTP
adapter lives in the sibling `jeffs_brain_memory.acl_openfga` package
so importing this module does not pull in `httpx`-based machinery.

The OpenFGA model the adapter speaks lives canonically at
`spec/openfga/schema.fga`.
"""

from __future__ import annotations

from .rbac import (
    InMemoryTupleStore,
    Role,
    TupleStore,
    create_rbac_provider,
    deny_tuple,
    grant_tuple,
    parent_subject,
    parent_tuple,
)
from .types import (
    AccessControlError,
    Action,
    CheckResult,
    ForbiddenError,
    Provider,
    ReadTuplesQuery,
    Resource,
    ResourceType,
    Subject,
    SubjectKind,
    Tuple,
    UnsupportedRelationError,
    WriteTuplesRequest,
    allow,
    deny,
    resource_key,
    subject_key,
    tuple_key,
)
from .wrap import wrap_store

__all__ = [
    "AccessControlError",
    "Action",
    "CheckResult",
    "ForbiddenError",
    "InMemoryTupleStore",
    "Provider",
    "ReadTuplesQuery",
    "Resource",
    "ResourceType",
    "Role",
    "Subject",
    "SubjectKind",
    "Tuple",
    "TupleStore",
    "UnsupportedRelationError",
    "WriteTuplesRequest",
    "allow",
    "create_rbac_provider",
    "deny",
    "deny_tuple",
    "grant_tuple",
    "parent_subject",
    "parent_tuple",
    "resource_key",
    "subject_key",
    "tuple_key",
    "wrap_store",
]
