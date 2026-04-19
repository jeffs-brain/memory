# SPDX-License-Identifier: Apache-2.0
"""OpenFGA HTTP adapter for `jeffs_brain_memory.acl`.

Importing this module pulls in `httpx` (already a hard dep). Keep the
`acl` package free of network dependencies by importing this only where
the OpenFGA backend is wanted.

The OpenFGA model the adapter speaks lives canonically at
`spec/openfga/schema.fga`.
"""

from __future__ import annotations

from .openfga import (
    OpenFgaHTTPError,
    OpenFgaOptions,
    OpenFgaRequestError,
    create_openfga_provider,
    decode_resource,
    decode_subject,
    encode_resource,
    encode_subject,
)

__all__ = [
    "OpenFgaHTTPError",
    "OpenFgaOptions",
    "OpenFgaRequestError",
    "create_openfga_provider",
    "decode_resource",
    "decode_subject",
    "encode_resource",
    "encode_subject",
]
