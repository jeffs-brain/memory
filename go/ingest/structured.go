// SPDX-License-Identifier: Apache-2.0

// Package ingest provides content extraction and chunking for the
// ingestion pipeline. The structured extractors handle CSV, TSV,
// JSON, XML, and JSONL documents with schema-aware chunking that
// preserves record boundaries and column context.
package ingest

import "reflect"

// defaultMaxInputSize is the upper bound on raw input accepted by any
// structured extractor. 50 MiB prevents unbounded memory consumption
// when the caller does not supply a per-config override.
const defaultMaxInputSize int64 = 50 * 1024 * 1024

// mapPointer extracts the underlying data pointer of a Go map so
// circular reference detection can compare map identity rather than
// the address of a local variable (which would differ on every call).
// This matches the TS approach of using Set<unknown> with object
// references.
func mapPointer(m map[string]any) uintptr {
	return reflect.ValueOf(m).Pointer()
}
