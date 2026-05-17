// SPDX-License-Identifier: Apache-2.0

// Package queue provides a concurrent worker pool for processing ingestion
// pipeline jobs claimed from a queue backend. The pool manages configurable
// parallelism, per-brain concurrency limits, backpressure detection, and
// graceful shutdown semantics.
//
// Types and the Adapter interface are defined in types.go, mirroring the
// canonical definitions from P3-1 (PostgreSQL queue). When P3-1 lands and
// both packages merge into the same Go package, the types.go here is
// replaced by P3-1's types.go with zero interface changes.
package queue
