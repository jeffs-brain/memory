// SPDX-License-Identifier: Apache-2.0

// Package memory is the root of the Jeffs Brain Go SDK.
//
// The SDK is split into focused subpackages:
//
//   - brain:     storage abstraction (Store interface, Path helpers, events)
//   - store/fs:  filesystem-backed Store
//   - store/git: git-backed Store
//   - store/mem: in-memory Store for tests
//   - store/http: HTTP-backed Store speaking spec/PROTOCOL.md
//   - search:    FTS5 / vector index interfaces
//   - query:     structured query AST + distillation
//   - retrieval: hybrid BM25 + vector + reranker retrieval
//   - memory:    remember / recall / reflect / consolidate
//   - knowledge: ingest / compile / search
//   - eval/lme:  long-memory-eval harness
//
// The cmd/memory binary is the reference CLI + HTTP daemon.
//
// Scaffold only; implementations are being ported from the upstream jeff
// project. See README.md for current status.
package memory
