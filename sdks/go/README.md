# Jeffs Brain Go SDK

Module path: `github.com/jeffs-brain/memory/go`

Status: **scaffold only**. Interfaces and types compile; no behaviour is
wired. Implementations are being ported from the upstream jeff project at
`~/code/jeff/apps/jeff/internal/`.

## Install

```bash
go install github.com/jeffs-brain/memory/go/cmd/memory@latest
```

## Build from source

```bash
cd sdks/go
go build ./...
go test ./...
```

## Layout

- `brain/` - storage abstraction (`Store`, `Path`, events, errors)
- `store/fs` | `store/git` | `store/mem` | `store/http` - Store backends
- `search/` - FTS5 + vector index interfaces
- `query/` - structured query AST + distillation interface
- `retrieval/` - hybrid BM25 + vector + reranker retrieval interface
- `memory/` - remember / recall / reflect / consolidate manager
- `knowledge/` - ingest / compile / search
- `eval/lme/` - long-memory-eval harness placeholder
- `cmd/memory/` - reference CLI + HTTP daemon
- `internal/httpd/` - shared HTTP daemon helpers

## SQLite vector search

The scaffold uses `modernc.org/sqlite` (pure-Go, no CGo) for FTS5. The
upstream jeff brain implements vector search in pure Go at
`search/vectors.go`, so the scaffold does not depend on sqlite-vec.

When the next implementation pass wants native sqlite-vec, add the
CGo-gated import:

```go
//go:build sqlite_vec

package search

import _ "github.com/asg017/sqlite-vec-go-bindings/cgo"
```

Leave the default build pure-Go so `go install` works without a C
toolchain.

## Protocol

`memory serve` speaks the wire protocol documented at
`spec/PROTOCOL.md`. Endpoints return `501 Problem+JSON` in the scaffold.

## Licence

Apache-2.0, see `LICENSE`.
