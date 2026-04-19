# Jeffs Brain Go SDK

The Go implementation of the Jeffs Brain memory and knowledge stack. Ships the public Go packages, the `memory` CLI with HTTP daemon, the `memory-mcp` stdio wrapper, and the `memory eval lme run` LongMemEval harness.

Module path: `github.com/jeffs-brain/memory/go`.

Wire-compatible with the TypeScript and Python SDKs over the [`spec/PROTOCOL.md`](../../spec/PROTOCOL.md) HTTP contract. Cross-SDK daemon parity today is `ask-basic`, `ask-augmented`, and `search-retrieve-only` through `memory serve`.

## Install

```bash
go get github.com/jeffs-brain/memory/go@latest
```

CLI binaries:

```bash
go install github.com/jeffs-brain/memory/go/cmd/memory@latest
go install github.com/jeffs-brain/memory/go/cmd/memory-mcp@latest
```

## Quickstart

Open a filesystem-backed brain, attach a search index, ingest a document, and run a hybrid search. Mirrors [`examples/go/hello-world`](../../examples/go/hello-world).

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"
    "path/filepath"

    "github.com/jeffs-brain/memory/go/brain"
    "github.com/jeffs-brain/memory/go/knowledge"
    "github.com/jeffs-brain/memory/go/search"
    "github.com/jeffs-brain/memory/go/store/fs"
)

func main() {
    ctx := context.Background()
    root, _ := filepath.Abs("./data/hello-world")
    if err := os.MkdirAll(root, 0o755); err != nil {
        log.Fatal(err)
    }

    store, err := fs.New(root)
    if err != nil {
        log.Fatal(err)
    }
    defer store.Close()

    b, err := brain.Open(ctx, brain.Options{ID: "hello-world", Root: root, Store: store})
    if err != nil {
        log.Fatal(err)
    }
    defer b.Close()

    db, err := search.OpenDB(filepath.Join(root, ".search.db"))
    if err != nil {
        log.Fatal(err)
    }
    defer search.CloseDB(db)

    idx, err := search.NewIndex(db, b.Store())
    if err != nil {
        log.Fatal(err)
    }
    unsub := idx.Subscribe(b.Store())
    defer unsub()

    kb, err := knowledge.New(knowledge.Options{
        BrainID: "hello-world",
        Store:   b.Store(),
        Index:   idx,
    })
    if err != nil {
        log.Fatal(err)
    }
    defer kb.Close()

    if _, err := kb.Ingest(ctx, knowledge.IngestRequest{Path: "./docs/hedgehogs.md"}); err != nil {
        log.Fatal(err)
    }

    resp, err := kb.Search(ctx, knowledge.SearchRequest{Query: "where do hedgehogs live?", MaxResults: 3})
    if err != nil {
        log.Fatal(err)
    }
    for i, h := range resp.Hits {
        fmt.Printf("%d. [%.3f] %s\n", i+1, h.Score, h.Path)
    }
}
```

## Packages

- `brain` - storage abstraction (`Store`, `Path`, events, errors), brain lifecycle.
- `acl` - authorisation contract (`Provider`), in-process RBAC, and `Wrap` to guard any `brain.Store`.
- `aclopenfga` - `acl.Provider` against an OpenFGA HTTP API.
- `llm` - cross-cutting LLM abstraction (Ollama, OpenAI, Anthropic).
- `search` - SQLite FTS5 + pure-Go vector index.
- `query` - structured query AST, stopword filtering, alias expansion, optional LLM distillation.
- `retrieval` - hybrid BM25 + vector + cross-encoder rerank with retry ladder.
- `knowledge` - ingest, compile, and hybrid search over markdown, URL, file, PDF.
- `memory` - extract, reflect, consolidate, recall, session buffers, episodes, feedback.
- `eval/lme` - LongMemEval benchmark harness with bulk, replay, and agentic ingest modes.
- `store/fs` - filesystem-backed `brain.Store`.
- `store/git` - git-backed `brain.Store`.
- `store/mem` - in-memory `brain.Store` for tests.
- `store/http` - HTTP-backed `brain.Store` speaking the wire protocol.
- `store/pt` - filesystem passthrough store mirroring the on-disk tree byte-for-byte; the layout the daemon and every SDK reads.

The `cmd/memory` binary is the reference CLI plus HTTP daemon. The `cmd/memory-mcp` binary exposes the canonical `memory_*` tools over MCP stdio.

## Documentation

- Getting started: <https://docs.jeffsbrain.com/getting-started/go/>
- Guides: <https://docs.jeffsbrain.com/guides/knowledge/>, [`/guides/retrieval/`](https://docs.jeffsbrain.com/guides/retrieval/), [`/guides/memory-lifecycle/`](https://docs.jeffsbrain.com/guides/memory-lifecycle/), [`/guides/stores/`](https://docs.jeffsbrain.com/guides/stores/), [`/guides/authorization/`](https://docs.jeffsbrain.com/guides/authorization/)
- CLI reference: <https://docs.jeffsbrain.com/reference/cli/>
- Configuration reference: <https://docs.jeffsbrain.com/reference/configuration/>
- Wire spec and algorithms: <https://docs.jeffsbrain.com/spec/protocol/>, [`/spec/algorithms/`](https://docs.jeffsbrain.com/spec/algorithms/), [`/spec/storage/`](https://docs.jeffsbrain.com/spec/storage/), [`/spec/query-dsl/`](https://docs.jeffsbrain.com/spec/query-dsl/), [`/spec/mcp-tools/`](https://docs.jeffsbrain.com/spec/mcp-tools/)

## Postgres store

The Go SDK does not currently ship a Postgres store adapter. A Postgres-backed store is provided by the TypeScript SDK only. Use `store/fs`, `store/git`, `store/http`, or `store/pt` from Go.

## CLI quickstart

```bash
memory init
memory ingest ./docs
memory search "question"
memory serve --addr 127.0.0.1:18841
```

## Authorisation

The `acl` package exposes `Provider`, in-process RBAC (`NewRbacProvider`), and a `Wrap` helper that turns any `brain.Store` into one that checks every read, write, and delete first.

```go
import (
    "github.com/jeffs-brain/memory/go/acl"
    "github.com/jeffs-brain/memory/go/aclopenfga"
)

provider, err := aclopenfga.NewProvider(aclopenfga.Options{
    APIURL:  "https://fga.example.com",
    StoreID: "store-1",
})
if err != nil { /* ... */ }
defer provider.Close()

guarded := acl.Wrap(store, provider, acl.Subject{Kind: acl.SubjectUser, ID: "alice"}, acl.WrapOptions{
    Resource: acl.Resource{Type: acl.ResourceBrain, ID: "notes"},
})
```

A denied call returns an `*acl.ForbiddenError` that satisfies `errors.Is(err, brain.ErrForbidden)`, so existing handlers keep matching.

## SQLite vector search

The default build uses `modernc.org/sqlite` (pure Go, no CGo) for FTS5 and implements vector search in pure Go at `search/vectors.go`, so `go install` works without a C toolchain.

When you want native `sqlite-vec`, add the CGo-gated binding by compiling with the `sqlite_vec` tag:

```go
//go:build sqlite_vec

package search

import _ "github.com/asg017/sqlite-vec-go-bindings/cgo"
```

## Cross-SDK daemon scenarios

Shared daemon scenarios verified in this SDK:

| Scenario | Request shape | Main local checks |
| -------- | ------------- | ----------------- |
| `ask-basic` | `POST /ask` with `question`, `topK`, `mode` | `cmd/memory/handler_ask_test.go` and `cmd/memory/serve_integration_test.go` |
| `ask-augmented` | `POST /ask` with `question`, `topK`, `mode`, `readerMode=augmented`, optional `questionDate` | `cmd/memory/handler_ask_test.go` and `eval/lme/actor_endpoint_test.go` |
| `search-retrieve-only` | `POST /search` with `query`, `topK`, `mode`, optional `questionDate`, `candidateK`, and `rerankTopN` | `cmd/memory/handler_search_test.go`, `cmd/memory/serve_integration_test.go`, and `eval/lme/actor_endpoint_test.go` |

Parity expectation is the same scenario request shape, transport shape, retrieval-mode handling, and temporal semantics as the TypeScript and Python daemons. It is not byte-identical model wording.

How we test it:

- `ask-basic` and `ask-augmented` are SSE answer scenarios. We verify `retrieve`, `answer_delta`, `citation`, and `done`.
- `search-retrieve-only` is a JSON retrieval scenario. We score the returned chunks only.
- `questionDate` is forwarded only for `ask-augmented` and `search-retrieve-only`.
- `candidateK` and `rerankTopN` are forwarded only for `search-retrieve-only`.
- `mode` is forwarded unchanged. The daemon resolves `auto` locally.
- The replay-backed tri-SDK run in `eval/scripts/run_tri_lme.sh` exercises `search-retrieve-only` only against a shared replay brain. Go extracts once, then calls each SDK daemon in `actor-endpoint-style=retrieve-only`, so each daemon returns retrieval payloads via `/search` while the shared augmented reader, judge, and manifests stay in Go.

Run the shared daemon scenario checks with:

```bash
cd sdks/go
go test ./cmd/memory ./eval/lme
```

To compare Go against the other SDKs on one shared scenario, use the runner in `eval/`:

```bash
cd eval
uv run python runner.py --sdk go --dataset datasets/smoke.jsonl --scorer exact --scenario ask-basic --output results/ask-basic
OPENAI_API_KEY=sk-... uv run python runner.py --sdk go --dataset datasets/lme.jsonl --scorer judge --scenario ask-augmented --brain eval --output results/ask-augmented
OPENAI_API_KEY=sk-... uv run python runner.py --sdk go --dataset datasets/lme.jsonl --scorer judge --scenario search-retrieve-only --brain eval --output results/search-retrieve-only
```

Use one output root per scenario so same-day runs do not overwrite `<output>/<date>/go.json`. For the full three-way comparison flow, see [`../../eval/README.md`](../../eval/README.md).

## LongMemEval replay

Go is the reference native LME runner and the coordinator for the replay-backed tri-SDK retrieve-only workflow. For the cross-SDK replay comparison, run [`../../eval/scripts/run_tri_lme.sh`](../../eval/scripts/run_tri_lme.sh) from the repo root rather than stitching the three daemons together by hand.

```bash
memory eval lme run \
  --dataset longmemeval_s.json \
  --ingest-mode replay \
  --concurrency 8 \
  --judge claude-haiku-4-5 \
  --actor gpt-4o
```

Env knobs: `JB_LME_JUDGE_MODEL`, `JB_LME_ACTOR_MODEL`, plus the standard `JB_LLM_PROVIDER`, `JB_LLM_MODEL`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OLLAMA_HOST`. A soft cost cap is enforced via `--max-cost-usd`.

## Build from source

```bash
cd sdks/go
go build ./...
go test ./...
```

## Examples

- [`examples/go/hello-world`](../../examples/go/hello-world) - ingest a markdown doc into a brain and run a hybrid search over it.

## Licence

Apache-2.0. See `LICENSE`.
