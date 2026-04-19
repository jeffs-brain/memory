# Jeffs Brain Go SDK

Module path: `github.com/jeffs-brain/memory/go`

The Go SDK is a full implementation of the [`spec/`](../../spec) wire contract. It ships the `memory` CLI with an HTTP daemon, the `memory-mcp` stdio wrapper, and a `memory eval lme run` benchmark harness that drives the full LongMemEval replay path.

Wire-compatible with the TypeScript and Python SDKs. Shared cross-SDK evaluation parity today is the daemon surface exposed by `memory serve`.

Cross-SDK daemon parity today is `ask-basic`, `ask-augmented`, and `search-retrieve-only` through `memory serve`. Full LongMemEval replay, replay ingest, and agentic loops stay in the native `memory eval lme run` path.

In the shared runner, `--mode auto` is the default, and the daemon resolves that to `hybrid` when embeddings are configured or `bm25` otherwise.

## Install

```bash
go install github.com/jeffs-brain/memory/go/cmd/memory@latest
go install github.com/jeffs-brain/memory/go/cmd/memory-mcp@latest
```

## Build from source

```bash
cd sdks/go
go build ./...
go test ./...
```

## Scenario verification

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
- The replay-backed tri-SDK run in `eval/scripts/run_tri_lme.sh` exercises `search-retrieve-only` only against a shared replay brain.

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

## Feature support

- Stores: `store/fs`, `store/git`, `store/mem`, `store/http` (spec/PROTOCOL.md wire client).
- Search: SQLite FTS5 BM25 plus pure-Go vector search (default, no CGo). Opt-in sqlite-vec binding via the `sqlite_vec` build tag.
- Query: structured AST, stopword filtering (en and nl), alias expansion, optional LLM distillation.
- Retrieval: hybrid BM25 + vector, five-rung retry ladder, intent reweight, real cross-encoder rerank (LLM and HTTP modes).
- Memory stages: extract, reflect, consolidate, recall, session buffers, episodes, feedback loop.
- Knowledge: markdown, URL, file, PDF ingest with frontmatter, wikilinks, compile passes.
- Cross-SDK daemon scenarios: `ask-basic`, `ask-augmented`, `search-retrieve-only`.
- Eval: full LongMemEval runner with bulk, replay, and agentic ingest modes.
- MCP: `cmd/memory-mcp` exposes the 11 canonical `memory_*` tools over MCP stdio.

## CLI quickstart

```bash
memory init
memory ingest ./docs
memory search "question"
memory serve --addr 127.0.0.1:18841
```

## LongMemEval replay

```bash
memory eval lme run \
  --dataset longmemeval_s.json \
  --ingest-mode replay \
  --concurrency 8 \
  --judge claude-haiku-4-5 \
  --actor gpt-4o
```

Env knobs: `JB_LME_JUDGE_MODEL`, `JB_LME_ACTOR_MODEL`, plus the standard `JB_LLM_PROVIDER`, `JB_LLM_MODEL`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OLLAMA_HOST`. A soft cost cap is enforced via `--max-cost-usd`.

## Layout

- `acl/` - authorisation contract (`Provider`), in-process RBAC, and `Wrap(brain.Store, provider, subject, opts)` for guarding any backend.
- `aclopenfga/` - OpenFGA HTTP adapter. See [`spec/openfga/schema.fga`](../../spec/openfga/schema.fga) for the model.
- `brain/` - storage abstraction (`Store`, `Path`, events, errors).
- `store/fs` | `store/git` | `store/mem` | `store/http` - Store backends.
- `search/` - FTS5 + vector index.
- `query/` - structured query AST + distillation.
- `retrieval/` - hybrid BM25 + vector + cross-encoder rerank.
- `memory/` - remember / recall / reflect / consolidate manager with extras (contextualiser, distiller, episodes, feedback).
- `knowledge/` - ingest / compile / search.
- `llm/` - provider abstraction (Ollama, OpenAI, Anthropic).
- `eval/lme/` - LongMemEval harness with replay and agentic modes.
- `cmd/memory/` - reference CLI + HTTP daemon.
- `cmd/memory-mcp/` - stdio MCP wrapper exposing the 11 `memory_*` tools.
- `internal/httpd/` - shared HTTP daemon helpers.

## Authorisation

The `acl` package exposes `Provider`, in-process RBAC (`NewRbacProvider`), and a `Wrap` helper that turns any `brain.Store` into one that checks every read/write/delete first.

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

Leave the default build pure-Go so `go install` works everywhere.

## Protocol

`memory serve` implements the wire contract documented at [`spec/PROTOCOL.md`](../../spec/PROTOCOL.md). The shared conformance suite at [`spec/conformance/http-contract.json`](../../spec/conformance/http-contract.json) drives the daemon through 28/29 green cases.

## Examples

- [`examples/go/hello-world`](../../examples/go/hello-world) - ingest a markdown doc into a brain and run a hybrid search over it.

## Licence

Apache-2.0. See `LICENSE`.
