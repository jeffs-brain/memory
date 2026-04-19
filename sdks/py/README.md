# jeffs-brain-memory

Python SDK for Jeffs Brain, the cross-language memory library for LLM agents.

Part of the polyglot [`jeffs-brain/memory`](https://github.com/jeffs-brain/memory) monorepo. This SDK tracks the same [`spec/`](../../spec/) and conformance fixtures as the TypeScript and Go SDKs.

Cross-SDK daemon parity today is `ask-basic`, `ask-augmented`, and `search-retrieve-only` through `memory serve`. Python does not currently ship a native `memory eval lme` CLI, so LongMemEval parity work goes through the daemon surface only.

## Feature support

Full port of the spec surface:

- Stores: `FsStore`, `MemStore`, `GitStore` (pygit2), `HttpStore` (spec/PROTOCOL.md wire client).
- Search: SQLite FTS5 BM25, sqlite-vec vector search, trigram fuzzy fallback.
- Query DSL: tokenisation, stopword filtering (en and nl), alias expansion, FTS5 compilation.
- Retrieval: hybrid BM25 + vector, Reciprocal Rank Fusion at `k=60`, five-rung retry ladder, intent reweight, cross-encoder rerank.
- Memory stages: extract, reflect, consolidate, recall, session buffers, episodes, feedback, contextualiser, distiller, procedural detection, wikilinks.
- Knowledge: markdown, URL, file, PDF ingest with frontmatter and compile passes.
- LLM providers: Ollama, OpenAI, Anthropic (all gated via the standard `JB_LLM_PROVIDER` / `JB_LLM_MODEL` env pair).
- HTTP daemon (`memory serve`) matching `spec/PROTOCOL.md`.
- Authorisation: `jeffs_brain_memory.acl` ships a `Provider` Protocol, in-process RBAC (workspace -> brain -> collection -> document hierarchy with `admin`/`writer`/`reader` roles and `deny:<role>` overrides), `wrap_store(...)` Store wrapper, and an idempotent `close()` lifecycle hook. The sibling `jeffs_brain_memory.acl_openfga` module ships an `httpx`-based OpenFGA HTTP adapter against the shared model in [`spec/openfga/`](../../spec/openfga).
- Cross-SDK daemon scenarios: `ask-basic`, `ask-augmented`, `search-retrieve-only`.
- MCP: see the companion [`jeffs-brain-memory-mcp`](https://pypi.org/project/jeffs-brain-memory-mcp/) package for the stdio wrapper.

## Install

```bash
pip install jeffs-brain-memory
# or
uv add jeffs-brain-memory
```

## CLI

```bash
memory init
memory ingest ./docs
memory search "question"
memory serve --addr 127.0.0.1:18842
memory ask --brain default --question "what did we decide?"
memory remember --brain default --path memory/notes.md --content "..."
memory recall --brain default --query "auth decision"
memory reflect --brain default
memory consolidate --brain default
memory create-brain --brain eval
memory list-brains
```

`memory serve` speaks the wire protocol documented in [`spec/PROTOCOL.md`](../../spec/PROTOCOL.md), so the cross-SDK eval runner and any TS or Go client drive it identically across `ask-basic`, `ask-augmented`, and `search-retrieve-only`. In the shared runner, `--mode auto` is the default, and the daemon resolves that to `hybrid` when embeddings are configured or `bm25` otherwise.

Native LME status today:

- Python does not currently ship a native `memory eval lme` command.
- The shared `eval/runner.py` scenarios are the direct cross-SDK parity checks for Python.
- The replay-backed tri-SDK retrieve-only workflow still runs from `eval/scripts/run_tri_lme.sh`, which extracts once with Go and then targets the Python daemon in `search-retrieve-only` / `actor-endpoint-style=retrieve-only` mode.
- In that tri-SDK flow the Python daemon returns retrieval payloads via `/search`; the shared augmented reader, judge, and manifests stay in Go.

## Environment variables

- `JB_HOME` - storage root (default `~/.jeffs-brain/`).
- `JB_TOKEN` - hosted bearer token; when set the SDK drives the platform backend via `HttpStore`.
- `JB_ENDPOINT` - hosted endpoint URL when `JB_TOKEN` is set.
- `JB_LLM_PROVIDER`, `JB_LLM_MODEL` - pin the LLM provider and model (`openai`, `anthropic`, `ollama`, or `fake`).
- `OLLAMA_HOST` - Ollama endpoint for local embeddings and chat (default `http://localhost:11434`).
- `JB_ADDR` - bind address for `memory serve` (default `:8080`).
- `JB_AUTH_TOKEN` - optional shared bearer on the daemon.

## Development

```bash
uv sync
uv run memory --version
uv run pytest -q
```

## Scenario verification

Shared daemon scenarios verified in this SDK:

| Scenario | Request shape | Main local checks |
| -------- | ------------- | ----------------- |
| `ask-basic` | `POST /ask` with `question`, `topK`, `mode` | `tests/test_serve_ask_augmented.py` |
| `ask-augmented` | `POST /ask` with `question`, `topK`, `mode`, `readerMode=augmented`, optional `questionDate` | `tests/test_serve_ask_augmented.py` |
| `search-retrieve-only` | `POST /search` with `query`, `topK`, `mode`, optional `questionDate`, `candidateK`, and `rerankTopN` | `tests/test_serve_handlers_real.py` and `tests/test_retrieval_temporal.py` |

Parity expectation is the same scenario request shape, transport shape, retrieval-mode handling, and temporal semantics as the Go and TypeScript daemons. It is not byte-identical model wording.

How we test it:

- `ask-basic` and `ask-augmented` are SSE answer scenarios. We verify `retrieve`, `answer_delta`, `citation`, and `done`.
- `search-retrieve-only` is a JSON retrieval scenario. We score the returned chunks only.
- `questionDate` is forwarded only for `ask-augmented` and `search-retrieve-only`.
- `candidateK` and `rerankTopN` are forwarded only for `search-retrieve-only`.
- `mode` is forwarded unchanged. The daemon resolves `auto` locally.
- The replay-backed tri-SDK run in `eval/scripts/run_tri_lme.sh` exercises `search-retrieve-only` only against a shared replay brain. Python participates there as a daemon target only.

Run the shared daemon scenario checks with:

```bash
cd sdks/py
uv sync
uv run pytest tests/test_serve_ask_augmented.py tests/test_serve_handlers_real.py tests/test_retrieval_temporal.py
```

To compare Python against the other SDKs on one shared scenario, use the runner in `eval/`:

```bash
cd eval
uv run python runner.py --sdk py --dataset datasets/smoke.jsonl --scorer exact --scenario ask-basic --output results/ask-basic
OPENAI_API_KEY=sk-... uv run python runner.py --sdk py --dataset datasets/lme.jsonl --scorer judge --scenario ask-augmented --brain eval --output results/ask-augmented
OPENAI_API_KEY=sk-... uv run python runner.py --sdk py --dataset datasets/lme.jsonl --scorer judge --scenario search-retrieve-only --brain eval --output results/search-retrieve-only
```

Use one output root per scenario so same-day runs do not overwrite `<output>/<date>/py.json`. For the full three-way comparison flow, see [`../../eval/README.md`](../../eval/README.md).

Python does not currently have a native LME runner README section to mirror here. For replay-backed tri-SDK parity, use the Go-orchestrated workflow in [`../../eval/scripts/run_tri_lme.sh`](../../eval/scripts/run_tri_lme.sh).

## Authorisation

```python
from jeffs_brain_memory import (
    wrap_store, Subject, Resource, create_rbac_provider,
    grant_tuple, parent_tuple, WriteTuplesRequest,
)

acl = create_rbac_provider()
ws = Resource(type="workspace", id="acme")
br = Resource(type="brain", id="notes")
await acl.write(WriteTuplesRequest(writes=[
    parent_tuple(br, ws),
    grant_tuple(Subject(kind="user", id="alice"), "writer", ws),
]))

guarded = wrap_store(store, acl, Subject(kind="user", id="alice"), resource=br)
# Every guarded.read/write/delete now runs through `acl.check` first.
```

For OpenFGA-backed checks swap `create_rbac_provider()` for `jeffs_brain_memory.acl_openfga.create_openfga_provider(...)` against the shared model at [`spec/openfga/schema.fga`](../../spec/openfga/schema.fga).

## Examples and docs

- [`spec/`](../../spec/) - protocol, query DSL, algorithms, storage, MCP tool contract.
- Docs site: https://docs.jeffsbrain.com
- MCP wrapper: [`jeffs-brain-memory-mcp`](../../mcp/py).

## Licence

Apache-2.0. See [LICENSE](./LICENSE).
