# @jeffs-brain/memory

Local-first, pluggable memory and retrieval library for LLM agents. Ships a `Store` abstraction over filesystem, Git, in-memory, and HTTP backends, hybrid BM25 plus pure-JS vector search, an extract, recall, reflect, and consolidate memory pipeline, RBAC plus an OpenFGA adapter, cross-encoder rerank, opt-in LLM query distillation, and a slim `memory` CLI that speaks the shared HTTP protocol. Runs entirely offline using an Ollama provider and the built-in hash embedder, or against OpenAI, Anthropic, or TEI when you want quality.

Part of the polyglot [`jeffs-brain/memory`](https://github.com/jeffs-brain/memory) repo. This SDK tracks the same [`spec/`](https://github.com/jeffs-brain/memory/tree/main/spec) and conformance fixtures as the Go and Python SDKs.

Cross-SDK daemon parity today is `ask-basic`, `ask-augmented`, and `search-retrieve-only` through `memory serve`. This package also ships native `memory eval lme` commands for single-SDK LongMemEval work, but the replay-backed tri-SDK benchmark is still coordinated from Go rather than from the TypeScript runner.

In the shared runner, `--mode auto` is the default, and the daemon resolves that to `hybrid` when embeddings are configured or `bm25` otherwise.

## Install

```bash
npm i @jeffs-brain/memory
# or
bun add @jeffs-brain/memory
```

The published `memory` binary runs on Node 20+ (its shebang is `#!/usr/bin/env node`). Bun is the preferred local development runtime for this package, but it is not required at install or runtime for end users.

## Feature support

- Stores: `FsStore`, `MemStore`, `GitStore`, `HttpStore` (spec/PROTOCOL.md wire client).
- Search: SQLite FTS5 BM25, pure-JS vector search, Reciprocal Rank Fusion (`k=60`).
- Query DSL: tokenisation, stopword filtering (en and nl), alias expansion, FTS5 compilation.
- Retrieval: hybrid BM25 + vector, five-rung retry ladder, intent reweight, cross-encoder rerank, opt-in query distill.
- Memory stages: extract, reflect, consolidate, recall, session buffers, episode recorder.
- Knowledge: markdown chunker, URL/file/PDF ingest, wikilinks, compile passes.
- SSE utilities: framework-agnostic frame formatting and heartbeat helpers via `@jeffs-brain/memory/sse`.
- Authorisation: pluggable `AccessControlProvider` contract (`@jeffs-brain/memory/acl`), in-process RBAC (workspace -> brain -> collection -> document hierarchy, `admin`/`writer`/`reader` roles, `deny:<role>` overrides), `withAccessControl(store, provider, subject, ...)` Store wrapper, optional `close()` lifecycle hook. Pair with [`@jeffs-brain/memory-openfga`](https://www.npmjs.com/package/@jeffs-brain/memory-openfga) for production tuple-store backed checks.
- Conformance: 28/29 cases green against `spec/conformance/http-contract.json`.
- Cross-SDK daemon scenarios: `ask-basic`, `ask-augmented`, `search-retrieve-only`.
- CLI: `memory init|ingest|search|extract|reflect|consolidate|eval|serve|acl|git`.

## SSE utilities

```ts
import { createSseHeartbeat, formatSseFrame } from '@jeffs-brain/memory/sse'

const write = (chunk: string): void => {
  response.write(chunk)
}

let nextEventId = 1

write(
  formatSseFrame({
    event: 'change',
    id: String(nextEventId++),
    data: JSON.stringify({ kind: 'updated', path: 'memory/notes.md' }),
  }),
)

const stopHeartbeat = createSseHeartbeat(25_000, () => {
  write(
    formatSseFrame({
      event: 'ping',
      id: String(nextEventId++),
      data: 'keepalive',
    }),
  )
})

request.on('close', stopHeartbeat)
```

These helpers expose the framing layer separately from the built-in `Response`-based daemon transport, so Express, Fastify, Hono, or plain Node handlers can emit SSE frames without reimplementing the wire format. They format `event`, `id`, and `data` lines for you, while protocol-specific sequencing such as the daemon's monotonic `/events` ids stays under the caller's control.

## Conformance runner

```ts
import { runConformanceSuite } from '@jeffs-brain/memory/conformance'

const result = await runConformanceSuite({
  baseUrl: 'http://127.0.0.1:18844/v1',
  authToken: process.env.JB_AUTH_TOKEN,
})

if (result.failed > 0) {
  throw new Error(
    result.cases
      .filter((testCase) => !testCase.ok)
      .map((testCase) => `${testCase.name}: ${testCase.error}`)
      .join('\n'),
  )
}
```

The runner packages the shared `spec/conformance/http-contract.json` fixture, provisions an isolated brain per case, replays the full HTTP store contract, and deletes every test brain afterwards.

## Embedded usage

```ts
import { createMemStore, createMemory, createHashEmbedder } from '@jeffs-brain/memory'

const store = createMemStore()
const embedder = createHashEmbedder()
const mem = createMemory({ store, provider, embedder, cursorStore, scope: 'project', actorId: 'me' })

await mem.extract({ messages })
const hits = await mem.recall({ query: 'what did we decide about auth?' })
console.log(hits)
```

Swap `createHashEmbedder()` for `OllamaEmbedder` or `TEIEmbedder` when you need real retrieval quality. The hash embedder is deterministic, zero-network, and intended for dev and CI only.

For a single orchestration surface covering pre-turn, post-turn, and session-end work:

```ts
import { createMemoryLifecycle } from '@jeffs-brain/memory'

const lifecycle = createMemoryLifecycle({ memory: mem })
const promptContext = await lifecycle.beforeTurn({ message: 'How should we handle auth?' })
const extracted = await lifecycle.afterTurn({ messages, sessionId: 'session-1' })
const ended = await lifecycle.endSession({ messages, sessionId: 'session-1', consolidate: true })
```

## CLI quickstart

```bash
memory init ./brain
memory ingest notes/meeting.md --brain ./brain
memory search "which database did we pick?" --brain ./brain
memory serve --addr 127.0.0.1:18844
```

`--brain` is optional once `JB_BRAIN` is exported. `memory serve` honours `JB_HOME` for its multi-brain root.

`memory serve` speaks the wire protocol documented at [`spec/PROTOCOL.md`](https://github.com/jeffs-brain/memory/blob/main/spec/PROTOCOL.md) so any language SDK or the cross-SDK eval runner can drive `ask-basic`, `ask-augmented`, and `search-retrieve-only` identically.

Native LME status today:

- TypeScript ships native `memory eval lme` commands for fetch, run, compare, and check.
- The replay-backed tri-SDK retrieve-only workflow still runs from `eval/scripts/run_tri_lme.sh`, which extracts once with Go and then targets the TS daemon in `search-retrieve-only` / `actor-endpoint-style=retrieve-only` mode.
- In that tri-SDK flow the TS daemon returns retrieval payloads via `/search`; the shared augmented reader, judge, and manifests stay in Go.

## Scenario verification

Shared daemon scenarios verified in this SDK:

| Scenario | Request shape | Main local checks |
| -------- | ------------- | ----------------- |
| `ask-basic` | `POST /ask` with `question`, `topK`, `mode` | `src/http/handlers.test.ts` and `src/http/daemon.test.ts` |
| `ask-augmented` | `POST /ask` with `question`, `topK`, `mode`, `readerMode=augmented`, optional `questionDate` | `src/http/handlers.test.ts` and `src/http/daemon.test.ts` |
| `search-retrieve-only` | `POST /search` with `query`, `topK`, `mode`, optional `questionDate`, `candidateK`, and `rerankTopN` | `src/http/daemon.test.ts` |

Parity expectation is the same scenario request shape, transport shape, retrieval-mode handling, and temporal semantics as the Go and Python daemons. It is not byte-identical model wording.

How we test it:

- `ask-basic` and `ask-augmented` are SSE answer scenarios. We verify `retrieve`, `answer_delta`, `citation`, and `done`.
- `search-retrieve-only` is a JSON retrieval scenario. We score the returned chunks only.
- `questionDate` is forwarded only for `ask-augmented` and `search-retrieve-only`.
- `candidateK` and `rerankTopN` are forwarded only for `search-retrieve-only`.
- `mode` is forwarded unchanged. The daemon resolves `auto` locally.
- The replay-backed tri-SDK run in `eval/scripts/run_tri_lme.sh` exercises `search-retrieve-only` only against a shared replay brain. TypeScript participates there as a daemon target, not as the shared reader or judge.

Run the shared daemon scenario checks with:

```bash
cd sdks/ts/memory
bun x vitest run src/http/handlers.test.ts src/http/daemon.test.ts
```

To compare TypeScript against the other SDKs on one shared scenario, use the runner in `eval/`:

```bash
cd eval
uv run python runner.py --sdk ts --dataset datasets/smoke.jsonl --scorer exact --scenario search-retrieve-only --mode bm25 --brain eval --seed-reference-brain --output results/smoke-search
OPENAI_API_KEY=sk-... uv run python runner.py --sdk ts --dataset datasets/lme.jsonl --scorer judge --scenario ask-augmented --brain eval --output results/ask-augmented
OPENAI_API_KEY=sk-... uv run python runner.py --sdk ts --dataset datasets/lme.jsonl --scorer judge --scenario search-retrieve-only --brain eval --output results/search-retrieve-only
```

Use one output root per scenario so same-day runs do not overwrite `<output>/<date>/ts.json`. For the full three-way comparison flow, see [`eval/README.md`](../../../eval/README.md).

For native TypeScript-only LongMemEval work, use the local `memory eval lme` commands. For apples-to-apples tri-SDK replay parity, use the Go-orchestrated workflow in [`../../../eval/scripts/run_tri_lme.sh`](../../../eval/scripts/run_tri_lme.sh).

## MCP server

To expose a brain to Claude Code, Claude Desktop, Cursor, Windsurf, or Zed, install [`@jeffs-brain/memory-mcp`](https://www.npmjs.com/package/@jeffs-brain/memory-mcp) (stdio server, 11 canonical tools). The [`@jeffs-brain/install`](https://www.npmjs.com/package/@jeffs-brain/install) orchestrator wires every host in one command:

```bash
npx @jeffs-brain/install
```

## Documentation

- TypeScript getting started: https://docs.jeffsbrain.com/getting-started/typescript/
- Memory lifecycle guide: https://docs.jeffsbrain.com/guides/memory-lifecycle/
- Retrieval guide: https://docs.jeffsbrain.com/guides/retrieval/
- Stores guide: https://docs.jeffsbrain.com/guides/stores/
- Authorisation guide: https://docs.jeffsbrain.com/guides/authorization/
- [`examples/ts/hello-world`](https://github.com/jeffs-brain/memory/tree/main/examples/ts/hello-world) - BM25 search over a markdown corpus.
- [`spec/`](https://github.com/jeffs-brain/memory/tree/main/spec) - protocol, storage, algorithms, query DSL, MCP tool contract.

## Companion packages

- `@jeffs-brain/memory-postgres` - Postgres + pgvector adapter.
- `@jeffs-brain/memory-openfga` - OpenFGA authorisation adapter.
- `@jeffs-brain/memory-mcp` - Model Context Protocol stdio server.
- `@jeffs-brain/install` - multi-agent installer.

## License

Apache-2.0. See [`LICENSE`](./LICENSE) and [`NOTICE`](./NOTICE).
