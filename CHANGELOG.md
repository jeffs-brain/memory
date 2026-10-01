# Changelog

All notable changes to the jeffs-brain memory libraries are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

## [Unreleased]

Proposed versions: `@jeffs-brain/memory` and `@jeffs-brain/memory-pi`
1.2.0, `@jeffs-brain/memory-mcp` 1.1.0, Go module and `memory-mcp` binary
1.2.0. The other npm packages have no shipped changes and stay at 1.0.0.

### Upgrade notes

- `memory serve` (TypeScript, Go, Python) listens on `127.0.0.1:8080` by
  default and refuses to start on a non-loopback address unless an auth
  token is set. A daemon without a token answers only requests addressed
  to a loopback host and refuses cross-origin browser requests.
- `POST /v1/brains/{id}/ingest/file` no longer reads a server-side `path`
  unless the daemon is started with `--ingest-root <dir>` (or
  `JB_INGEST_ROOT`); paths are then confined to that directory, symlinks
  included. Clients sending `contentBase64` are unaffected.
- The Go module requires Go 1.26 or newer.
- TypeScript cron schedules are evaluated in UTC (as in Go) instead of the
  host's local time zone. Pass `{ timeZone }` to `nextOccurrence` for a
  wall-clock zone.
- `memory-pi` accepts pi `>=0.78.1 <1.0.0` (was 0.74 only).

### Security

- Daemon, all three SDKs: arbitrary file read through `ingest/file` with a
  server-side path; SSRF through `ingest/url` (TypeScript and Python had
  no guard; Go's missed CGNAT, multicast and other reserved ranges, and
  did not cap redirects); bearer tokens compared in variable time
  (TypeScript, Python); 500 responses and `/ask` stream errors carried
  internal error detail; the TypeScript and Go daemons listened on every
  interface with no token by default.
- The SSRF guard checks every redirect hop, pins the connection to the
  vetted address (no DNS rebinding between check and connect), refuses
  credentials in URLs and bounds time and body size. It also covers the
  MCP servers' and the pi extension's `memory_ingest_url`.
- Dependencies: every reachable Go vulnerability (go-git, go-billy,
  x/crypto, x/net, x/text), and every advisory reported by `bun audit`
  and `pip-audit`, including starlette, mcp, cryptography, pyjwt,
  `@stackone/defender`'s pinned nanoid, vitest and the docs site's Astro.

### memory

#### Added

- `@jeffs-brain/memory/net`: `safeFetch`, `isBlockedAddress`,
  `UnsafeUrlError`, `FetchFailedError`.
- `memory serve --ingest-root`; `nextOccurrence(schedule, after,
  { timeZone })`.

#### Fixed

- Video extraction from a buffer always failed: the temporary directory
  was deleted before `ffprobe` read the file.
- Cron: starting on the 29th to 31st skipped the following month, and
  schedules followed the host time zone.
- `memory --version` errored instead of printing the version.
- `/ask` reported `done: { ok: true }` after a generation error.

#### Changed

- Error responses: `bad_gateway` (502) for upstream fetch failures,
  `misdirected_request` (421) for a foreign `Host`, `forbidden` (403) for
  refused paths and cross-origin requests.
- `@stackone/defender` ^0.8.3.

### memory-mcp

#### Fixed

- Every tool advertised an empty input schema, so clients saw no
  parameters. Schemas now carry fields, types, limits and descriptions.
- The server reports its package version (it said 1.0.0).

### memory-pi

#### Fixed

- `memory_ingest_url` uses the core SSRF guard; its own guard re-resolved
  DNS after checking and followed redirects unchecked.

### Go

#### Added

- `knowledge.ErrInvalidURL`, `ErrInvalidContent`, `ErrBlockedAddress` and
  `FetchError`; `httpd.ResolveBindAddr`, `LoopbackGuard`, `IsLoopbackHost`,
  `BadGateway` and `MisdirectedRequest`.

#### Fixed

- The planned CLI commands exited 0 after printing "not implemented";
  they exit 1.
- `memory-mcp` and the CLI share one version constant.
- `/events` sent `ready` before subscribing, so a change made as soon as
  a client saw `ready` could be lost.

### Python (unpublished)

- The same daemon hardening; `jeffs_brain_memory.net.safe_fetch`.
- `/extract` kept a raw `sessionDate` where Go and TypeScript use the
  parsed ISO date.
- `Provider.complete_stream` is declared `async`, as every implementation
  is; the daemon's store honours the `Store.list` contract; the daemon's
  knowledge base can search its brain's index.
- Planned CLI commands exit 1 and are hidden from `--help`.
- Ships `py.typed`; ruff and strict mypy are clean.

### Specification

- `PROTOCOL.md` gains a daemon security section and the new error codes,
  and states that `ready` means the subscription is active.
- `MCP-TOOLS.md`: the batch tool's schema sat under a duplicated
  directory heading; the intro named paths and a variable from another
  repository. Every MCP server is now tested against the spec.
- The retrieval golden fixtures are a synthetic public set.

### Repository

- Removed retrieval captures that embedded a private brain's paths, a
  committed debug binary and stale example and eval output.
- CI gates lint, typecheck (tests included), tests and build for
  TypeScript; ruff, format, strict mypy and tests for Python on 3.11 and
  3.13; the docs build; and vulnerability scans for Go, npm and Python.
  Dependabot covers every ecosystem.
- Releases publish only from a matching tag, and every gate blocks.

## [1.1.0] - 2026-08-25

`@jeffs-brain/memory` and `@jeffs-brain/memory-pi` only. Published from
`2cd4452` without a release tag or changelog entry; recorded here after
the fact.

### memory

#### Added

- `OpenAIConfig.defaultExtraBody` merges into every request body, with a
  per-request `extraBody` winning on collisions. Lets callers pin fields
  such as `reasoning_effort: none` for hybrid thinking models whose
  reasoning otherwise burns small token budgets.
- CLI: `JB_LLM_EXTRA_BODY` (a JSON object) feeds the same setting for the
  `openai` provider kind.

#### Fixed

- The OpenAI provider and embedder tolerate a `baseURL` that already ends
  in `/v1` (the OpenAI SDK convention) instead of doubling the segment.

### memory-pi

#### Added

- The `openai` provider config passes `extraBody` through.

## [1.0.1] - 2026-08-05

### memory-pi

#### Fixed

- Republish with build output. `memory-pi@1.0.0` shipped with no `dist/`
  (two files: `package.json` and `README.md`) because the release
  workflow builds via the root `tsc -b`, which does not reference
  memory-pi, and publishes with `--ignore-scripts`, which skips the
  package's own `prepublishOnly` build. 1.0.0 is deprecated on npm.

### Publishing

#### Fixed

- The root `build` script now also builds memory-pi, so the release
  workflow produces its artefacts.
- `publish-if-needed.sh` refuses to publish any package whose `main`
  entry point does not exist on disk, closing the empty-tarball class of
  failure for every package.
- The GitHub-release step only runs on tag refs, so a
  `workflow_dispatch` republish no longer fails after publishing.

## [1.0.0] - 2026-08-04

First stable major release. Every published package moves to 1.0.0 in
lockstep: `@jeffs-brain/memory`, `@jeffs-brain/memory-pi`,
`@jeffs-brain/memory-postgres`, `@jeffs-brain/memory-openfga`,
`@jeffs-brain/memory-mcp`, `@jeffs-brain/install`, and the Go module
(tagged `go/v1.0.0`). The `rc` npm channel is retired; `latest` points at
1.0.0 for all packages. Development moves to trunk-based flow on `main`;
the long-running `develop` branch is retired.

### Publishing

#### Fixed

- Cross-package dependency ranges are now concrete (`^1.0.0`) instead of
  `workspace:*`. `npm publish` does not rewrite workspace specifiers, so
  `memory-postgres@0.2.0-rc.6` and `memory-mcp@0.1.0` shipped with a
  literal `workspace:*` peer range on `@jeffs-brain/memory` that npm
  consumers cannot resolve. The same defect was fixed by hand for
  `memory-pi` 0.2.1; concrete ranges remove the bug class.
- The MCP servers now report their real version. The TypeScript server
  advertised `0.0.1` regardless of package version, and the Go
  `memory-mcp` did the same.

### memory

#### Added

- OKF (Open Knowledge Format) v0.1 content profile. New `./okf` subpath export
  (`@jeffs-brain/memory/okf`) ships pure, I/O-free helpers: `parseOkfDocument`,
  `normaliseOkfMetadata`, `deriveOkfTypeFromPath`, `conceptIdFromPath`, link
  extraction, and soft validation. The profile is permissive by design — `type`
  is the only field required for strict conformance, with legacy aliases
  (`title`/`name`/heading; `description`/`summary`; `timestamp`/`modified`/
  `updated_at`; `resource`/`url`/`source_url`; `tags` list or CSV) and
  path-derived presentation types (`wiki/`→Article, `memory/`→Memory, `raw/`→Raw
  Document, `codec.md`→Codec). Both `[[wikilinks]]` and OKF markdown links are
  treated as graph edges; broken links are tolerated. No I/O, no DDL. (#80)
  (TypeScript)

### memory-postgres

#### Added

- OKF-aware document metadata extraction. The write path now extracts a
  non-lossy frontmatter superset via `extractDocumentMetadata(content, { path,
  normaliseOkf: true })` so the persisted `metadata` jsonb preserves every
  scalar/list key and adds `okf_type`/`okf_title`/`okf_description`/
  `okf_resource`/`okf_timestamp`/`okf_tags`, back-filling `description`/
  `timestamp`/`resource` only when absent (never clobbering explicit values).
  No frontmatter is byte-identical to before (`{}`). (#80) (TypeScript)

#### Changed

- `computeWikilinkEdges` now also matches OKF markdown links
  (`[text](/path.md)`) in addition to `[[wikilinks]]`, with relative-path
  resolution, `#fragment`/`?query` stripping, and external-scheme/`../`
  exclusion. This yields additional `wikilink` rows in `document_edges` for
  documents written with markdown cross-links; the existing `wikilink` edge type
  already satisfies the table CHECK, so no schema change is required. (#80)
  (TypeScript)

#### Fixed

- Thread the caller's `AbortSignal` through the Postgres retrieve path to the
  query-embedding call. `PostgresRetrievalRequest` now accepts an optional
  `signal`, which `createPostgresRetriever().retrieve` forwards to
  `embedder.embed([query], signal)`. Previously the query embedding ran with no
  deadline, so a degraded embeddings service could keep a `/ask` or `/search`
  request in flight past the intended request deadline (a bounded but
  unintentionally slow request). With the signal supplied, an aborted deadline
  (e.g. `AbortSignal.any([request-close, AbortSignal.timeout(...)])`) cancels
  the in-flight embed and retrieval degrades to lexical via the existing
  fallback, rather than blocking. Omitting `signal` is byte-identical to the
  previous behaviour. The Go retriever already threaded its `context.Context`
  to `Embedder.Embed`; a mirrored parity test now guards that behaviour.
  (LLE-10559) (TypeScript, Go)

- Memoise `PostgresStore.init()` ensure-schema so the additive-column DDL runs
  at most once per store instance. `init()` is now single-flight: the first call
  assigns and awaits a memoised promise and every later call returns the same
  settled promise without issuing any SQL; a failed (e.g. transient
  `lock_timeout`) run is not cached, so a later call may retry, while a
  successful run is never repeated. Previously a host that constructed a store
  (or called `init()`) per request re-ran `ALTER TABLE memory.documents ADD
  COLUMN IF NOT EXISTS ...` on the hot recall path, taking a fresh momentary
  `ACCESS EXCLUSIVE` lock on `memory.documents` on every recall. Under load
  these locks convoyed behind one another and stalled all reads, the root cause
  of the 2026-06-10 outage.
- Bound the ensure-schema transaction with a `SET LOCAL lock_timeout` (via
  `set_config('lock_timeout', ..., true)`, default
  `DEFAULT_INIT_LOCK_TIMEOUT_MS = 3000`ms, configurable via the new
  `initLockTimeoutMs` option). If `memory.documents` is held by a long-running
  transaction the blocked DDL now fails fast with a Postgres `lock_timeout`
  error (SQLSTATE 55P03) instead of convoying behind it. The bound is scoped to
  the ensure-schema transaction only; normal query transactions are unaffected.
  (#77, LLE-10529) (TypeScript)

- Persist the `metadata` jsonb column on document write. `PostgresStore`
  previously inserted only path/content_hash/size/source/content/updated_at and
  dropped `metadata`, so the column stayed `{}` and the metadata-keyed graph
  edges (`document_ontology`, `shared_tag`, `same_session`, `supersedes`) could
  never fire. The store now derives metadata from the document's own `---`
  frontmatter (via a non-lossy YAML-subset extractor — the typed
  `parseFrontmatter` intentionally drops keys like `ontology_type`) and persists
  it via `${json}::text::jsonb` (a plain `::jsonb` bind double-encodes a JS
  string into a jsonb string scalar, defeating `->>'key'`), with
  `on conflict ... metadata = excluded.metadata`. Empty/absent frontmatter is
  byte-identical to previous behaviour (no migration; the column already exists).
  A Go reader test pins that `computeDocumentOntologyEdges` consumes the
  persisted field. (#76, LLE-10520) (TypeScript)

### memory: codec extraction priors (Go + TypeScript)

#### Added

- Optional codec priors on the memory-note extraction path. `extract`
  now accepts a `CodecPriors` value (plain string lists: `entities`,
  `relations`, `domainTerms`) that is folded into the extraction system
  prompt as a bounded, deduplicated, truncated block of SOFT
  known-entity hints, so extraction reuses the project's canonical
  entity / relation / term names. TypeScript: `ExtractArgs.priors` on
  `createMemory().extract` plus exported `buildCodecPriorsBlock`,
  `applyCodecPriors`, and `CodecPriorsError`. Go: new
  `ExtractFromMessagesWithPriors` entry point plus the `CodecPriors`
  type and `ErrInvalidCodecPriors`. Priors are deliberately distinct
  from the typed `ResolvedOntology` used by the ontology-type extractor.
  Omitting or supplying empty priors is byte-identical to previous
  behaviour. Malformed priors (line breaks; non-string items in
  TypeScript) raise a typed error before any LLM call. A Go↔TS golden
  parity test pins the rendered block byte-for-byte. (Go + TypeScript)
- Cancellation on the TypeScript memory-note extraction path.
  `ExtractArgs.signal` threads an `AbortSignal` into the extraction LLM
  call and is honoured before and during the call; the Go path already
  honoured `context.Context` cancellation. (TypeScript)

### memory (Go)

Shipped early as `go/v0.3.1` on 2026-05-30.

#### Added

- `conversations` storage scope. A first-class top-level `conversations/`
  tree (laid out by channel then date) is now discovered, indexed, and
  retrievable alongside `wiki`, `memory`, `raw`, and `sources`. Adds
  `brain.ConversationsPrefix()` and `brain.Conversation(rel)`, wires the
  scope through `search.Index` (classify, discover, FTS filter, scope
  matching) and `retrieval` (exact-scope and scope-filter aliases), and
  parses conversation-article frontmatter (`title` / `summary` /
  `modified`) the same as wiki articles so search results carry titles
  and summaries. Lets hosts persist synthesised session-learning
  articles under the brain and surface them through hybrid retrieval
  with no host-side index workaround. (Go)

### memory-pi

The 0.2.x line shipped incrementally to npm from the development branch;
notable entries are consolidated below. Versions 0.2.3 through 0.2.7
shipped without changelog entries.

#### memory-pi 0.2.2: Added

- `vectorExtensionPath` config option (also `MEMORY_PI_VECTOR_EXTENSION_PATH`
  env var) that overrides the path to sqlite-vec's loadable extension.
  Threaded straight through to `createSearchIndex`. Required for hosts
  that ship a bun-compiled single-file binary: sqlite-vec's
  `import.meta.resolve('sqlite-vec-<platform>/vec0.<ext>')` fails inside
  the virtual fs, so the host must copy the native extension next to
  the executable and point memory-pi at it. (TS)

#### memory-pi 0.2.1: Fixed

- Replace `"@jeffs-brain/memory": "workspace:*"` in the published tarball
  with `"^0.3.0"` so consumers installing via `npm` / `bun add` outside
  the source monorepo can actually resolve the core SDK. `npm publish`
  does not rewrite workspace specifiers (unlike `bun publish` /
  `pnpm publish`), so the 0.2.0 tarball was effectively uninstallable
  outside this repo. (TS)

#### memory-pi 0.2.0: Added

- `flatLayout` configuration option on `createMemoryExtension`. When
  `true`, the extension treats `brainRoot` as the brain directly and
  skips the `brainId` subdirectory join. Aimed at single-brain hosts
  that manage one brain per identity at a fixed path. (TS)
- `searchIndexPath` configuration option to redirect the SQLite FTS
  index outside the brain root. Lets hosts that keep brain content in
  a git working tree keep machine-local state out of the tree. (TS)
- `bootstrapScanDirs` option (default `['wiki', 'memory', 'raw']`) and
  a one-shot indexer (`bootstrap-flat.ts`) that walks the configured
  directories on first boot, chunks every markdown file, and upserts
  the chunks into the FTS index via `SearchIndex.upsertChunks`. The
  Store is bypassed entirely so source files are never duplicated or
  rewritten. Idempotent on re-entry. (TS)
- Internal SQLite `SearchIndex` is now wired into the `Memory` recall
  pipeline through an adapter so `memory_recall` returns BM25 hits
  instead of relying on the scope-prefix fallback. (TS)
- Environment variables `MEMORY_PI_FLAT_LAYOUT`,
  `MEMORY_PI_SEARCH_INDEX_PATH`, `MEMORY_PI_BRAIN_ROOT`,
  `MEMORY_PI_BRAIN_ID` for ops-friendly configuration. (TS)

#### memory-pi 0.2.0: Changed

- `resolveBrainPaths(root, brainId)` now accepts an optional third
  argument `{ flat?: boolean; searchIndexPath?: string }`. Existing
  two-argument calls keep working unchanged. (TS)
- `@earendil-works/pi-coding-agent` and `typebox` are now declared as
  `peerDependencies` so pi-bundled copies are used instead of installed
  duplicates. Required by pi's package-loading model. (TS)

## [0.4.0-rc.1] - 2026-05-21

### Added
- **P3: Queue Infrastructure** — PostgreSQL ingest queue (FOR UPDATE SKIP LOCKED), worker pool with crash recovery, dead letter queue with error history, shared rate limiter with circuit breaker
- **P4: Multimodal Extraction** — OCR (PaddleOCR/Tesseract), scanned PDF extraction, audio transcription (faster-whisper), streaming video extraction with keyframe OCR, schema-aware structured data (CSV/JSON/JSONL/XML)
- **P5: Connector Framework** — OAuth2 + SecureTokenStore (AES-256-GCM), connector registry, Slack/Google Drive/Notion connectors, webhook receiver (HMAC-SHA256)
- **Ontology** — 10 new edge types (applies_to, contains, assigned_to, implements, created_by, supersedes, derived_from, governs, requires, maps_to), removed rule.combined node type
- **Rate Limiting** — Token bucket with adaptive header-based throttling, per-tenant factory with TTL eviction, Netflix Hystrix circuit breaker pattern
- **Pipeline** — State machine with crash recovery, chunk delta computation, BLAKE3 migration, multi-language stemmers, reconciliation

### Changed
- Ontology: 31 node types → 30 (removed rule.combined), 19 edge types → 29 (+10)
- memory-postgres migrations renumbered sequentially (0001-0008)

### Fixed
- All 28 critical + 70 major review findings addressed
- Advisory lock scope (transaction-scoped unlock)
- Goroutine leak in rate limiter retry-after
- Worker pool retry count on per-brain rejection (uses Requeue)


## [0.3.0] - 2026-05-12

### Added

- Diversity-aware recall reranking with MMR-style greedy selection, Jaccard similarity penalties, and date-bucket diversity (Go, #22)
- Parallel rerank batching with configurable concurrency via errgroup (Go, #23)
- Full episode management CRUD — create, get, list, query by date range, participant, and topic (Go, #24)
- Age-based heuristic confidence with 90-day stale demotion, 180-day force-low, and reinforcement-span promotion (Go, #25)
- Feedback classifier for detecting positive, negative, and correction feedback in user messages (TS, #26)
- Cost accounting with BigInt microcents for drift-free LLM cost tracking in eval framework (TS, #27)
- Pipeline state tracking for crash recovery — documents resume from last completed stage on re-ingest (TS, #28)
- Prompt injection safety scanner with ML-based detection via @stackone/defender (TS) and Scanner interface with preprocessing and content isolation (Go, #29)

### Fixed

- Timing-unsafe bearer token comparison replaced with crypto/subtle.ConstantTimeCompare and SHA-256 pre-hashing (#10)
- SSRF in URL ingestion blocked with DNS-level IP validation via custom DialContext (#11)
- BrainID path traversal prevented with ValidateBrainID shared validation (#12)
- OpenAI embedder now includes Dimensions field in API requests (#13)
- Anthropic streaming tool_use content blocks handled via state machine (#14)
- PT store batch List correctly overlays journal state (#15)
- HTTP client timeouts added to all LLM providers via ResponseHeaderTimeout (#16)
- HTTP store New returns error instead of panicking (#17)
- RRF fusion skips empty-id candidates (#18)
- Composite-concrete query detection now requires both first-person and verb regexes, matching TS behaviour (#19)
- Stale-superseded multiplier aligned between Go and TS — metadata check, text-regex check, and unconditional application scope (#20)
- Markdown fence stripping added to TS LLM reranker JSON extraction (#21)

## [0.2.3] - 2026-04-29

### Fixed

- Made the Go retrieval retry ladder run a real index refresh through `RefreshSource` instead of treating the refresh rung as a no-op.

## [0.2.2] - 2026-04-29

### Fixed

- Made the Go path-slug fallback tests portable across macOS temp-directory symlinks.

## [0.2.1] - 2026-04-29

### Added

- First installable Go module release under `github.com/jeffs-brain/memory/go`, tagged as `go/v0.2.1`.
- Go release workflow validation for `go/vX.Y.Z` tags.

### Changed

- Moved the Go module to the repository `go/` directory so the public module path resolves through the standard Go toolchain.

## [0.2.0] - 2026-04-22

### Added

- `@jeffs-brain/memory/conformance`: reusable HTTP conformance runner with the shared `spec/conformance/http-contract.json` fixture bundled into the published package.
- `@jeffs-brain/memory/sse`: framework-agnostic SSE frame formatting and heartbeat helpers for custom daemons and transports.

### Fixed

- `memory serve` SSE streams now emit monotonic event ids and reuse the exported framing and heartbeat helpers across the built-in HTTP transport.

## [0.1.0] - 2026-04-18

### Added

- First public release of `@jeffs-brain/memory` TypeScript SDK: FsStore, MemStore, GitStore, HttpStore; SQLite BM25 plus pure-JS vector search; query DSL with alias tables; RRF hybrid retrieval with five-rung retry ladder; memory stages (extract, reflect, consolidate); knowledge ingest (markdown, URL, file, PDF).
- `@jeffs-brain/memory-postgres`: Postgres sibling for high-scale deployments.
- `@jeffs-brain/memory-openfga`: pure-fetch OpenFGA adapter for authorisation.
- `@jeffs-brain/memory-mcp`: Model Context Protocol stdio server exposing 11 tools; zero-config local mode (fs plus sqlite plus Ollama auto-detect) and hosted mode (`JB_TOKEN` plus HttpStore).
- Authoritative wire spec at `spec/` covering PROTOCOL, STORAGE, QUERY-DSL, ALGORITHMS, MCP-TOOLS.
- Conformance harness with 29 wire cases to drive SDK parity.
- Cross-language eval runner skeleton (`eval/`) with smoke and nightly matrices.

### Notes

- Go and Python SDKs are in the pipeline. See `go/` and `sdks/py/` README files.
- Platform integration (multi-tenant backend) is private and unpublished.

[Unreleased]: https://github.com/jeffs-brain/memory/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/jeffs-brain/memory/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/jeffs-brain/memory/compare/v0.4.0-rc.1...v1.0.0
[0.4.0-rc.1]: https://github.com/jeffs-brain/memory/compare/v0.3.0...v0.4.0-rc.1
[0.3.0]: https://github.com/jeffs-brain/memory/compare/go/v0.2.3...v0.3.0
[0.2.3]: https://github.com/jeffs-brain/memory/compare/go/v0.2.2...go/v0.2.3
[0.2.2]: https://github.com/jeffs-brain/memory/compare/go/v0.2.1...go/v0.2.2
[0.2.1]: https://github.com/jeffs-brain/memory/compare/v0.2.0...go/v0.2.1
[0.2.0]: https://github.com/jeffs-brain/memory/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jeffs-brain/memory/releases/tag/v0.1.0
