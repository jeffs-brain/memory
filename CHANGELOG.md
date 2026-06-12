# Changelog

All notable changes to the jeffs-brain memory libraries are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

## [Unreleased]

### memory-postgres 0.2.0-rc.5

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

### memory-postgres 0.2.0-rc.4

#### Fixed

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

### memory-postgres 0.2.0-rc.3

#### Fixed

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

### memory (Go + TypeScript) — codec extraction priors

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

### memory (Go) 0.3.1

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

### memory-pi 0.2.2

#### Added

- `vectorExtensionPath` config option (also `MEMORY_PI_VECTOR_EXTENSION_PATH`
  env var) that overrides the path to sqlite-vec's loadable extension.
  Threaded straight through to `createSearchIndex`. Required for hosts
  that ship a bun-compiled single-file binary: sqlite-vec's
  `import.meta.resolve('sqlite-vec-<platform>/vec0.<ext>')` fails inside
  the virtual fs, so the host must copy the native extension next to
  the executable and point memory-pi at it. (TS)

### memory-pi 0.2.1

#### Fixed

- Replace `"@jeffs-brain/memory": "workspace:*"` in the published tarball
  with `"^0.3.0"` so consumers installing via `npm` / `bun add` outside
  the source monorepo can actually resolve the core SDK. `npm publish`
  does not rewrite workspace specifiers (unlike `bun publish` /
  `pnpm publish`), so the 0.2.0 tarball was effectively uninstallable
  outside this repo. (TS)

### memory-pi 0.2.0

#### Added

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

#### Changed

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

[Unreleased]: https://github.com/jeffs-brain/memory/compare/v0.4.0-rc.1...HEAD
[0.4.0-rc.1]: https://github.com/jeffs-brain/memory/compare/v0.3.0...v0.4.0-rc.1
[0.3.0]: https://github.com/jeffs-brain/memory/compare/go/v0.2.3...v0.3.0
[0.2.3]: https://github.com/jeffs-brain/memory/compare/go/v0.2.2...go/v0.2.3
[0.2.2]: https://github.com/jeffs-brain/memory/compare/go/v0.2.1...go/v0.2.2
[0.2.1]: https://github.com/jeffs-brain/memory/compare/v0.2.0...go/v0.2.1
[0.2.0]: https://github.com/jeffs-brain/memory/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jeffs-brain/memory/releases/tag/v0.1.0
