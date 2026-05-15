# Changelog

All notable changes to the jeffs-brain memory libraries are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

## [Unreleased]

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

[Unreleased]: https://github.com/jeffs-brain/memory/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/jeffs-brain/memory/compare/go/v0.2.3...v0.3.0
[0.2.3]: https://github.com/jeffs-brain/memory/compare/go/v0.2.2...go/v0.2.3
[0.2.2]: https://github.com/jeffs-brain/memory/compare/go/v0.2.1...go/v0.2.2
[0.2.1]: https://github.com/jeffs-brain/memory/compare/v0.2.0...go/v0.2.1
[0.2.0]: https://github.com/jeffs-brain/memory/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jeffs-brain/memory/releases/tag/v0.1.0
