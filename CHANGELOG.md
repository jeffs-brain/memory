# Changelog

All notable changes to the jeffs-brain memory libraries are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

## [Unreleased]

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

- Go and Python SDKs are in the pipeline. See `sdks/go/` and `sdks/py/` README files.
- Platform integration (multi-tenant backend) is private and unpublished.

[Unreleased]: https://github.com/jeffs-brain/memory/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/jeffs-brain/memory/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jeffs-brain/memory/releases/tag/v0.1.0
