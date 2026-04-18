# Jeffs Brain specification

This directory is the **language-neutral source of truth** for the Jeffs Brain memory library. It defines the HTTP protocol, Server-Sent Events contract, query DSL, retrieval algorithms, storage semantics, MCP tool surface, and the fixtures and conformance suites that every SDK implementation must satisfy. The TypeScript implementation under `packages/memory/` is today the reference SDK and was extracted from a Go prototype; the Go and Python SDKs in flight are expected to read from this spec rather than from `packages/memory/`. When the two disagree, this directory wins.

The canonical contents of `spec/` will eventually move into a top-level `spec/` directory inside the open-source `jeffs-brain/memory` monorepo, which will host all three SDKs (TypeScript, Go, Python) alongside a shared conformance harness. Lifting the specification out of the SDK now lets the harness evolve independently of any one language and pre-stages that move.

## Layout

- `PROTOCOL.md` — HTTP store + SSE event stream wire contract consumed by the HTTP-backed `Store`.
- `QUERY-DSL.md` — query grammar, tokenisation rules, stopword filtering, alias expansion, FTS5 compilation.
- `ALGORITHMS.md` — hybrid retrieval pipeline, Reciprocal Rank Fusion, unanimity shortcut, rerank tail preservation, retry ladder.
- `STORAGE.md` — `Store` interface, path validation rules, batch semantics, error taxonomy, `ChangeEvent` shape.
- `MCP-TOOLS.md` — the `memory_*` MCP tool surface every SDK's MCP wrapper must expose.
- `fixtures/` — language-neutral test inputs: stopword lists, query parser cases.
- `conformance/` — replayable case files that drive any SDK's HTTP store implementation against the contract.

## Status

The current draft is lifted from the TypeScript reference and captures its observable behaviour. Sections that are ambiguous or under-specified in the reference are marked `TODO` inline so Go and Python implementers can flag them for clarification before coding.
