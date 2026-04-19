# Jeffs Brain specification

This directory is the **language-neutral source of truth** for the Jeffs Brain memory library. It defines the HTTP protocol, Server-Sent Events contract, query DSL, retrieval algorithms, storage semantics, MCP tool surface, and the fixtures and conformance suites every SDK implementation must satisfy.

All three SDKs (TypeScript, Go, Python) read from this spec. When any SDK disagrees with this directory, this directory wins.

## Layout

- `PROTOCOL.md`: HTTP store + SSE event stream wire contract consumed by the HTTP-backed `Store`.
- `QUERY-DSL.md`: query grammar, tokenisation rules, stopword filtering, alias expansion, FTS5 compilation.
- `ALGORITHMS.md`: hybrid retrieval pipeline, Reciprocal Rank Fusion, unanimity shortcut, rerank tail preservation, retry ladder.
- `STORAGE.md`: `Store` interface, path validation rules, batch semantics, error taxonomy, `ChangeEvent` shape.
- `MCP-TOOLS.md`: the `memory_*` MCP tool surface every SDK's MCP wrapper must expose.
- `fixtures/`: language-neutral test inputs: stopword lists, query parser cases, retrieval golden sets.
- `conformance/`: replayable case files that drive any SDK's HTTP store implementation against the contract.

## Status

- Spec stable; all three SDKs implement the full surface.
- HTTP conformance suite: 28/29 cases green across TypeScript, Go, and Python.
- Tri-SDK smoke benchmark: 19/20 (95%) on every SDK against Ollama `gemma3:latest`. See [`../eval/results/cross-sdk/`](../eval/results/cross-sdk).
- Sections that are intentionally under-specified (to give SDK authors room) are marked inline; everything else is observed behaviour of at least one shipping SDK.
