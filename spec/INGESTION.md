# Ingestion

This document defines the canonical chunking parameters and ingestion behaviour every Jeffs Brain SDK must implement. Conformance is verified by shared fixtures in `spec/fixtures/ingestion/`.

## Token estimation

All SDKs use a character-based approximation until a real tokeniser is wired in:

```
estimateTokens(text) = ceil(len(text) / 4)
```

Go implementation uses integer ceiling division: `(len(text) + 3) / 4`.
TypeScript implementation uses `Math.ceil(text.length / 4)`.

Both produce identical results for any given byte string.

## ChunkConfig

Every SDK exposes a `ChunkConfig` type (struct in Go, interface in TypeScript) with the following fields:

| Field | Type | Default | Constraint |
|-------|------|---------|------------|
| maxTokens | int | 512 | > 0, >= overlapTokens * 2 |
| overlapTokens | int | 64 | >= 0, < maxTokens |
| minTokens | int | 30 | >= 0, < maxTokens |
| strategy | enum | 'recursive' | see Strategy below |
| separators | string[] | see below | non-empty when strategy is 'recursive' |

### Defaults

```
DEFAULT_MAX_TOKENS     = 512
DEFAULT_OVERLAP_TOKENS = 64
DEFAULT_MIN_TOKENS     = 30
DEFAULT_STRATEGY       = 'recursive'
```

### Strategy enum

Strategies determine how the chunker identifies split points:

| Value | Description |
|-------|-------------|
| `recursive` | Split at separators in priority order, recurse into sub-chunks |
| `markdown` | Split at heading boundaries, fall back to recursive within sections |
| `code` | Split at function/class boundaries (language-aware) |
| `table` | Split at row boundaries preserving header context |
| `conversation` | Split at speaker-turn boundaries |

The initial implementation (P0-3) defines the enum and validates membership. Strategies beyond `recursive` and `markdown` are reserved for future tickets.

### Default separator hierarchy

When strategy is `recursive`, separators are tried in order until one produces chunks within budget:

```
["\n\n", "\n", ". ", " ", ""]
```

The empty string `""` is the final fallback: character-level splitting.

## Validation rules

`validateChunkConfig` (or `ValidateChunkConfig` in Go) enforces:

1. `maxTokens > 0`
2. `overlapTokens >= 0`
3. `overlapTokens < maxTokens`
4. `minTokens >= 0`
5. `minTokens < maxTokens`
6. `strategy` is a member of the Strategy enum
7. `separators` is non-empty when strategy is `recursive`

Invalid configs are rejected with a descriptive error. SDKs must never silently clamp invalid values.

## Conformance requirement

Given the same input text and the same `ChunkConfig`, both the Go and TypeScript SDKs must produce:

1. Identical token estimates for any string.
2. Identical default config values.
3. Identical validation outcomes (accept/reject) for any config.

Chunk boundary conformance (same offsets for the same document) is a Phase 1 goal tracked by P1-4. This spec defines the shared config and estimation that makes boundary alignment possible.

## Migration path from Go hardcoded constants

The Go `knowledge` package currently uses:

```
defaultChunkMaxChars = 1800  (character-based)
defaultChunkMinChars = 120   (character-based)
overlap              = 0     (none)
```

These will be replaced in P1-4 when the knowledge package adopts `ChunkConfig` from `go/ingest`. P0-3 creates the config type and canonical defaults without modifying the existing `knowledge` package behaviour.
