# Introduction

This is a reference document used for conformance testing across the Go and TypeScript SDKs. Both implementations must produce identical chunk boundaries when given the same ChunkConfig.

The introduction section contains enough text to form a standalone chunk above the minimum token threshold, ensuring it is not merged into any adjacent section.

## Architecture Overview

The system is organised into three layers: the store layer handles persistence, the index layer handles full-text search, and the retrieval layer handles hybrid ranking. Each layer is independently testable and can be swapped out without affecting the others.

Documents flow through the ingestion pipeline in four stages: extraction, normalisation, chunking, and indexing. The extraction stage converts source formats (HTML, PDF, markdown) into plain text. Normalisation strips control characters and enforces UTF-8 encoding. Chunking splits the normalised text into segments bounded by the configured token ceiling. Indexing writes the resulting chunks into the search index for later retrieval.

## Configuration

The chunking subsystem accepts three parameters: maxTokens controls the ceiling per chunk, overlapTokens controls how many trailing tokens from the previous chunk are prepended to the next, and minTokens sets the floor below which a chunk is merged into its neighbour.

## Short Section

Brief.
