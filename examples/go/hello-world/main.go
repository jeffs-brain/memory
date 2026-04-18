// SPDX-License-Identifier: Apache-2.0

// Hello World example for the Jeffs Brain Go SDK.
//
// Walks the canonical knowledge pipeline:
//
//  1. Open a filesystem-backed brain.
//  2. Open the SQLite FTS5 index next to the brain root and bind it
//     to a knowledge.Base.
//  3. Ingest a markdown file via knowledge.Ingest, which persists the
//     document under raw/documents/.
//  4. Run a knowledge.Search and print the top results.
//
// Search now hits the BM25 path end to end: the index walks
// raw/documents/ alongside memory/ and wiki/, so ingested content is
// immediately discoverable.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/knowledge"
	"github.com/jeffs-brain/memory/go/search"
	"github.com/jeffs-brain/memory/go/store/fs"
)

const brainID = "hello-world"

func main() {
	ctx := context.Background()

	root, err := filepath.Abs("./data/" + brainID)
	if err != nil {
		log.Fatalf("resolve brain root: %v", err)
	}
	if err := os.MkdirAll(root, 0o755); err != nil {
		log.Fatalf("mkdir brain root: %v", err)
	}

	store, err := fs.New(root)
	if err != nil {
		log.Fatalf("fs.New: %v", err)
	}
	defer func() { _ = store.Close() }()

	b, err := brain.Open(ctx, brain.Options{ID: brainID, Root: root, Store: store})
	if err != nil {
		log.Fatalf("brain.Open: %v", err)
	}
	defer func() { _ = b.Close() }()

	db, err := search.OpenDB(filepath.Join(root, ".search.db"))
	if err != nil {
		log.Fatalf("search.OpenDB: %v", err)
	}
	defer func() { _ = search.CloseDB(db) }()

	idx, err := search.NewIndex(db, b.Store())
	if err != nil {
		log.Fatalf("search.NewIndex: %v", err)
	}
	unsub := idx.Subscribe(b.Store())
	defer unsub()

	kb, err := knowledge.New(knowledge.Options{
		BrainID: brainID,
		Store:   b.Store(),
		Index:   idx,
	})
	if err != nil {
		log.Fatalf("knowledge.New: %v", err)
	}
	defer func() { _ = kb.Close() }()

	docPath, err := filepath.Abs("./docs/hedgehogs.md")
	if err != nil {
		log.Fatalf("resolve doc path: %v", err)
	}
	ingestResp, err := kb.Ingest(ctx, knowledge.IngestRequest{Path: docPath})
	if err != nil {
		log.Fatalf("ingest: %v", err)
	}
	fmt.Printf("Ingested %s (%d chunks, %d bytes)\n", ingestResp.Path, ingestResp.ChunkCount, ingestResp.Bytes)

	query := "where do hedgehogs live?"
	resp, err := kb.Search(ctx, knowledge.SearchRequest{Query: query, MaxResults: 3})
	if err != nil {
		log.Fatalf("search: %v", err)
	}

	fmt.Printf("Top %d results for %q:\n", len(resp.Hits), query)
	for i, h := range resp.Hits {
		snippet := h.Snippet
		if snippet == "" {
			snippet = h.Summary
		}
		if len(snippet) > 160 {
			snippet = snippet[:160] + "..."
		}
		fmt.Printf("%d. [%.3f] %s\n   %s\n", i+1, h.Score, h.Path, snippet)
	}
}
