// SPDX-License-Identifier: Apache-2.0

package knowledge

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/retrieval"
	"github.com/jeffs-brain/memory/go/search"
)

// Base is the top-level knowledge surface combining ingest, compile,
// and search. Implementations are expected to be safe for concurrent
// use; [New] returns an implementation that satisfies that contract.
type Base interface {
	// Ingest persists a single document sourced from a local file, a
	// URL, or an inline reader. Returns an [IngestResponse] summarising
	// what was written.
	Ingest(ctx context.Context, req IngestRequest) (IngestResponse, error)

	// IngestURL is a convenience wrapper that fetches url, extracts
	// plain text, and persists it. Delegates to [Base.Ingest].
	IngestURL(ctx context.Context, url string) (IngestResponse, error)

	// Compile chunks persisted documents and writes the resulting
	// chunks to the bound search index.
	Compile(ctx context.Context, opts CompileOptions) (CompileResult, error)

	// Search runs a hybrid retrieval against the bound index.
	Search(ctx context.Context, req SearchRequest) (SearchResponse, error)

	// Store returns the underlying [brain.Store] so callers can share
	// it with adjacent subsystems (memory, retrieval, FTS).
	Store() brain.Store

	// SetSearchIndex rebinds the FTS index used by Search and Compile.
	// Passing nil unbinds the index and forces the in-memory fallback.
	SetSearchIndex(idx *search.Index)

	// SetRetriever rebinds the hybrid retriever. Passing nil unbinds
	// the retriever; Search then degrades to the BM25 path.
	SetRetriever(r retrieval.Retriever)

	// Close releases resources held by the package. Does not close
	// the injected store; that lifecycle belongs to the caller.
	Close() error
}

// Options configures [New].
type Options struct {
	// BrainID tags every persisted document. Optional.
	BrainID string

	// Store is the backing [brain.Store]. Required.
	Store brain.Store

	// Index is the FTS5 index used by compile + BM25 search. Optional;
	// when nil, Search uses the in-memory fallback and Compile is a
	// no-op beyond document chunking.
	Index *search.Index

	// Retriever is the hybrid retriever used by Search when mode is
	// Auto or Hybrid. Optional; gracefully degrades to BM25 + FTS.
	Retriever retrieval.Retriever

	// HTTPFetcher overrides the URL fetcher used by IngestURL.
	// Defaults to [defaultFetcher]. Test shim.
	HTTPFetcher Fetcher
}

// Fetcher abstracts HTTP fetching so [Base.IngestURL] remains testable.
type Fetcher interface {
	Fetch(ctx context.Context, url string) (body []byte, contentType string, err error)
}

// kbase is the concrete [Base] implementation.
//
// All mutable fields (index, retriever, fetcher) are protected by mu so
// concurrent SetSearchIndex / SetRetriever calls are safe.
type kbase struct {
	brainID string
	store   brain.Store

	mu        sync.RWMutex
	index     *search.Index
	retriever retrieval.Retriever
	fetcher   Fetcher
}

// New constructs a [Base] backed by the supplied options. Returns an
// error when no store is supplied.
func New(opts Options) (Base, error) {
	if opts.Store == nil {
		return nil, errors.New("knowledge: Options.Store is required")
	}
	fetcher := opts.HTTPFetcher
	if fetcher == nil {
		fetcher = defaultFetcher{}
	}
	return &kbase{
		brainID:   opts.BrainID,
		store:     opts.Store,
		index:     opts.Index,
		retriever: opts.Retriever,
		fetcher:   fetcher,
	}, nil
}

// Store implements [Base].
func (k *kbase) Store() brain.Store { return k.store }

// SetSearchIndex implements [Base].
func (k *kbase) SetSearchIndex(idx *search.Index) {
	k.mu.Lock()
	k.index = idx
	k.mu.Unlock()
}

// SetRetriever implements [Base].
func (k *kbase) SetRetriever(r retrieval.Retriever) {
	k.mu.Lock()
	k.retriever = r
	k.mu.Unlock()
}

// Close implements [Base]. The store lifecycle is owned by the caller,
// so Close only releases package-local state. Safe to call multiple
// times.
func (k *kbase) Close() error { return nil }

// snapshot returns the current index and retriever under a read lock.
// Callers use the snapshot to avoid holding the lock across I/O.
func (k *kbase) snapshot() (*search.Index, retrieval.Retriever, Fetcher) {
	k.mu.RLock()
	defer k.mu.RUnlock()
	return k.index, k.retriever, k.fetcher
}

// requireStore returns an error for the unlikely case where a Base was
// constructed against a nil store (e.g. by direct struct literal in a
// test). [New] prevents this, but the check keeps the error path
// explicit.
func (k *kbase) requireStore() error {
	if k.store == nil {
		return fmt.Errorf("knowledge: nil store")
	}
	return nil
}
