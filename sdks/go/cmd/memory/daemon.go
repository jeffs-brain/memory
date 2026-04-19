// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/knowledge"
	"github.com/jeffs-brain/memory/go/llm"
	"github.com/jeffs-brain/memory/go/memory"
	"github.com/jeffs-brain/memory/go/retrieval"
	"github.com/jeffs-brain/memory/go/search"
	"github.com/jeffs-brain/memory/go/store/pt"
)

// Daemon owns every resource the HTTP daemon shares across handlers.
//
// A single Daemon backs `memory serve`. It resolves per-brain stores,
// search indices, retrievers, memory and knowledge facades on demand
// via [BrainManager]. The LLM provider and embedder come from
// environment configuration and are shared across brains.
type Daemon struct {
	Root       string
	AuthToken  string
	LLM        llm.Provider
	Embedder   llm.Embedder
	EmbedModel string
	Logger     *slog.Logger
	Brains     *BrainManager
}

// NewDaemon constructs a Daemon rooted at root. When provider is nil
// the daemon resolves one from the environment via
// [llm.ProviderFromEnv]. When embedder is nil it does the same via
// [llm.EmbedderFromEnv]. Failures resolving optional dependencies are
// non-fatal: the daemon falls back to deterministic stubs so callers
// can still exercise the wire surface.
func NewDaemon(ctx context.Context, root, authToken string, log *slog.Logger) (*Daemon, error) {
	if root == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, fmt.Errorf("daemon: resolve home: %w", err)
		}
		root = filepath.Join(home, ".jeffs-brain")
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("daemon: resolve root: %w", err)
	}
	if err := os.MkdirAll(abs, 0o755); err != nil {
		return nil, fmt.Errorf("daemon: create root %s: %w", abs, err)
	}
	if log == nil {
		log = slog.Default()
	}
	provider, perr := llm.ProviderFromEnv(llm.OSGetenv)
	if perr != nil {
		log.Warn("daemon: resolving llm provider", "err", perr)
		provider = llm.NewFake([]string{"ok"})
	}
	embedder, eerr := llm.EmbedderFromEnv(llm.OSGetenv)
	if eerr != nil {
		log.Debug("daemon: resolving embedder", "err", eerr)
		embedder = nil
	}
	embedModel := resolveEmbedModel(llm.OSGetenv, embedder)
	d := &Daemon{
		Root:       abs,
		AuthToken:  authToken,
		LLM:        provider,
		Embedder:   embedder,
		EmbedModel: embedModel,
		Logger:     log,
	}
	d.Brains = NewBrainManager(d)
	return d, nil
}

// Close releases every resource the daemon owns. Safe to call
// multiple times.
func (d *Daemon) Close() error {
	if d == nil {
		return nil
	}
	if d.Brains != nil {
		_ = d.Brains.Close()
	}
	if d.LLM != nil {
		_ = d.LLM.Close()
	}
	if d.Embedder != nil {
		_ = d.Embedder.Close()
	}
	return nil
}

func (d *Daemon) brainRoot(brainID string) string {
	return filepath.Join(d.Root, "brains", brainID)
}

func (d *Daemon) brainExists(brainID string) bool {
	info, err := os.Stat(d.brainRoot(brainID))
	if err != nil {
		return false
	}
	return info.IsDir()
}

// BrainResources is the set of per-brain runtime objects shared by
// every handler that operates on a brain. Resources are created once
// per brain and reused for the daemon's lifetime; [Close] releases
// them.
type BrainResources struct {
	ID        string
	Root      string
	Store     brain.Store
	Brain     *brain.Brain
	Search    *search.Index
	Retriever retrieval.Retriever
	Memory    *memory.Memory
	Knowledge knowledge.Base

	// initialScan resolves once the first full index scan completes.
	// Handlers that care about pre-seeded disk content (e.g. /search,
	// /ask) await this channel so the tri-SDK extract-once flow returns
	// hits on the first query. After the initial scan, Subscribe
	// handles mutations incrementally so no per-request re-scan is
	// needed.
	initialScan chan struct{}
}

// WaitReady blocks until the brain's initial search-index scan has
// completed, or the context is cancelled. Cheap after the first call:
// the channel is closed once and every subsequent wait returns
// immediately.
func (br *BrainResources) WaitReady(ctx context.Context) {
	if br == nil || br.initialScan == nil {
		return
	}
	select {
	case <-br.initialScan:
	case <-ctx.Done():
	}
}

// Close tears down every resource owned by the brain entry. Safe to
// call multiple times.
func (br *BrainResources) Close() error {
	if br == nil {
		return nil
	}
	var firstErr error
	if br.Knowledge != nil {
		if err := br.Knowledge.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	// search.Index owns the *sql.DB through a shared cache that the
	// daemon closes via BrainManager.Close, so do not close it here.
	if br.Brain != nil {
		if err := br.Brain.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// BrainManager lazily constructs and caches per-brain resources.
//
// Concurrency model: an RWMutex guards the map; readers walk the map
// in parallel, writers serialise create/delete. Construction of a
// fresh entry is double-checked under the write lock to avoid racing
// callers building two stores against the same root.
type BrainManager struct {
	d   *Daemon
	mu  sync.RWMutex
	all map[string]*BrainResources
}

func NewBrainManager(d *Daemon) *BrainManager {
	return &BrainManager{d: d, all: map[string]*BrainResources{}}
}

// ErrBrainNotFound is returned by [BrainManager.Get] when no brain
// exists on disk for the supplied id.
var ErrBrainNotFound = errors.New("brain manager: not found")

// List returns every brain id discovered under the daemon root by
// walking the brains/ subdirectory. Entries are returned in
// lexicographic order.
func (bm *BrainManager) List() ([]string, error) {
	dir := filepath.Join(bm.d.Root, "brains")
	entries, err := os.ReadDir(dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	out := make([]string, 0, len(entries))
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		out = append(out, e.Name())
	}
	return out, nil
}

// Create initialises a new brain on disk and warms its resource
// cache. Returns [brain.ErrConflict] (wrapped) if the brain root
// already exists.
func (bm *BrainManager) Create(ctx context.Context, brainID string) (*BrainResources, error) {
	if brainID == "" {
		return nil, errors.New("brain manager: brainID required")
	}
	root := bm.d.brainRoot(brainID)
	if _, err := os.Stat(root); err == nil {
		return nil, fmt.Errorf("brain manager: brain %s: %w", brainID, brain.ErrConflict)
	}
	if err := os.MkdirAll(root, 0o755); err != nil {
		return nil, fmt.Errorf("brain manager: create %s: %w", root, err)
	}
	return bm.Get(ctx, brainID)
}

// Get returns the cached BrainResources for brainID, building one on
// the first hit. Returns [ErrBrainNotFound] when no on-disk root
// exists.
func (bm *BrainManager) Get(ctx context.Context, brainID string) (*BrainResources, error) {
	bm.mu.RLock()
	if entry, ok := bm.all[brainID]; ok {
		bm.mu.RUnlock()
		return entry, nil
	}
	bm.mu.RUnlock()

	bm.mu.Lock()
	defer bm.mu.Unlock()
	if entry, ok := bm.all[brainID]; ok {
		return entry, nil
	}
	if !bm.d.brainExists(brainID) {
		return nil, ErrBrainNotFound
	}
	entry, err := bm.build(ctx, brainID)
	if err != nil {
		return nil, err
	}
	bm.all[brainID] = entry
	return entry, nil
}

// Delete tears down the cached resources for brainID and removes the
// on-disk root. Returns [ErrBrainNotFound] when the brain is not
// present.
func (bm *BrainManager) Delete(brainID string) error {
	bm.mu.Lock()
	defer bm.mu.Unlock()
	if entry, ok := bm.all[brainID]; ok {
		_ = entry.Close()
		delete(bm.all, brainID)
	}
	root := bm.d.brainRoot(brainID)
	info, err := os.Stat(root)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return ErrBrainNotFound
		}
		return err
	}
	if !info.IsDir() {
		return ErrBrainNotFound
	}
	return os.RemoveAll(root)
}

func (bm *BrainManager) Close() error {
	bm.mu.Lock()
	defer bm.mu.Unlock()
	for id, entry := range bm.all {
		_ = entry.Close()
		delete(bm.all, id)
	}
	return nil
}

// build wires a fresh BrainResources for an existing on-disk root.
// Search and retrieval are best-effort: if SQLite fails to open,
// search is left nil and the retriever falls back to BM25-via-search
// (or no-op when no index is present).
func (bm *BrainManager) build(ctx context.Context, brainID string) (*BrainResources, error) {
	root := bm.d.brainRoot(brainID)
	store, err := pt.New(root)
	if err != nil {
		return nil, fmt.Errorf("brain manager: store %s: %w", root, err)
	}
	br, err := brain.Open(ctx, brain.Options{ID: brainID, Root: root, Store: store})
	if err != nil {
		_ = store.Close()
		return nil, fmt.Errorf("brain manager: open %s: %w", brainID, err)
	}
	mem := memory.New(store)

	idx, vecIdx, src, retr := buildSearchAndRetrieval(ctx, root, store, bm.d, bm.d.Logger)

	kbase, kerr := knowledge.New(knowledge.Options{
		BrainID:   brainID,
		Store:     store,
		Index:     idx,
		Retriever: retr,
	})
	if kerr != nil {
		_ = br.Close()
		return nil, fmt.Errorf("brain manager: knowledge %s: %w", brainID, kerr)
	}

	initialScan := make(chan struct{})
	if idx != nil && store != nil {
		// Subscribe the index to brain mutations so writes propagate
		// into the FTS table without a second pass.
		_ = idx.Subscribe(store)
		// Initial scan so a newly opened brain with prior content
		// surfaces on the first search. Close the channel when done so
		// callers awaiting WaitReady() unblock. Vector backfill runs
		// afterwards so FTS queries unblock immediately while the
		// slower remote embed batches populate the vector index in the
		// background.
		go func() {
			defer close(initialScan)
			if err := idx.Update(context.Background()); err != nil {
				bm.d.Logger.Debug("search: initial update failed", "brain", brainID, "err", err)
			}
			if vecIdx != nil && bm.d.Embedder != nil && bm.d.EmbedModel != "" {
				backfillVectors(context.Background(), brainID, store, idx, vecIdx,
					bm.d.Embedder, bm.d.EmbedModel, bm.d.Logger)
			}
		}()
	} else {
		close(initialScan)
	}
	_ = src

	return &BrainResources{
		ID:          brainID,
		Root:        root,
		Store:       store,
		Brain:       br,
		Search:      idx,
		Retriever:   retr,
		Memory:      mem,
		Knowledge:   kbase,
		initialScan: initialScan,
	}, nil
}

// buildSearchAndRetrieval constructs the search index and retriever
// for a brain. Returns nil values when SQLite cannot be initialised;
// callers handle the absence gracefully.
func buildSearchAndRetrieval(
	ctx context.Context,
	root string,
	store brain.Store,
	d *Daemon,
	log *slog.Logger,
) (*search.Index, *search.VectorIndex, retrieval.Source, retrieval.Retriever) {
	var embedder llm.Embedder
	var embedModel string
	if d != nil {
		embedder = d.Embedder
		embedModel = d.EmbedModel
	}
	dbPath := filepath.Join(root, ".search.db")
	db, err := search.OpenDB(dbPath)
	if err != nil {
		log.Warn("daemon: open search db", "root", root, "err", err)
		return nil, nil, nil, nil
	}
	idx, err := search.NewIndex(db, store)
	if err != nil {
		log.Warn("daemon: build index", "root", root, "err", err)
		_ = search.CloseDB(db)
		return nil, nil, nil, nil
	}
	// Vector index lives in the same SQLite file as the FTS tables so
	// WAL writer serialisation and backup stay trivial. Failures here
	// fall back to BM25-only retrieval rather than blocking startup.
	var vecIdx *search.VectorIndex
	if embedder != nil && embedModel != "" {
		vi, verr := search.NewVectorIndex(db)
		if verr != nil {
			log.Warn("daemon: build vector index", "root", root, "err", verr)
		} else {
			vecIdx = vi
		}
	}
	srcOpts := retrieval.IndexSourceOptions{Embedder: embedder}
	if vecIdx != nil {
		srcOpts.Vectors = vecIdx
		srcOpts.Model = embedModel
	}
	src, err := retrieval.NewIndexSource(idx, srcOpts)
	if err != nil {
		log.Warn("daemon: build index source", "root", root, "err", err)
		return idx, vecIdx, nil, nil
	}
	reranker := buildReranker(d, log)
	retr, err := retrieval.New(retrieval.Config{Source: src, Embedder: embedder, Reranker: reranker})
	if err != nil {
		log.Warn("daemon: build retriever", "root", root, "err", err)
		return idx, vecIdx, src, nil
	}
	return idx, vecIdx, src, retr
}

// buildReranker wires an optional cross-encoder rerank pass based on
// environment configuration. Returns nil when the operator has not
// selected a reranker; the retrieval pipeline treats a nil reranker as
// "rerank disabled" and records RerankSkipReason="no_reranker" on the
// trace.
//
// Recognised JB_RERANK_PROVIDER values:
//   - "llm": LLMReranker backed by the daemon's configured LLM
//     provider. Respects JB_RERANK_MODEL (forwarded to the provider).
//   - "http": HTTPReranker pointing at JB_RERANK_URL. Optional
//     JB_RERANK_API_KEY forwards a bearer token; JB_RERANK_MODEL is
//     sent in the request body (defaults to bge-reranker-v2-m3).
//
// Any other value (or an empty variable) leaves the reranker nil.
func buildReranker(d *Daemon, log *slog.Logger) retrieval.Reranker {
	provider := strings.ToLower(strings.TrimSpace(os.Getenv("JB_RERANK_PROVIDER")))
	if provider == "" {
		return nil
	}
	switch provider {
	case "llm":
		if d == nil || d.LLM == nil {
			log.Warn("daemon: rerank provider=llm but no llm provider configured")
			return nil
		}
		model := strings.TrimSpace(os.Getenv("JB_RERANK_MODEL"))
		return retrieval.NewLLMReranker(d.LLM, model)
	case "http":
		endpoint := strings.TrimSpace(os.Getenv("JB_RERANK_URL"))
		if endpoint == "" {
			log.Warn("daemon: rerank provider=http but JB_RERANK_URL is empty")
			return nil
		}
		rr, err := retrieval.NewHTTPReranker(retrieval.HTTPRerankerConfig{
			Endpoint: endpoint,
			APIKey:   strings.TrimSpace(os.Getenv("JB_RERANK_API_KEY")),
			Model:    strings.TrimSpace(os.Getenv("JB_RERANK_MODEL")),
			Logger:   log,
		})
		if err != nil {
			log.Warn("daemon: build http reranker", "err", err)
			return nil
		}
		return rr
	default:
		log.Warn("daemon: unknown rerank provider", "provider", provider)
		return nil
	}
}
