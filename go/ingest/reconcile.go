// SPDX-License-Identifier: Apache-2.0

// Package ingest provides safety scanning and self-healing reconciliation
// for the ingestion pipeline.
package ingest

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/search"
)

// ReconcileReport summarises the outcome of a single reconciliation pass.
type ReconcileReport struct {
	MissingReindexed int
	OrphanedDeleted  int
	TotalDocuments   int
	TotalIndexed     int
	DriftDetected    bool
	Errors           int
	Elapsed          time.Duration
}

// ReconcileConfig configures the reconciler.
type ReconcileConfig struct {
	// Store is the brain store that holds raw documents.
	Store brain.Store
	// Index is the search index that holds FTS entries.
	Index *search.Index
	// Interval is the period between periodic reconciliation runs.
	// Defaults to 5 minutes when zero.
	Interval time.Duration
	// MaxRepairs caps the number of repair operations per run (circuit
	// breaker). Zero means 1000.
	MaxRepairs int
	// MaxScanTime caps the wall-clock time of a single reconciliation
	// pass. Zero means 5 minutes.
	MaxScanTime time.Duration
	// Logger receives operational messages. Uses slog.Default() when nil.
	Logger *slog.Logger
	// ReindexFn is called for each document that needs re-indexing. In
	// production this triggers search.Index.Update(); tests inject a
	// stub. When nil, the reconciler calls Index.Update(ctx) which
	// rescans the entire store -- callers should supply a targeted
	// function for efficiency.
	ReindexFn func(ctx context.Context, path brain.Path) error
}

// Reconciler detects and repairs drift between the brain store and the
// search index. It is safe for concurrent use; at most one RunOnce
// executes at a time per instance.
type Reconciler struct {
	cfg    ReconcileConfig
	mu     sync.Mutex
	logger *slog.Logger
}

// NewReconciler constructs a reconciler. Returns an error when
// mandatory dependencies are missing.
func NewReconciler(cfg ReconcileConfig) (*Reconciler, error) {
	if cfg.Store == nil {
		return nil, fmt.Errorf("reconcile: Store is required")
	}
	if cfg.Index == nil {
		return nil, fmt.Errorf("reconcile: Index is required")
	}
	if cfg.Interval == 0 {
		cfg.Interval = 5 * time.Minute
	}
	if cfg.MaxRepairs == 0 {
		cfg.MaxRepairs = 1000
	}
	if cfg.MaxScanTime == 0 {
		cfg.MaxScanTime = 5 * time.Minute
	}
	logger := cfg.Logger
	if logger == nil {
		logger = slog.Default()
	}
	return &Reconciler{
		cfg:    cfg,
		logger: logger,
	}, nil
}

// RunOnce performs a single reconciliation pass. It acquires a lock to
// prevent concurrent runs; if the lock is already held the call returns
// immediately with a zero report and no error.
func (r *Reconciler) RunOnce(ctx context.Context) (ReconcileReport, error) {
	if !r.mu.TryLock() {
		return ReconcileReport{}, nil
	}
	defer r.mu.Unlock()

	start := time.Now()
	deadline := start.Add(r.cfg.MaxScanTime)
	scanCtx, cancel := context.WithDeadline(ctx, deadline)
	defer cancel()

	report := ReconcileReport{}

	// Step 1: enumerate documents in the store under raw/documents.
	storePaths, err := r.enumerateStore(scanCtx)
	if err != nil {
		return report, fmt.Errorf("reconcile: listing store documents: %w", err)
	}
	report.TotalDocuments = len(storePaths)

	// Step 2: enumerate paths in the search index.
	indexedPaths, err := r.cfg.Index.IndexedPaths()
	if err != nil {
		return report, fmt.Errorf("reconcile: listing indexed paths: %w", err)
	}
	report.TotalIndexed = len(indexedPaths)

	// Step 3: build lookup sets and compute diff.
	storeSet := make(map[string]struct{}, len(storePaths))
	for _, p := range storePaths {
		storeSet[string(p)] = struct{}{}
	}
	indexSet := make(map[string]struct{}, len(indexedPaths))
	for _, p := range indexedPaths {
		indexSet[p] = struct{}{}
	}

	// Documents in store but missing from index.
	var missing []brain.Path
	for _, p := range storePaths {
		if _, ok := indexSet[string(p)]; !ok {
			missing = append(missing, p)
		}
	}

	// Paths in index but not in store (orphaned).
	var orphaned []string
	for _, p := range indexedPaths {
		if _, ok := storeSet[p]; !ok {
			orphaned = append(orphaned, p)
		}
	}

	report.DriftDetected = len(missing) > 0 || len(orphaned) > 0

	// Step 4: repair missing documents (re-index).
	repairsRemaining := r.cfg.MaxRepairs
	for _, p := range missing {
		if scanCtx.Err() != nil {
			break
		}
		if repairsRemaining <= 0 {
			r.logger.Warn("reconcile: max repairs reached, stopping", "limit", r.cfg.MaxRepairs)
			break
		}
		if err := r.reindex(scanCtx, p); err != nil {
			r.logger.Warn("reconcile: failed to re-index document", "path", string(p), "error", err.Error())
			report.Errors++
		} else {
			report.MissingReindexed++
		}
		repairsRemaining--
	}

	// Step 5: remove orphaned index entries.
	for _, p := range orphaned {
		if scanCtx.Err() != nil {
			break
		}
		if repairsRemaining <= 0 {
			r.logger.Warn("reconcile: max repairs reached, stopping orphan removal", "limit", r.cfg.MaxRepairs)
			break
		}
		if err := r.cfg.Index.Remove(p); err != nil {
			r.logger.Warn("reconcile: failed to remove orphaned index entry", "path", p, "error", err.Error())
			report.Errors++
		} else {
			report.OrphanedDeleted++
		}
		repairsRemaining--
	}

	report.Elapsed = time.Since(start)
	if report.DriftDetected {
		r.logger.Info("reconcile: drift repaired",
			"missing_reindexed", report.MissingReindexed,
			"orphaned_deleted", report.OrphanedDeleted,
			"errors", report.Errors,
			"elapsed", report.Elapsed.String(),
		)
	}
	return report, nil
}

// Start begins periodic reconciliation. It blocks until ctx is
// cancelled or the context expires. Each tick calls RunOnce.
func (r *Reconciler) Start(ctx context.Context) {
	ticker := time.NewTicker(r.cfg.Interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if _, err := r.RunOnce(ctx); err != nil {
				r.logger.Warn("reconcile: periodic run failed", "error", err.Error())
			}
		}
	}
}

// enumerateStore lists all files under raw/documents in the brain store.
func (r *Reconciler) enumerateStore(ctx context.Context) ([]brain.Path, error) {
	prefix := brain.RawDocumentsPrefix()
	entries, err := r.cfg.Store.List(ctx, prefix, brain.ListOpts{
		Recursive:        true,
		IncludeGenerated: true,
	})
	if err != nil {
		return nil, err
	}
	paths := make([]brain.Path, 0, len(entries))
	for _, e := range entries {
		if e.IsDir {
			continue
		}
		if !strings.HasSuffix(string(e.Path), ".md") {
			continue
		}
		paths = append(paths, e.Path)
	}
	return paths, nil
}

// reindex triggers re-indexing for a single document path. Uses the
// configured ReindexFn when set, otherwise falls back to a full
// Index.Update().
func (r *Reconciler) reindex(ctx context.Context, p brain.Path) error {
	if r.cfg.ReindexFn != nil {
		return r.cfg.ReindexFn(ctx, p)
	}
	return r.cfg.Index.Update(ctx)
}
