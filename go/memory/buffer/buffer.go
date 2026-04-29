// SPDX-License-Identifier: Apache-2.0

// Package buffer implements the L0 rolling observation buffer used by
// the memory stages. Observations are appended per turn and compacted
// when the rendered buffer exceeds a configured token budget.
package buffer

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
)

// ScopeKind is the buffer scope type.
type ScopeKind string

const (
	ScopeGlobal  ScopeKind = "global"
	ScopeProject ScopeKind = "project"
)

// Scope identifies which buffer to read/write.
type Scope struct {
	Kind ScopeKind
	Slug string
}

// Observation is a single turn's contribution to the buffer.
type Observation struct {
	At       time.Time
	Intent   string
	Entities []string
	Outcome  string
	Summary  string
}

// Config controls buffer behaviour.
type Config struct {
	TokenBudget       int
	CompactThreshold  int
	KeepRecentPercent int
	MaxObservationLen int
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() Config {
	return Config{
		TokenBudget:       8192,
		CompactThreshold:  100,
		KeepRecentPercent: 50,
		MaxObservationLen: 160,
	}
}

// Buffer is the L0 always-in-context rolling observation buffer.
type Buffer struct {
	store brain.Store
	scope Scope
	cfg   Config
	mu    sync.Mutex
}

// New creates a Buffer backed by the given store.
func New(store brain.Store, scope Scope, cfg Config) *Buffer {
	if cfg.TokenBudget <= 0 {
		cfg.TokenBudget = 8192
	}
	if cfg.CompactThreshold <= 0 {
		cfg.CompactThreshold = 100
	}
	if cfg.KeepRecentPercent <= 0 {
		cfg.KeepRecentPercent = 50
	}
	if cfg.MaxObservationLen <= 0 {
		cfg.MaxObservationLen = 160
	}
	return &Buffer{store: store, scope: scope, cfg: cfg}
}

// path returns the brain path for this buffer's scope.
func (b *Buffer) path() brain.Path {
	switch b.scope.Kind {
	case ScopeProject:
		return brain.MemoryBufferProject(b.scope.Slug)
	default:
		return brain.MemoryBufferGlobal()
	}
}

// Render returns the current buffer content as a string.
func (b *Buffer) Render(ctx context.Context) (string, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	content, err := b.store.Read(ctx, b.path())
	if err != nil {
		if errors.Is(err, brain.ErrNotFound) {
			return "", nil
		}
		return "", fmt.Errorf("buffer render: %w", err)
	}
	return string(content), nil
}

// Append adds an observation to the buffer.
func (b *Buffer) Append(ctx context.Context, obs Observation) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	summary := obs.Summary
	if len([]rune(summary)) > b.cfg.MaxObservationLen {
		summary = string([]rune(summary)[:b.cfg.MaxObservationLen])
	}

	line := formatObservation(obs.At, obs.Intent, obs.Outcome, summary, obs.Entities)

	return b.store.Append(ctx, b.path(), []byte(line+"\n"))
}

// TokenCount returns an approximate token count of the current
// buffer.
func (b *Buffer) TokenCount(ctx context.Context) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	content, err := b.store.Read(ctx, b.path())
	if err != nil {
		if errors.Is(err, brain.ErrNotFound) {
			return 0, nil
		}
		return 0, err
	}
	return len(content) / 4, nil
}

// NeedsCompaction returns true if the buffer has exceeded the token
// budget threshold.
func (b *Buffer) NeedsCompaction(ctx context.Context) (bool, error) {
	tokens, err := b.TokenCount(ctx)
	if err != nil {
		return false, err
	}
	threshold := (b.cfg.TokenBudget * b.cfg.CompactThreshold) / 100
	return tokens >= threshold, nil
}

// Compact removes older observations from the buffer.
func (b *Buffer) Compact(ctx context.Context) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	content, err := b.store.Read(ctx, b.path())
	if err != nil {
		if errors.Is(err, brain.ErrNotFound) {
			return 0, nil
		}
		return 0, err
	}

	lines := strings.Split(strings.TrimSpace(string(content)), "\n")
	if len(lines) <= 1 {
		return 0, nil
	}

	keepCount := max((len(lines)*b.cfg.KeepRecentPercent)/100, 1)
	if keepCount >= len(lines) {
		return 0, nil
	}

	removed := len(lines) - keepCount
	kept := lines[len(lines)-keepCount:]

	newContent := strings.Join(kept, "\n") + "\n"
	if err := b.store.Write(ctx, b.path(), []byte(newContent)); err != nil {
		return 0, fmt.Errorf("buffer compact: %w", err)
	}

	return removed, nil
}

func formatObservation(at time.Time, intent, outcome, summary string, entities []string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "- [%s]", at.Format("15:04:05"))
	if intent != "" {
		fmt.Fprintf(&b, " (%s)", intent)
	}
	if outcome != "" && outcome != "ok" {
		fmt.Fprintf(&b, " [%s]", outcome)
	}
	b.WriteString(" ")
	b.WriteString(summary)
	if len(entities) > 0 {
		fmt.Fprintf(&b, " {%s}", strings.Join(entities, ", "))
	}
	return b.String()
}
