// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"sync"
	"time"
)

// SubprocessResult holds the output of a subprocess invocation.
type SubprocessResult struct {
	Stdout   []byte
	Stderr   string
	ExitCode int
}

// SubprocessOptions configures subprocess invocation behaviour.
type SubprocessOptions struct {
	Timeout time.Duration
	Cwd     string
	Env     []string
}

// DefaultSubprocessTimeout is used when no timeout is specified.
const DefaultSubprocessTimeout = 60 * time.Second

// DefaultMaxConcurrentSubprocesses is the default cap for simultaneous
// subprocess invocations.
const DefaultMaxConcurrentSubprocesses = 8

// binaryCacheTTL is how long binary availability results are cached.
const binaryCacheTTL = 5 * time.Minute

// SubprocessRunner manages subprocess invocations with bounded
// concurrency and binary availability caching. Each instance owns
// its own semaphore and cache, eliminating module-global state.
type SubprocessRunner struct {
	semaphore chan struct{}
	cache     binaryAvailabilityCache
}

// NewSubprocessRunner creates a runner with the given concurrency
// limit. When maxConcurrency is zero, DefaultMaxConcurrentSubprocesses
// is used.
func NewSubprocessRunner(maxConcurrency int) *SubprocessRunner {
	if maxConcurrency <= 0 {
		maxConcurrency = DefaultMaxConcurrentSubprocesses
	}
	return &SubprocessRunner{
		semaphore: make(chan struct{}, maxConcurrency),
		cache: binaryAvailabilityCache{
			entries: make(map[string]binaryCacheEntry),
		},
	}
}

// Run executes a binary with the given arguments and optional stdin
// input. Timeout is enforced via context cancellation. Arguments are
// passed as an array -- no shell expansion occurs. Concurrency is
// limited by the runner's semaphore.
func (r *SubprocessRunner) Run(ctx context.Context, binary string, args []string, stdin []byte, opts SubprocessOptions) (SubprocessResult, error) {
	// Acquire semaphore slot, respecting context cancellation.
	select {
	case r.semaphore <- struct{}{}:
		defer func() { <-r.semaphore }()
	case <-ctx.Done():
		return SubprocessResult{ExitCode: -1}, fmt.Errorf("ingest: subprocess %q: context cancelled while waiting for concurrency slot: %w", binary, ctx.Err())
	}

	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = DefaultSubprocessTimeout
	}

	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, binary, args...)

	if opts.Cwd != "" {
		cmd.Dir = opts.Cwd
	}
	if len(opts.Env) > 0 {
		cmd.Env = opts.Env
	}

	if len(stdin) > 0 {
		cmd.Stdin = bytes.NewReader(stdin)
	}

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()

	exitCode := 0
	if err != nil {
		// Check for context cancellation (timeout) first. When the
		// context deadline is exceeded, the process is killed and we
		// want to surface a clear timeout error rather than a generic
		// exit error.
		if ctx.Err() != nil {
			return SubprocessResult{
				Stderr:   stderr.String(),
				ExitCode: -1,
			}, fmt.Errorf("ingest: subprocess %q timed out after %v: %w", binary, timeout, ctx.Err())
		}

		if exitErr, ok := err.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		} else {
			return SubprocessResult{
				Stderr:   stderr.String(),
				ExitCode: -1,
			}, fmt.Errorf("ingest: subprocess %q: %w", binary, err)
		}
	}

	return SubprocessResult{
		Stdout:   stdout.Bytes(),
		Stderr:   stderr.String(),
		ExitCode: exitCode,
	}, nil
}

// binaryAvailabilityCache caches binary availability lookups. Entries
// expire after binaryCacheTTL to avoid repeatedly probing the
// filesystem.
type binaryAvailabilityCache struct {
	mu      sync.RWMutex
	entries map[string]binaryCacheEntry
}

type binaryCacheEntry struct {
	available bool
	expiresAt time.Time
}

// CheckBinaryAvailable reports whether the named binary is present on
// the system PATH. Results are cached per runner instance.
func (r *SubprocessRunner) CheckBinaryAvailable(ctx context.Context, name string) bool {
	r.cache.mu.RLock()
	entry, ok := r.cache.entries[name]
	r.cache.mu.RUnlock()

	if ok && time.Now().Before(entry.expiresAt) {
		return entry.available
	}

	available := checkBinaryOnPath(ctx, name)

	r.cache.mu.Lock()
	r.cache.entries[name] = binaryCacheEntry{
		available: available,
		expiresAt: time.Now().Add(binaryCacheTTL),
	}
	r.cache.mu.Unlock()

	return available
}

// ResetBinaryCache clears the binary availability cache. Intended for
// testing only.
func (r *SubprocessRunner) ResetBinaryCache() {
	r.cache.mu.Lock()
	r.cache.entries = make(map[string]binaryCacheEntry)
	r.cache.mu.Unlock()
}

// defaultRunner is the default SubprocessRunner for backwards
// compatibility with callers that use the package-level functions.
var defaultRunner = NewSubprocessRunner(DefaultMaxConcurrentSubprocesses)

// RunSubprocess is a convenience function that delegates to the
// default SubprocessRunner. Prefer creating a dedicated runner via
// NewSubprocessRunner for production use.
func RunSubprocess(ctx context.Context, binary string, args []string, stdin []byte, opts SubprocessOptions) (SubprocessResult, error) {
	return defaultRunner.Run(ctx, binary, args, stdin, opts)
}

// CheckBinaryAvailable is a convenience function that delegates to
// the default SubprocessRunner.
func CheckBinaryAvailable(ctx context.Context, name string) bool {
	return defaultRunner.CheckBinaryAvailable(ctx, name)
}

// ResetBinaryCache is a convenience function that delegates to the
// default SubprocessRunner.
func ResetBinaryCache() {
	defaultRunner.ResetBinaryCache()
}

// checkBinaryOnPath checks if the binary exists on PATH using
// exec.LookPath.
func checkBinaryOnPath(_ context.Context, name string) bool {
	_, err := exec.LookPath(name)
	return err == nil
}
