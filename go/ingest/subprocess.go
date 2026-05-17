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

// RunSubprocess executes a binary with the given arguments and optional
// stdin input. Timeout is enforced via context cancellation. Arguments
// are passed as an array — no shell expansion occurs.
func RunSubprocess(ctx context.Context, binary string, args []string, stdin []byte, opts SubprocessOptions) (SubprocessResult, error) {
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
// expire after 5 minutes to avoid repeatedly probing the filesystem
// for binaries that are unlikely to appear or disappear during a
// single process lifetime.
type binaryAvailabilityCache struct {
	mu      sync.RWMutex
	entries map[string]binaryCacheEntry
}

type binaryCacheEntry struct {
	available bool
	expiresAt time.Time
}

const binaryCacheTTL = 5 * time.Minute

var globalBinaryCache = &binaryAvailabilityCache{
	entries: make(map[string]binaryCacheEntry),
}

// CheckBinaryAvailable reports whether the named binary is present on
// the system PATH. Results are cached for 5 minutes.
func CheckBinaryAvailable(ctx context.Context, name string) bool {
	globalBinaryCache.mu.RLock()
	entry, ok := globalBinaryCache.entries[name]
	globalBinaryCache.mu.RUnlock()

	if ok && time.Now().Before(entry.expiresAt) {
		return entry.available
	}

	available := checkBinaryOnPath(ctx, name)

	globalBinaryCache.mu.Lock()
	globalBinaryCache.entries[name] = binaryCacheEntry{
		available: available,
		expiresAt: time.Now().Add(binaryCacheTTL),
	}
	globalBinaryCache.mu.Unlock()

	return available
}

// ResetBinaryCache clears the binary availability cache. Intended for
// testing only.
func ResetBinaryCache() {
	globalBinaryCache.mu.Lock()
	globalBinaryCache.entries = make(map[string]binaryCacheEntry)
	globalBinaryCache.mu.Unlock()
}

// checkBinaryOnPath checks if the binary exists on PATH using
// exec.LookPath.
func checkBinaryOnPath(_ context.Context, name string) bool {
	_, err := exec.LookPath(name)
	return err == nil
}
