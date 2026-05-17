// SPDX-License-Identifier: Apache-2.0

package extract

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"sync"
	"time"
)

// SubprocessResult holds the output from a subprocess invocation.
type SubprocessResult struct {
	Stdout   []byte
	Stderr   string
	ExitCode int
}

// RunSubprocess executes a binary with the given arguments and optional
// stdin data. The context controls cancellation and the timeout is
// enforced as a hard wall-clock limit on top of the context deadline.
func RunSubprocess(ctx context.Context, binary string, args []string, stdin []byte, timeout time.Duration) (SubprocessResult, error) {
	if timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}

	cmd := exec.CommandContext(ctx, binary, args...)

	if len(stdin) > 0 {
		cmd.Stdin = bytes.NewReader(stdin)
	}

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	exitCode := 0
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		} else {
			return SubprocessResult{}, fmt.Errorf("extract: running %s: %w", binary, err)
		}
	}

	return SubprocessResult{
		Stdout:   stdout.Bytes(),
		Stderr:   stderr.String(),
		ExitCode: exitCode,
	}, nil
}

// binaryCache caches the result of binary availability checks for a
// short period to avoid repeated filesystem lookups.
var binaryCache = struct {
	mu      sync.Mutex
	entries map[string]binaryCacheEntry
}{
	entries: make(map[string]binaryCacheEntry),
}

type binaryCacheEntry struct {
	available bool
	checkedAt time.Time
}

const binaryCacheTTL = 5 * time.Minute

// CheckBinaryAvailable returns true when the named binary can be found
// on PATH. Results are cached for five minutes.
func CheckBinaryAvailable(name string) bool {
	binaryCache.mu.Lock()
	defer binaryCache.mu.Unlock()

	if entry, ok := binaryCache.entries[name]; ok {
		if time.Since(entry.checkedAt) < binaryCacheTTL {
			return entry.available
		}
	}

	_, err := exec.LookPath(name)
	available := err == nil

	binaryCache.entries[name] = binaryCacheEntry{
		available: available,
		checkedAt: time.Now(),
	}

	return available
}

// ResetBinaryCache clears the binary availability cache. Intended for
// use in tests only.
func ResetBinaryCache() {
	binaryCache.mu.Lock()
	defer binaryCache.mu.Unlock()
	binaryCache.entries = make(map[string]binaryCacheEntry)
}
