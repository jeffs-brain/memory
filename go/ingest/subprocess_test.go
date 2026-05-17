// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestRunSubprocess_Echo(t *testing.T) {
	t.Parallel()
	result, err := RunSubprocess(context.Background(), "echo", []string{"hello"}, nil, SubprocessOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.ExitCode != 0 {
		t.Fatalf("expected exit code 0, got %d", result.ExitCode)
	}
	if !strings.Contains(string(result.Stdout), "hello") {
		t.Fatalf("expected stdout to contain %q, got %q", "hello", string(result.Stdout))
	}
}

func TestRunSubprocess_NonZeroExit(t *testing.T) {
	t.Parallel()
	result, err := RunSubprocess(context.Background(), "sh", []string{"-c", "exit 42"}, nil, SubprocessOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.ExitCode != 42 {
		t.Fatalf("expected exit code 42, got %d", result.ExitCode)
	}
}

func TestRunSubprocess_Stdin(t *testing.T) {
	t.Parallel()
	result, err := RunSubprocess(context.Background(), "cat", nil, []byte("stdin data"), SubprocessOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if string(result.Stdout) != "stdin data" {
		t.Fatalf("expected stdout %q, got %q", "stdin data", string(result.Stdout))
	}
}

func TestRunSubprocess_Timeout(t *testing.T) {
	t.Parallel()
	_, err := RunSubprocess(context.Background(), "sleep", []string{"30"}, nil, SubprocessOptions{
		Timeout: 100 * time.Millisecond,
	})
	if err == nil {
		t.Fatal("expected timeout error, got nil")
	}
}

func TestRunSubprocess_NonexistentBinary(t *testing.T) {
	t.Parallel()
	_, err := RunSubprocess(context.Background(), "nonexistent-binary-xyz-abc", nil, nil, SubprocessOptions{})
	if err == nil {
		t.Fatal("expected error for nonexistent binary, got nil")
	}
}

func TestCheckBinaryAvailable_Echo(t *testing.T) {
	t.Parallel()
	ResetBinaryCache()
	defer ResetBinaryCache()

	available := CheckBinaryAvailable(context.Background(), "echo")
	if !available {
		t.Error("expected echo to be available")
	}
}

func TestCheckBinaryAvailable_Nonexistent(t *testing.T) {
	t.Parallel()
	ResetBinaryCache()
	defer ResetBinaryCache()

	available := CheckBinaryAvailable(context.Background(), "nonexistent-binary-xyz-abc")
	if available {
		t.Error("expected nonexistent binary to be unavailable")
	}
}

func TestCheckBinaryAvailable_Caching(t *testing.T) {
	t.Parallel()
	ResetBinaryCache()
	defer ResetBinaryCache()

	// First call should probe the binary.
	result1 := CheckBinaryAvailable(context.Background(), "echo")
	// Second call should hit the cache.
	result2 := CheckBinaryAvailable(context.Background(), "echo")

	if result1 != result2 {
		t.Errorf("cache inconsistency: first=%v, second=%v", result1, result2)
	}
}

func TestResetBinaryCache(t *testing.T) {
	t.Parallel()
	ResetBinaryCache()

	// Populate cache.
	CheckBinaryAvailable(context.Background(), "echo")

	// Reset should clear it.
	ResetBinaryCache()

	// Verify cache was cleared by checking internal state.
	globalBinaryCache.mu.RLock()
	count := len(globalBinaryCache.entries)
	globalBinaryCache.mu.RUnlock()

	if count != 0 {
		t.Errorf("expected empty cache after reset, got %d entries", count)
	}
}
