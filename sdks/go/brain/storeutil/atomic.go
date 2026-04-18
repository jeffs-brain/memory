// SPDX-License-Identifier: Apache-2.0

package storeutil

import (
	"fmt"
	"os"
	"path/filepath"
)

// AtomicWrite writes content to path with POSIX atomic semantics: a
// temp file in the same directory is written, fsync'd, then renamed
// over the target. Callers either see the full old content or the full
// new content, never a torn write. A crash at any point leaves either
// the old file or no file at all.
//
// The parent directory is fsync'd on a best-effort basis so the rename
// itself becomes durable on filesystems that support it. Any error from
// that step is swallowed; the caller has already observed a successful
// write.
func AtomicWrite(path string, content []byte, perm os.FileMode) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("mkdir %s: %w", dir, err)
	}

	tmp, err := os.CreateTemp(dir, ".brain-tmp-*")
	if err != nil {
		return fmt.Errorf("create temp in %s: %w", dir, err)
	}
	tmpName := tmp.Name()

	// Best-effort cleanup if anything below fails.
	defer func() {
		if _, statErr := os.Stat(tmpName); statErr == nil {
			_ = os.Remove(tmpName)
		}
	}()

	if _, err := tmp.Write(content); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("write temp: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("fsync temp: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close temp: %w", err)
	}
	if err := os.Chmod(tmpName, perm); err != nil {
		return fmt.Errorf("chmod temp: %w", err)
	}
	if err := os.Rename(tmpName, path); err != nil {
		return fmt.Errorf("rename temp to %s: %w", path, err)
	}

	if d, err := os.Open(dir); err == nil {
		_ = d.Sync()
		_ = d.Close()
	}
	return nil
}
