// SPDX-License-Identifier: Apache-2.0

package git

import (
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
	"github.com/jeffs-brain/memory/go/brain/storeutil"
)

// workingTree is a minimal filesystem helper scoped to the git package.
// It provides the read/write surface the gitstore needs over a brain
// root directory. Layout, listing and atomic-write rules are delegated
// to [storeutil] so the fs and git backends share one canonical layout.
type workingTree struct {
	root string
}

// newWorkingTree returns a helper rooted at an absolute path. The
// directory is created on demand.
func newWorkingTree(root string) (*workingTree, error) {
	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("gitstore: resolve root: %w", err)
	}
	if err := os.MkdirAll(abs, 0o755); err != nil {
		return nil, fmt.Errorf("gitstore: create root %s: %w", abs, err)
	}
	return &workingTree{root: abs}, nil
}

// Root returns the absolute filesystem path of the working tree root.
func (w *workingTree) Root() string { return w.root }

// resolve maps a logical [brain.Path] to its absolute on-disk path via
// [storeutil.Resolve].
func (w *workingTree) resolve(p brain.Path) (string, error) {
	return storeutil.Resolve(w.root, p)
}

// localPath returns the absolute on-disk path for p or ("", false) if
// the path is invalid.
func (w *workingTree) localPath(p brain.Path) (string, bool) {
	abs, err := w.resolve(p)
	if err != nil {
		return "", false
	}
	return abs, true
}

// wrapNotFound rewraps os.ErrNotExist as [brain.ErrNotFound] with path
// context so callers can use [errors.Is].
func wrapNotFound(p brain.Path, op string, err error) error {
	if errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("gitstore: %s %s: %w", op, p, brain.ErrNotFound)
	}
	return err
}

// ---- read side ----

func (w *workingTree) read(p brain.Path) ([]byte, error) {
	abs, err := w.resolve(p)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		return nil, wrapNotFound(p, "read", err)
	}
	return data, nil
}

func (w *workingTree) exists(p brain.Path) (bool, error) {
	abs, err := w.resolve(p)
	if err != nil {
		return false, err
	}
	_, err = os.Stat(abs)
	if errors.Is(err, fs.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return true, nil
}

func (w *workingTree) stat(p brain.Path) (brain.FileInfo, error) {
	abs, err := w.resolve(p)
	if err != nil {
		return brain.FileInfo{}, err
	}
	info, err := os.Stat(abs)
	if err != nil {
		return brain.FileInfo{}, wrapNotFound(p, "stat", err)
	}
	return brain.FileInfo{
		Path:    p,
		Size:    info.Size(),
		ModTime: info.ModTime(),
		IsDir:   info.IsDir(),
	}, nil
}

func (w *workingTree) list(dir brain.Path, opts brain.ListOpts) ([]brain.FileInfo, error) {
	absDir := w.root
	if dir != "" {
		resolved, err := w.resolve(dir)
		if err != nil {
			return nil, err
		}
		absDir = resolved
	}
	entries, err := storeutil.List(w.root, absDir, dir, opts)
	if err != nil {
		return nil, fmt.Errorf("gitstore: %w", err)
	}
	return entries, nil
}

// ---- write side ----

func (w *workingTree) write(ctx context.Context, p brain.Path, content []byte) (created bool, err error) {
	_ = ctx
	abs, err := w.resolve(p)
	if err != nil {
		return false, err
	}
	_, statErr := os.Stat(abs)
	existedBefore := statErr == nil
	if err := storeutil.AtomicWrite(abs, content, 0o644); err != nil {
		return false, err
	}
	return !existedBefore, nil
}

func (w *workingTree) append(ctx context.Context, p brain.Path, content []byte) (created bool, err error) {
	_ = ctx
	abs, err := w.resolve(p)
	if err != nil {
		return false, err
	}
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		return false, err
	}
	_, statErr := os.Stat(abs)
	existedBefore := statErr == nil
	f, err := os.OpenFile(abs, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return false, err
	}
	if _, err := f.Write(content); err != nil {
		_ = f.Close()
		return false, err
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		return false, err
	}
	if err := f.Close(); err != nil {
		return false, err
	}
	return !existedBefore, nil
}

func (w *workingTree) delete(ctx context.Context, p brain.Path) error {
	_ = ctx
	abs, err := w.resolve(p)
	if err != nil {
		return err
	}
	if err := os.Remove(abs); err != nil {
		return wrapNotFound(p, "delete", err)
	}
	return nil
}

func (w *workingTree) rename(ctx context.Context, src, dst brain.Path) error {
	_ = ctx
	srcAbs, err := w.resolve(src)
	if err != nil {
		return err
	}
	dstAbs, err := w.resolve(dst)
	if err != nil {
		return err
	}
	if _, err := os.Stat(srcAbs); err != nil {
		return wrapNotFound(src, "rename", err)
	}
	if err := os.MkdirAll(filepath.Dir(dstAbs), 0o755); err != nil {
		return err
	}
	if err := os.Rename(srcAbs, dstAbs); err != nil {
		return err
	}
	return nil
}

// now is captured so tests can override it. Not used today but defined
// here so future refactors keep a single clock source.
var now = func() time.Time { return time.Now() }
