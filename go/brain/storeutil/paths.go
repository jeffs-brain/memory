// SPDX-License-Identifier: Apache-2.0

// Package storeutil provides filesystem layout helpers shared by the
// [fs] and [git] stores. Both stores materialise a [brain.Store] on top
// of the same on-disk layout so the mapping between logical paths and
// on-disk relative paths lives here, not in each backend.
//
// The helpers are intentionally pure functions over [brain.Path] values;
// they do no I/O and hold no state.
package storeutil

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/jeffs-brain/memory/go/brain"
)

// Relative maps a logical [brain.Path] to its on-disk relative path
// under the store root. The layout matches the historical fsstore rules
// so fs and git backends are interoperable on the same directory.
//
//   - memory/global/X       -> memory/X
//   - memory/project/<s>/X  -> projects/<s>/memory/X
//   - wiki/X                -> wiki/X
//   - raw/X                 -> raw/X
//   - kb-schema.md          -> kb-schema.md
//   - schema-version.yaml   -> schema-version.yaml
func Relative(p brain.Path) (string, error) {
	lp := string(p)
	switch {
	case lp == "kb-schema.md", lp == "schema-version.yaml":
		return lp, nil
	case strings.HasPrefix(lp, "memory/global/"):
		return "memory/" + strings.TrimPrefix(lp, "memory/global/"), nil
	case lp == "memory/global":
		return "memory", nil
	case strings.HasPrefix(lp, "memory/project/"):
		rest := strings.TrimPrefix(lp, "memory/project/")
		slash := strings.IndexByte(rest, '/')
		if slash == -1 {
			return "projects/" + rest + "/memory", nil
		}
		slug := rest[:slash]
		tail := rest[slash+1:]
		return "projects/" + slug + "/memory/" + tail, nil
	case lp == "memory/project":
		return "projects", nil
	case strings.HasPrefix(lp, "wiki/") || lp == "wiki":
		return lp, nil
	case strings.HasPrefix(lp, "raw/") || lp == "raw":
		return lp, nil
	default:
		return "", fmt.Errorf("%w: unknown root in %q", brain.ErrInvalidPath, lp)
	}
}

// LogicalFromRel is the inverse of [Relative]. Given a path relative to
// the on-disk root, it returns the canonical logical path.
func LogicalFromRel(rel string) brain.Path {
	rel = filepath.ToSlash(rel)
	switch {
	case rel == "kb-schema.md", rel == "schema-version.yaml":
		return brain.Path(rel)
	case rel == "memory":
		return "memory/global"
	case strings.HasPrefix(rel, "memory/"):
		return brain.Path("memory/global/" + strings.TrimPrefix(rel, "memory/"))
	case rel == "projects":
		return "memory/project"
	case strings.HasPrefix(rel, "projects/"):
		rest := strings.TrimPrefix(rel, "projects/")
		slash := strings.IndexByte(rest, '/')
		if slash == -1 {
			return brain.Path("memory/project/" + rest)
		}
		slug := rest[:slash]
		tail := rest[slash+1:]
		if tail == "memory" {
			return brain.Path("memory/project/" + slug)
		}
		if strings.HasPrefix(tail, "memory/") {
			return brain.Path("memory/project/" + slug + "/" + strings.TrimPrefix(tail, "memory/"))
		}
		return brain.Path("memory/project/" + slug + "/" + tail)
	case rel == "wiki" || strings.HasPrefix(rel, "wiki/"):
		return brain.Path(rel)
	case rel == "raw" || strings.HasPrefix(rel, "raw/"):
		return brain.Path(rel)
	default:
		return brain.Path(rel)
	}
}

// Resolve turns a logical path into an absolute filesystem path under
// root. It validates the path shape first so callers never hit
// [os.PathError] on a malformed logical path.
func Resolve(root string, p brain.Path) (string, error) {
	if err := brain.ValidatePath(p); err != nil {
		return "", err
	}
	rel, err := Relative(p)
	if err != nil {
		return "", err
	}
	return filepath.Join(root, filepath.FromSlash(rel)), nil
}

// ShouldSkipDir reports whether a directory encountered during a List
// walk should be skipped entirely. Hides the atomic-write staging areas
// and the git metadata directory so callers never see backend internals
// leak into their result set.
func ShouldSkipDir(name string) bool {
	if name == ".git" {
		return true
	}
	if strings.HasPrefix(name, ".brain-staging") {
		return true
	}
	return false
}

// ShouldSkipFile reports whether a file encountered during a List walk
// should be skipped. Mirrors [ShouldSkipDir] for the file case (a .git
// file handle for a worktree, or a leftover atomic-write temp).
func ShouldSkipFile(name string) bool {
	if name == ".git" {
		return true
	}
	if strings.HasPrefix(name, ".brain-tmp-") {
		return true
	}
	return false
}
