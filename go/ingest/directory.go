// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"bufio"
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

const fileSizeLimit = 25 * 1024 * 1024 // 25 MiB

// knownTextExtensions lists file extensions treated as ingestable text.
var knownTextExtensions = map[string]bool{
	".md": true, ".markdown": true, ".txt": true, ".text": true,
	".pdf": true, ".json": true, ".html": true, ".htm": true,
	".csv": true, ".tsv": true, ".xml": true, ".yaml": true,
	".yml": true, ".toml": true, ".rst": true, ".adoc": true,
	".org": true, ".tex": true, ".rtf": true,
}

// EnumerateOptions configures directory file enumeration.
type EnumerateOptions struct {
	Directory string
	Glob      string
	Recursive bool
	MaxFiles  int
}

// EnumeratedFile describes a single file found during enumeration.
type EnumeratedFile struct {
	Path string
	Size int64
}

// EnumerateFiles walks a directory and returns files suitable for ingestion.
//
// It excludes hidden files (dot-prefix), does not follow symlinks, respects
// .gitignore patterns when present, applies glob filter if provided, enforces
// maxFiles limit, and skips files over 25 MiB.
func EnumerateFiles(ctx context.Context, opts EnumerateOptions) ([]EnumeratedFile, []string, error) {
	if opts.MaxFiles <= 0 {
		opts.MaxFiles = 100
	}

	gitignorePatterns := loadGitignore(opts.Directory)

	var files []EnumeratedFile
	var skipped []string
	limitReached := false

	walkFn := func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			skipped = append(skipped, fmt.Sprintf("%s: %s", path, err.Error()))
			return nil
		}

		if ctx.Err() != nil {
			return ctx.Err()
		}

		// Skip hidden entries
		if strings.HasPrefix(d.Name(), ".") {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		relPath, relErr := filepath.Rel(opts.Directory, path)
		if relErr != nil {
			relPath = path
		}

		// Check .gitignore
		if matchesGitignore(relPath, gitignorePatterns) {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		// Non-recursive: skip subdirectories
		if d.IsDir() {
			if !opts.Recursive && path != opts.Directory {
				return filepath.SkipDir
			}
			return nil
		}

		// Skip symlinks
		info, infoErr := d.Info()
		if infoErr != nil {
			skipped = append(skipped, fmt.Sprintf("%s: %s", relPath, infoErr.Error()))
			return nil
		}
		if info.Mode()&fs.ModeSymlink != 0 {
			return nil
		}

		// Max files check
		if len(files) >= opts.MaxFiles {
			if !limitReached {
				skipped = append(skipped, fmt.Sprintf("max files limit (%d) reached", opts.MaxFiles))
				limitReached = true
			}
			return nil
		}

		// Size check
		if info.Size() > fileSizeLimit {
			skipped = append(skipped, fmt.Sprintf("%s: exceeds 25 MiB limit (%d MiB)", relPath, info.Size()/(1024*1024)))
			return nil
		}

		// Glob filter
		if opts.Glob != "" && !matchGlob(relPath, opts.Glob) {
			return nil
		}

		// Extension filter
		ext := strings.ToLower(filepath.Ext(d.Name()))
		if ext != "" && !knownTextExtensions[ext] {
			return nil
		}

		files = append(files, EnumeratedFile{Path: path, Size: info.Size()})
		return nil
	}

	if err := filepath.WalkDir(opts.Directory, walkFn); err != nil {
		return files, skipped, err
	}

	return files, skipped, nil
}

// loadGitignore reads .gitignore from the directory root if present.
func loadGitignore(dir string) []string {
	f, err := os.Open(filepath.Join(dir, ".gitignore"))
	if err != nil {
		return nil
	}
	defer f.Close()

	var patterns []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, "!") {
			continue
		}
		patterns = append(patterns, line)
	}
	return patterns
}

// matchesGitignore checks if a relative path matches any gitignore pattern.
func matchesGitignore(relPath string, patterns []string) bool {
	segments := strings.Split(relPath, string(filepath.Separator))
	for _, pattern := range patterns {
		cleaned := strings.TrimRight(pattern, "/")
		// Segment match (e.g. "node_modules")
		for _, seg := range segments {
			if seg == cleaned {
				return true
			}
		}
		// Wildcard prefix (e.g. "*.log")
		if strings.HasPrefix(cleaned, "*") && strings.HasSuffix(relPath, cleaned[1:]) {
			return true
		}
		// Root-relative (e.g. "/dist")
		if strings.HasPrefix(cleaned, "/") && strings.HasPrefix(relPath, cleaned[1:]) {
			return true
		}
		// Direct prefix
		if strings.HasPrefix(relPath, cleaned) {
			return true
		}
	}
	return false
}

// matchGlob performs simple glob matching on a relative path.
func matchGlob(relPath string, pattern string) bool {
	// Handle **/ prefix
	normalised := strings.TrimPrefix(pattern, "**/")
	// Try matching against both full path and basename
	matched, _ := filepath.Match(normalised, relPath)
	if matched {
		return true
	}
	matched, _ = filepath.Match(normalised, filepath.Base(relPath))
	return matched
}
