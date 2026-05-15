// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func writeTestFile(t *testing.T, path, content string) {
	t.Helper()
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
}

func TestEnumerateFiles_FlatDirectory(t *testing.T) {
	dir := t.TempDir()
	writeTestFile(t, filepath.Join(dir, "a.md"), "# A")
	writeTestFile(t, filepath.Join(dir, "b.md"), "# B")
	writeTestFile(t, filepath.Join(dir, "c.md"), "# C")

	files, skipped, err := EnumerateFiles(context.Background(), EnumerateOptions{
		Directory: dir,
		Recursive: true,
		MaxFiles:  100,
	})
	if err != nil {
		t.Fatalf("enumerate: %v", err)
	}
	if len(files) != 3 {
		t.Errorf("expected 3 files, got %d", len(files))
	}
	if len(skipped) != 0 {
		t.Errorf("expected 0 skipped, got %d: %v", len(skipped), skipped)
	}
}

func TestEnumerateFiles_GlobFilter(t *testing.T) {
	dir := t.TempDir()
	writeTestFile(t, filepath.Join(dir, "a.md"), "markdown")
	writeTestFile(t, filepath.Join(dir, "b.txt"), "text")

	files, _, err := EnumerateFiles(context.Background(), EnumerateOptions{
		Directory: dir,
		Glob:      "*.md",
		Recursive: true,
		MaxFiles:  100,
	})
	if err != nil {
		t.Fatalf("enumerate: %v", err)
	}
	if len(files) != 1 {
		t.Errorf("expected 1 file, got %d", len(files))
	}
}

func TestEnumerateFiles_NonRecursive(t *testing.T) {
	dir := t.TempDir()
	writeTestFile(t, filepath.Join(dir, "top.md"), "top")
	writeTestFile(t, filepath.Join(dir, "sub", "nested.md"), "nested")

	files, _, err := EnumerateFiles(context.Background(), EnumerateOptions{
		Directory: dir,
		Recursive: false,
		MaxFiles:  100,
	})
	if err != nil {
		t.Fatalf("enumerate: %v", err)
	}
	if len(files) != 1 {
		t.Errorf("expected 1 file, got %d", len(files))
	}
}

func TestEnumerateFiles_MaxFilesLimit(t *testing.T) {
	dir := t.TempDir()
	for i := 0; i < 5; i++ {
		writeTestFile(t, filepath.Join(dir, filepath.Base(t.Name())+string(rune('a'+i))+".md"), "content")
	}

	files, skipped, err := EnumerateFiles(context.Background(), EnumerateOptions{
		Directory: dir,
		Recursive: true,
		MaxFiles:  2,
	})
	if err != nil {
		t.Fatalf("enumerate: %v", err)
	}
	if len(files) != 2 {
		t.Errorf("expected 2 files, got %d", len(files))
	}
	found := false
	for _, s := range skipped {
		if s == "max files limit (2) reached" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected max files limit skip reason, got %v", skipped)
	}
}

func TestEnumerateFiles_HiddenFilesExcluded(t *testing.T) {
	dir := t.TempDir()
	writeTestFile(t, filepath.Join(dir, ".hidden.md"), "hidden")
	writeTestFile(t, filepath.Join(dir, "visible.md"), "visible")

	files, _, err := EnumerateFiles(context.Background(), EnumerateOptions{
		Directory: dir,
		Recursive: true,
		MaxFiles:  100,
	})
	if err != nil {
		t.Fatalf("enumerate: %v", err)
	}
	if len(files) != 1 {
		t.Errorf("expected 1 file, got %d", len(files))
	}
}

func TestEnumerateFiles_GitignoreRespected(t *testing.T) {
	dir := t.TempDir()
	writeTestFile(t, filepath.Join(dir, ".gitignore"), "node_modules\n*.log\n")
	writeTestFile(t, filepath.Join(dir, "node_modules", "pkg.txt"), "pkg")
	writeTestFile(t, filepath.Join(dir, "debug.log"), "logs")
	writeTestFile(t, filepath.Join(dir, "readme.md"), "readme")

	files, _, err := EnumerateFiles(context.Background(), EnumerateOptions{
		Directory: dir,
		Recursive: true,
		MaxFiles:  100,
	})
	if err != nil {
		t.Fatalf("enumerate: %v", err)
	}
	if len(files) != 1 {
		t.Errorf("expected 1 file (readme.md), got %d", len(files))
	}
}

func TestEnumerateFiles_RecursiveByDefault(t *testing.T) {
	dir := t.TempDir()
	writeTestFile(t, filepath.Join(dir, "top.md"), "top")
	writeTestFile(t, filepath.Join(dir, "sub", "nested.md"), "nested")

	files, _, err := EnumerateFiles(context.Background(), EnumerateOptions{
		Directory: dir,
		Recursive: true,
		MaxFiles:  100,
	})
	if err != nil {
		t.Fatalf("enumerate: %v", err)
	}
	if len(files) != 2 {
		t.Errorf("expected 2 files, got %d", len(files))
	}
}

func TestEnumerateFiles_UnknownExtensionsSkipped(t *testing.T) {
	dir := t.TempDir()
	writeTestFile(t, filepath.Join(dir, "image.png"), "binary")
	writeTestFile(t, filepath.Join(dir, "doc.md"), "markdown")

	files, _, err := EnumerateFiles(context.Background(), EnumerateOptions{
		Directory: dir,
		Recursive: true,
		MaxFiles:  100,
	})
	if err != nil {
		t.Fatalf("enumerate: %v", err)
	}
	if len(files) != 1 {
		t.Errorf("expected 1 file, got %d", len(files))
	}
}
