// SPDX-License-Identifier: Apache-2.0

package brain

import (
	"context"
	"errors"
	"sync"
	"testing"
)

// fakeStore is a minimal in-package [Store] stub used by the schema tests
// so the brain package does not take a test-time dependency on a backend.
// It supports Read, Write, Exists, and Batch (single-version stamp).
type fakeStore struct {
	mu     sync.Mutex
	files  map[Path][]byte
	closed bool
}

func newFakeStore() *fakeStore {
	return &fakeStore{files: make(map[Path][]byte)}
}

func (s *fakeStore) Read(ctx context.Context, p Path) ([]byte, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, ErrReadOnly
	}
	data, ok := s.files[p]
	if !ok {
		return nil, ErrNotFound
	}
	out := make([]byte, len(data))
	copy(out, data)
	return out, nil
}

func (s *fakeStore) Write(ctx context.Context, p Path, content []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ErrReadOnly
	}
	s.files[p] = append([]byte(nil), content...)
	return nil
}

func (s *fakeStore) Append(ctx context.Context, p Path, content []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return ErrReadOnly
	}
	s.files[p] = append(s.files[p], content...)
	return nil
}

func (s *fakeStore) Delete(ctx context.Context, p Path) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.files[p]; !ok {
		return ErrNotFound
	}
	delete(s.files, p)
	return nil
}

func (s *fakeStore) Rename(ctx context.Context, src, dst Path) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	data, ok := s.files[src]
	if !ok {
		return ErrNotFound
	}
	delete(s.files, src)
	s.files[dst] = data
	return nil
}

func (s *fakeStore) Exists(ctx context.Context, p Path) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, ok := s.files[p]
	return ok, nil
}

func (s *fakeStore) Stat(ctx context.Context, p Path) (FileInfo, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	data, ok := s.files[p]
	if !ok {
		return FileInfo{}, ErrNotFound
	}
	return FileInfo{Path: p, Size: int64(len(data))}, nil
}

func (s *fakeStore) List(ctx context.Context, dir Path, opts ListOpts) ([]FileInfo, error) {
	return nil, nil
}

func (s *fakeStore) Batch(ctx context.Context, opts BatchOptions, fn func(Batch) error) error {
	b := &fakeBatch{store: s}
	if err := fn(b); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for p, data := range b.writes {
		s.files[p] = append([]byte(nil), data...)
	}
	return nil
}

func (s *fakeStore) Subscribe(sink EventSink) func() { return func() {} }
func (s *fakeStore) LocalPath(p Path) (string, bool)  { return "", false }
func (s *fakeStore) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	return nil
}

type fakeBatch struct {
	store  *fakeStore
	writes map[Path][]byte
}

func (b *fakeBatch) Read(ctx context.Context, p Path) ([]byte, error) {
	if data, ok := b.writes[p]; ok {
		return append([]byte(nil), data...), nil
	}
	return b.store.Read(ctx, p)
}

func (b *fakeBatch) Write(ctx context.Context, p Path, content []byte) error {
	if b.writes == nil {
		b.writes = make(map[Path][]byte)
	}
	b.writes[p] = append([]byte(nil), content...)
	return nil
}

func (b *fakeBatch) Append(ctx context.Context, p Path, content []byte) error {
	return errors.New("not used by schema tests")
}
func (b *fakeBatch) Delete(ctx context.Context, p Path) error {
	return errors.New("not used by schema tests")
}
func (b *fakeBatch) Rename(ctx context.Context, src, dst Path) error {
	return errors.New("not used by schema tests")
}
func (b *fakeBatch) Exists(ctx context.Context, p Path) (bool, error) { return false, nil }
func (b *fakeBatch) Stat(ctx context.Context, p Path) (FileInfo, error) {
	return FileInfo{}, ErrNotFound
}
func (b *fakeBatch) List(ctx context.Context, dir Path, opts ListOpts) ([]FileInfo, error) {
	return nil, nil
}

var _ Store = (*fakeStore)(nil)
var _ Batch = (*fakeBatch)(nil)

func TestCheckSchemaVersion_MissingCreatesV1(t *testing.T) {
	s := newFakeStore()
	defer s.Close()
	ctx := context.Background()

	if err := CheckSchemaVersion(ctx, s); err != nil {
		t.Fatalf("CheckSchemaVersion on empty store: %v", err)
	}

	data, err := s.Read(ctx, SchemaVersionPath())
	if err != nil {
		t.Fatalf("reading schema-version.yaml after check: %v", err)
	}
	if want := "version: 1\n"; string(data) != want {
		t.Fatalf("schema-version.yaml content = %q, want %q", data, want)
	}
}

func TestCheckSchemaVersion_CurrentIsNoop(t *testing.T) {
	s := newFakeStore()
	defer s.Close()
	ctx := context.Background()

	if err := s.Write(ctx, SchemaVersionPath(), []byte("version: 1\n")); err != nil {
		t.Fatal(err)
	}

	if err := CheckSchemaVersion(ctx, s); err != nil {
		t.Fatalf("CheckSchemaVersion with current version: %v", err)
	}
}

func TestCheckSchemaVersion_NewerReturnsError(t *testing.T) {
	s := newFakeStore()
	defer s.Close()
	ctx := context.Background()

	if err := s.Write(ctx, SchemaVersionPath(), []byte("version: 99\n")); err != nil {
		t.Fatal(err)
	}

	err := CheckSchemaVersion(ctx, s)
	if err == nil {
		t.Fatal("CheckSchemaVersion with future version returned nil")
	}
	if !errors.Is(err, ErrSchemaVersion) {
		t.Fatalf("error does not wrap ErrSchemaVersion: %v", err)
	}
}

func TestMigrateSchema_MissingIsNoop(t *testing.T) {
	s := newFakeStore()
	defer s.Close()
	ctx := context.Background()

	if err := MigrateSchema(ctx, s); err != nil {
		t.Fatalf("MigrateSchema on empty store: %v", err)
	}

	// MigrateSchema with v==0 treats it as v1==SchemaVersion, so it's a
	// no-op and the file stays absent. CheckSchemaVersion is what stamps.
	exists, err := s.Exists(ctx, SchemaVersionPath())
	if err != nil {
		t.Fatalf("Exists: %v", err)
	}
	if exists {
		t.Fatal("MigrateSchema should be a no-op when version is already current")
	}
}

func TestMigrateSchema_CurrentIsNoop(t *testing.T) {
	s := newFakeStore()
	defer s.Close()
	ctx := context.Background()

	if err := s.Write(ctx, SchemaVersionPath(), []byte("version: 1\n")); err != nil {
		t.Fatal(err)
	}

	if err := MigrateSchema(ctx, s); err != nil {
		t.Fatalf("MigrateSchema at current version: %v", err)
	}
}

func TestMigrateSchema_NewerReturnsError(t *testing.T) {
	s := newFakeStore()
	defer s.Close()
	ctx := context.Background()

	if err := s.Write(ctx, SchemaVersionPath(), []byte("version: 99\n")); err != nil {
		t.Fatal(err)
	}

	err := MigrateSchema(ctx, s)
	if err == nil {
		t.Fatal("MigrateSchema with future version returned nil")
	}
	if !errors.Is(err, ErrSchemaVersion) {
		t.Fatalf("error does not wrap ErrSchemaVersion: %v", err)
	}
}

func TestSchemaVersionPath_IsValid(t *testing.T) {
	if err := ValidatePath(SchemaVersionPath()); err != nil {
		t.Fatalf("SchemaVersionPath() is not a valid brain.Path: %v", err)
	}
}
