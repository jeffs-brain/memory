// SPDX-License-Identifier: Apache-2.0

package acl

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/jeffs-brain/memory/go/brain"
)

// stubStore records every call made against it. The recorded values
// are checked by the wrap tests to prove that denied calls never
// reach the underlying store.
type stubStore struct {
	mu    sync.Mutex
	calls []stubCall
}

type stubCall struct {
	method string
	args   []any
}

func (s *stubStore) record(method string, args ...any) {
	s.mu.Lock()
	s.calls = append(s.calls, stubCall{method: method, args: args})
	s.mu.Unlock()
}

func (s *stubStore) hasCall(method string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, c := range s.calls {
		if c.method == method {
			return true
		}
	}
	return false
}

func (s *stubStore) Read(ctx context.Context, p brain.Path) ([]byte, error) {
	s.record("read", p)
	return []byte("hello"), nil
}

func (s *stubStore) Write(ctx context.Context, p brain.Path, content []byte) error {
	s.record("write", p, content)
	return nil
}

func (s *stubStore) Append(ctx context.Context, p brain.Path, content []byte) error {
	s.record("append", p, content)
	return nil
}

func (s *stubStore) Delete(ctx context.Context, p brain.Path) error {
	s.record("delete", p)
	return nil
}

func (s *stubStore) Rename(ctx context.Context, src, dst brain.Path) error {
	s.record("rename", src, dst)
	return nil
}

func (s *stubStore) Exists(ctx context.Context, p brain.Path) (bool, error) {
	s.record("exists", p)
	return true, nil
}

func (s *stubStore) Stat(ctx context.Context, p brain.Path) (brain.FileInfo, error) {
	s.record("stat", p)
	return brain.FileInfo{Path: p, Size: 1, ModTime: time.Unix(0, 0)}, nil
}

func (s *stubStore) List(ctx context.Context, dir brain.Path, opts brain.ListOpts) ([]brain.FileInfo, error) {
	s.record("list", dir)
	return nil, nil
}

func (s *stubStore) Batch(ctx context.Context, opts brain.BatchOptions, fn func(brain.Batch) error) error {
	s.record("batch")
	inner := &stubBatch{parent: s}
	return fn(inner)
}

func (s *stubStore) Subscribe(sink brain.EventSink) func() {
	s.record("subscribe")
	return func() {}
}

func (s *stubStore) LocalPath(p brain.Path) (string, bool) {
	s.record("localPath", p)
	return "/tmp/x", true
}

func (s *stubStore) Close() error {
	s.record("close")
	return nil
}

type stubBatch struct {
	parent *stubStore
}

func (b *stubBatch) Read(ctx context.Context, p brain.Path) ([]byte, error) {
	b.parent.record("batch.read", p)
	return []byte("x"), nil
}

func (b *stubBatch) Write(ctx context.Context, p brain.Path, content []byte) error {
	b.parent.record("batch.write", p, content)
	return nil
}

func (b *stubBatch) Append(ctx context.Context, p brain.Path, content []byte) error {
	b.parent.record("batch.append", p, content)
	return nil
}

func (b *stubBatch) Delete(ctx context.Context, p brain.Path) error {
	b.parent.record("batch.delete", p)
	return nil
}

func (b *stubBatch) Rename(ctx context.Context, src, dst brain.Path) error {
	b.parent.record("batch.rename", src, dst)
	return nil
}

func (b *stubBatch) Exists(ctx context.Context, p brain.Path) (bool, error) {
	b.parent.record("batch.exists", p)
	return true, nil
}

func (b *stubBatch) Stat(ctx context.Context, p brain.Path) (brain.FileInfo, error) {
	b.parent.record("batch.stat", p)
	return brain.FileInfo{Path: p}, nil
}

func (b *stubBatch) List(ctx context.Context, dir brain.Path, opts brain.ListOpts) ([]brain.FileInfo, error) {
	b.parent.record("batch.list", dir)
	return nil, nil
}

// staticProvider is a Provider that always returns the same result.
type staticProvider struct {
	name   string
	result CheckResult
	err    error
	calls  []checkCall
	mu     sync.Mutex
}

type checkCall struct {
	subject  Subject
	action   Action
	resource Resource
}

func (s *staticProvider) Name() string { return s.name }

func (s *staticProvider) Check(ctx context.Context, subject Subject, action Action, resource Resource) (CheckResult, error) {
	s.mu.Lock()
	s.calls = append(s.calls, checkCall{subject, action, resource})
	s.mu.Unlock()
	return s.result, s.err
}

func (s *staticProvider) Write(ctx context.Context, req WriteTuplesRequest) error { return nil }

func (s *staticProvider) Read(ctx context.Context, query ReadTuplesQuery) ([]Tuple, error) {
	return nil, nil
}

func (s *staticProvider) Close() error { return nil }

var (
	denyAll  = &staticProvider{name: "deny-all", result: Deny("deny-all")}
	allowAll = &staticProvider{name: "allow-all", result: Allow("")}

	aliceSubject = Subject{Kind: SubjectUser, ID: "alice"}
	brainTarget  = Resource{Type: ResourceBrain, ID: "notes"}
)

func newDeny() *staticProvider  { return &staticProvider{name: "deny", result: Deny("nope")} }
func newAllow() *staticProvider { return &staticProvider{name: "allow", result: Allow("")} }

func TestWrapDeniedReadReturnsForbiddenAndSkipsStore(t *testing.T) {
	inner := &stubStore{}
	wrapped := Wrap(inner, newDeny(), aliceSubject, WrapOptions{Resource: brainTarget})
	_, err := wrapped.Read(context.Background(), "foo")
	var fe *ForbiddenError
	if !errors.As(err, &fe) {
		t.Fatalf("Read error = %v, want *ForbiddenError", err)
	}
	if inner.hasCall("read") {
		t.Fatal("denied Read still hit the underlying store")
	}
}

func TestWrapDeniedWriteReturnsForbiddenAndSkipsStore(t *testing.T) {
	inner := &stubStore{}
	wrapped := Wrap(inner, newDeny(), aliceSubject, WrapOptions{Resource: brainTarget})
	err := wrapped.Write(context.Background(), "foo", []byte("x"))
	var fe *ForbiddenError
	if !errors.As(err, &fe) {
		t.Fatalf("Write error = %v, want *ForbiddenError", err)
	}
	if inner.hasCall("write") {
		t.Fatal("denied Write still hit the underlying store")
	}
}

func TestWrapDeniedDeleteReturnsForbiddenAndSkipsStore(t *testing.T) {
	inner := &stubStore{}
	wrapped := Wrap(inner, newDeny(), aliceSubject, WrapOptions{Resource: brainTarget})
	err := wrapped.Delete(context.Background(), "foo")
	var fe *ForbiddenError
	if !errors.As(err, &fe) {
		t.Fatalf("Delete error = %v, want *ForbiddenError", err)
	}
	if inner.hasCall("delete") {
		t.Fatal("denied Delete still hit the underlying store")
	}
}

func TestWrapAllowedDelegates(t *testing.T) {
	inner := &stubStore{}
	wrapped := Wrap(inner, newAllow(), aliceSubject, WrapOptions{Resource: brainTarget})
	got, err := wrapped.Read(context.Background(), "foo")
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if string(got) != "hello" {
		t.Fatalf("Read returned %q, want hello", got)
	}
	if !inner.hasCall("read") {
		t.Fatal("allowed Read did not delegate")
	}
}

func TestWrapPassesCorrectSubjectActionResource(t *testing.T) {
	provider := newAllow()
	inner := &stubStore{}
	wrapped := Wrap(inner, provider, aliceSubject, WrapOptions{
		ResolveResource: func(p brain.Path) Resource {
			return Resource{Type: ResourceDocument, ID: string(p)}
		},
	})
	if err := wrapped.Delete(context.Background(), "doc-1"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if len(provider.calls) != 1 {
		t.Fatalf("Check called %d times, want 1", len(provider.calls))
	}
	got := provider.calls[0]
	if got.subject != aliceSubject {
		t.Errorf("subject = %+v, want %+v", got.subject, aliceSubject)
	}
	if got.action != ActionDelete {
		t.Errorf("action = %s, want delete", got.action)
	}
	if got.resource != (Resource{Type: ResourceDocument, ID: "doc-1"}) {
		t.Errorf("resource = %+v, want document:doc-1", got.resource)
	}
}

func TestWrapRenameGuardsSrcAndDst(t *testing.T) {
	provider := newAllow()
	inner := &stubStore{}
	wrapped := Wrap(inner, provider, aliceSubject, WrapOptions{
		ResolveResource: func(p brain.Path) Resource {
			return Resource{Type: ResourceDocument, ID: string(p)}
		},
	})
	if err := wrapped.Rename(context.Background(), "src", "dst"); err != nil {
		t.Fatalf("Rename: %v", err)
	}
	if len(provider.calls) != 2 {
		t.Fatalf("Check called %d times, want 2", len(provider.calls))
	}
	if provider.calls[0].action != ActionWrite || provider.calls[0].resource.ID != "src" {
		t.Errorf("first Check = %+v, want write src", provider.calls[0])
	}
	if provider.calls[1].action != ActionWrite || provider.calls[1].resource.ID != "dst" {
		t.Errorf("second Check = %+v, want write dst", provider.calls[1])
	}
}

func TestWrapBatchGuardsEveryOp(t *testing.T) {
	inner := &stubStore{}
	wrapped := Wrap(inner, newDeny(), aliceSubject, WrapOptions{Resource: brainTarget})
	err := wrapped.Batch(context.Background(), brain.BatchOptions{Reason: "test"}, func(b brain.Batch) error {
		_, rerr := b.Read(context.Background(), "foo")
		return rerr
	})
	var fe *ForbiddenError
	if !errors.As(err, &fe) {
		t.Fatalf("Batch error = %v, want *ForbiddenError", err)
	}
	if inner.hasCall("batch.read") {
		t.Fatal("denied batch.Read still hit the underlying batch")
	}
	// The outer Batch entry itself runs so the inner callback can be
	// invoked; only the nested op is blocked.
	if !inner.hasCall("batch") {
		t.Fatal("expected outer Batch call to run")
	}
}

func TestWrapLocalPathAlwaysReturnsEmpty(t *testing.T) {
	inner := &stubStore{}
	wrapped := Wrap(inner, newAllow(), aliceSubject, WrapOptions{Resource: brainTarget})
	path, ok := wrapped.LocalPath("foo")
	if ok {
		t.Errorf("LocalPath ok = true, want false")
	}
	if path != "" {
		t.Errorf("LocalPath path = %q, want empty", path)
	}
}

func TestWrapDeniedErrorIsBrainSentinel(t *testing.T) {
	inner := &stubStore{}
	wrapped := Wrap(inner, newDeny(), aliceSubject, WrapOptions{Resource: brainTarget})
	_, err := wrapped.Read(context.Background(), "foo")
	if !errors.Is(err, brain.ErrForbidden) {
		t.Fatalf("errors.Is(err, brain.ErrForbidden) = false; err = %v", err)
	}
}

func TestWrapNoResolverReturnsForbidden(t *testing.T) {
	inner := &stubStore{}
	wrapped := Wrap(inner, newAllow(), aliceSubject, WrapOptions{})
	_, err := wrapped.Read(context.Background(), "foo")
	var fe *ForbiddenError
	if !errors.As(err, &fe) {
		t.Fatalf("Read error = %v, want *ForbiddenError", err)
	}
	if fe.Action != ActionRead {
		t.Errorf("error action = %s, want read", fe.Action)
	}
	if fe.Reason != "no resource resolver configured" {
		t.Errorf("reason = %q", fe.Reason)
	}
	if inner.hasCall("read") {
		t.Fatal("Read with no resolver still hit the store")
	}
}

func TestWrapListEmptyDirIsUnguarded(t *testing.T) {
	inner := &stubStore{}
	provider := newDeny()
	wrapped := Wrap(inner, provider, aliceSubject, WrapOptions{Resource: brainTarget})
	if _, err := wrapped.List(context.Background(), "", brain.ListOpts{}); err != nil {
		t.Fatalf("root List should not error: %v", err)
	}
	if !inner.hasCall("list") {
		t.Fatal("root List did not delegate")
	}
	if len(provider.calls) != 0 {
		t.Fatalf("provider invoked %d times, want 0", len(provider.calls))
	}
}

func TestWrapBatchAllowedDelegates(t *testing.T) {
	inner := &stubStore{}
	wrapped := Wrap(inner, newAllow(), aliceSubject, WrapOptions{Resource: brainTarget})
	err := wrapped.Batch(context.Background(), brain.BatchOptions{}, func(b brain.Batch) error {
		if err := b.Write(context.Background(), "foo", []byte("x")); err != nil {
			return err
		}
		_, err := b.Read(context.Background(), "foo")
		return err
	})
	if err != nil {
		t.Fatalf("Batch: %v", err)
	}
	if !inner.hasCall("batch.write") || !inner.hasCall("batch.read") {
		t.Fatal("allowed batch ops did not reach the underlying batch")
	}
}
