// SPDX-License-Identifier: Apache-2.0

package acl

import (
	"context"

	"github.com/jeffs-brain/memory/go/brain"
)

// WrapOptions configures the resource resolution strategy used by
// [Wrap]. Callers MUST set either Resource (for the common single-
// brain case) or ResolveResource (for stores that span multiple
// resources). Passing both is allowed; ResolveResource wins.
type WrapOptions struct {
	// Resource is the static resource that every guarded operation
	// targets. Used when one [brain.Store] maps to one brain or
	// collection.
	Resource Resource
	// ResolveResource maps a path to its owning resource. Useful when
	// a single store covers many resources (for example, a store that
	// fronts an entire workspace).
	ResolveResource func(brain.Path) Resource
}

// Wrap returns a [brain.Store] that consults provider before every
// read or write. Method-to-action mapping mirrors the TypeScript
// reference implementation:
//
//	read, exists, stat, list, batch.read, batch.exists, batch.stat,
//	batch.list                                  -> ActionRead
//	write, append, batch.write, batch.append    -> ActionWrite
//	delete, batch.delete                        -> ActionDelete
//	rename, batch.rename                        -> ActionWrite (src + dst)
//
// Subscribe and Close pass through unchanged. List with an empty
// directory is unguarded so root listings work without a resolver.
//
// If neither WrapOptions.Resource nor WrapOptions.ResolveResource is
// configured, every guarded call returns a [*ForbiddenError]; the
// wrapped store is never invoked. We deliberately do not panic so
// misconfiguration in production cannot crash the host.
func Wrap(store brain.Store, provider Provider, subject Subject, opts WrapOptions) brain.Store {
	return &aclStore{store: store, provider: provider, subject: subject, opts: opts}
}

type aclStore struct {
	store    brain.Store
	provider Provider
	subject  Subject
	opts     WrapOptions
}

// resolve computes the resource targeted by p. When neither resolver
// is configured we return a placeholder document resource keyed by p
// so the resulting [*ForbiddenError] still names something useful in
// logs.
func (a *aclStore) resolve(p brain.Path) (Resource, *ForbiddenError) {
	if a.opts.ResolveResource != nil {
		return a.opts.ResolveResource(p), nil
	}
	if a.opts.Resource != (Resource{}) {
		return a.opts.Resource, nil
	}
	placeholder := Resource{Type: ResourceDocument, ID: string(p)}
	return placeholder, &ForbiddenError{
		Subject:  a.subject,
		Action:   ActionRead,
		Resource: placeholder,
		Reason:   "no resource resolver configured",
	}
}

// guard runs a check for action against the resource targeting p. On
// any failure it returns a non-nil error; callers must not invoke the
// underlying store when guard returns non-nil.
func (a *aclStore) guard(ctx context.Context, action Action, p brain.Path) error {
	resource, ferr := a.resolve(p)
	if ferr != nil {
		// The placeholder action is not necessarily what the caller
		// asked for; rewrite it so the error reflects the real call.
		ferr.Action = action
		return ferr
	}
	res, err := a.provider.Check(ctx, a.subject, action, resource)
	if err != nil {
		return err
	}
	if !res.Allowed {
		return &ForbiddenError{
			Subject:  a.subject,
			Action:   action,
			Resource: resource,
			Reason:   res.Reason,
		}
	}
	return nil
}

func (a *aclStore) Read(ctx context.Context, p brain.Path) ([]byte, error) {
	if err := a.guard(ctx, ActionRead, p); err != nil {
		return nil, err
	}
	return a.store.Read(ctx, p)
}

func (a *aclStore) Write(ctx context.Context, p brain.Path, content []byte) error {
	if err := a.guard(ctx, ActionWrite, p); err != nil {
		return err
	}
	return a.store.Write(ctx, p, content)
}

func (a *aclStore) Append(ctx context.Context, p brain.Path, content []byte) error {
	if err := a.guard(ctx, ActionWrite, p); err != nil {
		return err
	}
	return a.store.Append(ctx, p, content)
}

func (a *aclStore) Delete(ctx context.Context, p brain.Path) error {
	if err := a.guard(ctx, ActionDelete, p); err != nil {
		return err
	}
	return a.store.Delete(ctx, p)
}

func (a *aclStore) Rename(ctx context.Context, src, dst brain.Path) error {
	if err := a.guard(ctx, ActionWrite, src); err != nil {
		return err
	}
	if err := a.guard(ctx, ActionWrite, dst); err != nil {
		return err
	}
	return a.store.Rename(ctx, src, dst)
}

func (a *aclStore) Exists(ctx context.Context, p brain.Path) (bool, error) {
	if err := a.guard(ctx, ActionRead, p); err != nil {
		return false, err
	}
	return a.store.Exists(ctx, p)
}

func (a *aclStore) Stat(ctx context.Context, p brain.Path) (brain.FileInfo, error) {
	if err := a.guard(ctx, ActionRead, p); err != nil {
		return brain.FileInfo{}, err
	}
	return a.store.Stat(ctx, p)
}

func (a *aclStore) List(ctx context.Context, dir brain.Path, opts brain.ListOpts) ([]brain.FileInfo, error) {
	// Root listings are unguarded so callers can enumerate top-level
	// entries without configuring a resolver. This mirrors the TS
	// wrapper exactly.
	if dir != "" {
		if err := a.guard(ctx, ActionRead, dir); err != nil {
			return nil, err
		}
	}
	return a.store.List(ctx, dir, opts)
}

func (a *aclStore) Batch(ctx context.Context, opts brain.BatchOptions, fn func(brain.Batch) error) error {
	return a.store.Batch(ctx, opts, func(inner brain.Batch) error {
		return fn(&aclBatch{batch: inner, parent: a})
	})
}

func (a *aclStore) Subscribe(sink brain.EventSink) func() {
	return a.store.Subscribe(sink)
}

// LocalPath always returns ("", false) because the [Provider] contract
// is context-bound and may not be cheap; leaking a filesystem path
// without consulting the provider would defeat the whole point of the
// wrapper.
func (a *aclStore) LocalPath(p brain.Path) (string, bool) { return "", false }

func (a *aclStore) Close() error { return a.store.Close() }

// aclBatch mirrors aclStore for the batch surface so every nested
// operation is guarded with the same rules.
type aclBatch struct {
	batch  brain.Batch
	parent *aclStore
}

func (b *aclBatch) Read(ctx context.Context, p brain.Path) ([]byte, error) {
	if err := b.parent.guard(ctx, ActionRead, p); err != nil {
		return nil, err
	}
	return b.batch.Read(ctx, p)
}

func (b *aclBatch) Write(ctx context.Context, p brain.Path, content []byte) error {
	if err := b.parent.guard(ctx, ActionWrite, p); err != nil {
		return err
	}
	return b.batch.Write(ctx, p, content)
}

func (b *aclBatch) Append(ctx context.Context, p brain.Path, content []byte) error {
	if err := b.parent.guard(ctx, ActionWrite, p); err != nil {
		return err
	}
	return b.batch.Append(ctx, p, content)
}

func (b *aclBatch) Delete(ctx context.Context, p brain.Path) error {
	if err := b.parent.guard(ctx, ActionDelete, p); err != nil {
		return err
	}
	return b.batch.Delete(ctx, p)
}

func (b *aclBatch) Rename(ctx context.Context, src, dst brain.Path) error {
	if err := b.parent.guard(ctx, ActionWrite, src); err != nil {
		return err
	}
	if err := b.parent.guard(ctx, ActionWrite, dst); err != nil {
		return err
	}
	return b.batch.Rename(ctx, src, dst)
}

func (b *aclBatch) Exists(ctx context.Context, p brain.Path) (bool, error) {
	if err := b.parent.guard(ctx, ActionRead, p); err != nil {
		return false, err
	}
	return b.batch.Exists(ctx, p)
}

func (b *aclBatch) Stat(ctx context.Context, p brain.Path) (brain.FileInfo, error) {
	if err := b.parent.guard(ctx, ActionRead, p); err != nil {
		return brain.FileInfo{}, err
	}
	return b.batch.Stat(ctx, p)
}

func (b *aclBatch) List(ctx context.Context, dir brain.Path, opts brain.ListOpts) ([]brain.FileInfo, error) {
	if dir != "" {
		if err := b.parent.guard(ctx, ActionRead, dir); err != nil {
			return nil, err
		}
	}
	return b.batch.List(ctx, dir, opts)
}

// compile-time interface checks.
var (
	_ brain.Store = (*aclStore)(nil)
	_ brain.Batch = (*aclBatch)(nil)
)
