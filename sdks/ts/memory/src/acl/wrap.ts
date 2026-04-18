// SPDX-License-Identifier: Apache-2.0

/**
 * `withAccessControl(store, acl, subject, opts)` - wrap a `Store` with an
 * authorisation check on every read/write surface.
 *
 * Each `Store` method is mapped to an `Action`:
 *
 *   read, exists, stat, list, batch-read, localPath  -> 'read'
 *   write, append, batch-write, batch-append         -> 'write'
 *   delete, batch-delete                             -> 'delete'
 *   rename, batch-rename                             -> 'write' (both sides)
 *
 * The resource targeted by a given path is resolved via
 * `opts.resolveResource`. If not supplied, we fall back to `opts.resource`,
 * which covers the common case of one `Store` per brain/collection.
 */

import type {
  Batch,
  BatchOptions,
  EventSink,
  FileInfo,
  ListOpts,
  Path,
  Store,
  Unsubscribe,
} from '../store/index.js'
import type { AccessControlProvider, Action, Resource, Subject } from './index.js'
import { ForbiddenError } from './index.js'

export type WithAclOptions = {
  readonly resource?: Resource
  readonly resolveResource?: (path: Path) => Resource
}

export const withAccessControl = (
  store: Store,
  acl: AccessControlProvider,
  subject: Subject,
  opts: WithAclOptions = {},
): Store => {
  const resolve = (path: Path): Resource => {
    if (opts.resolveResource) return opts.resolveResource(path)
    if (opts.resource) return opts.resource
    throw new ForbiddenError(
      subject,
      'read',
      { type: 'document', id: path },
      'no resource resolver configured',
    )
  }

  const guard = async (action: Action, path: Path): Promise<void> => {
    const resource = resolve(path)
    const result = await acl.check(subject, action, resource)
    if (!result.allowed) {
      throw new ForbiddenError(subject, action, resource, result.reason)
    }
  }

  const wrappedBatch = (batch: Batch): Batch => ({
    read: async (p) => {
      await guard('read', p)
      return batch.read(p)
    },
    write: async (p, c) => {
      await guard('write', p)
      return batch.write(p, c)
    },
    append: async (p, c) => {
      await guard('write', p)
      return batch.append(p, c)
    },
    delete: async (p) => {
      await guard('delete', p)
      return batch.delete(p)
    },
    rename: async (src, dst) => {
      await guard('write', src)
      await guard('write', dst)
      return batch.rename(src, dst)
    },
    exists: async (p) => {
      await guard('read', p)
      return batch.exists(p)
    },
    stat: async (p) => {
      await guard('read', p)
      return batch.stat(p)
    },
    list: async (dir, listOpts) => {
      if (dir !== '') await guard('read', dir)
      return batch.list(dir, listOpts)
    },
  })

  return {
    read: async (path: Path): Promise<Buffer> => {
      await guard('read', path)
      return store.read(path)
    },
    write: async (path: Path, content: Buffer): Promise<void> => {
      await guard('write', path)
      return store.write(path, content)
    },
    append: async (path: Path, content: Buffer): Promise<void> => {
      await guard('write', path)
      return store.append(path, content)
    },
    delete: async (path: Path): Promise<void> => {
      await guard('delete', path)
      return store.delete(path)
    },
    rename: async (src: Path, dst: Path): Promise<void> => {
      await guard('write', src)
      await guard('write', dst)
      return store.rename(src, dst)
    },
    exists: async (path: Path): Promise<boolean> => {
      await guard('read', path)
      return store.exists(path)
    },
    stat: async (path: Path): Promise<FileInfo> => {
      await guard('read', path)
      return store.stat(path)
    },
    list: async (dir: Path | '', listOpts?: ListOpts): Promise<FileInfo[]> => {
      if (dir !== '') await guard('read', dir)
      return store.list(dir, listOpts)
    },
    batch: async (bopts: BatchOptions, fn: (batch: Batch) => Promise<void>): Promise<void> => {
      return store.batch(bopts, (inner) => fn(wrappedBatch(inner)))
    },
    subscribe: (sink: EventSink): Unsubscribe => store.subscribe(sink),
    // `localPath` cannot enforce ACL safely because the provider contract is
    // async. Return no path at all rather than leak a filesystem location.
    localPath: (_path: Path): string | undefined => undefined,
    close: (): Promise<void> => store.close(),
  }
}
