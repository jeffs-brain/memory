import { describe, expect, it, vi } from 'vitest'
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
import type { AccessControlProvider, Resource, Subject } from './index.js'
import { ForbiddenError } from './index.js'
import { withAccessControl } from './wrap.js'

const path = (s: string): Path => s as Path

const makeStub = (): Store & {
  readonly calls: { method: string; args: unknown[] }[]
} => {
  const calls: { method: string; args: unknown[] }[] = []
  const record =
    <T>(method: string, value: T) =>
    async (...args: unknown[]): Promise<T> => {
      calls.push({ method, args })
      return value
    }
  const stub: Store = {
    read: record('read', Buffer.from('hello')),
    write: record('write', undefined),
    append: record('append', undefined),
    delete: record('delete', undefined),
    rename: record('rename', undefined),
    exists: record('exists', true),
    stat: record('stat', {
      path: path('a'),
      size: 1,
      modTime: new Date(0),
      isDir: false,
    } satisfies FileInfo),
    list: record('list', [] as FileInfo[]),
    batch: async (_: BatchOptions, fn: (b: Batch) => Promise<void>) => {
      calls.push({ method: 'batch', args: [] })
      const inner: Batch = {
        read: record('batch.read', Buffer.from('x')),
        write: record('batch.write', undefined),
        append: record('batch.append', undefined),
        delete: record('batch.delete', undefined),
        rename: record('batch.rename', undefined),
        exists: record('batch.exists', true),
        stat: record('batch.stat', {
          path: path('a'),
          size: 1,
          modTime: new Date(0),
          isDir: false,
        } satisfies FileInfo),
        list: record('batch.list', [] as FileInfo[]),
      }
      await fn(inner)
    },
    subscribe:
      (_sink: EventSink): Unsubscribe =>
      () => {},
    localPath: () => undefined,
    close: async () => {},
  }
  return Object.assign(stub, { calls })
}

const denyAll: AccessControlProvider = {
  name: 'deny-all',
  check: async () => ({ allowed: false, reason: 'deny-all' }),
}

const allowAll: AccessControlProvider = {
  name: 'allow-all',
  check: async () => ({ allowed: true }),
}

const alice: Subject = { kind: 'user', id: 'alice' }
const brainResource: Resource = { type: 'brain', id: 'notes' }

describe('withAccessControl', () => {
  it('throws ForbiddenError and never hits the store on denied read', async () => {
    const inner = makeStub()
    const wrapped = withAccessControl(inner, denyAll, alice, { resource: brainResource })
    await expect(wrapped.read(path('foo'))).rejects.toBeInstanceOf(ForbiddenError)
    expect(inner.calls.find((c) => c.method === 'read')).toBeUndefined()
  })

  it('throws ForbiddenError and never hits the store on denied write', async () => {
    const inner = makeStub()
    const wrapped = withAccessControl(inner, denyAll, alice, { resource: brainResource })
    await expect(wrapped.write(path('foo'), Buffer.from('x'))).rejects.toBeInstanceOf(
      ForbiddenError,
    )
    expect(inner.calls.find((c) => c.method === 'write')).toBeUndefined()
  })

  it('delegates to the underlying store when allowed', async () => {
    const inner = makeStub()
    const wrapped = withAccessControl(inner, allowAll, alice, { resource: brainResource })
    const out = await wrapped.read(path('foo'))
    expect(out.toString()).toBe('hello')
    expect(inner.calls).toEqual([{ method: 'read', args: [path('foo')] }])
  })

  it('calls check with the correct subject/action/resource tuple', async () => {
    const check = vi.fn().mockResolvedValue({ allowed: true })
    const acl: AccessControlProvider = { name: 'spy', check }
    const inner = makeStub()
    const wrapped = withAccessControl(inner, acl, alice, {
      resolveResource: (p) => ({ type: 'document', id: p }),
    })
    await wrapped.delete(path('doc-1'))
    expect(check).toHaveBeenCalledWith(alice, 'delete', { type: 'document', id: 'doc-1' })
  })

  it('guards both src and dst on rename', async () => {
    const check = vi.fn().mockResolvedValue({ allowed: true })
    const acl: AccessControlProvider = { name: 'spy', check }
    const inner = makeStub()
    const wrapped = withAccessControl(inner, acl, alice, {
      resolveResource: (p) => ({ type: 'document', id: p }),
    })
    await wrapped.rename(path('src'), path('dst'))
    expect(check).toHaveBeenCalledTimes(2)
    expect(check).toHaveBeenNthCalledWith(1, alice, 'write', { type: 'document', id: 'src' })
    expect(check).toHaveBeenNthCalledWith(2, alice, 'write', { type: 'document', id: 'dst' })
  })

  it('guards every op inside a batch', async () => {
    const inner = makeStub()
    const wrapped = withAccessControl(inner, denyAll, alice, { resource: brainResource })
    await expect(
      wrapped.batch({ reason: 'test' }, async (b) => {
        await b.read(path('foo'))
      }),
    ).rejects.toBeInstanceOf(ForbiddenError)
    // The batch wrapper itself is entered, but the nested read must be blocked
    // before the stub's batch.read runs.
    expect(inner.calls.find((c) => c.method === 'batch.read')).toBeUndefined()
  })

  it('never exposes local paths through the ACL wrapper', () => {
    const inner = makeStub()
    const wrapped = withAccessControl(inner, allowAll, alice, { resource: brainResource })
    expect(wrapped.localPath(path('foo'))).toBeUndefined()
  })
})
