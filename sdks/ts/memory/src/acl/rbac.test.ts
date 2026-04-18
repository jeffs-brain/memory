import { describe, expect, it } from 'vitest'
import type { Resource, Subject, Tuple } from './index.js'
import {
  createRbacProvider,
  denyTuple,
  grantTuple,
  parentTuple,
  type RbacRole,
} from './rbac.js'

const user = (id: string): Subject => ({ kind: 'user', id })
const workspace = (id: string): Resource => ({ type: 'workspace', id })
const brain = (id: string): Resource => ({ type: 'brain', id })
const collection = (id: string): Resource => ({ type: 'collection', id })
const document = (id: string): Resource => ({ type: 'document', id })

const seed = async (tuples: readonly Tuple[]) => {
  const acl = createRbacProvider()
  if (acl.write) await acl.write({ writes: tuples })
  return acl
}

describe('rbac adapter', () => {
  it('admin on workspace implies admin on brain via parent chain', async () => {
    const alice = user('alice')
    const ws = workspace('acme')
    const br = brain('notes')
    const acl = await seed([
      parentTuple(br, ws),
      grantTuple(alice, 'admin', ws),
    ])

    const result = await acl.check(alice, 'admin', br)
    expect(result.allowed).toBe(true)

    // Inheritance cascades further down.
    const col = collection('meetings')
    const doc = document('doc-1')
    const acl2 = await seed([
      parentTuple(br, ws),
      parentTuple(col, br),
      parentTuple(doc, col),
      grantTuple(alice, 'admin', ws),
    ])
    expect((await acl2.check(alice, 'write', doc)).allowed).toBe(true)
    expect((await acl2.check(alice, 'delete', doc)).allowed).toBe(true)
  })

  it('revoke at brain level overrides workspace grant', async () => {
    const alice = user('alice')
    const ws = workspace('acme')
    const br = brain('notes')
    const acl = await seed([
      parentTuple(br, ws),
      grantTuple(alice, 'admin', ws),
      denyTuple(alice, 'admin' as RbacRole, br),
    ])

    const adminCheck = await acl.check(alice, 'admin', br)
    expect(adminCheck.allowed).toBe(false)
    expect(adminCheck.reason).toMatch(/deny:admin/)

    // The deny on admin also covers any lower action that requires admin
    // (delete). A narrower deny would need to be crafted to leave
    // writer/reader untouched.
    const writeCheck = await acl.check(alice, 'write', br)
    expect(writeCheck.allowed).toBe(false)
  })

  it('document reader cannot write', async () => {
    const bob = user('bob')
    const doc = document('doc-1')
    const acl = await seed([grantTuple(bob, 'reader', doc)])

    expect((await acl.check(bob, 'read', doc)).allowed).toBe(true)
    const write = await acl.check(bob, 'write', doc)
    expect(write.allowed).toBe(false)
    expect(write.reason).toMatch(/insufficient/)
  })

  it('unknown subject denied by default', async () => {
    const eve = user('eve')
    const doc = document('doc-1')
    const acl = await seed([grantTuple(user('alice'), 'admin', doc)])

    const result = await acl.check(eve, 'read', doc)
    expect(result.allowed).toBe(false)
    expect(result.reason).toMatch(/no grant/)
  })

  it('rejects unsupported relations on write', async () => {
    const acl = createRbacProvider()
    await expect(
      acl.write?.({ writes: [{ subject: user('a'), relation: 'bogus', resource: brain('x') }] }),
    ).rejects.toThrow(/unsupported relation/)
  })

  it('read returns the stored tuples', async () => {
    const alice = user('alice')
    const br = brain('notes')
    const acl = await seed([grantTuple(alice, 'writer', br)])
    const found = await acl.read?.({ subject: alice })
    expect(found?.length).toBe(1)
    expect(found?.[0]?.relation).toBe('writer')
  })
})
