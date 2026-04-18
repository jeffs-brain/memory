/**
 * Simple RBAC adapter.
 *
 * Stores `Tuple` rows in memory (or in an injected `RbacTupleStore`) and
 * evaluates `check` against a fixed hierarchy:
 *
 *   workspace -> brain -> collection -> document
 *
 * Roles collapse to `admin`, `writer`, `reader` at every level, with
 * inheritance `admin -> writer -> reader`. Granting `admin` on a workspace
 * implicitly grants `admin` on every brain/collection/document that names
 * that workspace as its parent. A matching `deny` tuple at a lower level
 * overrides the parent grant.
 *
 * Parent relationships are themselves expressed as tuples
 * (`resource -> parent <- workspace` style) so the store stays uniform.
 *
 * Semantics invented for this adapter (documented here so reviewers can
 * poke at them):
 *   - The only supported relations for grants are `admin`, `writer`,
 *     `reader`. Any `write` call with another relation is rejected.
 *   - `parent` tuples wire child resources to their parent. Subject must
 *     be the parent resource encoded as `Subject`-shaped data
 *     (`kind: 'service'`, `id: 'workspace:<id>'` etc.). We expose a helper
 *     `parentTuple` to make that readable from callers.
 *   - `deny` tuples use the reserved relation `deny:<role>` and take
 *     precedence over any grant reachable via inheritance, but only at
 *     the exact level the deny is attached to and below.
 */

import type {
  AccessControlProvider,
  Action,
  CheckResult,
  ReadTuplesQuery,
  Resource,
  Subject,
  Tuple,
  WriteTuplesRequest,
} from './index.js'
import { AccessControlError, allow, deny, resourceKey, subjectKey, tupleKey } from './index.js'

export type RbacRole = 'admin' | 'writer' | 'reader'

const ROLES: readonly RbacRole[] = ['admin', 'writer', 'reader'] as const
const ROLE_RANK: Readonly<Record<RbacRole, number>> = { admin: 3, writer: 2, reader: 1 }
const DENY_PREFIX = 'deny:'

const ACTION_TO_MIN_ROLE: Readonly<Record<Action, RbacRole>> = {
  read: 'reader',
  write: 'writer',
  delete: 'admin',
  admin: 'admin',
  export: 'reader',
}

const isRole = (s: string): s is RbacRole => s === 'admin' || s === 'writer' || s === 'reader'

/**
 * Minimal tuple-persistence surface. The default implementation is an
 * in-memory `Map`, but consumers can back it with Postgres (or anything
 * else) by implementing this type.
 */
export type RbacTupleStore = {
  add(tuple: Tuple): void
  remove(tuple: Tuple): void
  list(query: ReadTuplesQuery): readonly Tuple[]
}

export type RbacOptions = {
  readonly store?: RbacTupleStore
}

export const createInMemoryTupleStore = (): RbacTupleStore => {
  const rows = new Map<string, Tuple>()
  return {
    add: (t) => {
      rows.set(tupleKey(t), t)
    },
    remove: (t) => {
      rows.delete(tupleKey(t))
    },
    list: (q) => {
      const out: Tuple[] = []
      for (const t of rows.values()) {
        if (q.subject && subjectKey(q.subject) !== subjectKey(t.subject)) continue
        if (q.relation !== undefined && q.relation !== t.relation) continue
        if (q.resource && resourceKey(q.resource) !== resourceKey(t.resource)) continue
        out.push(t)
      }
      return out
    },
  }
}

/**
 * Encode a parent resource as a pseudo-`Subject` so it can be stored in
 * the same tuple shape.
 */
export const parentSubject = (parent: Resource): Subject => ({
  kind: 'service',
  id: `${parent.type}:${parent.id}`,
})

export const parentTuple = (child: Resource, parent: Resource): Tuple => ({
  subject: parentSubject(parent),
  relation: 'parent',
  resource: child,
})

export const grantTuple = (subject: Subject, role: RbacRole, resource: Resource): Tuple => ({
  subject,
  relation: role,
  resource,
})

export const denyTuple = (subject: Subject, role: RbacRole, resource: Resource): Tuple => ({
  subject,
  relation: `${DENY_PREFIX}${role}`,
  resource,
})

export const createRbacProvider = (opts: RbacOptions = {}): AccessControlProvider => {
  const store = opts.store ?? createInMemoryTupleStore()

  const walkAncestors = (resource: Resource): Resource[] => {
    const chain: Resource[] = [resource]
    const seen = new Set<string>([resourceKey(resource)])
    let cursor: Resource | undefined = resource
    while (cursor !== undefined) {
      const parents = store.list({ resource: cursor, relation: 'parent' })
      const firstParent = parents[0]
      if (firstParent === undefined) break
      // `parent` tuples use a service-subject encoding of the parent.
      const parent = decodeParentSubject(firstParent.subject)
      if (parent === undefined) break
      const key = resourceKey(parent)
      if (seen.has(key)) break
      seen.add(key)
      chain.push(parent)
      cursor = parent
    }
    return chain
  }

  const highestGrantedRole = (subject: Subject, chain: readonly Resource[]): RbacRole | undefined => {
    let best: RbacRole | undefined
    for (const resource of chain) {
      for (const role of ROLES) {
        const found = store.list({ subject, relation: role, resource })
        if (found.length > 0) {
          if (best === undefined || ROLE_RANK[role] > ROLE_RANK[best]) best = role
        }
      }
    }
    return best
  }

  const deniedRolesAt = (subject: Subject, resource: Resource): Set<RbacRole> => {
    const denied = new Set<RbacRole>()
    for (const role of ROLES) {
      const found = store.list({
        subject,
        relation: `${DENY_PREFIX}${role}`,
        resource,
      })
      if (found.length > 0) denied.add(role)
    }
    return denied
  }

  const check = async (
    subject: Subject,
    action: Action,
    resource: Resource,
  ): Promise<CheckResult> => {
    const required = ACTION_TO_MIN_ROLE[action]
    const chain = walkAncestors(resource)

    // A deny tuple anywhere from the target resource upwards that
    // covers the required role blocks the request outright.
    for (const node of chain) {
      const denied = deniedRolesAt(subject, node)
      for (const role of denied) {
        if (ROLE_RANK[role] >= ROLE_RANK[required]) {
          return deny(`deny:${role} on ${resourceKey(node)}`)
        }
      }
    }

    const granted = highestGrantedRole(subject, chain)
    if (granted === undefined) {
      return deny(`no grant for ${subjectKey(subject)} on ${resourceKey(resource)}`)
    }
    if (ROLE_RANK[granted] >= ROLE_RANK[required]) {
      return allow(`${granted} grant satisfies ${required}`)
    }
    return deny(`${granted} insufficient for ${required}`)
  }

  const write = async (req: WriteTuplesRequest): Promise<void> => {
    for (const t of req.writes ?? []) assertSupportedRelation(t.relation)
    for (const t of req.deletes ?? []) assertSupportedRelation(t.relation)
    for (const t of req.writes ?? []) store.add(t)
    for (const t of req.deletes ?? []) store.remove(t)
  }

  const read = async (query: ReadTuplesQuery): Promise<readonly Tuple[]> => store.list(query)

  return { name: 'rbac', check, write, read }
}

const assertSupportedRelation = (relation: string): void => {
  if (relation === 'parent') return
  if (relation.startsWith(DENY_PREFIX)) {
    const rest = relation.slice(DENY_PREFIX.length)
    if (isRole(rest)) return
  }
  if (isRole(relation)) return
  throw new AccessControlError(`rbac: unsupported relation: ${relation}`)
}

const decodeParentSubject = (s: Subject): Resource | undefined => {
  if (s.kind !== 'service') return undefined
  const idx = s.id.indexOf(':')
  if (idx === -1) return undefined
  const type = s.id.slice(0, idx)
  const id = s.id.slice(idx + 1)
  if (type !== 'workspace' && type !== 'brain' && type !== 'collection' && type !== 'document') {
    return undefined
  }
  return { type, id }
}
