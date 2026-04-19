// SPDX-License-Identifier: Apache-2.0

/**
 * ACL primitives for `@jeffs-brain/memory`.
 *
 * Defines the `AccessControlProvider` contract used by the memory core to
 * authorise reads and writes against a `Store`. Ships with two in-box
 * adapters:
 *   - `rbac.ts`   - a simple role-based adapter for small installs.
 *   - `openfga.ts` - an OpenFGA HTTP adapter for production use.
 *
 * See `spec/openfga/schema.fga` for the authorisation model the OpenFGA adapter speaks.
 */

export type ResourceType = 'workspace' | 'brain' | 'collection' | 'document'

export type Resource = {
  readonly type: ResourceType
  readonly id: string
}

export type SubjectKind = 'user' | 'api_key' | 'service'

export type Subject = {
  readonly kind: SubjectKind
  readonly id: string
}

/**
 * Actions map roughly to OpenFGA relations. `read` and `write` are the two
 * actions any `Store`-level wrapper needs; the remaining actions are carried
 * for adapters that want to express richer policy (delete, export, admin).
 */
export type Action = 'read' | 'write' | 'delete' | 'admin' | 'export'

export type CheckResult = {
  readonly allowed: boolean
  readonly reason?: string
}

/**
 * A single relationship tuple, lifted straight from the Zanzibar/OpenFGA
 * vocabulary so adapters can round-trip without translation.
 */
export type Tuple = {
  readonly subject: Subject
  readonly relation: string
  readonly resource: Resource
}

export type WriteTuplesRequest = {
  readonly writes?: readonly Tuple[]
  readonly deletes?: readonly Tuple[]
}

export type ReadTuplesQuery = {
  readonly subject?: Subject
  readonly relation?: string
  readonly resource?: Resource
}

/**
 * Pluggable authorisation provider.
 *
 * Any adapter MUST implement `check`. `write`, `read` and `close` are
 * optional: `write`/`read` because some providers (e.g. a stateless policy
 * evaluator) do not persist tuples, and `close` because providers that
 * own no transport state have nothing to release.
 */
export type AccessControlProvider = {
  readonly name: string
  check(subject: Subject, action: Action, resource: Resource): Promise<CheckResult>
  write?(req: WriteTuplesRequest): Promise<void>
  read?(query: ReadTuplesQuery): Promise<readonly Tuple[]>
  /**
   * Release any resources held by the provider (network clients, file
   * handles, etc.). Implementations that own no resources may omit this
   * method.
   */
  close?(): Promise<void>
}

export class AccessControlError extends Error {
  override readonly name: string = 'AccessControlError'
  constructor(
    message: string,
    override readonly cause?: unknown,
  ) {
    super(message)
  }
}

export class ForbiddenError extends AccessControlError {
  override readonly name = 'ForbiddenError'
  constructor(
    readonly subject: Subject,
    readonly action: Action,
    readonly resource: Resource,
    reason?: string,
  ) {
    super(
      `acl: forbidden: ${subject.kind}:${subject.id} ${action} ${resource.type}:${resource.id}${
        reason ? ` (${reason})` : ''
      }`,
    )
  }
}

export const isForbidden = (err: unknown): err is ForbiddenError => err instanceof ForbiddenError

export const allow = (reason?: string): CheckResult =>
  reason === undefined ? { allowed: true } : { allowed: true, reason }

export const deny = (reason?: string): CheckResult =>
  reason === undefined ? { allowed: false } : { allowed: false, reason }

export const subjectKey = (s: Subject): string => `${s.kind}:${s.id}`
export const resourceKey = (r: Resource): string => `${r.type}:${r.id}`
export const tupleKey = (t: Tuple): string =>
  `${subjectKey(t.subject)}|${t.relation}|${resourceKey(t.resource)}`

export type { AccessControlProvider as default }

export { createRbacProvider, type RbacOptions, type RbacTupleStore } from './rbac.js'
export { withAccessControl, type WithAclOptions } from './wrap.js'
