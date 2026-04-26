// SPDX-License-Identifier: Apache-2.0

/**
 * ACL primitives for `@jeffs-brain/memory-react-native`.
 *
 * Defines the `AccessControlProvider` contract used by the memory core to
 * authorise reads and writes against a `Store`.
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

export type Action = 'read' | 'write' | 'delete' | 'admin' | 'export'

export type CheckResult = {
  readonly allowed: boolean
  readonly reason?: string
}

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

export type AccessControlProvider = {
  readonly name: string
  check(subject: Subject, action: Action, resource: Resource): Promise<CheckResult>
  write?(req: WriteTuplesRequest): Promise<void>
  read?(query: ReadTuplesQuery): Promise<readonly Tuple[]>
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
