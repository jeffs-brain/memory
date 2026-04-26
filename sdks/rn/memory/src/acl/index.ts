// SPDX-License-Identifier: Apache-2.0

export type {
  AccessControlProvider,
  Action,
  CheckResult,
  ReadTuplesQuery,
  Resource,
  ResourceType,
  Subject,
  SubjectKind,
  Tuple,
  WriteTuplesRequest,
} from './primitives.js'
export {
  AccessControlError,
  ForbiddenError,
  allow,
  deny,
  isForbidden,
  resourceKey,
  subjectKey,
  tupleKey,
} from './primitives.js'
export { createRbacProvider, type RbacOptions, type RbacTupleStore } from './rbac.js'
export { withAccessControl, type WithAclOptions } from './wrap.js'
