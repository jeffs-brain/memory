// SPDX-License-Identifier: Apache-2.0

export type { DispatchFn, MutationHook, MutationHookOptions, PathMatcher } from './types.js'
export {
  createMutationHook,
  defaultPathMatcher,
  globPathMatcher,
  prefixPathMatcher,
} from './mutation-hook.js'
