// SPDX-License-Identifier: Apache-2.0

/**
 * Lightweight logger contract shared by the runtime and the bootstrap
 * helpers. Lives in its own module so other modules can depend on the
 * type without pulling in the rest of `runtime.ts`.
 */

export type RuntimeLogger = {
  info(message: string, ctx?: Record<string, unknown>): void
  warn(message: string, ctx?: Record<string, unknown>): void
  error(message: string, ctx?: Record<string, unknown>): void
}

export const noopRuntimeLogger: RuntimeLogger = {
  info: () => {},
  warn: () => {},
  error: () => {},
}
