// SPDX-License-Identifier: Apache-2.0

/**
 * Common helpers used by every writer adapter. Handles MCP server spec
 * construction, backup naming, and JSON read/write with directory bootstrap.
 */

import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import type { InstallConfig, McpServerSpec } from '../types.js'
import { MCP_PACKAGE } from '../types.js'

export const buildServerSpec = (config: InstallConfig): McpServerSpec => {
  if (config.mode === 'hosted') {
    const token = config.token ?? ''
    return {
      command: 'npx',
      args: ['-y', MCP_PACKAGE],
      env: {
        JB_TOKEN: token,
        JB_ENDPOINT: config.endpoint,
      },
    }
  }
  return {
    command: 'npx',
    args: ['-y', MCP_PACKAGE],
    env: {
      JB_HOME: config.storage,
    },
  }
}

export const toPlainSpec = (spec: McpServerSpec): Record<string, unknown> => ({
  command: spec.command,
  args: [...spec.args],
  env: { ...spec.env },
})

export const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value)

export type Fs = {
  readonly exists: (p: string) => boolean
  readonly read: (p: string) => string
  readonly write: (p: string, data: string) => void
  readonly rename: (from: string, to: string) => void
  readonly mkdir: (p: string) => void
  readonly now: () => Date
}

export const nodeFs: Fs = {
  exists: (p) => existsSync(p),
  read: (p) => readFileSync(p, 'utf8'),
  write: (p, data) => writeFileSync(p, data, 'utf8'),
  rename: (from, to) => renameSync(from, to),
  mkdir: (p) => mkdirSync(p, { recursive: true }),
  now: () => new Date(),
}

export const backupPath = (target: string, now: Date): string => {
  const pad = (n: number) => n.toString().padStart(2, '0')
  const stamp =
    `${now.getUTCFullYear()}${pad(now.getUTCMonth() + 1)}${pad(now.getUTCDate())}` +
    `${pad(now.getUTCHours())}${pad(now.getUTCMinutes())}${pad(now.getUTCSeconds())}`
  return `${target}.jbpre-${stamp}.bak`
}

export const readJsonOrEmpty = (fs: Fs, target: string): Record<string, unknown> => {
  if (!fs.exists(target)) return {}
  const raw = fs.read(target)
  if (raw.trim().length === 0) return {}
  try {
    const parsed: unknown = JSON.parse(raw)
    return isObject(parsed) ? parsed : {}
  } catch (err) {
    throw new Error(
      `Failed to parse ${target} as JSON: ${err instanceof Error ? err.message : String(err)}`,
    )
  }
}

export const ensureParentDir = (fs: Fs, target: string): void => {
  const dir = path.dirname(target)
  if (!fs.exists(dir)) fs.mkdir(dir)
}

export type WriteJsonResult = {
  readonly created: boolean
  readonly backup?: string
}

export const writeJsonWithBackup = (
  fs: Fs,
  target: string,
  next: Record<string, unknown>,
): WriteJsonResult => {
  ensureParentDir(fs, target)
  const existed = fs.exists(target)
  let backup: string | undefined
  if (existed) {
    backup = backupPath(target, fs.now())
    fs.rename(target, backup)
  }
  const json = `${JSON.stringify(next, null, 2)}\n`
  fs.write(target, json)
  return backup ? { created: !existed, backup } : { created: !existed }
}
