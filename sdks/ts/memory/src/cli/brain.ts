// SPDX-License-Identifier: Apache-2.0

/**
 * Brain directory helpers. `openBrain` opens the git-backed store for an
 * existing brain and reads its on-disk config. `initBrain` creates a new
 * brain directory with config + git init.
 */

import { mkdir, readFile, writeFile, access } from 'node:fs/promises'
import { join, resolve } from 'node:path'
import type { GitStore } from '../store/index.js'
import { createGitStore } from '../store/index.js'
import { CliError } from './config.js'

export const CONFIG_FILENAME = 'config.json'

export type BrainConfig = {
  readonly version: number
  readonly createdAt: string
  readonly actorId: string
}

export const defaultBrainConfig = (): BrainConfig => ({
  version: 1,
  createdAt: new Date().toISOString(),
  actorId: 'default',
})

export const readBrainConfig = async (dir: string): Promise<BrainConfig> => {
  const path = join(dir, CONFIG_FILENAME)
  try {
    const raw = await readFile(path, 'utf8')
    const parsed = JSON.parse(raw) as Partial<BrainConfig>
    return {
      version: parsed.version ?? 1,
      createdAt: parsed.createdAt ?? new Date(0).toISOString(),
      actorId: parsed.actorId ?? 'default',
    }
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
      return defaultBrainConfig()
    }
    throw new CliError(
      `failed to read ${path}: ${err instanceof Error ? err.message : String(err)}`,
    )
  }
}

export const writeBrainConfig = async (
  dir: string,
  cfg: BrainConfig,
): Promise<void> => {
  await mkdir(dir, { recursive: true })
  const path = join(dir, CONFIG_FILENAME)
  await writeFile(path, `${JSON.stringify(cfg, null, 2)}\n`, 'utf8')
}

export const initBrain = async (dir: string): Promise<GitStore> => {
  const abs = resolve(dir)
  await mkdir(abs, { recursive: true })
  const cfgPath = join(abs, CONFIG_FILENAME)
  if (!(await pathExists(cfgPath))) {
    await writeBrainConfig(abs, defaultBrainConfig())
  }
  return createGitStore({ dir: abs, init: true })
}

export const openBrain = async (dir: string): Promise<GitStore> => {
  const abs = resolve(dir)
  if (!(await pathExists(abs))) {
    throw new CliError(`brain directory does not exist: ${abs}`)
  }
  return createGitStore({ dir: abs, init: true })
}

const pathExists = async (p: string): Promise<boolean> => {
  try {
    await access(p)
    return true
  } catch {
    return false
  }
}
