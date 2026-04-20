// SPDX-License-Identifier: Apache-2.0

import { createHash } from 'node:crypto'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

import type { Provider } from '../llm/index.js'
import { CONTEXTUAL_PREFIX_SYSTEM_PROMPT } from './prompts.js'
import type { ContextualPrefixBuilder, ContextualPrefixRequest } from './types.js'

const CONTEXTUAL_PREFIX_MAX_TOKENS = 120

export type ContextualPrefixBuilderConfig = {
  readonly provider?: Provider
  readonly model?: string
  readonly cacheDir?: string
  readonly maxTokens?: number
}

type CacheRecord = {
  readonly prefix?: unknown
}

class LiveContextualPrefixBuilder implements ContextualPrefixBuilder {
  private readonly provider: Provider | undefined
  private readonly model: string
  private readonly cacheDir: string | undefined
  private readonly maxTokens: number
  private readonly cache = new Map<string, string>()
  private readonly pending = new Map<string, Promise<string>>()

  constructor(cfg: ContextualPrefixBuilderConfig) {
    this.provider = cfg.provider
    this.model = cfg.model ?? ''
    this.cacheDir = cfg.cacheDir?.trim() || undefined
    this.maxTokens = cfg.maxTokens ?? CONTEXTUAL_PREFIX_MAX_TOKENS
  }

  enabled(): boolean {
    return this.provider !== undefined
  }

  modelName(): string {
    return this.model
  }

  async buildPrefix(args: ContextualPrefixRequest, signal?: AbortSignal): Promise<string> {
    if (!this.enabled()) return ''
    const factBody = args.factBody.trim()
    if (factBody === '') return ''

    const key = cacheKey(this.model, args.sessionId, factBody)
    const cached = this.cache.get(key)
    if (cached !== undefined) return cached

    const pending = this.pending.get(key)
    if (pending !== undefined) return pending

    const task = this.loadOrBuildPrefix(key, args, signal).finally(() => {
      this.pending.delete(key)
    })
    this.pending.set(key, task)
    return task
  }

  private async loadOrBuildPrefix(
    key: string,
    args: ContextualPrefixRequest,
    signal?: AbortSignal,
  ): Promise<string> {
    const fileCached = await this.readCache(key)
    if (fileCached !== undefined) {
      this.cache.set(key, fileCached)
      return fileCached
    }

    const provider = this.provider
    if (provider === undefined) return ''

    const sessionSummary =
      args.sessionSummary.trim() !== ''
        ? args.sessionSummary.trim()
        : '(no session header supplied)'
    const response = await provider.complete(
      {
        ...(this.model !== '' ? { model: this.model } : {}),
        messages: [
          {
            role: 'system',
            content: CONTEXTUAL_PREFIX_SYSTEM_PROMPT,
          },
          {
            role: 'user',
            content: `Session header:\n${sessionSummary}\n\nFact body:\n${factBodySlice(args.factBody)}\n`,
          },
        ],
        maxTokens: this.maxTokens,
        temperature: 0,
      },
      signal,
    )
    const prefix = sanitisePrefix(response.content)
    if (prefix === '') return ''

    this.cache.set(key, prefix)
    await this.writeCache(key, prefix).catch(() => undefined)
    return prefix
  }

  private async readCache(key: string): Promise<string | undefined> {
    const cacheDir = this.cacheDir
    if (cacheDir === undefined) return undefined
    try {
      const raw = await readFile(cachePath(cacheDir, key), 'utf8')
      const parsed = JSON.parse(raw) as CacheRecord
      return typeof parsed.prefix === 'string' ? parsed.prefix : undefined
    } catch {
      return undefined
    }
  }

  private async writeCache(key: string, prefix: string): Promise<void> {
    const cacheDir = this.cacheDir
    if (cacheDir === undefined) return
    const filePath = cachePath(cacheDir, key)
    await mkdir(join(cacheDir, key.slice(0, 2)), { recursive: true })
    await writeFile(
      filePath,
      JSON.stringify({
        prefix,
        model: this.model,
        written: new Date().toISOString(),
      }),
      'utf8',
    )
  }
}

const factBodySlice = (value: string): string => value.slice(0, 4096)

const sanitisePrefix = (value: string): string => {
  let trimmed = value.trim()
  if (trimmed.toLowerCase().startsWith('context:')) {
    trimmed = trimmed.slice('context:'.length).trim()
  }
  return trimmed.replace(/\s+/g, ' ')
}

const cacheKey = (model: string, sessionId: string | undefined, factBody: string): string =>
  createHash('sha256')
    .update('v1\n')
    .update(`model=${model}\n`)
    .update(`session=${sessionId ?? ''}\n`)
    .update(`body=${factBody}`)
    .digest('hex')

const cachePath = (cacheDir: string, key: string): string =>
  join(cacheDir, key.slice(0, 2), `${key}.json`)

export const createContextualPrefixBuilder = (
  cfg: ContextualPrefixBuilderConfig,
): ContextualPrefixBuilder | undefined => {
  if (cfg.provider === undefined) return undefined
  return new LiveContextualPrefixBuilder(cfg)
}
