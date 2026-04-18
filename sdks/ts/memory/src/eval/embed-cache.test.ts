// SPDX-License-Identifier: Apache-2.0

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { LMEEmbedCache } from './embed-cache.js'

const tempDirs: string[] = []

afterEach(async () => {
  await Promise.all(
    tempDirs.splice(0).map((dir) =>
      rm(dir, { recursive: true, force: true }),
    ),
  )
})

describe('LMEEmbedCache', () => {
  it('returns undefined on cache misses and round-trips stored vectors', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'lme-embed-cache-'))
    tempDirs.push(dir)
    const cache = await LMEEmbedCache.open({
      path: join(dir, 'cache.sqlite'),
    })

    expect(cache.get('test-model', 'missing text')).toBeUndefined()

    cache.put('test-model', 'hello world', [0.25, 0.5, 0.75])
    expect(cache.get('test-model', 'hello world')).toEqual([0.25, 0.5, 0.75])

    cache.close()
  })
})
