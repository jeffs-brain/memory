// SPDX-License-Identifier: Apache-2.0

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { buildTools, createMemoryRuntime } from '../src/index.js'

describe('memory_ingest_url', () => {
  let brainRoot: string

  beforeEach(async () => {
    brainRoot = await mkdtemp(join(tmpdir(), 'memory-pi-ingest-url-'))
  })

  afterEach(async () => {
    await rm(brainRoot, { recursive: true, force: true })
  })

  it.each([
    ['http://127.0.0.1:9/internal', /non-public address/],
    ['http://169.254.169.254/latest/meta-data/', /non-public address/],
    ['http://[::1]/', /non-public address/],
    ['file:///etc/passwd', /unsupported scheme/],
  ])('refuses %s', async (url, reason) => {
    const runtime = await createMemoryRuntime({
      brainRoot,
      brainId: 'ingest-url',
      store: { kind: 'fs' },
      embedder: { kind: 'off' },
    })
    try {
      const tools = buildTools(runtime)
      await expect(tools.ingest_url.execute('call-1', { url })).rejects.toThrow(reason)
    } finally {
      await runtime.close()
    }
  })
})
