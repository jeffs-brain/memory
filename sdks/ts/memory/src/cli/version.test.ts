// SPDX-License-Identifier: Apache-2.0

import { readFile } from 'node:fs/promises'
import { describe, expect, it } from 'vitest'

import { CLI_VERSION, rootCommand } from './main.js'

describe('memory --version', () => {
  it('reports the package.json version', async () => {
    const pkg: unknown = JSON.parse(
      await readFile(new URL('../../package.json', import.meta.url), 'utf8'),
    )
    expect(pkg).toMatchObject({ version: CLI_VERSION })
    const meta =
      typeof rootCommand.meta === 'function' ? await rootCommand.meta() : await rootCommand.meta
    expect(meta?.version).toBe(CLI_VERSION)
  })
})
