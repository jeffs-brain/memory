// SPDX-License-Identifier: Apache-2.0

import { mkdir, symlink, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { mkdtemp, rm } from 'node:fs/promises'
import { enumerateFiles } from './directory.js'

describe('enumerateFiles', () => {
  let tmp: string

  beforeEach(async () => {
    tmp = await mkdtemp(join(tmpdir(), 'dir-enum-test-'))
  })

  afterEach(async () => {
    await rm(tmp, { recursive: true, force: true })
  })

  it('enumerates markdown files in a flat directory', async () => {
    await writeFile(join(tmp, 'a.md'), '# A\nContent A')
    await writeFile(join(tmp, 'b.md'), '# B\nContent B')
    await writeFile(join(tmp, 'c.md'), '# C\nContent C')

    const result = await enumerateFiles({ directory: tmp })
    expect(result.files).toHaveLength(3)
    expect(result.skipped).toHaveLength(0)
  })

  it('applies glob filter to only match *.md files', async () => {
    await writeFile(join(tmp, 'a.md'), 'markdown')
    await writeFile(join(tmp, 'b.txt'), 'text')

    const result = await enumerateFiles({ directory: tmp, glob: '*.md' })
    expect(result.files).toHaveLength(1)
    expect(result.files[0]?.path).toContain('a.md')
  })

  it('respects recursive=false and only returns top-level files', async () => {
    await writeFile(join(tmp, 'top.md'), 'top')
    await mkdir(join(tmp, 'sub'))
    await writeFile(join(tmp, 'sub', 'nested.md'), 'nested')

    const result = await enumerateFiles({ directory: tmp, recursive: false })
    expect(result.files).toHaveLength(1)
    expect(result.files[0]?.path).toContain('top.md')
  })

  it('respects maxFiles limit and reports skipped reason', async () => {
    await writeFile(join(tmp, 'a.md'), 'a')
    await writeFile(join(tmp, 'b.md'), 'b')
    await writeFile(join(tmp, 'c.md'), 'c')
    await writeFile(join(tmp, 'd.md'), 'd')
    await writeFile(join(tmp, 'e.md'), 'e')

    const result = await enumerateFiles({ directory: tmp, maxFiles: 2 })
    expect(result.files).toHaveLength(2)
    expect(result.skipped.some((s) => s.includes('max files limit'))).toBe(true)
  })

  it('excludes hidden files', async () => {
    await writeFile(join(tmp, '.hidden.md'), 'hidden')
    await writeFile(join(tmp, 'visible.md'), 'visible')

    const result = await enumerateFiles({ directory: tmp })
    expect(result.files).toHaveLength(1)
    expect(result.files[0]?.path).toContain('visible.md')
  })

  it('skips files with unknown extensions', async () => {
    await writeFile(join(tmp, 'image.png'), 'binary')
    await writeFile(join(tmp, 'doc.md'), 'markdown')

    const result = await enumerateFiles({ directory: tmp })
    expect(result.files).toHaveLength(1)
    expect(result.files[0]?.path).toContain('doc.md')
  })

  it('respects .gitignore patterns', async () => {
    await writeFile(join(tmp, '.gitignore'), 'node_modules\n*.log\n')
    await mkdir(join(tmp, 'node_modules'))
    await writeFile(join(tmp, 'node_modules', 'pkg.txt'), 'pkg')
    await writeFile(join(tmp, 'debug.log'), 'logs')
    await writeFile(join(tmp, 'readme.md'), 'readme')

    const result = await enumerateFiles({ directory: tmp })
    expect(result.files).toHaveLength(1)
    expect(result.files[0]?.path).toContain('readme.md')
  })

  it('handles non-existent directory gracefully', async () => {
    const result = await enumerateFiles({ directory: join(tmp, 'nonexistent') })
    expect(result.files).toHaveLength(0)
    expect(result.skipped.length).toBeGreaterThan(0)
  })

  it('walks subdirectories recursively by default', async () => {
    await mkdir(join(tmp, 'sub'))
    await writeFile(join(tmp, 'top.md'), 'top')
    await writeFile(join(tmp, 'sub', 'nested.md'), 'nested')

    const result = await enumerateFiles({ directory: tmp })
    expect(result.files).toHaveLength(2)
  })

  it('skips symlinks and only returns real files', async () => {
    await writeFile(join(tmp, 'real.md'), 'real content')
    const outsideDir = await mkdtemp(join(tmpdir(), 'dir-enum-outside-'))
    await writeFile(join(outsideDir, 'secret.md'), 'secret')
    try {
      await symlink(join(outsideDir, 'secret.md'), join(tmp, 'link.md'))
    } catch {
      // Symlinks may not be supported on all platforms; skip test.
      return
    }

    const result = await enumerateFiles({ directory: tmp })
    expect(result.files).toHaveLength(1)
    expect(result.files[0]?.path).toContain('real.md')
    await rm(outsideDir, { recursive: true, force: true })
  })

  it('maxFiles > 500 still works at library level (cap is in MCP tool layer)', async () => {
    await writeFile(join(tmp, 'a.md'), 'content')

    const result = await enumerateFiles({ directory: tmp, maxFiles: 501 })
    expect(result.files).toHaveLength(1)
  })
})
