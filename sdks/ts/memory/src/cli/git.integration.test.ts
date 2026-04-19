// SPDX-License-Identifier: Apache-2.0

import { execFile as execFileCallback } from 'node:child_process'
import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { promisify } from 'node:util'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createGitStore } from '../store/index.js'
import { toPath } from '../store/path.js'
import { gitCommand } from './commands/git.js'

const createdDirs: string[] = []
const execFile = promisify(execFileCallback)

type RunnableCommand = {
  readonly run?: (ctx: { readonly args: Record<string, unknown> }) => Promise<void> | void
  readonly subCommands?: Record<string, RunnableCommand> | ((...args: readonly unknown[]) => unknown)
}

const makeTempDir = async (): Promise<string> => {
  const dir = await mkdtemp(join(tmpdir(), 'memory-git-cli-'))
  createdDirs.push(dir)
  return dir
}

const runGit = async (args: readonly string[], cwd?: string): Promise<string> => {
  const { stdout } = await execFile('git', [...args], {
    ...(cwd !== undefined ? { cwd } : {}),
    encoding: 'utf8',
  })
  return stdout
}

const pickSub = (parent: RunnableCommand, name: string): RunnableCommand => {
  const subs = parent.subCommands
  if (subs === undefined || typeof subs === 'function') {
    throw new Error('expected subCommands')
  }
  const entry = subs[name]
  if (entry === undefined) throw new Error(`no subcommand ${name}`)
  return entry
}

const runGitCommand = async (
  subcommand: string,
  args: Record<string, unknown>,
): Promise<Record<string, unknown>> => {
  const chunks: string[] = []
  const spy = vi
    .spyOn(process.stdout, 'write')
    .mockImplementation(((chunk: string | Uint8Array) => {
      chunks.push(typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8'))
      return true
    }) as typeof process.stdout.write)
  try {
    const cmd = pickSub(gitCommand as unknown as RunnableCommand, subcommand)
    await cmd.run?.({ args })
  } finally {
    spy.mockRestore()
  }
  return JSON.parse(chunks.join('').trim()) as Record<string, unknown>
}

afterEach(async () => {
  while (createdDirs.length > 0) {
    const dir = createdDirs.pop()
    if (dir !== undefined) await rm(dir, { recursive: true, force: true })
  }
  vi.restoreAllMocks()
})

describe('memory git operator commands', () => {
  it('reports dirty working-tree changes via `git diff`', async () => {
    const brainDir = await makeTempDir()
    const store = await createGitStore({ dir: brainDir, init: true })
    try {
      await store.write(toPath('memory/note.md'), Buffer.from('clean', 'utf8'))
    } finally {
      await store.close()
    }

    await writeFile(join(brainDir, 'memory', 'note.md'), 'dirty', 'utf8')

    const payload = await runGitCommand('diff', { brain: brainDir })
    expect(payload['operation']).toBe('diff')
    expect(payload['clean']).toBe(false)
    expect(payload['summary']).toMatchObject({ modified: 1, total: 1 })
    expect(payload['changes']).toEqual([
      expect.objectContaining({ label: 'Modified', path: 'memory/note.md' }),
    ])
  })

  it('exposes commit history and commit file inspection', async () => {
    const brainDir = await makeTempDir()
    const store = await createGitStore({ dir: brainDir, init: true })
    try {
      await store.write(toPath('memory/one.md'), Buffer.from('one', 'utf8'))
      await store.write(toPath('wiki/topic.md'), Buffer.from('two', 'utf8'))
    } finally {
      await store.close()
    }

    const logPayload = await runGitCommand('log', { brain: brainDir, limit: '1' })
    const entries = logPayload['entries'] as Array<Record<string, unknown>>
    expect(entries.length).toBe(1)
    const oid = entries[0]?.['oid']
    expect(typeof oid).toBe('string')

    const showPayload = await runGitCommand('show', { brain: brainDir, commit: oid })
    expect(showPayload['operation']).toBe('show')
    expect(showPayload['commit']).toMatchObject({
      oid,
      files: ['wiki/topic.md'],
    })

    const filesPayload = await runGitCommand('files', { brain: brainDir, commit: oid })
    expect(filesPayload['files']).toEqual(['wiki/topic.md'])
  })

  it('returns healthy verification output and section stats', async () => {
    const brainDir = await makeTempDir()
    const store = await createGitStore({ dir: brainDir, init: true })
    try {
      await store.write(toPath('memory/one.md'), Buffer.from('one', 'utf8'))
      await store.write(toPath('wiki/topic.md'), Buffer.from('two', 'utf8'))
      await store.write(toPath('raw/documents/source.md'), Buffer.from('three', 'utf8'))
    } finally {
      await store.close()
    }

    const verifyPayload = await runGitCommand('verify', { brain: brainDir })
    expect(verifyPayload['ok']).toBe(true)

    const statsPayload = await runGitCommand('stats', { brain: brainDir })
    expect(statsPayload['operation']).toBe('stats')
    expect(statsPayload['sections']).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ name: 'memory', count: 1 }),
        expect.objectContaining({ name: 'wiki', count: 1 }),
        expect.objectContaining({ name: 'raw/documents', count: 1 }),
      ]),
    )
  })

  it('cleans tracked junk files and records the cleanup', async () => {
    const brainDir = await makeTempDir()
    const store = await createGitStore({ dir: brainDir, init: true })
    try {
      await store.write(toPath('raw/documents/dist/app.map'), Buffer.from('map', 'utf8'))
      await store.write(toPath('raw/documents/vendor/bundle.min.js'), Buffer.from('js', 'utf8'))
    } finally {
      await store.close()
    }

    const dryRun = await runGitCommand('clean', { brain: brainDir, scope: 'raw' })
    expect(dryRun['totalFound']).toBe(2)
    expect(dryRun['committed']).toBe(false)

    const applied = await runGitCommand('clean', {
      brain: brainDir,
      scope: 'raw',
      apply: true,
    })
    expect(applied['totalFound']).toBe(2)
    expect(applied['committed']).toBe(true)
    await expect(stat(join(brainDir, 'raw', 'documents', 'dist', 'app.map'))).rejects.toThrow()
    await expect(stat(join(brainDir, 'raw', 'documents', 'vendor', 'bundle.min.js'))).rejects.toThrow()
  })

  it('resets a scoped section and leaves unrelated files alone', async () => {
    const brainDir = await makeTempDir()
    const store = await createGitStore({ dir: brainDir, init: true })
    try {
      await store.write(toPath('memory/keep.md'), Buffer.from('keep', 'utf8'))
      await store.write(toPath('wiki/drop.md'), Buffer.from('drop', 'utf8'))
    } finally {
      await store.close()
    }

    const payload = await runGitCommand('reset', {
      brain: brainDir,
      scope: 'wiki',
      confirm: true,
    })
    expect(payload['deleted']).toBe(1)
    expect(payload['committed']).toBe(true)
    expect(await readFile(join(brainDir, 'memory', 'keep.md'), 'utf8')).toBe('keep')
    await expect(stat(join(brainDir, 'wiki', 'drop.md'))).rejects.toThrow()
  })

  it('reports no-op conflict resolution cleanly when there are no conflicts', async () => {
    const brainDir = await makeTempDir()
    const store = await createGitStore({ dir: brainDir, init: true })
    await store.close()

    const payload = await runGitCommand('resolve', { brain: brainDir, auto: true })
    expect(payload).toMatchObject({
      operation: 'resolve',
      resolved: 0,
      remaining: 0,
      committed: false,
    })
  })
})
