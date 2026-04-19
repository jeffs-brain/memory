// SPDX-License-Identifier: Apache-2.0

/**
 * End-to-end integration test for the CLI core:
 *   init -> ingest -> search
 *
 * We drive the underlying helpers directly rather than spawning a child
 * bun process. That keeps the test hermetic and avoids depending on
 * `dist/cli.js` existing at test time.
 */

import { execFile as execFileCallback } from 'node:child_process'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { promisify } from 'node:util'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createIngest } from '../knowledge/ingest.js'
import { noopLogger } from '../llm/index.js'
import type { Reranker } from '../rerank/index.js'
import { createGitStore } from '../store/index.js'
import { toPath } from '../store/path.js'
import { initBrain, openBrain, readBrainConfig, writeBrainConfig } from './brain.js'
import { gitCommand } from './commands/git.js'
import { runSearch } from './search-runner.js'

const createdDirs: string[] = []
const execFile = promisify(execFileCallback)

afterEach(async () => {
  while (createdDirs.length > 0) {
    const dir = createdDirs.pop()
    if (dir !== undefined) await rm(dir, { recursive: true, force: true })
  }
  vi.restoreAllMocks()
})

const makeTempDir = async (): Promise<string> => {
  const dir = await mkdtemp(join(tmpdir(), 'memory-it-'))
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

const makePathPreferringReranker = (pathFragment: string): Reranker => ({
  name: () => 'test-reranker',
  rerank: async ({ documents }) =>
    documents
      .map((document, index) => ({
        id: document.id,
        index,
        score: document.id.includes(pathFragment) ? 100 : documents.length - index,
      }))
      .sort((left, right) => right.score - left.score),
})

type RunnableCommand = {
  readonly run?: (ctx: { readonly args: Record<string, unknown> }) => Promise<void> | void
  readonly subCommands?: Record<string, RunnableCommand> | ((...args: readonly unknown[]) => unknown)
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

describe('memory init + ingest + search', () => {
  it('initialises a brain, ingests a file, and surfaces it via BM25 search', async () => {
    const root = await makeTempDir()
    const brainDir = join(root, 'brain')

    // init
    const initStore = await initBrain(brainDir)
    await initStore.close()

    const cfg = await readBrainConfig(brainDir)
    expect(cfg.version).toBe(1)

    // ingest
    const filePath = join(root, 'hedgehog.md')
    await writeFile(
      filePath,
      '# Hedgehogs\nHedgehogs are small spiny mammals native to Europe.',
      'utf8',
    )

    const ingestStore = await openBrain(brainDir)
    try {
      const ingest = createIngest({ store: ingestStore, logger: noopLogger })
      const result = await ingest(
        await (await import('node:fs/promises')).readFile(filePath),
        { name: 'hedgehog.md' },
      )
      expect(result.path).toContain('raw/documents/')
    } finally {
      await ingestStore.close()
    }

    // search (bm25, no embedder needed)
    const searchStore = await openBrain(brainDir)
    try {
      const hits = await runSearch('hedgehog', {
        store: searchStore,
        limit: 5,
        mode: 'bm25',
      })
      expect(hits.length).toBeGreaterThan(0)
      const top = hits[0]
      expect(top).toBeDefined()
      if (top !== undefined) {
        expect(top.path).toContain('hedgehog')
        expect(top.snippet.toLowerCase()).toContain('hedgehog')
      }
    } finally {
      await searchStore.close()
    }
  })

  it('handles recommendation-style queries with punctuation via the shared search path', async () => {
    const root = await makeTempDir()
    const brainDir = join(root, 'brain')

    const initStore = await initBrain(brainDir)
    await initStore.close()

    const store = await openBrain(brainDir)
    try {
      await store.write(
        toPath('memory/global/user-cocktail-preference.md'),
        Buffer.from(
          [
            '---',
            'name: User Cocktail Preference',
            "description: User already made a classic Pimm's Cup and wants a twist on a familiar cocktail.",
            'type: user',
            'scope: global',
            '---',
            '',
            "I like the idea of the Pimm's Cup with a Twist. For a cocktail suggestion, I want a twist on a familiar cocktail rather than a basic recipe.",
          ].join('\n'),
          'utf8',
        ),
      )

      const hits = await runSearch(
        "I've been thinking about making a cocktail for an upcoming get-together, but I'm not sure which one to choose. Any suggestions?",
        {
          store,
          limit: 5,
          mode: 'bm25',
        },
      )

      expect(hits.length).toBeGreaterThan(0)
      expect(hits[0]?.path).toContain('user-cocktail-preference')
    } finally {
      await store.close()
    }
  })

  it('prioritises durable user preference notes for recommendation queries', async () => {
    const root = await makeTempDir()
    const brainDir = join(root, 'brain')

    const initStore = await initBrain(brainDir)
    await initStore.close()

    const store = await openBrain(brainDir)
    try {
      await store.write(
        toPath('memory/global/user-hotel-preference.md'),
        Buffer.from(
          [
            '---',
            'name: User Hotel Preference',
            'description: User prefers hotels with great views and standout amenities.',
            'type: user',
            'scope: global',
            '---',
            '',
            'The user prefers hotels with great views and unique features such as rooftop pools or balcony hot tubs.',
            '',
            'Evidence: Besides great views, I also like hotels with unique features, such as a rooftop pool or a hot tub on the balcony.',
          ].join('\n'),
          'utf8',
        ),
      )
      await store.write(
        toPath('memory/project/demo/reference-miami-hotels.md'),
        Buffer.from(
          [
            '---',
            'name: Miami Hotels',
            'description: Generic Miami hotel options near the airport.',
            'type: reference',
            'scope: project',
            '---',
            '',
            'Budget and airport hotel recommendations in Miami.',
          ].join('\n'),
          'utf8',
        ),
      )

      const hits = await runSearch('Can you suggest a hotel for my upcoming trip to Miami?', {
        store,
        limit: 5,
        mode: 'bm25',
      })

      expect(hits.length).toBeGreaterThan(0)
      expect(hits[0]?.path).toContain('user-hotel-preference')
    } finally {
      await store.close()
    }
  })

  it('applies hybrid-rerank mode even when retrieval falls back to BM25', async () => {
    const root = await makeTempDir()
    const brainDir = join(root, 'brain')

    const initStore = await initBrain(brainDir)
    await initStore.close()

    const store = await openBrain(brainDir)
    try {
      await store.write(
        toPath('memory/project/demo/hotel-a.md'),
        Buffer.from('Hotel hotel hotel in Miami with a pool.', 'utf8'),
      )
      await store.write(
        toPath('memory/project/demo/hotel-b.md'),
        Buffer.from('Hotel in Miami with a balcony hot tub.', 'utf8'),
      )

      const baseline = await runSearch('hotel miami', {
        store,
        limit: 2,
        mode: 'bm25',
      })
      expect(baseline[0]?.path).toContain('hotel-a')

      const reranked = await runSearch('hotel miami', {
        store,
        limit: 2,
        mode: 'hybrid-rerank',
        reranker: makePathPreferringReranker('hotel-b'),
      })
      expect(reranked[0]?.path).toContain('hotel-b')
    } finally {
      await store.close()
    }
  })

  it('writeBrainConfig + readBrainConfig round-trip', async () => {
    const dir = await makeTempDir()
    await writeBrainConfig(dir, {
      version: 1,
      createdAt: '2026-04-17T00:00:00.000Z',
      actorId: 'alex',
    })
    const loaded = await readBrainConfig(dir)
    expect(loaded.actorId).toBe('alex')
    expect(loaded.createdAt).toBe('2026-04-17T00:00:00.000Z')
  })

  it('runs `git sync` against a configured remote', async () => {
    const root = await makeTempDir()
    const brainDir = join(root, 'brain')
    const remoteDir = join(root, 'remote.git')
    const cloneDir = join(root, 'clone')
    const freshDir = join(root, 'fresh')

    await runGit(['init', '--bare', '--initial-branch=main', remoteDir])

    const store = await createGitStore({ dir: brainDir, init: true, remoteUrl: remoteDir })
    try {
      await store.write(toPath('memory/local.md'), Buffer.from('local seed', 'utf8'))
      await store.push()
      await store.write(toPath('memory/pending.md'), Buffer.from('pending push', 'utf8'))
    } finally {
      await store.close()
    }

    await runGit(['clone', remoteDir, cloneDir])
    await runGit(['config', 'user.name', 'Remote Tester'], cloneDir)
    await runGit(['config', 'user.email', 'remote@example.com'], cloneDir)
    await mkdir(join(cloneDir, 'memory'), { recursive: true })
    await writeFile(join(cloneDir, 'memory', 'remote.md'), 'remote change', 'utf8')
    await runGit(['add', '.'], cloneDir)
    await runGit(['commit', '-m', 'remote change'], cloneDir)
    await runGit(['push'], cloneDir)

    const stdout: string[] = []
    vi.spyOn(process.stdout, 'write').mockImplementation(((chunk: string | Uint8Array) => {
      stdout.push(typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8'))
      return true
    }) as typeof process.stdout.write)

    const sync = pickSub(gitCommand as unknown as RunnableCommand, 'sync')
    await sync.run?.({ args: { brain: brainDir } })

    await runGit(['clone', remoteDir, freshDir])
    expect(await readFile(join(freshDir, 'memory', 'pending.md'), 'utf8')).toBe('pending push')
    expect(await readFile(join(freshDir, 'memory', 'remote.md'), 'utf8')).toBe('remote change')

    expect(JSON.parse(stdout.join('').trim())).toMatchObject({
      brain: brainDir,
      operation: 'sync',
      ok: true,
    })
  })
})
