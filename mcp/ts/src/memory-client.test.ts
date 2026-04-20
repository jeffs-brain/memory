// SPDX-License-Identifier: Apache-2.0

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import type { LocalConfig } from './config.js'
import { createMemoryClient } from './memory-client.js'

type SearchResponse = {
  readonly hits: Array<{
    readonly path: string
  }>
}

type RecallResponse = {
  readonly chunks: Array<{
    readonly path: string
  }>
}

const sleep = async (ms: number): Promise<void> =>
  await new Promise((resolve) => setTimeout(resolve, ms))

const tempRoots: string[] = []

const makeLocalClient = async () => {
  const root = await mkdtemp(join(tmpdir(), 'memory-mcp-local-'))
  tempRoots.push(root)
  const cfg: LocalConfig = {
    kind: 'local',
    brainRoot: root,
    defaultBrain: 'test-brain',
    ollamaBaseUrl: 'http://127.0.0.1:9',
  }
  return createMemoryClient(cfg)
}

afterEach(async () => {
  while (tempRoots.length > 0) {
    const root = tempRoots.pop()
    if (root !== undefined) await rm(root, { recursive: true, force: true })
  }
})

describe('createMemoryClient (local)', () => {
  it('applies strict scope filtering to search and recall', async () => {
    const client = await makeLocalClient()
    await client.remember({
      title: 'Project conventions',
      content: 'Project workflow notes for this repo.',
      path: 'memory/project/tenant-a/project-workflow.md',
    })
    await client.remember({
      title: 'Global conventions',
      content: 'Global workflow notes for every repo.',
      path: 'memory/global/global-workflow.md',
    })

    const projectSearch = (await client.search({
      query: 'workflow',
      scope: 'project',
      topK: 5,
    })) as SearchResponse
    const projectRecall = (await client.recall({
      query: 'workflow',
      scope: 'project',
      topK: 5,
    })) as RecallResponse

    expect(projectSearch.hits.map((hit) => hit.path)).toEqual([
      'memory/project/tenant-a/project-workflow.md',
    ])
    expect(projectRecall.chunks.map((chunk) => chunk.path)).toEqual([
      'memory/project/tenant-a/project-workflow.md',
    ])

    await client.close()
  })

  it('sorts local search results by recency when requested', async () => {
    const client = await makeLocalClient()
    await client.remember({
      title: 'Older release notes',
      content: 'Release checklist and deployment steps.',
      path: 'memory/project/tenant-a/release-older.md',
    })
    await sleep(20)
    await client.remember({
      title: 'Newer release notes',
      content: 'Release checklist and deployment steps.',
      path: 'memory/project/tenant-a/release-newer.md',
    })

    const results = (await client.search({
      query: 'release checklist',
      scope: 'project',
      sort: 'recency',
      topK: 2,
    })) as SearchResponse

    expect(results.hits.map((hit) => hit.path)).toEqual([
      'memory/project/tenant-a/release-newer.md',
      'memory/project/tenant-a/release-older.md',
    ])

    await client.close()
  })
})
