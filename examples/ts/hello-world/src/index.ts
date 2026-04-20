// SPDX-License-Identifier: Apache-2.0

/**
 * Minimal hello world for @jeffs-brain/memory.
 *
 * 1. Create a filesystem-backed brain store.
 * 2. Ingest a markdown doc: chunk it, persist it via the store, then
 *    upsert the chunks into a SQLite search index (BM25 only — no
 *    embedder wired in this example).
 * 3. Run a BM25 query and print the top-k matches.
 */

import { mkdir, readFile } from 'node:fs/promises'
import { basename, resolve } from 'node:path'
import {
  type SqliteSearchIndexChunk,
  chunkMarkdown,
  createFsStore,
  createSearchIndex,
  toPath,
} from '@jeffs-brain/memory'

const BRAIN_ID = 'hello-world'

async function main(): Promise<void> {
  const dataRoot = resolve('./data')
  const brainRoot = resolve(dataRoot, BRAIN_ID)
  await mkdir(brainRoot, { recursive: true })

  // 1. Store. Lives under ./data/<brain>/ on disk.
  const store = await createFsStore({ root: brainRoot })

  // 2. Search index. SQLite file alongside the store. vectorDim matches
  //    the package default; we never supply embeddings so its value is
  //    not load-bearing.
  const index = await createSearchIndex({ dbPath: resolve(brainRoot, 'search.db') })

  try {
    // 3. Ingest docs/hedgehogs.md.
    const docPath = resolve('./docs/hedgehogs.md')
    const raw = await readFile(docPath)
    const storedPath = `raw/documents/${basename(docPath)}`
    await store.batch({ reason: 'ingest' }, async (batch) => {
      await batch.write(toPath(storedPath), raw)
    })

    const chunks = chunkMarkdown(raw.toString('utf8'))
    const indexChunks: SqliteSearchIndexChunk[] = chunks.map((chunk) => ({
      id: `${BRAIN_ID}:${basename(docPath)}:${chunk.ordinal}`,
      path: storedPath,
      ordinal: chunk.ordinal,
      title: chunk.headingPath.join(' > '),
      content: chunk.content,
    }))
    index.upsertChunks(indexChunks)
    console.log(`Ingested ${storedPath} (${chunks.length} chunks, ${raw.byteLength} bytes)`)

    // 4. Search.
    const query = 'where do hedgehogs live?'
    const topK = 3
    const results = index.searchBM25(query, topK)

    console.log(`Top ${results.length} results for "${query}":`)
    for (const [i, hit] of results.entries()) {
      const snippet = hit.chunk.content.replace(/\s+/g, ' ').trim().slice(0, 160)
      console.log(`${i + 1}. [${hit.score.toFixed(3)}] ${hit.chunk.path}`)
      console.log(`   ${snippet}${hit.chunk.content.length > 160 ? '...' : ''}`)
    }
  } finally {
    await index.close()
    await store.close()
  }
}

main().catch((err: unknown) => {
  console.error(err)
  process.exit(1)
})
