// SPDX-License-Identifier: Apache-2.0

import { createHash } from 'node:crypto'
import { createReadStream, createWriteStream } from 'node:fs'
import { access, mkdir, rename, rm } from 'node:fs/promises'
import * as nodeFs from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import * as git from 'isomorphic-git'
import http from 'isomorphic-git/http/node'

type FetchLike = (input: string, init?: RequestInit) => Promise<Response>

export type LMEUpstreamDatasetName = 'oracle' | 's' | 'm'
export type LMEUpstreamBundleName = 'legacy' | 'cleaned'

export type LMEUpstreamDatasetSpec = {
  readonly bundle: LMEUpstreamBundleName
  readonly split: LMEUpstreamDatasetName
  readonly filename: string
  readonly url: string
  readonly sha256: string
  readonly sizeBytes: number
}

export type LMEUpstreamBundleSpec = {
  readonly bundle: LMEUpstreamBundleName
  readonly datasetId: string
  readonly repoUrl: string
  readonly files: Readonly<Record<LMEUpstreamDatasetName, LMEUpstreamDatasetSpec>>
}

export const LONG_MEM_EVAL_UPSTREAM_DATASETS: Readonly<
  Record<LMEUpstreamBundleName, LMEUpstreamBundleSpec>
> = {
  legacy: {
    bundle: 'legacy',
    datasetId: 'xiaowu0162/LongMemEval',
    repoUrl: 'https://huggingface.co/datasets/xiaowu0162/LongMemEval',
    files: {
      oracle: {
        bundle: 'legacy',
        split: 'oracle',
        filename: 'longmemeval_oracle',
        url: 'https://huggingface.co/datasets/xiaowu0162/LongMemEval/resolve/main/longmemeval_oracle',
        sha256: '821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c',
        sizeBytes: 15_388_478,
      },
      s: {
        bundle: 'legacy',
        split: 's',
        filename: 'longmemeval_s',
        url: 'https://huggingface.co/datasets/xiaowu0162/LongMemEval/resolve/main/longmemeval_s',
        sha256: '08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894',
        sizeBytes: 278_025_796,
      },
      m: {
        bundle: 'legacy',
        split: 'm',
        filename: 'longmemeval_m',
        url: 'https://huggingface.co/datasets/xiaowu0162/LongMemEval/resolve/main/longmemeval_m',
        sha256: 'fb5413e3b077c62927daab794836991a2fcfa61ceacab57dc679fb02daaff2d9',
        sizeBytes: 2_745_274_681,
      },
    },
  },
  cleaned: {
    bundle: 'cleaned',
    datasetId: 'xiaowu0162/longmemeval-cleaned',
    repoUrl: 'https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned',
    files: {
      oracle: {
        bundle: 'cleaned',
        split: 'oracle',
        filename: 'longmemeval_oracle.json',
        url: 'https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json',
        sha256: '821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c',
        sizeBytes: 15_388_478,
      },
      s: {
        bundle: 'cleaned',
        split: 's',
        filename: 'longmemeval_s_cleaned.json',
        url: 'https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json',
        sha256: 'd6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442',
        sizeBytes: 277_383_467,
      },
      m: {
        bundle: 'cleaned',
        split: 'm',
        filename: 'longmemeval_m_cleaned.json',
        url: 'https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json',
        sha256: '9d79e5524794a2e6900a3aa9cb7d9152c5a3e8319c9a87c25494ba1eacee495f',
        sizeBytes: 2_737_100_077,
      },
    },
  },
}

export const LONG_MEM_EVAL_OFFICIAL_REPO_URL =
  'https://github.com/xiaowu0162/LongMemEval'
export const LONG_MEM_EVAL_OFFICIAL_REPO_REF =
  '982fbd7045c9977e9119b5424cab0d7790d19413'

export type LMEUpstreamFetchMetadata = {
  readonly bundle: LMEUpstreamBundleName
  readonly split: LMEUpstreamDatasetName
  readonly url: string
  readonly path: string
  readonly fetchedAt: string
  readonly status: number
  readonly sizeBytes: number
  readonly sha256: string
}

export type LMEUpstreamFetchResult = {
  readonly bundle: LMEUpstreamBundleName
  readonly dir: string
  readonly files: Readonly<Record<LMEUpstreamDatasetName, LMEUpstreamFetchMetadata>>
}

export type FetchUpstreamDatasetOptions = {
  readonly dir: string
  readonly splits?: readonly LMEUpstreamDatasetName[]
  readonly fetcher?: FetchLike
  readonly now?: () => Date
}

export type LMEOfficialRepoFetchResult = {
  readonly dir: string
  readonly url: string
  readonly ref: string
  readonly commit: string
}

export type FetchOfficialRepoOptions = {
  readonly dir: string
  readonly url?: string
  readonly ref?: string
}

export const resolveUpstreamDatasetPath = (
  bundle: LMEUpstreamBundleName,
  split: LMEUpstreamDatasetName,
  dir: string,
): string => join(resolve(dir), LONG_MEM_EVAL_UPSTREAM_DATASETS[bundle].files[split].filename)

export const fetchUpstreamDatasets = async (
  bundle: LMEUpstreamBundleName,
  opts: FetchUpstreamDatasetOptions,
): Promise<LMEUpstreamFetchResult> => {
  const spec = LONG_MEM_EVAL_UPSTREAM_DATASETS[bundle]
  const fetcher = resolveFetch(opts.fetcher)
  const now = opts.now ?? (() => new Date())
  const dir = resolve(opts.dir)
  await mkdir(dir, { recursive: true })
  const splits = opts.splits ?? (['oracle', 's', 'm'] as const)
  const files = {} as Record<LMEUpstreamDatasetName, LMEUpstreamFetchMetadata>

  for (const split of splits) {
    const file = spec.files[split]
    const destPath = join(dir, file.filename)
    const existingSha = await fileSha256(destPath)
    if (existingSha === file.sha256) {
      files[split] = {
        bundle,
        split,
        url: file.url,
        path: destPath,
        fetchedAt: now().toISOString(),
        status: 200,
        sizeBytes: file.sizeBytes,
        sha256: existingSha,
      }
      continue
    }

    const response = await fetcher(file.url, { method: 'GET' })
    if (!response.ok || response.body === null) {
      throw new Error(
        `fetchUpstreamDatasets: failed to fetch ${file.url}: ${response.status} ${response.statusText}`,
      )
    }
    const tmpPath = `${destPath}.tmp`
    await rm(tmpPath, { force: true })
    await mkdir(dirname(destPath), { recursive: true })

    const hash = createHash('sha256')
    const sink = createWriteStream(tmpPath)
    const source = Readable.fromWeb(response.body as ReadableStream<Uint8Array>)
    source.on('data', (chunk: Buffer | string) => {
      hash.update(chunk)
    })
    await pipeline(source, sink)
    const sha256 = hash.digest('hex')
    if (sha256 !== file.sha256) {
      await rm(tmpPath, { force: true })
      throw new Error(
        `fetchUpstreamDatasets: sha256 mismatch for ${file.filename}: got ${sha256}, expected ${file.sha256}`,
      )
    }
    await rm(destPath, { force: true })
    await rename(tmpPath, destPath)
    files[split] = {
      bundle,
      split,
      url: file.url,
      path: destPath,
      fetchedAt: now().toISOString(),
      status: response.status,
      sizeBytes: file.sizeBytes,
      sha256,
    }
  }

  return { bundle, dir, files }
}

export const fetchOfficialRepo = async (
  opts: FetchOfficialRepoOptions,
): Promise<LMEOfficialRepoFetchResult> => {
  const dir = resolve(opts.dir)
  const url = opts.url ?? LONG_MEM_EVAL_OFFICIAL_REPO_URL
  const ref = opts.ref ?? LONG_MEM_EVAL_OFFICIAL_REPO_REF
  await mkdir(dirname(dir), { recursive: true })

  if (!(await pathExists(join(dir, '.git')))) {
    await git.clone({
      fs: nodeFs,
      http,
      dir,
      url,
      ref,
      singleBranch: true,
      depth: 1,
    })
  } else {
    await git.fetch({
      fs: nodeFs,
      http,
      dir,
      url,
      ref,
      singleBranch: true,
      depth: 1,
      prune: true,
      tags: false,
    })
    await git.checkout({
      fs: nodeFs,
      dir,
      ref,
      force: true,
    })
  }

  const commit = await git.resolveRef({
    fs: nodeFs,
    dir,
    ref: 'HEAD',
  })
  return { dir, url, ref, commit }
}

const resolveFetch = (fetcher: FetchLike | undefined): FetchLike => {
  if (fetcher !== undefined) return fetcher
  if (typeof globalThis.fetch === 'function') {
    return globalThis.fetch.bind(globalThis) as FetchLike
  }
  throw new Error(
    'fetchUpstreamDatasets: no global fetch is available; inject one via the `fetcher` option',
  )
}

const fileSha256 = async (filePath: string): Promise<string | undefined> => {
  if (!(await pathExists(filePath))) return undefined
  const hash = createHash('sha256')
  const stream = createReadStream(filePath)
  for await (const chunk of stream) {
    hash.update(chunk as Buffer)
  }
  return hash.digest('hex')
}

const pathExists = async (filePath: string): Promise<boolean> => {
  try {
    await access(filePath)
    return true
  } catch {
    return false
  }
}
