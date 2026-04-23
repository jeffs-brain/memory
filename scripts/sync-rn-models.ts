#!/usr/bin/env bun

import { createHash } from 'node:crypto'
import { createReadStream, createWriteStream } from 'node:fs'
import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import {
  type ModelFileManifest,
  type ModelManifest,
  createHostedModelManifests,
  createUpstreamModelManifests,
  listModelFiles,
  modelBundleSizeBytes,
} from '../sdks/rn/memory/src/model/registry.ts'

type SyncOptions = {
  readonly bucket: string
  readonly prefix?: string
  readonly baseUrl: string
  readonly project?: string
  readonly location?: string
  readonly cacheControl: string
  readonly createBucket: boolean
  readonly makePublic: boolean
  readonly dryRun: boolean
  readonly selectedModelIds?: ReadonlySet<string>
  readonly manifestOut?: string
  readonly tempRoot?: string
  readonly huggingFaceToken?: string
}

type UploadedTextFile = {
  readonly destination: string
  readonly content: string
  readonly contentType: string
  readonly cacheControl?: string
}

const GEMMA_TERMS_URL = 'https://ai.google.dev/gemma/terms'
const GEMMA_PROHIBITED_USE_POLICY_URL = 'https://ai.google.dev/gemma/prohibited_use_policy'
const GEMMA_NOTICE_TEXT =
  'Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms\n'

const usage = (): never => {
  console.error(
    [
      'Usage: bun scripts/sync-rn-models.ts --bucket=gs://bucket-name [options]',
      '',
      'Options:',
      '  --prefix=react-native        Object prefix under the bucket',
      '  --base-url=https://...       Public base URL for hosted manifests',
      '  --project=<gcp-project>      GCP project for bucket commands',
      '  --location=eu                Bucket location when creating a bucket',
      '  --create-bucket              Create the bucket when missing',
      '  --make-public                Grant allUsers objectViewer on the bucket',
      '  --models=id1,id2             Mirror only the selected model ids',
      '  --manifest-out=path          Write hosted manifest JSON to this path',
      '  --temp-root=/tmp             Override the temporary staging directory',
      '  --dry-run                    Print actions without downloading or uploading',
      '',
      'Env fallbacks:',
      '  RN_MODEL_BUCKET, RN_MODEL_PREFIX, RN_MODEL_BASE_URL, RN_MODEL_PROJECT,',
      '  RN_MODEL_LOCATION, RN_MODEL_MANIFEST_OUT, RN_MODEL_TEMP_ROOT,',
      '  HF_TOKEN, HUGGING_FACE_HUB_TOKEN, or ~/.cache/huggingface/token',
    ].join('\n'),
  )
  process.exit(1)
}

const getFlagValue = (arg: string, prefix: string): string | undefined => {
  if (!arg.startsWith(`${prefix}=`)) return undefined
  return arg.slice(prefix.length + 1)
}

const normaliseBucket = (value: string): string => {
  const trimmed = value.trim()
  if (trimmed === '') throw new Error('bucket must not be empty')
  return trimmed.startsWith('gs://') ? trimmed : `gs://${trimmed}`
}

const bucketNameFrom = (bucket: string): string =>
  bucket.replace(/^gs:\/\//, '').replace(/\/+$/, '')

const buildDestination = (
  options: SyncOptions,
  manifest: ModelManifest,
  file: ModelFileManifest,
): string => {
  const parts = [options.bucket, options.prefix ?? '', manifest.id, file.filename]
  return parts
    .map((part, index) => (index === 0 ? part.replace(/\/+$/, '') : part.replace(/^\/+|\/+$/g, '')))
    .filter((part) => part !== '')
    .join('/')
}

const runCommand = async (command: string, args: readonly string[]): Promise<void> => {
  const proc = Bun.spawn([command, ...args], {
    stdout: 'inherit',
    stderr: 'inherit',
    stdin: 'inherit',
  })
  const exitCode = await proc.exited
  if (exitCode !== 0) {
    throw new Error(`${command} ${args.join(' ')} failed with exit code ${exitCode}`)
  }
}

const ensureBucket = async (options: SyncOptions): Promise<void> => {
  const describeArgs = ['storage', 'buckets', 'describe', options.bucket]
  if (options.project !== undefined) describeArgs.push('--project', options.project)
  const describe = Bun.spawn(['gcloud', ...describeArgs], {
    stdout: 'ignore',
    stderr: 'ignore',
  })
  const describeExit = await describe.exited
  if (describeExit === 0) return
  if (!options.createBucket) {
    throw new Error(
      `bucket ${options.bucket} is not accessible; rerun with --create-bucket once gcloud auth is refreshed`,
    )
  }

  const createArgs = ['storage', 'buckets', 'create', options.bucket]
  if (options.location !== undefined) createArgs.push('--location', options.location)
  if (options.project !== undefined) createArgs.push('--project', options.project)
  createArgs.push('--uniform-bucket-level-access')
  await runCommand('gcloud', createArgs)
}

const parseOptions = (): SyncOptions => {
  const args = process.argv.slice(2)
  if (args.includes('--help') || args.includes('-h')) usage()

  const bucket =
    args.map((arg) => getFlagValue(arg, '--bucket')).find((value) => value !== undefined) ??
    process.env.RN_MODEL_BUCKET
  if (bucket === undefined) usage()

  const prefix =
    args.map((arg) => getFlagValue(arg, '--prefix')).find((value) => value !== undefined) ??
    process.env.RN_MODEL_PREFIX ??
    'react-native'
  const project =
    args.map((arg) => getFlagValue(arg, '--project')).find((value) => value !== undefined) ??
    process.env.RN_MODEL_PROJECT
  const location =
    args.map((arg) => getFlagValue(arg, '--location')).find((value) => value !== undefined) ??
    process.env.RN_MODEL_LOCATION
  const baseUrl =
    args.map((arg) => getFlagValue(arg, '--base-url')).find((value) => value !== undefined) ??
    process.env.RN_MODEL_BASE_URL ??
    `https://storage.googleapis.com/${bucketNameFrom(normaliseBucket(bucket))}`
  const manifestOut =
    args.map((arg) => getFlagValue(arg, '--manifest-out')).find((value) => value !== undefined) ??
    process.env.RN_MODEL_MANIFEST_OUT
  const tempRoot =
    args.map((arg) => getFlagValue(arg, '--temp-root')).find((value) => value !== undefined) ??
    process.env.RN_MODEL_TEMP_ROOT
  const modelList = args
    .map((arg) => getFlagValue(arg, '--models'))
    .find((value) => value !== undefined)
  const selectedModelIds =
    modelList === undefined || modelList.trim() === ''
      ? undefined
      : new Set(
          modelList
            .split(',')
            .map((entry) => entry.trim())
            .filter((entry) => entry !== ''),
        )

  return {
    bucket: normaliseBucket(bucket),
    prefix,
    baseUrl,
    project,
    location,
    manifestOut,
    tempRoot,
    selectedModelIds,
    cacheControl: 'public, max-age=31536000, immutable',
    createBucket: args.includes('--create-bucket'),
    makePublic: args.includes('--make-public'),
    dryRun: args.includes('--dry-run'),
    huggingFaceToken: undefined,
  }
}

const resolveHuggingFaceToken = async (): Promise<string | undefined> => {
  const fromEnv = process.env.HF_TOKEN ?? process.env.HUGGING_FACE_HUB_TOKEN
  if (fromEnv !== undefined && fromEnv.trim() !== '') return fromEnv.trim()

  const tokenPath = join(process.env.HOME ?? '', '.cache', 'huggingface', 'token')
  if (tokenPath.trim() === '') return undefined
  try {
    const file = await readFile(tokenPath, 'utf8')
    const trimmed = file.trim()
    return trimmed === '' ? undefined : trimmed
  } catch {
    return undefined
  }
}

const verifyChecksum = async (path: string, checksum: string | undefined): Promise<void> => {
  if (checksum === undefined || checksum === '') return
  const expected = checksum.replace(/^sha256:/, '')
  const hash = createHash('sha256')
  for await (const chunk of createReadStream(path)) {
    hash.update(chunk)
  }
  const actual = hash.digest('hex')
  if (actual !== expected) {
    throw new Error(`checksum mismatch for ${path}: expected ${expected}, got ${actual}`)
  }
}

const downloadFile = async (path: string, file: ModelFileManifest): Promise<void> => {
  if (file.downloadUrl === undefined || file.downloadUrl === '') {
    throw new Error(`missing download URL for ${file.filename}`)
  }
  const response = await fetch(file.downloadUrl, {
    headers: file.downloadHeaders,
  })
  if (!response.ok || response.body === null) {
    throw new Error(
      `failed to download ${file.filename}: ${response.status} ${response.statusText}`,
    )
  }
  await mkdir(dirname(path), { recursive: true })
  await pipeline(Readable.fromWeb(response.body), createWriteStream(path))
  await verifyChecksum(path, file.checksum)
}

const uploadFile = async (
  options: SyncOptions,
  sourcePath: string,
  destination: string,
  contentType: string,
  cacheControl?: string,
): Promise<void> => {
  const args = [
    'storage',
    'cp',
    sourcePath,
    destination,
    '--cache-control',
    cacheControl ?? options.cacheControl,
    '--content-type',
    contentType,
  ]
  if (options.project !== undefined) args.push('--project', options.project)
  await runCommand('gcloud', args)
}

const uploadTextFile = async (
  options: SyncOptions,
  tempDir: string,
  file: UploadedTextFile,
): Promise<void> => {
  const localPath = join(
    tempDir,
    '.generated',
    file.destination.replace(/^gs:\/\//, '').replace(/\//g, '__'),
  )
  await mkdir(dirname(localPath), { recursive: true })
  await writeFile(localPath, file.content, 'utf8')
  await uploadFile(options, localPath, file.destination, file.contentType, file.cacheControl)
}

const inferContentType = (filename: string): string => {
  if (filename.endsWith('.json')) return 'application/json'
  if (filename.endsWith('.txt')) return 'text/plain; charset=utf-8'
  if (filename.endsWith('.html')) return 'text/html; charset=utf-8'
  return 'application/octet-stream'
}

const isGemmaManifest = (manifest: ModelManifest): boolean => {
  return (
    manifest.source?.provider === 'huggingface' &&
    (manifest.source.repo.startsWith('google/gemma-') ||
      manifest.source.repo.startsWith('litert-community/gemma-'))
  )
}

const fetchText = async (url: string): Promise<string> => {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`failed to fetch ${url}: ${response.status} ${response.statusText}`)
  }
  return await response.text()
}

const buildGemmaLegalFiles = async (
  options: SyncOptions,
  manifest: ModelManifest,
): Promise<readonly UploadedTextFile[]> => {
  if (!isGemmaManifest(manifest)) return []
  const baseDestination = buildDestination(options, manifest, {
    filename: 'NOTICE.txt',
    sizeBytes: GEMMA_NOTICE_TEXT.length,
  })
  const prefix = baseDestination.replace(/\/NOTICE\.txt$/, '')
  const [termsHtml, prohibitedUsePolicyHtml] = await Promise.all([
    fetchText(GEMMA_TERMS_URL),
    fetchText(GEMMA_PROHIBITED_USE_POLICY_URL),
  ])
  return [
    {
      destination: `${prefix}/NOTICE.txt`,
      content: GEMMA_NOTICE_TEXT,
      contentType: 'text/plain; charset=utf-8',
      cacheControl: 'public, max-age=86400',
    },
    {
      destination: `${prefix}/GEMMA_TERMS_OF_USE.html`,
      content: termsHtml,
      contentType: 'text/html; charset=utf-8',
      cacheControl: 'public, max-age=86400',
    },
    {
      destination: `${prefix}/GEMMA_PROHIBITED_USE_POLICY.html`,
      content: prohibitedUsePolicyHtml,
      contentType: 'text/html; charset=utf-8',
      cacheControl: 'public, max-age=86400',
    },
  ]
}

const mirrorManifest = async (
  options: SyncOptions,
  tempDir: string,
  manifest: ModelManifest,
): Promise<void> => {
  const files = listModelFiles(manifest)
  console.log(
    `Mirroring ${manifest.id} (${files.length} file${files.length === 1 ? '' : 's'}, ${modelBundleSizeBytes(manifest)} bytes)`,
  )
  for (const file of files) {
    const tempPath = join(tempDir, manifest.id, file.filename)
    const destination = buildDestination(options, manifest, file)
    if (options.dryRun) {
      console.log(`  download ${file.downloadUrl}`)
      console.log(`  upload   ${destination}`)
      continue
    }
    await mkdir(join(tempPath, '..'), { recursive: true })
    await downloadFile(tempPath, file)
    const info = await stat(tempPath)
    if (info.size !== file.sizeBytes) {
      throw new Error(
        `size mismatch for ${file.filename}: expected ${file.sizeBytes}, got ${info.size}`,
      )
    }
    await uploadFile(options, tempPath, destination, inferContentType(file.filename))
  }

  const legalFiles = await buildGemmaLegalFiles(options, manifest)
  for (const legalFile of legalFiles) {
    if (options.dryRun) {
      console.log(`  upload   ${legalFile.destination}`)
      continue
    }
    await uploadTextFile(options, tempDir, legalFile)
  }
}

const main = async (): Promise<void> => {
  const token = await resolveHuggingFaceToken()
  const options = {
    ...parseOptions(),
    ...(token === undefined ? {} : { huggingFaceToken: token }),
  }
  const upstream = createUpstreamModelManifests({
    huggingFaceToken: options.huggingFaceToken,
  })
  const manifests = upstream.filter((manifest) =>
    options.selectedModelIds === undefined ? true : options.selectedModelIds.has(manifest.id),
  )
  if (manifests.length === 0) {
    throw new Error('no manifests selected')
  }
  for (const manifest of manifests) {
    for (const file of listModelFiles(manifest)) {
      if (file.source?.provider === 'huggingface' && file.source.gated === true) {
        const authorisation =
          file.downloadHeaders?.authorization ?? file.downloadHeaders?.Authorization
        if (authorisation === undefined || authorisation.trim() === '') {
          throw new Error(
            `selected manifests include gated Hugging Face artefacts; set HF_TOKEN or HUGGING_FACE_HUB_TOKEN before mirroring ${manifest.id}`,
          )
        }
      }
    }
  }

  if (!options.dryRun) {
    await ensureBucket(options)
  }

  const tempDir = await mkdtemp(join(options.tempRoot ?? tmpdir(), 'rn-model-sync-'))
  try {
    for (const manifest of manifests) {
      await mirrorManifest(options, tempDir, manifest)
    }

    if (!options.dryRun && options.makePublic) {
      const args = [
        'storage',
        'buckets',
        'add-iam-policy-binding',
        options.bucket,
        '--member=allUsers',
        '--role=roles/storage.objectViewer',
      ]
      if (options.project !== undefined) args.push('--project', options.project)
      await runCommand('gcloud', args)
    }

    const hostedManifests = createHostedModelManifests({
      baseUrl: options.baseUrl,
      prefix: options.prefix,
    }).filter((manifest) =>
      options.selectedModelIds === undefined ? true : options.selectedModelIds.has(manifest.id),
    )
    const manifestJson = `${JSON.stringify(hostedManifests, null, 2)}\n`
    if (options.manifestOut !== undefined) {
      await mkdir(dirname(options.manifestOut), { recursive: true })
      await writeFile(options.manifestOut, manifestJson, 'utf8')
      console.log(`Hosted manifest written to ${options.manifestOut}`)
    } else {
      console.log(manifestJson)
    }
  } finally {
    await rm(tempDir, { recursive: true, force: true })
  }
}

await main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error)
  console.error(message)
  process.exit(1)
})
