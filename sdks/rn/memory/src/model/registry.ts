export type ModelSource = {
  readonly provider: 'huggingface'
  readonly repo: string
  readonly revision?: string
  readonly path: string
  readonly gated?: boolean
}

export type ModelFileManifest = {
  readonly filename: string
  readonly sizeBytes: number
  readonly downloadUrl?: string
  readonly downloadHeaders?: Readonly<Record<string, string>>
  readonly checksum?: string
  readonly source?: ModelSource
}

export type ModelManifest = {
  readonly id: string
  readonly name: string
  readonly filename: string
  readonly sizeBytes: number
  readonly bundleSizeBytes?: number
  readonly type: 'llm' | 'embedding'
  readonly platform: 'android' | 'ios' | 'both'
  readonly backend: 'litert' | 'mediapipe' | 'onnx'
  readonly downloadUrl?: string
  readonly downloadHeaders?: Readonly<Record<string, string>>
  readonly checksum?: string
  readonly source?: ModelSource
  readonly supportFiles?: readonly ModelFileManifest[]
  readonly metadata: {
    readonly parameterCount?: string
    readonly contextLength?: number
    readonly quantisation?: string
    readonly dimension?: number
  }
}

export type UpstreamModelManifestOptions = {
  readonly huggingFaceToken?: string
}

export type HostedModelManifestOptions = {
  readonly baseUrl: string
  readonly prefix?: string
}

export type DefaultModelManifestOptions = {
  readonly huggingFaceToken?: string
  readonly hostedBaseUrl?: string
  readonly hostedPrefix?: string
}

const HUGGING_FACE_BASE_URL = 'https://huggingface.co'

const BASE_MODEL_MANIFESTS: readonly ModelManifest[] = [
  {
    id: 'gemma-4-e4b-it',
    name: 'Gemma 4 E4B IT',
    filename: 'gemma-4-E4B-it.litertlm',
    sizeBytes: 3_654_467_584,
    type: 'llm',
    platform: 'both',
    backend: 'litert',
    checksum: 'sha256:f335f2bfd1b758dc6476db16c0f41854bd6237e2658d604cbe566bcefd00a7bc',
    source: {
      provider: 'huggingface',
      repo: 'litert-community/gemma-4-E4B-it-litert-lm',
      path: 'gemma-4-E4B-it.litertlm',
      gated: false,
    },
    metadata: {
      parameterCount: '4B',
      contextLength: 32_768,
    },
  },
  {
    id: 'gemma-4-e2b-it',
    name: 'Gemma 4 E2B IT',
    filename: 'gemma-4-E2B-it.litertlm',
    sizeBytes: 2_583_085_056,
    type: 'llm',
    platform: 'both',
    backend: 'litert',
    checksum: 'sha256:ab7838cdfc8f77e54d8ca45eadceb20452d9f01e4bfade03e5dce27911b27e42',
    source: {
      provider: 'huggingface',
      repo: 'litert-community/gemma-4-E2B-it-litert-lm',
      path: 'gemma-4-E2B-it.litertlm',
      gated: false,
    },
    metadata: {
      parameterCount: '2B',
      contextLength: 32_768,
    },
  },
  {
    id: 'all-minilm-l6-v2-onnx',
    name: 'all-MiniLM-L6-v2 ONNX',
    filename: 'model.onnx',
    sizeBytes: 90_387_630,
    bundleSizeBytes: 91_333_577,
    type: 'embedding',
    platform: 'both',
    backend: 'onnx',
    checksum: 'sha256:bbd7b466f6d58e646fdc2bd5fd67b2f5e93c0b687011bd4548c420f7bd46f0c5',
    source: {
      provider: 'huggingface',
      repo: 'Qdrant/all-MiniLM-L6-v2-onnx',
      path: 'model.onnx',
      gated: false,
    },
    supportFiles: [
      {
        filename: 'config.json',
        sizeBytes: 650,
        source: {
          provider: 'huggingface',
          repo: 'Qdrant/all-MiniLM-L6-v2-onnx',
          path: 'config.json',
          gated: false,
        },
      },
      {
        filename: 'special_tokens_map.json',
        sizeBytes: 695,
        source: {
          provider: 'huggingface',
          repo: 'Qdrant/all-MiniLM-L6-v2-onnx',
          path: 'special_tokens_map.json',
          gated: false,
        },
      },
      {
        filename: 'tokenizer.json',
        sizeBytes: 711_661,
        source: {
          provider: 'huggingface',
          repo: 'Qdrant/all-MiniLM-L6-v2-onnx',
          path: 'tokenizer.json',
          gated: false,
        },
      },
      {
        filename: 'tokenizer_config.json',
        sizeBytes: 1_433,
        source: {
          provider: 'huggingface',
          repo: 'Qdrant/all-MiniLM-L6-v2-onnx',
          path: 'tokenizer_config.json',
          gated: false,
        },
      },
      {
        filename: 'vocab.txt',
        sizeBytes: 231_508,
        source: {
          provider: 'huggingface',
          repo: 'Qdrant/all-MiniLM-L6-v2-onnx',
          path: 'vocab.txt',
          gated: false,
        },
      },
    ],
    metadata: {
      dimension: 384,
    },
  },
]

const trimSlashes = (value: string): string => value.replace(/^\/+|\/+$/g, '')

const encodePath = (value: string): string => {
  return value
    .split('/')
    .filter((segment) => segment !== '')
    .map((segment) => encodeURIComponent(segment))
    .join('/')
}

const appendUrlPath = (baseUrl: string, ...segments: readonly string[]): string => {
  const trimmedBase = baseUrl.replace(/\/+$/, '')
  const rest = segments
    .map((segment) => trimSlashes(segment))
    .filter((segment) => segment !== '')
    .map((segment) => encodePath(segment))
    .join('/')
  return rest === '' ? trimmedBase : `${trimmedBase}/${rest}`
}

const authHeadersFor = (
  source: ModelSource | undefined,
  token: string | undefined,
): Readonly<Record<string, string>> | undefined => {
  if (source?.provider !== 'huggingface' || source.gated !== true) return undefined
  if (token === undefined || token.trim() === '') return undefined
  return {
    authorization: `Bearer ${token.trim()}`,
  }
}

const withResolvedFile = (
  file: ModelFileManifest,
  resolver: (file: ModelFileManifest) => string,
  headers: Readonly<Record<string, string>> | undefined,
): ModelFileManifest => {
  return {
    ...file,
    downloadUrl: resolver(file),
    ...(headers === undefined ? {} : { downloadHeaders: headers }),
  }
}

export const createHuggingFaceResolveUrl = (source: ModelSource): string => {
  if (source.provider !== 'huggingface') {
    throw new Error(`unsupported model source provider: ${source.provider}`)
  }
  return appendUrlPath(
    HUGGING_FACE_BASE_URL,
    source.repo,
    'resolve',
    source.revision ?? 'main',
    source.path,
  )
}

export const createHostedModelDownloadUrl = (args: {
  readonly baseUrl: string
  readonly manifestId: string
  readonly filename: string
  readonly prefix?: string
}): string => {
  return appendUrlPath(args.baseUrl, args.prefix ?? '', args.manifestId, args.filename)
}

export const listModelFiles = (manifest: ModelManifest): readonly ModelFileManifest[] => {
  return [
    {
      filename: manifest.filename,
      sizeBytes: manifest.sizeBytes,
      ...(manifest.downloadUrl === undefined ? {} : { downloadUrl: manifest.downloadUrl }),
      ...(manifest.downloadHeaders === undefined
        ? {}
        : { downloadHeaders: manifest.downloadHeaders }),
      ...(manifest.checksum === undefined ? {} : { checksum: manifest.checksum }),
      ...(manifest.source === undefined ? {} : { source: manifest.source }),
    },
    ...(manifest.supportFiles ?? []),
  ]
}

export const modelBundleSizeBytes = (manifest: ModelManifest): number => {
  return (
    manifest.bundleSizeBytes ??
    listModelFiles(manifest).reduce((total, file) => total + file.sizeBytes, 0)
  )
}

export const createUpstreamModelManifests = (
  options: UpstreamModelManifestOptions = {},
): readonly ModelManifest[] => {
  return BASE_MODEL_MANIFESTS.map((manifest) => {
    const headers = authHeadersFor(manifest.source, options.huggingFaceToken)
    return {
      ...manifest,
      ...(manifest.source === undefined
        ? {}
        : { downloadUrl: createHuggingFaceResolveUrl(manifest.source) }),
      ...(headers === undefined ? {} : { downloadHeaders: headers }),
      ...(manifest.supportFiles === undefined
        ? {}
        : {
            supportFiles: manifest.supportFiles.map((file) =>
              withResolvedFile(
                file,
                (entry) =>
                  entry.source === undefined
                    ? (entry.downloadUrl ?? '')
                    : createHuggingFaceResolveUrl(entry.source),
                authHeadersFor(file.source, options.huggingFaceToken),
              ),
            ),
          }),
    }
  })
}

export const createHostedModelManifests = (
  options: HostedModelManifestOptions,
): readonly ModelManifest[] => {
  const hostedUrl = (manifestId: string, filename: string): string => {
    return createHostedModelDownloadUrl({
      baseUrl: options.baseUrl,
      manifestId,
      filename,
      ...(options.prefix === undefined ? {} : { prefix: options.prefix }),
    })
  }

  return BASE_MODEL_MANIFESTS.map((manifest) => {
    return {
      ...manifest,
      downloadUrl: hostedUrl(manifest.id, manifest.filename),
      ...(manifest.supportFiles === undefined
        ? {}
        : {
            supportFiles: manifest.supportFiles.map((file) =>
              withResolvedFile(file, (entry) => hostedUrl(manifest.id, entry.filename), undefined),
            ),
          }),
    }
  })
}

export const resolveDefaultModelManifests = (
  options: DefaultModelManifestOptions = {},
): readonly ModelManifest[] => {
  if (options.hostedBaseUrl !== undefined && options.hostedBaseUrl.trim() !== '') {
    return createHostedModelManifests({
      baseUrl: options.hostedBaseUrl,
      ...(options.hostedPrefix === undefined ? {} : { prefix: options.hostedPrefix }),
    })
  }

  return createUpstreamModelManifests({
    ...(options.huggingFaceToken === undefined
      ? {}
      : { huggingFaceToken: options.huggingFaceToken }),
  })
}

export const DEFAULT_MODEL_MANIFESTS: readonly ModelManifest[] = resolveDefaultModelManifests()

export const listModelManifests = (
  platform?: ModelManifest['platform'],
): readonly ModelManifest[] => {
  if (platform === undefined) return DEFAULT_MODEL_MANIFESTS
  return DEFAULT_MODEL_MANIFESTS.filter(
    (manifest) => manifest.platform === 'both' || manifest.platform === platform,
  )
}
