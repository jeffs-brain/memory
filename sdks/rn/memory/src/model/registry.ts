export type ModelManifest = {
  readonly id: string
  readonly name: string
  readonly filename: string
  readonly sizeBytes: number
  readonly type: 'llm' | 'embedding'
  readonly platform: 'android' | 'ios' | 'both'
  readonly backend: 'litert' | 'mediapipe' | 'onnx'
  readonly downloadUrl?: string
  readonly checksum?: string
  readonly metadata: {
    readonly parameterCount?: string
    readonly contextLength?: number
    readonly quantisation?: string
    readonly dimension?: number
  }
}

export const DEFAULT_MODEL_MANIFESTS: readonly ModelManifest[] = [
  {
    id: 'gemma-4-e4b-4bit',
    name: 'Gemma 4 E4B 4-bit',
    filename: 'gemma-4-e4b-4bit.task',
    sizeBytes: 5_300_000_000,
    type: 'llm',
    platform: 'both',
    backend: 'litert',
    metadata: {
      parameterCount: '4B',
      contextLength: 128_000,
      quantisation: '4bit',
    },
  },
  {
    id: 'gemma-4-e2b-4bit',
    name: 'Gemma 4 E2B 4-bit',
    filename: 'gemma-4-e2b-4bit.task',
    sizeBytes: 3_200_000_000,
    type: 'llm',
    platform: 'both',
    backend: 'litert',
    metadata: {
      parameterCount: '2B',
      contextLength: 128_000,
      quantisation: '4bit',
    },
  },
  {
    id: 'all-minilm-l6-v2',
    name: 'all-MiniLM-L6-v2',
    filename: 'all-minilm-l6-v2.onnx',
    sizeBytes: 22_000_000,
    type: 'embedding',
    platform: 'both',
    backend: 'onnx',
    metadata: {
      dimension: 384,
    },
  },
]

export const listModelManifests = (
  platform?: ModelManifest['platform'],
): readonly ModelManifest[] => {
  if (platform === undefined) return DEFAULT_MODEL_MANIFESTS
  return DEFAULT_MODEL_MANIFESTS.filter(
    (manifest) => manifest.platform === 'both' || manifest.platform === platform,
  )
}
