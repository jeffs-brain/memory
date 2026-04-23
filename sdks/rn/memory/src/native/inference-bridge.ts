import { resolveInferenceModule } from './registry.js'
import type {
  GenerateParams,
  GenerateResult,
  InferenceModule,
  ModelConfig,
  ModelInfo,
  StreamChunk,
} from './types.js'

export type InferenceBridge = {
  loadModel(config: ModelConfig): Promise<void>
  unloadModel(): Promise<void>
  isModelLoaded(): boolean
  generate(params: GenerateParams): Promise<GenerateResult>
  generateStream?(params: GenerateParams): AsyncIterable<StreamChunk>
  getModelInfo(): ModelInfo | null
}

export const createInferenceBridge = (module?: InferenceModule): InferenceBridge => {
  const resolved = resolveInferenceModule(module)
  const base: InferenceBridge = {
    loadModel: async (config) => await resolved.loadModel(config),
    unloadModel: async () => await resolved.unloadModel(),
    isModelLoaded: () => resolved.isModelLoaded(),
    generate: async (params) => await resolved.generate(params),
    getModelInfo: () => resolved.getModelInfo(),
  }
  const generateStream = resolved.generateStream
  if (generateStream !== undefined) {
    return {
      ...base,
      generateStream: async function* (params: GenerateParams) {
        yield* generateStream(params)
      },
    }
  }
  return base
}
