import { resolveEmbeddingModule } from './registry.js'
import type { EmbeddingModelConfig, EmbeddingModule } from './types.js'

export type EmbeddingBridge = {
  loadModel(config: EmbeddingModelConfig): Promise<void>
  unloadModel(): Promise<void>
  embed(texts: readonly string[]): Promise<number[][]>
  dimension(): number
}

export const createEmbeddingBridge = (module?: EmbeddingModule): EmbeddingBridge => {
  const resolved = resolveEmbeddingModule(module)
  return {
    loadModel: async (config) => await resolved.loadModel(config),
    unloadModel: async () => await resolved.unloadModel(),
    embed: async (texts) => await resolved.embed(texts),
    dimension: () => resolved.dimension(),
  }
}
