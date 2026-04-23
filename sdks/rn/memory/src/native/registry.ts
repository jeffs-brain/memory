import type { EmbeddingModule, InferenceModule } from './types.js'

type NativeRegistry = {
  inference?: InferenceModule
  embedding?: EmbeddingModule
}

const registry: NativeRegistry = {}

export const registerInferenceModule = (module: InferenceModule): void => {
  registry.inference = module
}

export const registerEmbeddingModule = (module: EmbeddingModule): void => {
  registry.embedding = module
}

export const resolveInferenceModule = (module?: InferenceModule): InferenceModule => {
  const resolved = module ?? registry.inference
  if (resolved === undefined) {
    throw new Error('react-native memory: inference module not registered')
  }
  return resolved
}

export const resolveEmbeddingModule = (module?: EmbeddingModule): EmbeddingModule => {
  const resolved = module ?? registry.embedding
  if (resolved === undefined) {
    throw new Error('react-native memory: embedding module not registered')
  }
  return resolved
}
