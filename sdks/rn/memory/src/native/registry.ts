import type { EmbeddingModule, InferenceModule } from './types.js'

type NativeRegistry = {
  inference?: InferenceModule
  embedding?: EmbeddingModule
}

const registry: NativeRegistry = {}

const missingNativeModuleError = (kind: 'inference' | 'embedding'): Error => {
  const packageHint =
    kind === 'inference'
      ? 'registerInferenceModule(yourModule)'
      : 'registerEmbeddingModule(yourModule)'
  return new Error(
    `react-native memory: ${kind} module not registered. Local on-device inference requires your app to provide a native module and call ${packageHint} during startup. Use the cloud provider path instead if you do not want to ship native local inference yet.`,
  )
}

export const registerInferenceModule = (module: InferenceModule): void => {
  registry.inference = module
}

export const registerEmbeddingModule = (module: EmbeddingModule): void => {
  registry.embedding = module
}

export const resolveInferenceModule = (module?: InferenceModule): InferenceModule => {
  const resolved = module ?? registry.inference
  if (resolved === undefined) {
    throw missingNativeModuleError('inference')
  }
  return resolved
}

export const resolveEmbeddingModule = (module?: EmbeddingModule): EmbeddingModule => {
  const resolved = module ?? registry.embedding
  if (resolved === undefined) {
    throw missingNativeModuleError('embedding')
  }
  return resolved
}

export const hasRegisteredInferenceModule = (): boolean => registry.inference !== undefined

export const hasRegisteredEmbeddingModule = (): boolean => registry.embedding !== undefined
