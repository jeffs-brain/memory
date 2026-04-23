import { describe, expect, it } from 'vitest'

import {
  DEFAULT_MODEL_MANIFESTS,
  createHostedModelManifests,
  createUpstreamModelManifests,
  listModelFiles,
  modelBundleSizeBytes,
  resolveDefaultModelManifests,
} from './registry.js'

describe('model registry', () => {
  it('resolves Hugging Face upstream URLs for Gemma 4 LiteRT artefacts', () => {
    const manifests = createUpstreamModelManifests({ huggingFaceToken: 'hf_demo' })
    const gemma = manifests.find((manifest) => manifest.id === 'gemma-4-e2b-it')

    expect(gemma).toMatchObject({
      filename: 'gemma-4-E2B-it.litertlm',
      downloadUrl:
        'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm',
      checksum: 'sha256:ab7838cdfc8f77e54d8ca45eadceb20452d9f01e4bfade03e5dce27911b27e42',
    })
    expect(gemma?.downloadHeaders).toBeUndefined()
  })

  it('includes support files for the ONNX embedding bundle', () => {
    const manifests = createUpstreamModelManifests()
    const embedder = manifests.find((manifest) => manifest.id === 'all-minilm-l6-v2-onnx')
    if (embedder === undefined) {
      throw new Error('expected all-minilm-l6-v2-onnx manifest')
    }

    expect(embedder?.supportFiles?.map((file) => file.filename)).toEqual([
      'config.json',
      'special_tokens_map.json',
      'tokenizer.json',
      'tokenizer_config.json',
      'vocab.txt',
    ])
    expect(listModelFiles(embedder)).toHaveLength(6)
    expect(modelBundleSizeBytes(embedder)).toBe(91_333_577)
  })

  it('builds hosted URLs without upstream auth headers', () => {
    const manifests = createHostedModelManifests({
      baseUrl: 'https://storage.googleapis.com/demo-models',
      prefix: 'react-native',
    })
    const gemma = manifests.find((manifest) => manifest.id === 'gemma-4-e4b-it')
    const embedder = manifests.find((manifest) => manifest.id === 'all-minilm-l6-v2-onnx')

    expect(gemma?.downloadUrl).toBe(
      'https://storage.googleapis.com/demo-models/react-native/gemma-4-e4b-it/gemma-4-E4B-it.litertlm',
    )
    expect(gemma?.downloadHeaders).toBeUndefined()
    expect(embedder?.supportFiles?.[2]?.downloadUrl).toBe(
      'https://storage.googleapis.com/demo-models/react-native/all-minilm-l6-v2-onnx/tokenizer.json',
    )
  })

  it('ships upstream defaults for the open source package', () => {
    const gemma = DEFAULT_MODEL_MANIFESTS.find((manifest) => manifest.id === 'gemma-4-e2b-it')

    expect(gemma?.downloadUrl).toBe(
      'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm',
    )
    expect(gemma?.downloadHeaders).toBeUndefined()
  })

  it('resolves hosted defaults only when a base URL is provided', () => {
    const manifests = resolveDefaultModelManifests({
      hostedBaseUrl: 'https://storage.googleapis.com/private-models',
      hostedPrefix: 'react-native',
    })
    const gemma = manifests.find((manifest) => manifest.id === 'gemma-4-e2b-it')

    expect(gemma?.downloadUrl).toBe(
      'https://storage.googleapis.com/private-models/react-native/gemma-4-e2b-it/gemma-4-E2B-it.litertlm',
    )
    expect(gemma?.downloadHeaders).toBeUndefined()
  })
})
