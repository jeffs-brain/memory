import { describe, expect, it } from 'vitest'

import { createMemoryFileAdapter } from '../testing/memory-file-adapter.js'
import { createModelManager } from './manager.js'

describe('createModelManager', () => {
  it('downloads, lists, sizes, cancels, and deletes models', async () => {
    const adapter = createMemoryFileAdapter()
    const cancelled: string[] = []
    const progress: number[] = []
    const requests: Array<{ readonly url: string; readonly destination: string }> = []

    const manager = createModelManager({
      root: '/models',
      adapter,
      manifests: [
        {
          id: 'demo-model',
          name: 'Demo model',
          filename: 'demo.bin',
          sizeBytes: 10,
          bundleSizeBytes: 20,
          type: 'llm',
          platform: 'both',
          backend: 'litert',
          downloadUrl: 'https://example.com/demo.bin',
          supportFiles: [
            {
              filename: 'tokenizer.json',
              sizeBytes: 10,
              downloadUrl: 'https://example.com/tokenizer.json',
            },
          ],
          metadata: {
            parameterCount: '1B',
            contextLength: 8192,
            quantisation: '4bit',
          },
        },
      ],
      downloadTransport: {
        download: async (request) => {
          requests.push({ url: request.url, destination: request.destination })
          request.onProgress?.({ downloadedBytes: 5, totalBytes: 10 })
          request.onProgress?.({ downloadedBytes: 10, totalBytes: 10 })
          await adapter.writeText(
            request.destination,
            request.destination.endsWith('demo.bin') ? '1234567890' : 'abcdefghij',
          )
        },
        cancel: (modelId) => {
          cancelled.push(modelId)
        },
      },
    })

    await manager.download('demo-model', (pct) => {
      progress.push(pct)
    })

    expect(progress).toEqual([25, 50, 50, 75, 100, 100])
    expect(requests).toEqual([
      {
        url: 'https://example.com/demo.bin',
        destination: '/models/demo-model/demo.bin',
      },
      {
        url: 'https://example.com/tokenizer.json',
        destination: '/models/demo-model/tokenizer.json',
      },
    ])
    expect(manager.getModelPath('demo-model')).toBe('/models/demo-model/demo.bin')
    expect((await manager.listDownloaded()).map((entry) => entry.path)).toEqual([
      '/models/demo-model/demo.bin',
    ])
    expect((await manager.listDownloaded()).map((entry) => entry.sizeBytes)).toEqual([20])
    expect(await manager.storageUsed()).toBe(20)

    manager.cancelDownload('demo-model')
    expect(cancelled).toEqual(['demo-model'])

    await manager.deleteModel('demo-model')
    expect(await manager.listDownloaded()).toEqual([])
  })

  it('fails fast for gated upstream models without Hugging Face auth', async () => {
    const adapter = createMemoryFileAdapter()
    const manifest = {
      id: 'private-gemma',
      name: 'Private Gemma',
      filename: 'model.litertlm',
      sizeBytes: 10,
      type: 'llm' as const,
      platform: 'both' as const,
      backend: 'litert' as const,
      downloadUrl: 'https://huggingface.co/private/repo/resolve/main/model.litertlm',
      source: {
        provider: 'huggingface' as const,
        repo: 'private/repo',
        path: 'model.litertlm',
        gated: true,
      },
      metadata: {
        parameterCount: '2B',
      },
    }

    const manager = createModelManager({
      root: '/models',
      adapter,
      manifests: [manifest],
      downloadTransport: {
        download: async () => {
          throw new Error('should not attempt download')
        },
        cancel: () => {},
      },
    })

    await expect(manager.download('private-gemma')).rejects.toThrow(/gated Hugging Face artefact/)
  })
})
