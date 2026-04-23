import { describe, expect, it } from 'vitest'

import { createMemoryFileAdapter } from '../testing/memory-file-adapter.js'
import { createModelManager } from './manager.js'

describe('createModelManager', () => {
  it('downloads, lists, sizes, cancels, and deletes models', async () => {
    const adapter = createMemoryFileAdapter()
    const cancelled: string[] = []
    const progress: number[] = []

    const manager = createModelManager({
      root: '/models',
      adapter,
      manifests: [
        {
          id: 'demo-model',
          name: 'Demo model',
          filename: 'demo.bin',
          sizeBytes: 10,
          type: 'llm',
          platform: 'both',
          backend: 'litert',
          downloadUrl: 'https://example.com/demo.bin',
          metadata: {
            parameterCount: '1B',
            contextLength: 8192,
            quantisation: '4bit',
          },
        },
      ],
      downloadTransport: {
        download: async (request) => {
          request.onProgress?.({ downloadedBytes: 5, totalBytes: 10 })
          request.onProgress?.({ downloadedBytes: 10, totalBytes: 10 })
          await adapter.writeText(request.destination, '1234567890')
        },
        cancel: (modelId) => {
          cancelled.push(modelId)
        },
      },
    })

    await manager.download('demo-model', (pct) => {
      progress.push(pct)
    })

    expect(progress).toEqual([50, 100])
    expect(manager.getModelPath('demo-model')).toBe('/models/demo-model/demo.bin')
    expect((await manager.listDownloaded()).map((entry) => entry.path)).toEqual([
      '/models/demo-model/demo.bin',
    ])
    expect(await manager.storageUsed()).toBe(10)

    manager.cancelDownload('demo-model')
    expect(cancelled).toEqual(['demo-model'])

    await manager.deleteModel('demo-model')
    expect(await manager.listDownloaded()).toEqual([])
  })
})
