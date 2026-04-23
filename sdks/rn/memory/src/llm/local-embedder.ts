import type { EmbeddingBridge } from '../native/embedding-bridge.js'
import type { Embedder } from './types.js'

export class LocalEmbedder implements Embedder {
  constructor(
    private readonly bridge: EmbeddingBridge,
    private readonly embedderName: string,
    private readonly embedderModel: string,
  ) {}

  name(): string {
    return this.embedderName
  }

  model(): string {
    return this.embedderModel
  }

  dimension(): number {
    return this.bridge.dimension()
  }

  async embed(texts: readonly string[]): Promise<number[][]> {
    return await this.bridge.embed(texts)
  }
}
