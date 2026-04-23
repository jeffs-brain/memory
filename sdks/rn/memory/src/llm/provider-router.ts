import type { ConnectivityMonitor } from '../connectivity/monitor.js'
import type {
  CompletionRequest,
  CompletionResponse,
  Embedder,
  Provider,
  StreamEvent,
  StructuredRequest,
} from './types.js'

export type RoutingDecision = {
  readonly kind: 'provider' | 'embedder'
  readonly route: 'local' | 'cloud'
  readonly reason: string
  readonly name: string
  readonly model: string
}

export type ProviderRouterConfig = {
  readonly localProvider?: Provider
  readonly cloudProvider?: Provider
  readonly localEmbedder?: Embedder
  readonly cloudEmbedder?: Embedder
  readonly connectivity?: ConnectivityMonitor
  readonly strategy: 'local-only' | 'cloud-only' | 'auto'
  readonly autoConfig?: {
    readonly preferLocal?: boolean
    readonly cloudTriggers?: {
      readonly maxInputTokens?: number
      readonly taskTypes?: readonly string[]
    }
  }
}

const estimateRequestTokens = (request: CompletionRequest | StructuredRequest): number => {
  const text = [
    request.systemStatic ?? '',
    request.systemDynamic ?? '',
    request.system ?? '',
    ...request.messages.map((message) => message.content ?? ''),
  ].join('\n')
  return Math.ceil(text.length / 4)
}

export class ProviderRouter implements Provider, Embedder {
  private lastDecisionValue: RoutingDecision | null = null

  constructor(private readonly config: ProviderRouterConfig) {}

  name(): string {
    return 'provider-router'
  }

  modelName(): string {
    return (
      this.lastDecisionValue?.model ??
      this.config.localProvider?.modelName() ??
      this.config.cloudProvider?.modelName() ??
      'unknown'
    )
  }

  model(): string {
    return (
      this.lastDecisionValue?.model ??
      this.config.localEmbedder?.model() ??
      this.config.cloudEmbedder?.model() ??
      'unknown'
    )
  }

  dimension(): number {
    return this.config.localEmbedder?.dimension() ?? this.config.cloudEmbedder?.dimension() ?? 0
  }

  supportsStructuredDecoding(): boolean {
    return (
      this.config.localProvider?.supportsStructuredDecoding() === true ||
      this.config.cloudProvider?.supportsStructuredDecoding() === true
    )
  }

  lastRoute(): RoutingDecision | null {
    return this.lastDecisionValue
  }

  async complete(request: CompletionRequest, signal?: AbortSignal): Promise<CompletionResponse> {
    const selected = await this.selectProvider(request)
    this.lastDecisionValue = selected.decision
    return await selected.provider.complete(request, signal)
  }

  async structured(request: StructuredRequest, signal?: AbortSignal): Promise<string> {
    const selected = await this.selectProvider(request)
    this.lastDecisionValue = selected.decision
    return await selected.provider.structured(request, signal)
  }

  async *stream(request: CompletionRequest, signal?: AbortSignal): AsyncIterable<StreamEvent> {
    const selected = await this.selectProvider(request)
    this.lastDecisionValue = selected.decision
    if (selected.provider.stream === undefined) {
      const response = await selected.provider.complete(request, signal)
      if (response.content !== '') {
        yield { type: 'text_delta', text: response.content }
      }
      yield { type: 'done', usage: response.usage, stopReason: response.stopReason }
      return
    }
    yield* selected.provider.stream(request, signal)
  }

  async embed(texts: readonly string[], signal?: AbortSignal): Promise<number[][]> {
    const selected = await this.selectEmbedder()
    this.lastDecisionValue = selected.decision
    return await selected.embedder.embed(texts, signal)
  }

  private async selectProvider(
    request: CompletionRequest | StructuredRequest,
  ): Promise<{ readonly provider: Provider; readonly decision: RoutingDecision }> {
    const online = this.config.connectivity?.snapshot().online ?? true
    const preferLocal = this.config.autoConfig?.preferLocal ?? true
    const taskType = request.taskType ?? 'chat'
    const shouldEscalate =
      online &&
      this.config.cloudProvider !== undefined &&
      ((this.config.autoConfig?.cloudTriggers?.taskTypes?.includes(taskType) ?? false) ||
        (this.config.autoConfig?.cloudTriggers?.maxInputTokens !== undefined &&
          estimateRequestTokens(request) > this.config.autoConfig.cloudTriggers.maxInputTokens))

    switch (this.config.strategy) {
      case 'local-only':
        if (this.config.localProvider !== undefined) {
          return {
            provider: this.config.localProvider,
            decision: {
              kind: 'provider',
              route: 'local',
              reason: 'strategy-local-only',
              name: this.config.localProvider.name(),
              model: this.config.localProvider.modelName(),
            },
          }
        }
        break
      case 'cloud-only':
        if (online && this.config.cloudProvider !== undefined) {
          return {
            provider: this.config.cloudProvider,
            decision: {
              kind: 'provider',
              route: 'cloud',
              reason: 'strategy-cloud-only',
              name: this.config.cloudProvider.name(),
              model: this.config.cloudProvider.modelName(),
            },
          }
        }
        if (this.config.localProvider !== undefined) {
          return {
            provider: this.config.localProvider,
            decision: {
              kind: 'provider',
              route: 'local',
              reason: 'offline-fallback',
              name: this.config.localProvider.name(),
              model: this.config.localProvider.modelName(),
            },
          }
        }
        break
      case 'auto':
        if (!online && this.config.localProvider !== undefined) {
          return {
            provider: this.config.localProvider,
            decision: {
              kind: 'provider',
              route: 'local',
              reason: 'offline',
              name: this.config.localProvider.name(),
              model: this.config.localProvider.modelName(),
            },
          }
        }
        if (shouldEscalate && this.config.cloudProvider !== undefined) {
          return {
            provider: this.config.cloudProvider,
            decision: {
              kind: 'provider',
              route: 'cloud',
              reason: 'cloud-trigger',
              name: this.config.cloudProvider.name(),
              model: this.config.cloudProvider.modelName(),
            },
          }
        }
        if (preferLocal && this.config.localProvider !== undefined) {
          return {
            provider: this.config.localProvider,
            decision: {
              kind: 'provider',
              route: 'local',
              reason: 'prefer-local',
              name: this.config.localProvider.name(),
              model: this.config.localProvider.modelName(),
            },
          }
        }
        if (online && this.config.cloudProvider !== undefined) {
          return {
            provider: this.config.cloudProvider,
            decision: {
              kind: 'provider',
              route: 'cloud',
              reason: 'prefer-cloud',
              name: this.config.cloudProvider.name(),
              model: this.config.cloudProvider.modelName(),
            },
          }
        }
        if (this.config.localProvider !== undefined) {
          return {
            provider: this.config.localProvider,
            decision: {
              kind: 'provider',
              route: 'local',
              reason: 'fallback-local',
              name: this.config.localProvider.name(),
              model: this.config.localProvider.modelName(),
            },
          }
        }
        break
    }

    throw new Error('provider-router: no provider available for the current strategy')
  }

  private async selectEmbedder(): Promise<{
    readonly embedder: Embedder
    readonly decision: RoutingDecision
  }> {
    const online = this.config.connectivity?.snapshot().online ?? true
    if (this.config.localEmbedder !== undefined) {
      return {
        embedder: this.config.localEmbedder,
        decision: {
          kind: 'embedder',
          route: 'local',
          reason: 'prefer-local',
          name: this.config.localEmbedder.name(),
          model: this.config.localEmbedder.model(),
        },
      }
    }
    if (online && this.config.cloudEmbedder !== undefined) {
      return {
        embedder: this.config.cloudEmbedder,
        decision: {
          kind: 'embedder',
          route: 'cloud',
          reason: 'local-unavailable',
          name: this.config.cloudEmbedder.name(),
          model: this.config.cloudEmbedder.model(),
        },
      }
    }
    throw new Error('provider-router: no embedder available for the current strategy')
  }
}
