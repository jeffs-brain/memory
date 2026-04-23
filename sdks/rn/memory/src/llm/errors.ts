export class LlmError extends Error {
  override readonly cause?: unknown

  constructor(message: string, cause?: unknown) {
    super(message)
    this.name = 'LlmError'
    if (cause !== undefined) this.cause = cause
  }
}

export class TransportError extends LlmError {
  constructor(message: string, cause?: unknown) {
    super(message, cause)
    this.name = 'TransportError'
  }
}

export class ProviderError extends LlmError {
  constructor(
    message: string,
    readonly status: number,
    readonly body?: string,
    cause?: unknown,
  ) {
    super(message, cause)
    this.name = 'ProviderError'
  }
}

export class PromptTooLongError extends LlmError {
  constructor(message: string, cause?: unknown) {
    super(message, cause)
    this.name = 'PromptTooLongError'
  }
}

export class StreamIdleTimeoutError extends LlmError {
  constructor(message: string, cause?: unknown) {
    super(message, cause)
    this.name = 'StreamIdleTimeoutError'
  }
}

export class SchemaValidationError extends LlmError {
  constructor(
    readonly rawPayload: string,
    readonly reason: string,
  ) {
    super(`schema validation failed: ${reason}`)
    this.name = 'SchemaValidationError'
  }
}

export { LlmError as LLMError }
