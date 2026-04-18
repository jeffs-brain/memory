// SPDX-License-Identifier: Apache-2.0

/**
 * Canonical error taxonomy for the provider layer. Callers can decide
 * which failures are retryable by inspecting the class rather than
 * scraping message strings.
 */

export class LLMError extends Error {
  override readonly cause?: unknown

  constructor(message: string, cause?: unknown) {
    super(message)
    this.name = 'LLMError'
    if (cause !== undefined) this.cause = cause
  }
}

/**
 * Transport-level failures (DNS, TCP reset, aborted stream, unexpected
 * EOF). Typically safe to retry once, provided no assistant content has
 * yet been committed downstream.
 */
export class TransportError extends LLMError {
  constructor(message: string, cause?: unknown) {
    super(message, cause)
    this.name = 'TransportError'
  }
}

/**
 * Protocol-level failures (non-2xx HTTP, malformed payload, schema
 * mismatch). The provider rejected the request; retrying is a waste
 * unless the caller changes the request.
 */
export class ProviderError extends LLMError {
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

/**
 * The API rejected the prompt because it exceeds the model's context
 * window. Callers trigger compaction in response.
 */
export class PromptTooLongError extends LLMError {
  constructor(message: string, cause?: unknown) {
    super(message, cause)
    this.name = 'PromptTooLongError'
  }
}

/**
 * The SSE stream stalled for longer than the configured idle timeout.
 * Derives from TransportError so the caller's retry logic treats it the
 * same as any other wire-level failure.
 */
export class StreamIdleTimeoutError extends TransportError {
  constructor(timeoutMs: number) {
    super(`stream idle timeout: no data received for ${timeoutMs}ms`)
    this.name = 'StreamIdleTimeoutError'
  }
}

/**
 * The structured-output retry loop exhausted its budget. The last raw
 * payload and the validator's reason are preserved on the instance so
 * callers can show them to the user or log them.
 */
export class SchemaValidationError extends LLMError {
  constructor(
    readonly rawPayload: string,
    readonly reason: string,
  ) {
    super(`schema validation failed: ${reason}`)
    this.name = 'SchemaValidationError'
  }
}
