// SPDX-License-Identifier: Apache-2.0

/**
 * Prompt injection safety scanner for the ingest pipeline. Uses
 * @stackone/defender for ML-based detection (Tier 2: MiniLM classifier).
 *
 * The scanner is OPTIONAL: if the defender package is unavailable or the
 * model fails to load, ingestion continues with a warning. Flagged
 * documents are never blocked — they receive metadata annotations so
 * downstream consumers can handle them appropriately.
 */

import type { Logger } from '../llm/types.js'

/** Confidence threshold below which content is considered safe. */
const INJECTION_CONFIDENCE_THRESHOLD = 0.5

/** Tag used to wrap ingested content for isolation. */
const ISOLATION_TAG = 'ingested-document'

/** Characters that should be stripped during preprocessing (zero-width). */
const ZERO_WIDTH_CHARS_RE =
  /(?:[\u200B\u200C\u200E\u200F\u2000\u2001\u2060\u2061\u2062\u2063\u2064\uFEFF\u00AD]|\u200D)/g

/** Pattern to detect closing isolation tags that must be escaped in content. */
const CLOSING_TAG_RE = /<\/ingested-document>/gi

/** Escaped replacement for closing tags found in content. */
const CLOSING_TAG_ESCAPED = '&lt;/ingested-document&gt;'

export type SafetyScanResult = {
  readonly injectionDetected: boolean
  readonly confidence: number
  readonly detections: readonly string[]
}

export type IsolatedContent = {
  readonly content: string
  readonly source: string
  readonly hash: string
}

export type SafetyMetadata = {
  readonly injection_risk: boolean
  readonly injection_confidence: number
}

export type SafetyScannerConfig = {
  readonly logger: Logger
  readonly confidenceThreshold?: number
}

/**
 * Normalise text for consistent scanning: NFKC normalisation and removal
 * of zero-width characters that could evade pattern detection.
 *
 * Time: O(n) where n = text length.
 * Space: O(n) for the normalised copy.
 */
export const preprocessText = (text: string): string => {
  const normalised = text.normalize('NFKC')
  return normalised.replace(ZERO_WIDTH_CHARS_RE, '')
}

/**
 * Wrap content in isolation delimiters to prevent prompt injection
 * breakout. Any existing closing tags in the content are escaped.
 *
 * Time: O(n) where n = content length.
 * Space: O(n) for the wrapped string.
 */
export const wrapInIsolation = ({ content, source, hash }: IsolatedContent): string => {
  const escapedContent = content.replace(CLOSING_TAG_RE, CLOSING_TAG_ESCAPED)
  return `<${ISOLATION_TAG} source="${escapeAttr(source)}" hash="${hash}">${escapedContent}</${ISOLATION_TAG}>`
}

/**
 * Escape XML attribute characters to prevent attribute injection.
 */
const escapeAttr = (value: string): string =>
  value.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

/**
 * Create a safety scanner that uses @stackone/defender for ML-based
 * prompt injection detection. The scanner loads lazily: the ONNX model
 * is loaded on first scan. If the model is unavailable, the scanner
 * degrades gracefully and logs a warning.
 *
 * Returns undefined when the defender package cannot be imported.
 */
export const createSafetyScanner = (
  config: SafetyScannerConfig,
): { scan: (text: string, signal?: AbortSignal) => Promise<SafetyScanResult> } | undefined => {
  const { logger } = config
  const threshold = config.confidenceThreshold ?? INJECTION_CONFIDENCE_THRESHOLD

  let defenderInstance: DefenderLike | undefined
  let loadAttempted = false
  let loadFailed = false

  const ensureDefender = async (): Promise<DefenderLike | undefined> => {
    if (defenderInstance !== undefined) return defenderInstance
    if (loadFailed) return undefined
    if (loadAttempted) return undefined

    loadAttempted = true
    try {
      const mod = await import('@stackone/defender')
      defenderInstance = mod.createPromptDefense({
        enableTier1: true,
        enableTier2: true,
        blockHighRisk: false,
      })
      await defenderInstance.warmupTier2()
      return defenderInstance
    } catch (err: unknown) {
      loadFailed = true
      logger.warn('safety scanner: defender unavailable, continuing without ML detection', {
        error: String(err),
      })
      return undefined
    }
  }

  const scan = async (text: string, _signal?: AbortSignal): Promise<SafetyScanResult> => {
    const preprocessed = preprocessText(text)
    const defender = await ensureDefender()

    if (defender === undefined) {
      return { injectionDetected: false, confidence: 0, detections: [] }
    }

    try {
      const result = await defender.defendToolResult(preprocessed, 'ingest_document')
      const confidence = result.tier2Score ?? 0
      const injectionDetected = confidence >= threshold
      const detections: string[] = [...result.detections]

      if (injectionDetected) {
        logger.info('safety scanner: injection detected', {
          confidence,
          detections,
          riskLevel: result.riskLevel,
        })
      }

      return { injectionDetected, confidence, detections }
    } catch (err: unknown) {
      logger.warn('safety scanner: scan failed, treating as safe', {
        error: String(err),
      })
      return { injectionDetected: false, confidence: 0, detections: [] }
    }
  }

  // Verify import is possible synchronously by attempting a dynamic import
  // We return the scanner optimistically; the actual load happens on first scan
  return { scan }
}

/**
 * Build metadata fields for a flagged document. Returns an empty object
 * when the scan did not detect injection, so spreading into metadata is
 * always safe.
 */
export const buildSafetyMetadata = (scanResult: SafetyScanResult): Partial<SafetyMetadata> => {
  if (!scanResult.injectionDetected) return {}
  return {
    injection_risk: true,
    injection_confidence: scanResult.confidence,
  }
}

/**
 * Minimal interface matching the subset of PromptDefense we consume.
 * Avoids coupling to the full @stackone/defender type surface.
 */
type DefenderLike = {
  warmupTier2(): Promise<void>
  defendToolResult(
    value: unknown,
    toolName: string,
  ): Promise<{
    allowed: boolean
    riskLevel: string
    detections: string[]
    tier2Score?: number
  }>
}
