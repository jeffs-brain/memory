export type DocumentBodyLimits = {
  readonly singleDocumentBytes: number
  readonly batchDecodedBytes: number
  readonly batchOpCount: number
}

export const DEFAULT_DOCUMENT_BODY_LIMITS: DocumentBodyLimits = {
  singleDocumentBytes: 2 * 1024 * 1024,
  batchDecodedBytes: 8 * 1024 * 1024,
  batchOpCount: 1024,
}

const sanitisePositiveInt = (value: number | undefined, fallback: number): number => {
  if (value === undefined) return fallback
  if (!Number.isInteger(value) || value <= 0) return fallback
  return value
}

export const normaliseDocumentBodyLimits = (
  overrides: Partial<DocumentBodyLimits> = {},
): DocumentBodyLimits => ({
  singleDocumentBytes: sanitisePositiveInt(
    overrides.singleDocumentBytes,
    DEFAULT_DOCUMENT_BODY_LIMITS.singleDocumentBytes,
  ),
  batchDecodedBytes: sanitisePositiveInt(
    overrides.batchDecodedBytes,
    DEFAULT_DOCUMENT_BODY_LIMITS.batchDecodedBytes,
  ),
  batchOpCount: sanitisePositiveInt(
    overrides.batchOpCount,
    DEFAULT_DOCUMENT_BODY_LIMITS.batchOpCount,
  ),
})
