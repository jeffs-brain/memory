// SPDX-License-Identifier: Apache-2.0

/**
 * ExtractionResult holds extracted ontology types from a document.
 * Defined in a shared file so that both template matching (P6-6) and
 * the proposal workflow (P6-7) can reference it without duplication.
 * When P6-4 is merged, the canonical definition in extract.ts will
 * supersede this one.
 */

import type { TypeEntry } from './templates.js'

export type ExtractionResult = {
  readonly nodeTypes: readonly TypeEntry[]
  readonly edgeTypes: readonly TypeEntry[]
  readonly businessCategories: readonly string[]
  readonly domain: string
  readonly confidence: number
}
