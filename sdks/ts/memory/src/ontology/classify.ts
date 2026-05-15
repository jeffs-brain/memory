// SPDX-License-Identifier: Apache-2.0

/**
 * Document classification and entity tagging for the ontology system.
 * Classifies documents as structured (JSON), tabular (CSV/TSV), or
 * unstructured (prose), and tags chunks with ontology entity types.
 *
 * Classification cascade: JSON detection -> tabular detection -> LLM.
 * Port of go/ontology/classify.go.
 */

import type { Provider } from '../llm/types.js'
import type { ResolvedOntology } from './store.js'
import { isValidBusinessCategory } from './validation.js'

/**
 * Minimum proportion of entity type matches required for a business
 * category to win classification via weighted voting.
 */
export const CATEGORY_WINNER_THRESHOLD = 0.4

export type DocumentClass = 'structured' | 'tabular' | 'unstructured'

export type ClassificationResult = {
  readonly class: DocumentClass
  readonly category: string
  readonly confidence: number
  readonly isStructured: boolean
}

export type ClassifierOptions = {
  readonly provider?: Provider
}

/**
 * Classifier classifies documents and tags chunks with type metadata.
 * When a Provider is supplied, unstructured documents use LLM-powered
 * classification. Without a Provider, heuristic-only mode is used.
 */
export class Classifier {
  private readonly provider: Provider | undefined

  constructor(options: ClassifierOptions) {
    this.provider = options.provider
  }

  /**
   * Determines the document class and business category.
   * Classification cascade: JSON -> tabular -> LLM fallback.
   */
  async classify(
    content: string,
    fileName: string,
    signal?: AbortSignal,
  ): Promise<ClassificationResult> {
    if (isJsonDocument(content)) {
      return {
        class: 'structured',
        category: inferCategoryFromJSON(content),
        confidence: 0.95,
        isStructured: true,
      }
    }

    if (isTabularDocument(content)) {
      return {
        class: 'tabular',
        category: inferCategoryFromFileName(fileName),
        confidence: 0.85,
        isStructured: true,
      }
    }

    if (this.provider !== undefined) {
      return this.classifyWithLLM(content, fileName, signal)
    }

    return {
      class: 'unstructured',
      category: 'general',
      confidence: 0.5,
      isStructured: false,
    }
  }

  private async classifyWithLLM(
    content: string,
    fileName: string,
    signal?: AbortSignal,
  ): Promise<ClassificationResult> {
    const preview = content.length > 2000 ? content.slice(0, 2000) : content
    const prompt = buildClassificationPrompt(preview, fileName)

    try {
      const resp = await this.provider!.complete(
        {
          messages: [
            { role: 'system', content: CLASSIFICATION_SYSTEM_PROMPT },
            { role: 'user', content: prompt },
          ],
          temperature: 0.1,
          maxTokens: 256,
          jsonMode: true,
        },
        signal,
      )

      return parseClassificationResponse(resp.content)
    } catch {
      return {
        class: 'unstructured',
        category: 'general',
        confidence: 0.3,
        isStructured: false,
      }
    }
  }
}

const CLASSIFICATION_SYSTEM_PROMPT = `You are a document classifier. Analyse the provided document content and classify it into one of these categories:
- entity: Documents about customers, products, suppliers, or other business entities
- rule: Documents describing business rules, constraints, policies, or validation logic
- exception: Documents about workarounds, overrides, or special cases
- decision: Documents about decision trees, escalation paths, or branch logic
- process: Documents about workflows, procedures, approval chains, or integrations
- reference: General reference documentation that does not fit the above

Respond with a JSON object containing:
- "category": one of "entity", "rule", "exception", "decision", "process", "reference"
- "confidence": a number between 0 and 1 indicating your confidence
- "reasoning": a brief explanation of why you chose this category`

function buildClassificationPrompt(content: string, fileName: string): string {
  const parts: string[] = []
  if (fileName !== '') {
    parts.push(`File: ${fileName}\n\n`)
  }
  parts.push('Document content:\n<ingested-document>\n')
  parts.push(content)
  parts.push('\n</ingested-document>')
  return parts.join('')
}

type LLMClassification = {
  readonly category?: string
  readonly confidence?: number
}

/**
 * Finds the first balanced JSON object in text using depth-tracking.
 * Returns undefined if no valid object is found.
 */
function extractJSONObject(text: string): string | undefined {
  const start = text.indexOf('{')
  if (start < 0) return undefined

  let depth = 0
  let inString = false
  let escaping = false

  for (let i = start; i < text.length; i++) {
    const ch = text[i]
    if (escaping) {
      escaping = false
      continue
    }
    if (inString) {
      if (ch === '\\') escaping = true
      else if (ch === '"') inString = false
      continue
    }
    if (ch === '"') {
      inString = true
      continue
    }
    if (ch === '{') {
      depth++
    } else if (ch === '}') {
      depth--
      if (depth === 0) {
        const candidate = text.slice(start, i + 1)
        try {
          JSON.parse(candidate)
          return candidate
        } catch {
          return undefined
        }
      }
    }
  }

  return undefined
}

function parseClassificationResponse(text: string): ClassificationResult {
  const trimmed = text.trim()
  const jsonStr = extractJSONObject(trimmed)

  if (jsonStr === undefined) {
    return { class: 'unstructured', category: 'general', confidence: 0.3, isStructured: false }
  }

  try {
    const parsed = JSON.parse(jsonStr) as LLMClassification
    const category = mapLLMCategory(parsed.category ?? '')
    let confidence = parsed.confidence ?? 0.7
    if (confidence <= 0 || confidence > 1) confidence = 0.7

    return {
      class: 'unstructured',
      category,
      confidence,
      isStructured: false,
    }
  } catch {
    return { class: 'unstructured', category: 'general', confidence: 0.3, isStructured: false }
  }
}

function mapLLMCategory(raw: string): string {
  const categoryMap: Record<string, string> = {
    entity: 'entity',
    rule: 'rule',
    exception: 'exception',
    decision: 'decision',
    process: 'process',
    reference: 'reference',
    customer: 'customer',
    order: 'order',
    product: 'product',
    address: 'address',
    document: 'document',
  }
  const lower = raw.toLowerCase().trim()
  const mapped = categoryMap[lower]
  if (mapped !== undefined) return mapped
  if (isValidBusinessCategory(lower)) return lower
  return 'general'
}

/**
 * Returns true if content parses as JSON with business-relevant
 * structure (object or array, not a bare primitive).
 */
export function isJsonDocument(content: string): boolean {
  const trimmed = content.trim()
  if (trimmed.length === 0) return false

  const first = trimmed[0]
  if (first !== '{' && first !== '[') return false

  try {
    const parsed: unknown = JSON.parse(trimmed)
    if (parsed === null || typeof parsed !== 'object') return false

    // Require non-empty structure
    if (Array.isArray(parsed)) {
      // Empty arrays and primitive arrays are not business-relevant JSON
      if (parsed.length === 0) return false
      // At least one element must be an object or nested array
      return parsed.some(
        (item) => typeof item === 'object' && item !== null,
      )
    }

    // Non-empty object
    return Object.keys(parsed).length > 0
  } catch {
    return false
  }
}

/**
 * Returns true if content appears to be CSV/TSV data. Uses a heuristic:
 * checks the first 5 lines for consistent comma or pipe delimiters
 * (>= 2 delimiters per line on >= 3 of the first 5 lines).
 */
export function isTabularDocument(content: string): boolean {
  const lines = content.split('\n').slice(0, 5)
  if (lines.length < 3) return false

  let commaCount = 0
  let pipeCount = 0

  for (const line of lines) {
    const trimmed = line.trim()
    if (trimmed === '') continue

    const commas = countOccurrences(trimmed, ',')
    const pipes = countOccurrences(trimmed, '|')

    if (commas >= 2) commaCount++
    if (pipes >= 2) pipeCount++
  }

  return commaCount >= 3 || pipeCount >= 3
}

function countOccurrences(str: string, char: string): number {
  let count = 0
  for (let i = 0; i < str.length; i++) {
    if (str[i] === char) count++
  }
  return count
}

function inferCategoryFromJSON(content: string): string {
  const lower = content.toLowerCase()
  const categoryKeywords: Record<string, readonly string[]> = {
    customer: ['customer', 'client', 'account'],
    order: ['order', 'purchase', 'transaction'],
    product: ['product', 'item', 'sku', 'catalog'],
    address: ['address', 'location', 'postal', 'zip'],
    document: ['document', 'file', 'attachment'],
    authorization: ['authorization', 'permission', 'role', 'access'],
    integration: ['integration', 'api', 'endpoint', 'webhook'],
  }

  let bestCategory = ''
  let bestCount = 0

  for (const [category, keywords] of Object.entries(categoryKeywords)) {
    let count = 0
    for (const kw of keywords) {
      count += countStringOccurrences(lower, kw)
    }
    if (count > bestCount) {
      bestCount = count
      bestCategory = category
    }
  }

  if (bestCategory !== '' && bestCount >= 2) return bestCategory
  return 'general'
}

function inferCategoryFromFileName(fileName: string): string {
  const lower = fileName.toLowerCase()
  const keywords: Record<string, string> = {
    customer: 'customer',
    order: 'order',
    product: 'product',
    address: 'address',
    invoice: 'order',
    shipping: 'order',
    price: 'product',
    auth: 'authorization',
    api: 'integration',
  }
  for (const [kw, category] of Object.entries(keywords)) {
    if (lower.includes(kw)) return category
  }
  return 'general'
}

function countStringOccurrences(haystack: string, needle: string): number {
  let count = 0
  let idx = 0
  while (true) {
    idx = haystack.indexOf(needle, idx)
    if (idx < 0) break
    count++
    idx += needle.length
  }
  return count
}

/**
 * Uses ontology-aware weighted voting to determine the best business
 * category for content.
 */
export function determineCategory(content: string, ontology: ResolvedOntology | undefined): string {
  if (ontology === undefined) return 'general'

  const counts = buildCategoryCounts(content, ontology)
  return categoryWinner(counts)
}

function buildCategoryCounts(
  content: string,
  ontology: ResolvedOntology,
): Record<string, number> {
  const counts: Record<string, number> = {}
  const lower = content.toLowerCase()

  for (const nt of ontology.nodeTypes) {
    const dotIdx = nt.type.indexOf('.')
    if (dotIdx < 0) continue
    const name = nt.type.slice(dotIdx + 1)
    const spaced = name.replace(/_/g, ' ')

    if (lower.includes(name) || lower.includes(spaced)) {
      for (const cat of ontology.businessCategories) {
        if (nt.description.toLowerCase().includes(cat)) {
          counts[cat] = (counts[cat] ?? 0) + 1
        }
      }
    }
  }

  return counts
}

function categoryWinner(counts: Record<string, number>): string {
  let total = 0
  for (const c of Object.values(counts)) {
    total += c
  }
  if (total === 0) return 'general'

  let bestCategory = ''
  let bestCount = 0
  for (const [cat, count] of Object.entries(counts)) {
    if (count > bestCount) {
      bestCount = count
      bestCategory = cat
    }
  }

  if (bestCount / total >= CATEGORY_WINNER_THRESHOLD) return bestCategory
  return 'general'
}
