// SPDX-License-Identifier: Apache-2.0

/**
 * LLM-powered ontology extraction from documents. Analyses document
 * content to discover node types, edge types, and business categories.
 * Handles multi-section splitting for large documents and merges results
 * via noisy-OR confidence aggregation.
 *
 * Port of apps/intelligence-service/src/services/document-processing/
 * ontology-extraction.service.ts adapted for the standalone memory
 * package.
 */

import { Buffer } from 'node:buffer'
import type { Provider } from '../llm/types.js'
import { isValidNodeType, isValidEdgeType, type TypeEntry } from './templates.js'
import type { ResolvedOntology, ResolvedType } from './store.js'

import { jaroWinklerDistance } from './similarity.js'
import { extractJSON } from '../llm/structured.js'

/** Byte count above which content is split into multiple sections. */
export const SINGLE_SECTION_THRESHOLD = 8000

/** LLM temperature used for extraction. */
export const EXTRACTION_TEMPERATURE = 0.1

/** Jaro-Winkler threshold for merging similarly-labelled types during multi-section merge. */
export const FUZZY_LABEL_MERGE = 0.88

/** Minimum confidence value considered in noisy-OR aggregation. */
export const CONFIDENCE_FLOOR = 0.3

/** Maximum confidence value returned by noisy-OR aggregation. */
export const CONFIDENCE_CAP = 0.99

/** Maximum number of retries when the LLM returns malformed output. */
const DEFAULT_MAX_RETRIES = 2

export type ExtractionResult = {
  readonly nodeTypes: TypeEntry[]
  readonly edgeTypes: TypeEntry[]
  readonly businessCategories: string[]
  readonly domain: string
  readonly confidence: number
}

export type ExtractionParams = {
  readonly content: string
  readonly fileName: string
  readonly existingTypes?: ResolvedOntology
}

export type ExtractorOptions = {
  readonly provider: Provider | undefined
  /** Override the default temperature (0.1). */
  readonly temperature?: number
  /** Override the default max retries (2). */
  readonly maxRetries?: number
}

/**
 * Extractor performs LLM-powered ontology extraction from documents.
 * Returns empty results without error when no provider is configured.
 */
export class Extractor {
  private readonly provider: Provider | undefined
  private readonly temperature: number
  private readonly maxRetries: number

  constructor(options: ExtractorOptions) {
    this.provider = options.provider
    this.temperature = options.temperature ?? EXTRACTION_TEMPERATURE
    this.maxRetries = options.maxRetries ?? DEFAULT_MAX_RETRIES
  }

  /**
   * Analyse document content and return discovered ontology types.
   * Returns an empty result without error when the provider is undefined.
   */
  async extract(params: ExtractionParams, signal?: AbortSignal): Promise<ExtractionResult> {
    if (this.provider === undefined) {
      return emptyExtractionResult()
    }

    const content = params.content
    if (content.length === 0) {
      return emptyExtractionResult()
    }

    if (content.length <= SINGLE_SECTION_THRESHOLD) {
      return this.extractSingleSection(content, params, signal)
    }

    return this.extractMultiSection(content, params, signal)
  }

  private async extractSingleSection(
    content: string,
    params: ExtractionParams,
    signal?: AbortSignal,
  ): Promise<ExtractionResult> {
    const systemPrompt = buildOntologyExtractionPrompt(params.existingTypes)
    const userMsg = buildUserMessage(content, params.fileName)
    let result = await this.callLLMWithRetry(systemPrompt, userMsg, signal)

    if (params.existingTypes !== undefined) {
      result = filterExistingTypes(result, params.existingTypes)
    }

    return result
  }

  private async extractMultiSection(
    content: string,
    params: ExtractionParams,
    signal?: AbortSignal,
  ): Promise<ExtractionResult> {
    const sections = splitContent(content, SINGLE_SECTION_THRESHOLD)
    const allResults: ExtractionResult[] = []
    const discoveredNodeTypes: TypeEntry[] = []
    const discoveredEdgeTypes: TypeEntry[] = []

    for (let i = 0; i < sections.length; i++) {
      let systemPrompt = buildOntologyExtractionPrompt(params.existingTypes)
      if (discoveredNodeTypes.length > 0 || discoveredEdgeTypes.length > 0) {
        systemPrompt += buildContextPrefix(discoveredNodeTypes, discoveredEdgeTypes)
      }

      let userMsg = buildUserMessage(sections[i]!, params.fileName)
      if (sections.length > 1) {
        userMsg = `[Section ${i + 1} of ${sections.length}]\n\n${userMsg}`
      }

      const result = await this.callLLMWithRetry(systemPrompt, userMsg, signal)
      allResults.push(result)
      discoveredNodeTypes.push(...result.nodeTypes)
      discoveredEdgeTypes.push(...result.edgeTypes)
    }

    if (allResults.length === 0) {
      return emptyExtractionResult()
    }

    let merged = mergeOntologyExtractions(allResults)

    if (params.existingTypes !== undefined) {
      merged = filterExistingTypes(merged, params.existingTypes)
    }

    return merged
  }

  private async callLLMWithRetry(
    systemPrompt: string,
    userMsg: string,
    signal?: AbortSignal,
  ): Promise<ExtractionResult> {
    const messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userMsg },
    ]

    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      signal?.throwIfAborted()

      const resp = await this.provider!.complete(
        {
          messages,
          temperature: this.temperature,
          maxTokens: 4096,
          jsonMode: true,
        },
        signal,
      )

      const parseResult = parseExtractionResponse(resp.content)
      if (parseResult !== undefined) {
        return parseResult
      }

      if (attempt < this.maxRetries) {
        messages.push(
          { role: 'assistant', content: resp.content },
          {
            role: 'user',
            content:
              'Your previous response was not valid JSON matching the expected schema. Return only valid JSON with the keys: domain, confidence, nodeTypes, edgeTypes, businessCategories.',
          },
        )
        continue
      }

      throw new Error(`ontology: extraction failed after ${this.maxRetries} retries`)
    }

    throw new Error('ontology: extraction exhausted retries')
  }
}

/**
 * Parse an LLM response into an ExtractionResult.
 * Returns undefined when the response cannot be parsed.
 */
function parseExtractionResponse(text: string): ExtractionResult | undefined {
  let jsonStr: string
  try {
    jsonStr = extractJSON(text)
  } catch {
    return undefined
  }

  let raw: RawExtractionResult
  try {
    raw = JSON.parse(jsonStr) as RawExtractionResult
  } catch {
    return undefined
  }

  if (typeof raw.domain !== 'string' || raw.domain === '') {
    return undefined
  }

  const nodeTypes: TypeEntry[] = []
  if (Array.isArray(raw.nodeTypes)) {
    for (const nt of raw.nodeTypes) {
      if (
        typeof nt === 'object' &&
        nt !== null &&
        typeof nt.type === 'string' &&
        typeof nt.label === 'string' &&
        typeof nt.description === 'string' &&
        nt.type !== '' &&
        nt.label !== '' &&
        nt.description !== '' &&
        isValidNodeType(nt.type)
      ) {
        nodeTypes.push({ type: nt.type, label: nt.label, description: nt.description })
      }
    }
  }

  const edgeTypes: TypeEntry[] = []
  if (Array.isArray(raw.edgeTypes)) {
    for (const et of raw.edgeTypes) {
      if (
        typeof et === 'object' &&
        et !== null &&
        typeof et.type === 'string' &&
        typeof et.label === 'string' &&
        typeof et.description === 'string' &&
        et.type !== '' &&
        et.label !== '' &&
        et.description !== '' &&
        isValidEdgeType(et.type)
      ) {
        edgeTypes.push({ type: et.type, label: et.label, description: et.description })
      }
    }
  }

  const businessCategories: string[] = []
  if (Array.isArray(raw.businessCategories)) {
    for (const cat of raw.businessCategories) {
      if (typeof cat === 'string' && cat !== '') {
        businessCategories.push(cat)
      }
    }
  }

  return {
    domain: raw.domain,
    confidence: typeof raw.confidence === 'number' ? raw.confidence : 0,
    nodeTypes,
    edgeTypes,
    businessCategories,
  }
}

type RawExtractionResult = {
  domain: string
  confidence: number
  nodeTypes: ReadonlyArray<{ type: string; label: string; description: string }>
  edgeTypes: ReadonlyArray<{ type: string; label: string; description: string }>
  businessCategories: readonly string[]
}

/**
 * Compute the noisy-OR confidence aggregation.
 * Filters values below CONFIDENCE_FLOOR, caps at CONFIDENCE_CAP.
 * Returns 0 for empty input.
 */
export function noisyOr(confidences: readonly number[]): number {
  if (confidences.length === 0) {
    return 0
  }

  const filtered = confidences.filter((c) => c >= CONFIDENCE_FLOOR)

  if (filtered.length === 0) {
    return 0
  }

  if (filtered.length === 1) {
    return Math.min(filtered[0]!, CONFIDENCE_CAP)
  }

  let product = 1.0
  for (const c of filtered) {
    product *= 1.0 - c
  }

  const result = 1.0 - product
  return Math.min(result, CONFIDENCE_CAP)
}

function emptyExtractionResult(): ExtractionResult {
  return {
    nodeTypes: [],
    edgeTypes: [],
    businessCategories: [],
    domain: '',
    confidence: 0,
  }
}

/**
 * Merge results from multi-section extraction. Deduplicates by type key
 * (keeps longest description), fuzzy-deduplicates labels within the same
 * prefix (Jaro-Winkler >= 0.88), and aggregates confidence via noisy-OR.
 */
function mergeOntologyExtractions(results: readonly ExtractionResult[]): ExtractionResult {
  if (results.length === 0) {
    return emptyExtractionResult()
  }
  if (results.length === 1) {
    return results[0]!
  }

  const nodeMap = new Map<string, TypeEntry>()
  const edgeMap = new Map<string, TypeEntry>()
  const catSet = new Set<string>()
  const domains = new Map<string, number>()
  const confidences: number[] = []

  for (const r of results) {
    for (const nt of r.nodeTypes) {
      const existing = nodeMap.get(nt.type)
      if (existing === undefined || nt.description.length > existing.description.length) {
        nodeMap.set(nt.type, nt)
      }
    }
    for (const et of r.edgeTypes) {
      const existing = edgeMap.get(et.type)
      if (existing === undefined || et.description.length > existing.description.length) {
        edgeMap.set(et.type, et)
      }
    }
    for (const cat of r.businessCategories) {
      catSet.add(cat)
    }
    if (r.domain !== '') {
      domains.set(r.domain, (domains.get(r.domain) ?? 0) + 1)
    }
    if (r.confidence > 0) {
      confidences.push(r.confidence)
    }
  }

  fuzzyDedupByPrefix(nodeMap)
  fuzzyDedupEdges(edgeMap)

  let bestDomain = ''
  let bestCount = 0
  for (const [domain, count] of domains) {
    if (count > bestCount) {
      bestCount = count
      bestDomain = domain
    }
  }

  return {
    nodeTypes: [...nodeMap.values()],
    edgeTypes: [...edgeMap.values()],
    businessCategories: [...catSet],
    domain: bestDomain,
    confidence: noisyOr(confidences),
  }
}

function fuzzyDedupByPrefix(nodeMap: Map<string, TypeEntry>): void {
  const byPrefix = new Map<string, string[]>()
  for (const key of nodeMap.keys()) {
    const prefix = typePrefix(key)
    const keys = byPrefix.get(prefix)
    if (keys !== undefined) {
      keys.push(key)
    } else {
      byPrefix.set(prefix, [key])
    }
  }

  for (const keys of byPrefix.values()) {
    for (let i = 0; i < keys.length; i++) {
      for (let j = i + 1; j < keys.length; j++) {
        const entryI = nodeMap.get(keys[i]!)
        const entryJ = nodeMap.get(keys[j]!)
        if (entryI === undefined || entryJ === undefined) continue
        const sim = jaroWinklerDistance(entryI.label, entryJ.label)
        if (sim >= FUZZY_LABEL_MERGE) {
          if (entryJ.description.length > entryI.description.length) {
            nodeMap.delete(keys[i]!)
          } else {
            nodeMap.delete(keys[j]!)
          }
        }
      }
    }
  }
}

function fuzzyDedupEdges(edgeMap: Map<string, TypeEntry>): void {
  const keys = [...edgeMap.keys()]
  for (let i = 0; i < keys.length; i++) {
    for (let j = i + 1; j < keys.length; j++) {
      const entryI = edgeMap.get(keys[i]!)
      const entryJ = edgeMap.get(keys[j]!)
      if (entryI === undefined || entryJ === undefined) continue
      const sim = jaroWinklerDistance(entryI.label, entryJ.label)
      if (sim >= FUZZY_LABEL_MERGE) {
        if (entryJ.description.length > entryI.description.length) {
          edgeMap.delete(keys[i]!)
        } else {
          edgeMap.delete(keys[j]!)
        }
      }
    }
  }
}

function filterExistingTypes(
  result: ExtractionResult,
  existing: ResolvedOntology,
): ExtractionResult {
  const existingNodeSet = new Set<string>()
  for (const nt of existing.nodeTypes) {
    existingNodeSet.add(nt.type)
  }

  const existingEdgeSet = new Set<string>()
  for (const et of existing.edgeTypes) {
    existingEdgeSet.add(et.type)
  }

  const existingCatSet = new Set<string>(existing.businessCategories)

  return {
    domain: result.domain,
    confidence: result.confidence,
    nodeTypes: result.nodeTypes.filter((nt) => !existingNodeSet.has(nt.type)),
    edgeTypes: result.edgeTypes.filter((et) => !existingEdgeSet.has(et.type)),
    businessCategories: result.businessCategories.filter((cat) => !existingCatSet.has(cat)),
  }
}

function typePrefix(typeID: string): string {
  const dotIndex = typeID.indexOf('.')
  if (dotIndex < 0) return typeID
  return typeID.slice(0, dotIndex)
}

/**
 * Build the system prompt for ontology extraction.
 */
export function buildOntologyExtractionPrompt(existingTypes?: ResolvedOntology): string {
  let prompt = `You are a domain ontology analyst. Analyse this document and extract the SCHEMA -- the types of entities, rules, and relationships that exist in this domain.

Do NOT extract individual data items or instances. Instead, identify the CATEGORIES and TYPES of things described.

For each discovered type, provide:
- type: A dotted identifier following the pattern prefix.name where prefix is one of: entity, rule, exception, decision, process. The name should be snake_case.
- label: A human-readable name
- description: A one-sentence explanation of what this type represents

Also identify:
- Edge types: the kinds of relationships between entities (use snake_case identifiers like requires, compatible_with, belongs_to)
- Business categories: the high-level domain areas covered (use snake_case like server_hardware, customer_management)

Analyse the document structure, column headers, data patterns, and content to infer the domain ontology.

Examples of good ontology design patterns:

Pattern 1 - Entity-Rule binding:
entity.product is constrained by rule.pricing via the "constrains" edge.

Pattern 2 - Process-Decision-Exception:
process.approval_chain contains decision.escalation points that may trigger exception.override.

Pattern 3 - Classification hierarchy:
entity.category relates to entity.subcategory via "belongs_to" edge.

Return your analysis as a JSON object with the following structure:
{
  "domain": "string describing the domain",
  "confidence": 0.0 to 1.0,
  "nodeTypes": [{"type": "prefix.name", "label": "Human Label", "description": "One sentence"}],
  "edgeTypes": [{"type": "snake_case_name", "label": "Human Label", "description": "One sentence"}],
  "businessCategories": ["snake_case_category"]
}`

  if (
    existingTypes !== undefined &&
    (existingTypes.nodeTypes.length > 0 || existingTypes.edgeTypes.length > 0)
  ) {
    prompt +=
      '\n\nThe following types already exist in the ontology. Do NOT include these in your response -- only return NEW types that are not already present:\n'
    prompt += buildExistingTypesSection(existingTypes)
  }

  return prompt
}

function buildExistingTypesSection(existing: ResolvedOntology): string {
  const parts: string[] = []
  if (existing.nodeTypes.length > 0) {
    parts.push('\nExisting node types:')
    for (const nt of existing.nodeTypes) {
      parts.push(`- ${nt.type}: ${nt.label}`)
    }
  }
  if (existing.edgeTypes.length > 0) {
    parts.push('\nExisting edge types:')
    for (const et of existing.edgeTypes) {
      parts.push(`- ${et.type}: ${et.label}`)
    }
  }
  return parts.join('\n') + '\n'
}

function buildUserMessage(content: string, fileName: string): string {
  const sanitised = sanitiseFileName(fileName)
  if (sanitised !== '') {
    return `Document: ${sanitised}\n\n<ingested-document>\n${content}\n</ingested-document>`
  }
  return `<ingested-document>\n${content}\n</ingested-document>`
}

/**
 * Strip newlines, control characters, and trim the filename to
 * prevent prompt injection via crafted file names.
 */
function sanitiseFileName(name: string): string {
  const cleaned = name.replace(/[\x00-\x1f\x7f-\x9f]/g, '').trim()
  return cleaned.length > 256 ? cleaned.slice(0, 256) : cleaned
}

function buildContextPrefix(
  nodeTypes: readonly TypeEntry[],
  edgeTypes: readonly TypeEntry[],
): string {
  const lines = [
    '\n\nTypes discovered from earlier sections of this document (avoid duplicating these):',
  ]
  if (nodeTypes.length > 0) {
    lines.push('\nDiscovered node types:')
    for (const t of nodeTypes) {
      lines.push(`- ${t.type}: ${t.label}`)
    }
  }
  if (edgeTypes.length > 0) {
    lines.push('\nDiscovered edge types:')
    for (const t of edgeTypes) {
      lines.push(`- ${t.type}: ${t.label}`)
    }
  }
  return lines.join('\n') + '\n'
}

/**
 * Split content into sections of approximately maxBytes.
 * Detects tabular content and samples header + data rows per section.
 * Otherwise splits on markdown headings or at byte boundaries.
 */
export function splitContent(content: string, maxBytes: number): string[] {
  if (Buffer.byteLength(content, 'utf8') <= maxBytes) {
    return [content]
  }

  if (isTabularContent(content)) {
    return splitTabularContent(content, maxBytes)
  }

  return splitByHeadingsOrBytes(content, maxBytes)
}

/**
 * Returns true if the content looks like CSV/TSV data.
 * Heuristic: >= 2 delimiters per line on >= 3 of the first 5 lines.
 */
export function isTabularContent(content: string): boolean {
  const lines = content.split('\n', 6)
  if (lines.length < 3) return false

  const limit = Math.min(5, lines.length)
  let tabularLines = 0
  for (let i = 0; i < limit; i++) {
    const line = lines[i]!
    const commas = (line.match(/,/g) ?? []).length
    const pipes = (line.match(/\|/g) ?? []).length
    if (commas >= 2 || pipes >= 2) {
      tabularLines++
    }
  }

  return tabularLines >= 3
}

function splitTabularContent(content: string, maxBytes: number): string[] {
  const lines = content.split('\n')
  if (lines.length === 0) return [content]

  const header = lines[0]!
  const dataLines = lines.slice(1)
  if (dataLines.length === 0) return [content]

  const headerBytes = Buffer.byteLength(header, 'utf8') + 1 // +1 for newline
  const sections: string[] = []
  let sectionLines: string[] = []
  let currentBytes = headerBytes

  for (const line of dataLines) {
    const lineBytes = Buffer.byteLength(line, 'utf8') + 1
    if (currentBytes + lineBytes > maxBytes && sectionLines.length > 0) {
      sections.push(header + '\n' + sectionLines.join('\n'))
      sectionLines = []
      currentBytes = headerBytes
    }
    sectionLines.push(line)
    currentBytes += lineBytes
  }

  if (sectionLines.length > 0) {
    sections.push(header + '\n' + sectionLines.join('\n'))
  }

  return sections
}

function splitByHeadingsOrBytes(content: string, maxBytes: number): string[] {
  const lines = content.split('\n')
  const sections: string[] = []
  let currentSection = ''
  let currentBytes = 0

  for (const line of lines) {
    const lineBytes = Buffer.byteLength(line, 'utf8') + 1

    const isHeading = line.startsWith('# ') || line.startsWith('## ') || line.startsWith('### ')

    if (isHeading && currentBytes > 0) {
      sections.push(currentSection)
      currentSection = ''
      currentBytes = 0
    }

    if (currentBytes + lineBytes > maxBytes && currentBytes > 0) {
      sections.push(currentSection)
      currentSection = ''
      currentBytes = 0
    }

    if (currentBytes > 0) {
      currentSection += '\n'
      currentBytes++
    }
    currentSection += line
    currentBytes += Buffer.byteLength(line, 'utf8')
  }

  if (currentBytes > 0) {
    sections.push(currentSection)
  }

  return sections
}
