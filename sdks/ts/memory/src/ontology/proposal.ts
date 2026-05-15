// SPDX-License-Identifier: Apache-2.0

/**
 * Proposal workflow for ontology type discovery and review. Creates
 * proposals from extraction results, runs deduplication against the
 * resolved ontology, and provides accept/merge/reject operations with
 * full audit trail.
 *
 * Port of go/ontology/proposal.go.
 */

import type { Store } from '../store/index.js'
import { isNotFound } from '../store/errors.js'
import { type Path, toPath } from '../store/path.js'
import type { OntologyTypeDefinition, TypeStatus } from './types.js'
import type { ResolvedOntology } from './store.js'
import type { Registry } from './registry.js'
import type { TypeEntry } from './templates.js'
import type { ExtractionResult } from './types-extraction.js'
import { hashString } from '../ingest/hash.js'
import { Deduplicator } from './dedup.js'
import { formatEdgeTypeLabel, formatNodeTypeLabel } from './format.js'
import { hasPrefix } from './validation.js'

export type { ExtractionResult } from './types-extraction.js'

export type ProposalStatus = 'proposed' | 'accepted' | 'merged' | 'rejected'

export type Proposal = {
  readonly id: string
  readonly type: string
  readonly label: string
  readonly description: string
  readonly category: 'nodeType' | 'edgeType'
  readonly discoveredFrom: string
  readonly createdAt: string
  readonly status: ProposalStatus
  readonly reviewedBy?: string
  readonly reviewedAt?: string
  readonly mergedInto?: string
}

export type ProposalBatch = {
  readonly id: string
  readonly proposals: readonly Proposal[]
  readonly domain: string
  readonly sourceDocument: string
  readonly createdAt: string
}

export type ProposalFilter = {
  readonly status?: ProposalStatus
  readonly category?: 'nodeType' | 'edgeType'
}

export type ProposalWorkflowConfig = {
  readonly registry: Registry
  readonly store: Store
  readonly clock?: () => Date
}

type MutableProposal = {
  id: string
  type: string
  label: string
  description: string
  category: 'nodeType' | 'edgeType'
  discoveredFrom: string
  createdAt: string
  status: ProposalStatus
  reviewedBy?: string
  reviewedAt?: string
  mergedInto?: string
}

type MutableProposalBatch = {
  id: string
  proposals: MutableProposal[]
  domain: string
  sourceDocument: string
  createdAt: string
}

function proposalBatchPath(batchId: string): Path {
  return toPath(`ontology/proposals/${batchId}.json`)
}

const PROPOSALS_DIR: Path = toPath('ontology/proposals')

function computeProposalId(typeName: string, batchId: string): string {
  return hashString(`${typeName}:${batchId}`).slice(0, 16)
}

function computeBatchId(domain: string, sourceDocument: string, timestamp: string): string {
  return hashString(`${domain}:${sourceDocument}:${timestamp}`).slice(0, 16)
}

/**
 * ProposalWorkflow manages the type proposal lifecycle. Creates
 * proposals from extraction results, stores them as JSON in the
 * brain store, and provides accept/merge/reject operations.
 */
export class ProposalWorkflow {
  private readonly registry: Registry
  private readonly store: Store
  private readonly clock: () => Date

  constructor(config: ProposalWorkflowConfig) {
    this.registry = config.registry
    this.store = config.store
    this.clock = config.clock ?? (() => new Date())
  }

  /**
   * Creates proposals for unique types from an extraction result.
   * Runs deduplication against the resolved ontology first.
   */
  async proposeFromExtraction(
    result: ExtractionResult,
    sourceDocument: string,
  ): Promise<ProposalBatch> {
    const resolved = await this.registry.resolve('', '', '')
    const existingTypes = resolvedToDefinitions(resolved)
    const extractedDefs = extractionToDefinitions(result, this.clock)

    const dedup = new Deduplicator({})
    const dedupResult = await dedup.deduplicate(extractedDefs, existingTypes)

    const now = this.clock().toISOString()
    const batchId = computeBatchId(result.domain, sourceDocument, now)

    const proposals: Proposal[] = dedupResult.unique.map((unique) => {
      const category: 'nodeType' | 'edgeType' = hasPrefix(unique.type) !== undefined ? 'nodeType' : 'edgeType'
      return {
        id: computeProposalId(unique.type, batchId),
        type: unique.type,
        label: unique.label,
        description: unique.description,
        category,
        discoveredFrom: sourceDocument,
        createdAt: now,
        status: 'proposed' as ProposalStatus,
      }
    })

    const batch: ProposalBatch = {
      id: batchId,
      proposals,
      domain: result.domain,
      sourceDocument,
      createdAt: now,
    }

    await this.writeBatch(batch)
    return batch
  }

  /**
   * Activates a proposed type in the registry at brain scope.
   * Idempotent for already-accepted proposals.
   */
  async accept(batchId: string, proposalId: string, reviewedBy: string): Promise<void> {
    const batch = await this.readBatch(batchId)
    const { proposal, index } = findProposal(batch, proposalId)

    if (proposal.status === 'accepted') return

    if (proposal.status !== 'proposed') {
      throw new Error(`ontology: cannot accept proposal "${proposalId}" with status "${proposal.status}"`)
    }

    const now = this.clock().toISOString()
    const label = proposal.label !== ''
      ? proposal.label
      : proposal.category === 'nodeType'
        ? formatNodeTypeLabel(proposal.type)
        : formatEdgeTypeLabel(proposal.type)

    const def: OntologyTypeDefinition = {
      type: proposal.type,
      label,
      description: proposal.description,
      createdAt: now,
      status: 'active' as TypeStatus,
    }

    await this.registry.registerType('brain', def)

    const mutable = toMutableBatch(batch)
    const target = mutable.proposals[index]
    if (target === undefined) throw new Error('ontology: proposal index out of bounds')
    target.status = 'accepted'
    target.reviewedBy = reviewedBy
    target.reviewedAt = now

    await this.writeBatch(mutable)
  }

  /**
   * Marks a proposal as merged into an existing type. No registry
   * change; the decision is recorded for audit purposes.
   * Idempotent for already-merged proposals.
   */
  async merge(batchId: string, proposalId: string, targetType: string, reviewedBy: string): Promise<void> {
    const batch = await this.readBatch(batchId)
    const { proposal, index } = findProposal(batch, proposalId)

    if (proposal.status === 'merged') return

    if (proposal.status !== 'proposed') {
      throw new Error(`ontology: cannot merge proposal "${proposalId}" with status "${proposal.status}"`)
    }

    const now = this.clock().toISOString()
    const mutable = toMutableBatch(batch)
    const target = mutable.proposals[index]
    if (target === undefined) throw new Error('ontology: proposal index out of bounds')
    target.status = 'merged'
    target.mergedInto = targetType
    target.reviewedBy = reviewedBy
    target.reviewedAt = now

    await this.writeBatch(mutable)
  }

  /**
   * Marks a proposal as rejected. No registry change.
   * Idempotent for already-rejected proposals.
   */
  async reject(batchId: string, proposalId: string, reviewedBy: string): Promise<void> {
    const batch = await this.readBatch(batchId)
    const { proposal, index } = findProposal(batch, proposalId)

    if (proposal.status === 'rejected') return

    if (proposal.status !== 'proposed') {
      throw new Error(`ontology: cannot reject proposal "${proposalId}" with status "${proposal.status}"`)
    }

    const now = this.clock().toISOString()
    const mutable = toMutableBatch(batch)
    const target = mutable.proposals[index]
    if (target === undefined) throw new Error('ontology: proposal index out of bounds')
    target.status = 'rejected'
    target.reviewedBy = reviewedBy
    target.reviewedAt = now

    await this.writeBatch(mutable)
  }

  /**
   * Bulk-accepts all pending proposals in a batch. Individual
   * registration failures are collected rather than short-circuiting,
   * and the batch is written atomically at the end regardless of
   * partial failures so that successfully accepted proposals persist.
   */
  async acceptAll(batchId: string, reviewedBy: string): Promise<void> {
    const batch = await this.readBatch(batchId)
    const mutable = toMutableBatch(batch)
    const now = this.clock().toISOString()
    const failures: string[] = []

    for (const p of mutable.proposals) {
      if (p.status !== 'proposed') continue

      const label = p.label !== ''
        ? p.label
        : p.category === 'nodeType'
          ? formatNodeTypeLabel(p.type)
          : formatEdgeTypeLabel(p.type)

      const def: OntologyTypeDefinition = {
        type: p.type,
        label,
        description: p.description,
        createdAt: now,
        status: 'active' as TypeStatus,
      }

      try {
        await this.registry.registerType('brain', def)
        p.status = 'accepted'
        p.reviewedBy = reviewedBy
        p.reviewedAt = now
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err)
        failures.push(`register "${p.type}": ${msg}`)
      }
    }

    await this.writeBatch(mutable)

    if (failures.length > 0) {
      throw new Error(`ontology: accept all batch "${batchId}": ${failures.join('; ')}`)
    }
  }

  /**
   * Returns proposal batches matching the filter.
   */
  async list(filter?: ProposalFilter): Promise<readonly ProposalBatch[]> {
    const batches = await this.listBatches()

    if (!filter?.status && !filter?.category) {
      return batches
    }

    const filtered: ProposalBatch[] = []
    for (const batch of batches) {
      const matchedProposals = batch.proposals.filter((p) => {
        if (filter?.status && p.status !== filter.status) return false
        if (filter?.category && p.category !== filter.category) return false
        return true
      })
      if (matchedProposals.length > 0) {
        filtered.push({ ...batch, proposals: matchedProposals })
      }
    }

    return filtered
  }

  /**
   * Reads a single proposal batch by ID.
   */
  async getBatch(batchId: string): Promise<ProposalBatch> {
    return this.readBatch(batchId)
  }

  private async readBatch(id: string): Promise<ProposalBatch> {
    validateBatchId(id)
    const data = await this.store.read(proposalBatchPath(id))
    const parsed: unknown = JSON.parse(data.toString('utf-8'))
    return validateProposalBatch(parsed)
  }

  private async writeBatch(batch: ProposalBatch | MutableProposalBatch): Promise<void> {
    validateBatchId(batch.id)
    const data = Buffer.from(JSON.stringify(batch), 'utf-8')
    await this.store.write(proposalBatchPath(batch.id), data)
  }

  private async listBatches(): Promise<ProposalBatch[]> {
    let entries: ReadonlyArray<{ readonly path: Path }>
    try {
      entries = await this.store.list(PROPOSALS_DIR, {})
    } catch (err: unknown) {
      if (isNotFound(err)) return []
      throw err
    }

    const batches: ProposalBatch[] = []
    for (const entry of entries) {
      if (!String(entry.path).endsWith('.json')) continue
      try {
        const data = await this.store.read(entry.path)
        const parsed: unknown = JSON.parse(data.toString('utf-8'))
        const validated = validateProposalBatch(parsed)
        batches.push(validated)
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err)
        console.warn(`ontology: skipping corrupt proposal batch at ${String(entry.path)}: ${msg}`)
        continue
      }
    }

    batches.sort((a, b) => a.createdAt.localeCompare(b.createdAt))
    return batches
  }
}

function findProposal(
  batch: ProposalBatch,
  id: string,
): { proposal: Proposal; index: number } {
  for (let i = 0; i < batch.proposals.length; i++) {
    const p = batch.proposals[i]
    if (p !== undefined && p.id === id) {
      return { proposal: p, index: i }
    }
  }
  throw new Error(`ontology: proposal "${id}" not found in batch "${batch.id}"`)
}

function toMutableBatch(batch: ProposalBatch): MutableProposalBatch {
  return {
    id: batch.id,
    proposals: batch.proposals.map((p) => ({ ...p })),
    domain: batch.domain,
    sourceDocument: batch.sourceDocument,
    createdAt: batch.createdAt,
  }
}

function resolvedToDefinitions(resolved: ResolvedOntology): OntologyTypeDefinition[] {
  const defs: OntologyTypeDefinition[] = []
  for (const rt of resolved.nodeTypes) {
    const def: OntologyTypeDefinition = {
      type: rt.type,
      label: rt.label,
      description: rt.description,
      createdAt: rt.createdAt,
      status: rt.status,
    }
    if (rt.discoveredFrom !== undefined) {
      defs.push({ ...def, discoveredFrom: rt.discoveredFrom })
    } else {
      defs.push(def)
    }
  }
  for (const rt of resolved.edgeTypes) {
    const def: OntologyTypeDefinition = {
      type: rt.type,
      label: rt.label,
      description: rt.description,
      createdAt: rt.createdAt,
      status: rt.status,
    }
    if (rt.discoveredFrom !== undefined) {
      defs.push({ ...def, discoveredFrom: rt.discoveredFrom })
    } else {
      defs.push(def)
    }
  }
  return defs
}

const VALID_BATCH_ID_PATTERN = /^[a-zA-Z0-9][a-zA-Z0-9_-]*$/

/**
 * Validates that a batch ID is safe for path interpolation. Prevents
 * path traversal attacks via ".." or "/" in IDs.
 */
function validateBatchId(id: string): void {
  if (id === '') {
    throw new Error('ontology: batch ID must not be empty')
  }
  if (!VALID_BATCH_ID_PATTERN.test(id)) {
    throw new Error(`ontology: invalid batch ID "${id}": must match ^[a-zA-Z0-9][a-zA-Z0-9_-]*$`)
  }
}

/**
 * Runtime validation for parsed ProposalBatch JSON. Ensures the
 * parsed value has the required shape rather than relying on a bare
 * `as` cast.
 */
function validateProposalBatch(parsed: unknown): ProposalBatch {
  if (typeof parsed !== 'object' || parsed === null) {
    throw new Error('ontology: proposal batch must be an object')
  }
  const obj = parsed as Record<string, unknown>
  if (typeof obj['id'] !== 'string' || obj['id'] === '') {
    throw new Error('ontology: proposal batch missing or empty "id"')
  }
  if (!Array.isArray(obj['proposals'])) {
    throw new Error('ontology: proposal batch missing "proposals" array')
  }
  if (typeof obj['domain'] !== 'string') {
    throw new Error('ontology: proposal batch missing "domain"')
  }
  if (typeof obj['sourceDocument'] !== 'string') {
    throw new Error('ontology: proposal batch missing "sourceDocument"')
  }
  if (typeof obj['createdAt'] !== 'string') {
    throw new Error('ontology: proposal batch missing "createdAt"')
  }
  return obj as unknown as ProposalBatch
}

function extractionToDefinitions(result: ExtractionResult, clock: () => Date): OntologyTypeDefinition[] {
  const now = clock().toISOString()
  const defs: OntologyTypeDefinition[] = []
  for (const nt of result.nodeTypes) {
    defs.push({
      type: nt.type,
      label: nt.label,
      description: nt.description,
      createdAt: now,
      status: 'proposed',
    })
  }
  for (const et of result.edgeTypes) {
    defs.push({
      type: et.type,
      label: et.label,
      description: et.description,
      createdAt: now,
      status: 'proposed',
    })
  }
  return defs
}
