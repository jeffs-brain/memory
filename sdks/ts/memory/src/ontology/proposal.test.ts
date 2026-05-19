// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, beforeEach } from 'vitest'
import { createMemStore } from '../store/memstore.js'
import { createFileOntologyStore } from './store.js'
import { Registry } from './registry.js'
import { ProposalWorkflow, type ExtractionResult, type ProposalBatch } from './proposal.js'
import type { TypeEntry } from './templates.js'

function fixedClock(date: Date): () => Date {
  return () => date
}

function makeWorkflow(): { workflow: ProposalWorkflow; registry: Registry } {
  const backingStore = createMemStore()
  const ontologyStore = createFileOntologyStore(backingStore, {
    brainId: 'brain-1',
    projectId: 'proj-1',
    orgId: 'org-1',
  })
  const registry = new Registry({ store: ontologyStore })
  const workflow = new ProposalWorkflow({
    registry,
    store: backingStore,
    brainId: 'brain-1',
    projectId: 'proj-1',
    orgId: 'org-1',
    clock: fixedClock(new Date('2026-05-15T12:00:00.000Z')),
  })
  return { workflow, registry }
}

function sampleExtraction(): ExtractionResult {
  return {
    nodeTypes: [
      { type: 'entity.invoice', label: 'Invoice', description: 'A financial invoice document' },
      { type: 'entity.warehouse', label: 'Warehouse', description: 'Physical storage location' },
    ],
    edgeTypes: [
      { type: 'ships_to', label: 'Ships To', description: 'Logistics shipping relationship' },
    ],
    businessCategories: ['logistics'],
    domain: 'logistics',
    confidence: 0.85,
  }
}

describe('ProposalWorkflow', () => {
  let workflow: ProposalWorkflow
  let registry: Registry

  beforeEach(() => {
    const setup = makeWorkflow()
    workflow = setup.workflow
    registry = setup.registry
  })

  describe('proposeFromExtraction', () => {
    it('creates proposals for unique types', async () => {
      const batch = await workflow.proposeFromExtraction(sampleExtraction(), 'test-doc.pdf')

      expect(batch.id).toBeTruthy()
      expect(batch.domain).toBe('logistics')
      expect(batch.sourceDocument).toBe('test-doc.pdf')
      expect(batch.proposals).toHaveLength(3)

      for (const p of batch.proposals) {
        expect(p.status).toBe('proposed')
        expect(p.id).toBeTruthy()
      }
    })

    it('excludes types matching existing ontology via dedup', async () => {
      const result: ExtractionResult = {
        nodeTypes: [
          { type: 'entity.customer', label: 'Customer (Entity)', description: 'A customer entity' },
          { type: 'entity.invoice', label: 'Invoice', description: 'A financial invoice' },
        ],
        edgeTypes: [
          { type: 'triggers', label: 'Triggers', description: 'Triggers relationship' },
        ],
        businessCategories: [],
        domain: 'mixed',
        confidence: 0.8,
      }

      const batch = await workflow.proposeFromExtraction(result, 'test.pdf')
      // entity.customer and triggers are built-in, only entity.invoice should be proposed
      expect(batch.proposals).toHaveLength(1)
      expect(batch.proposals[0].type).toBe('entity.invoice')
    })

    it('returns empty batch for empty extraction', async () => {
      const result: ExtractionResult = {
        nodeTypes: [],
        edgeTypes: [],
        businessCategories: [],
        domain: '',
        confidence: 0,
      }

      const batch = await workflow.proposeFromExtraction(result, 'empty.pdf')
      expect(batch.proposals).toHaveLength(0)
    })

    it('assigns correct categories', async () => {
      const batch = await workflow.proposeFromExtraction(sampleExtraction(), 'test.pdf')

      const nodeTypes = batch.proposals.filter((p) => p.category === 'nodeType')
      const edgeTypes = batch.proposals.filter((p) => p.category === 'edgeType')

      expect(nodeTypes).toHaveLength(2)
      expect(edgeTypes).toHaveLength(1)
    })
  })

  describe('accept', () => {
    it('activates type in registry', async () => {
      const batch = await workflow.proposeFromExtraction(sampleExtraction(), 'test.pdf')
      const invoiceProposal = batch.proposals.find((p) => p.type === 'entity.invoice')!

      await workflow.accept(batch.id, invoiceProposal.id, 'reviewer@test.com')

      const resolved = await registry.resolve('brain-1', 'proj-1', 'org-1')
      const invoice = resolved.nodeTypes.find((t) => t.type === 'entity.invoice')
      expect(invoice).toBeDefined()
    })

    it('is idempotent for already-accepted proposals', async () => {
      const batch = await workflow.proposeFromExtraction(sampleExtraction(), 'test.pdf')
      const pid = batch.proposals[0].id

      await workflow.accept(batch.id, pid, 'reviewer@test.com')
      await workflow.accept(batch.id, pid, 'reviewer@test.com')
      // No error thrown
    })

    it('records reviewer and timestamp', async () => {
      const batch = await workflow.proposeFromExtraction(sampleExtraction(), 'test.pdf')
      const pid = batch.proposals[0].id

      await workflow.accept(batch.id, pid, 'reviewer@test.com')

      const updated = await workflow.getBatch(batch.id)
      const accepted = updated.proposals.find((p) => p.id === pid)!
      expect(accepted.status).toBe('accepted')
      expect(accepted.reviewedBy).toBe('reviewer@test.com')
      expect(accepted.reviewedAt).toBeTruthy()
    })
  })

  describe('merge', () => {
    it('sets target type', async () => {
      const batch = await workflow.proposeFromExtraction(sampleExtraction(), 'test.pdf')
      const pid = batch.proposals[0].id

      await workflow.merge(batch.id, pid, 'entity.customer', 'reviewer@test.com')

      const updated = await workflow.getBatch(batch.id)
      const merged = updated.proposals.find((p) => p.id === pid)!
      expect(merged.status).toBe('merged')
      expect(merged.mergedInto).toBe('entity.customer')
      expect(merged.reviewedBy).toBe('reviewer@test.com')
    })
  })

  describe('reject', () => {
    it('sets status to rejected', async () => {
      const batch = await workflow.proposeFromExtraction(sampleExtraction(), 'test.pdf')
      const pid = batch.proposals[0].id

      await workflow.reject(batch.id, pid, 'reviewer@test.com')

      const updated = await workflow.getBatch(batch.id)
      const rejected = updated.proposals.find((p) => p.id === pid)!
      expect(rejected.status).toBe('rejected')
      expect(rejected.reviewedBy).toBe('reviewer@test.com')
    })

    it('does not affect registry', async () => {
      const batch = await workflow.proposeFromExtraction(sampleExtraction(), 'test.pdf')
      const pid = batch.proposals[0].id
      const typeName = batch.proposals[0].type

      await workflow.reject(batch.id, pid, 'reviewer@test.com')

      const resolved = await registry.resolve('brain-1', 'proj-1', 'org-1')
      const found = resolved.nodeTypes.find(
        (t) => t.type === typeName && t.scope === 'brain',
      )
      expect(found).toBeUndefined()
    })

    it('throws when rejecting an accepted proposal', async () => {
      const batch = await workflow.proposeFromExtraction(sampleExtraction(), 'test.pdf')
      const pid = batch.proposals[0].id

      await workflow.accept(batch.id, pid, 'reviewer')
      await expect(
        workflow.reject(batch.id, pid, 'reviewer'),
      ).rejects.toThrow('cannot reject')
    })
  })

  describe('acceptAll', () => {
    it('bulk-accepts all pending proposals', async () => {
      const batch = await workflow.proposeFromExtraction(sampleExtraction(), 'test.pdf')

      await workflow.acceptAll(batch.id, 'reviewer@test.com')

      const updated = await workflow.getBatch(batch.id)
      for (const p of updated.proposals) {
        expect(p.status).toBe('accepted')
        expect(p.reviewedBy).toBe('reviewer@test.com')
      }

      const resolved = await registry.resolve('brain-1', 'proj-1', 'org-1')
      const invoiceFound = resolved.nodeTypes.some((t) => t.type === 'entity.invoice')
      const warehouseFound = resolved.nodeTypes.some((t) => t.type === 'entity.warehouse')
      const shipsToFound = resolved.edgeTypes.some((t) => t.type === 'ships_to')

      expect(invoiceFound).toBe(true)
      expect(warehouseFound).toBe(true)
      expect(shipsToFound).toBe(true)
    })
  })

  describe('list', () => {
    it('returns all batches without filter', async () => {
      await workflow.proposeFromExtraction(sampleExtraction(), 'test.pdf')

      const all = await workflow.list()
      expect(all).toHaveLength(1)
      expect(all[0].proposals).toHaveLength(3)
    })

    it('filters by category', async () => {
      await workflow.proposeFromExtraction(sampleExtraction(), 'test.pdf')

      // sampleExtraction has 2 nodeType and 1 edgeType
      const nodeOnly = await workflow.list({ category: 'nodeType' })
      let totalNodeType = 0
      for (const b of nodeOnly) {
        totalNodeType += b.proposals.length
      }
      expect(totalNodeType).toBe(2)

      const edgeOnly = await workflow.list({ category: 'edgeType' })
      let totalEdgeType = 0
      for (const b of edgeOnly) {
        totalEdgeType += b.proposals.length
      }
      expect(totalEdgeType).toBe(1)
    })

    it('filters by status', async () => {
      const batch = await workflow.proposeFromExtraction(sampleExtraction(), 'test.pdf')

      await workflow.accept(batch.id, batch.proposals[0].id, 'reviewer')
      await workflow.reject(batch.id, batch.proposals[1].id, 'reviewer')

      const pending = await workflow.list({ status: 'proposed' })
      let totalPending = 0
      for (const b of pending) {
        totalPending += b.proposals.length
      }
      expect(totalPending).toBe(1)

      const accepted = await workflow.list({ status: 'accepted' })
      let totalAccepted = 0
      for (const b of accepted) {
        totalAccepted += b.proposals.length
      }
      expect(totalAccepted).toBe(1)
    })
  })

  describe('persistence', () => {
    it('survives workflow reconstruction', async () => {
      const backingStore = createMemStore()
      const ontologyStore = createFileOntologyStore(backingStore, {
        brainId: 'brain-1',
        projectId: 'proj-1',
        orgId: 'org-1',
      })
      const reg = new Registry({ store: ontologyStore })
      const clock = fixedClock(new Date('2026-05-15T12:00:00.000Z'))

      const wf1 = new ProposalWorkflow({ registry: reg, store: backingStore, brainId: 'brain-1', projectId: 'proj-1', orgId: 'org-1', clock })
      const batch = await wf1.proposeFromExtraction(sampleExtraction(), 'test.pdf')

      const wf2 = new ProposalWorkflow({ registry: reg, store: backingStore, brainId: 'brain-1', projectId: 'proj-1', orgId: 'org-1', clock })
      const readBack = await wf2.getBatch(batch.id)
      expect(readBack.proposals).toHaveLength(batch.proposals.length)
    })
  })

  describe('round-trip', () => {
    it('extract -> propose -> accept -> verify in resolved', async () => {
      const result: ExtractionResult = {
        nodeTypes: [
          { type: 'entity.invoice', label: 'Invoice', description: 'A financial invoice' },
        ],
        edgeTypes: [],
        businessCategories: [],
        domain: 'finance',
        confidence: 0.9,
      }

      const batch = await workflow.proposeFromExtraction(result, 'finance.pdf')
      expect(batch.proposals).toHaveLength(1)

      // Before accept: not in resolved
      let resolved = await registry.resolve('brain-1', 'proj-1', 'org-1')
      const beforeAccept = resolved.nodeTypes.find(
        (t) => t.type === 'entity.invoice' && t.scope === 'brain',
      )
      expect(beforeAccept).toBeUndefined()

      // Accept
      await workflow.accept(batch.id, batch.proposals[0].id, 'reviewer@test.com')

      // After accept: in resolved
      resolved = await registry.resolve('brain-1', 'proj-1', 'org-1')
      const afterAccept = resolved.nodeTypes.find((t) => t.type === 'entity.invoice')
      expect(afterAccept).toBeDefined()
      expect(afterAccept!.scope).toBe('brain')
    })
  })
})
