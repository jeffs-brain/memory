// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it, beforeEach } from 'vitest'
import { createMemStore } from '../store/memstore.js'
import type { OntologyTypeDefinition } from './types.js'
import { createFileOntologyStore, type OntologyStore } from './store.js'
import { Registry } from './registry.js'

function makeRegistry(): Registry {
  const backingStore = createMemStore()
  const ontologyStore = createFileOntologyStore(backingStore, {
    brainId: 'brain-1',
    projectId: 'proj-1',
    orgId: 'org-1',
  })
  return new Registry({ store: ontologyStore })
}

describe('Registry', () => {
  let registry: Registry

  beforeEach(() => {
    registry = makeRegistry()
  })

  describe('registerType', () => {
    it('adds a type to the store', async () => {
      const def: OntologyTypeDefinition = {
        type: 'entity.invoice',
        label: 'Invoice',
        description: 'Financial invoice',
        createdAt: '2026-01-01T00:00:00Z',
        status: 'active',
      }
      await registry.registerType('brain', def)

      const resolved = await registry.resolve('brain-1', 'proj-1', 'org-1')
      const invoice = resolved.nodeTypes.find((t) => t.type === 'entity.invoice')
      expect(invoice).toBeDefined()
      expect(invoice!.label).toBe('Invoice')
    })
  })

  describe('deprecateType', () => {
    it('marks a type as deprecated', async () => {
      const def: OntologyTypeDefinition = {
        type: 'entity.invoice',
        label: 'Invoice',
        description: 'Financial invoice',
        createdAt: '2026-01-01T00:00:00Z',
        status: 'active',
      }
      await registry.registerType('brain', def)
      await registry.deprecateType('brain', 'entity.invoice')

      const resolved = await registry.resolve('brain-1', 'proj-1', 'org-1')
      const invoice = resolved.nodeTypes.find((t) => t.type === 'entity.invoice')
      expect(invoice).toBeUndefined()
    })

    it('throws for nonexistent type', async () => {
      await expect(
        registry.deprecateType('brain', 'entity.missing'),
      ).rejects.toThrow('not found')
    })
  })

  describe('proposeType', () => {
    it('creates a proposed type for genuinely new types', async () => {
      await registry.proposeType('brain', 'nodeType', 'entity.invoice', 'Discovered in procurement docs')

      // Proposed types should NOT appear in resolved (only active resolve)
      const resolved = await registry.resolve('brain-1', 'proj-1', 'org-1')
      const invoice = resolved.nodeTypes.find((t) => t.type === 'entity.invoice' && t.scope === 'brain')
      expect(invoice).toBeUndefined()
    })

    it('skips when similar type already exists', async () => {
      // "entity.customer" is built-in. Proposing "entity.customers"
      // should be skipped since "Customers (Entity)" is very similar
      // to "Customer (Entity)".
      await registry.proposeType('brain', 'nodeType', 'entity.customers', 'Should be skipped')

      const resolved = await registry.resolve('brain-1', 'proj-1', 'org-1')
      const customers = resolved.nodeTypes.find((t) => t.type === 'entity.customers')
      expect(customers).toBeUndefined()
    })

    it('creates edge type proposals', async () => {
      await registry.proposeType('brain', 'edgeType', 'ships_to', 'Logistics relation')

      // ships_to is not a built-in edge type, so it should be created as proposed
      const resolved = await registry.resolve('brain-1', 'proj-1', 'org-1')
      // It's proposed so won't appear in resolved (active only)
      const shipsTo = resolved.edgeTypes.find((t) => t.type === 'ships_to' && t.scope === 'brain')
      expect(shipsTo).toBeUndefined()
    })
  })

  describe('resolve', () => {
    it('returns all built-in types when store is empty', async () => {
      const resolved = await registry.resolve('brain-1', 'proj-1', 'org-1')
      expect(resolved.nodeTypes).toHaveLength(30)
      expect(resolved.edgeTypes).toHaveLength(29)
      expect(resolved.businessCategories).toHaveLength(8)
    })

    it('includes registered types in resolution', async () => {
      const def: OntologyTypeDefinition = {
        type: 'entity.invoice',
        label: 'Invoice',
        description: 'Financial invoice',
        createdAt: '2026-01-01T00:00:00Z',
        status: 'active',
      }
      await registry.registerType('brain', def)

      const resolved = await registry.resolve('brain-1', 'proj-1', 'org-1')
      expect(resolved.nodeTypes).toHaveLength(31)
    })
  })
})
