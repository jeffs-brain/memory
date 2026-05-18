// SPDX-License-Identifier: Apache-2.0

import { beforeEach, describe, expect, it } from 'vitest'
import { createMemStore } from '../store/memstore.js'
import type { Store } from '../store/index.js'
import { toPath } from '../store/path.js'
import type { OntologyTypeDefinition } from './types.js'
import {
  createFileOntologyStore,
  type FileOntologyStoreConfig,
  type OntologyStore,
} from './store.js'

const TEST_CONFIG: FileOntologyStoreConfig = {
  brainId: 'brain-1',
  projectId: 'proj-1',
  orgId: 'org-1',
}

function makeStore(): { ontologyStore: OntologyStore; backingStore: Store } {
  const backingStore = createMemStore()
  const ontologyStore = createFileOntologyStore(backingStore, TEST_CONFIG)
  return { ontologyStore, backingStore }
}

function makeNodeDef(overrides: Partial<OntologyTypeDefinition> = {}): OntologyTypeDefinition {
  return {
    type: 'entity.invoice',
    label: 'Invoice',
    description: 'Financial invoice documents',
    createdAt: '2026-05-10T00:00:00.000Z',
    status: 'active',
    ...overrides,
  }
}

function makeEdgeDef(overrides: Partial<OntologyTypeDefinition> = {}): OntologyTypeDefinition {
  return {
    type: 'manages',
    label: 'Manages',
    description: 'Source manages target',
    createdAt: '2026-05-10T00:00:00.000Z',
    status: 'active',
    ...overrides,
  }
}

describe('FileOntologyStore', () => {
  let ontologyStore: OntologyStore
  let backingStore: Store

  beforeEach(() => {
    const stores = makeStore()
    ontologyStore = stores.ontologyStore
    backingStore = stores.backingStore
  })

  describe('getType', () => {
    it('returns built-in node type without setup', async () => {
      const def = await ontologyStore.getType('built-in', 'entity.customer')
      expect(def).toBeDefined()
      expect(def!.type).toBe('entity.customer')
      expect(def!.status).toBe('active')
      expect(def!.label).toBeTruthy()
      expect(def!.description).toBeTruthy()
    })

    it('returns built-in edge type without setup', async () => {
      const def = await ontologyStore.getType('built-in', 'triggers')
      expect(def).toBeDefined()
      expect(def!.type).toBe('triggers')
      expect(def!.status).toBe('active')
    })

    it('returns undefined for nonexistent built-in type', async () => {
      const def = await ontologyStore.getType('built-in', 'nonexistent.type')
      expect(def).toBeUndefined()
    })

    it('returns type after upsert at brain scope', async () => {
      await ontologyStore.upsertType('brain', makeNodeDef())
      const def = await ontologyStore.getType('brain', 'entity.invoice')
      expect(def).toBeDefined()
      expect(def!.type).toBe('entity.invoice')
      expect(def!.label).toBe('Invoice')
    })

    it('returns undefined for nonexistent type at brain scope', async () => {
      const def = await ontologyStore.getType('brain', 'entity.missing')
      expect(def).toBeUndefined()
    })
  })

  describe('upsertType', () => {
    it('creates a new node type at brain scope', async () => {
      await ontologyStore.upsertType('brain', makeNodeDef())
      const def = await ontologyStore.getType('brain', 'entity.invoice')
      expect(def).toBeDefined()
      expect(def!.label).toBe('Invoice')
    })

    it('overwrites existing type with same ID', async () => {
      await ontologyStore.upsertType('brain', makeNodeDef({ label: 'V1' }))
      await ontologyStore.upsertType('brain', makeNodeDef({ label: 'V2' }))
      const def = await ontologyStore.getType('brain', 'entity.invoice')
      expect(def!.label).toBe('V2')
    })

    it('creates edge type correctly', async () => {
      await ontologyStore.upsertType('brain', makeEdgeDef())
      const def = await ontologyStore.getType('brain', 'manages')
      expect(def).toBeDefined()
      expect(def!.type).toBe('manages')
    })

    it('rejects upsert to built-in scope', async () => {
      await expect(
        ontologyStore.upsertType('built-in', makeNodeDef()),
      ).rejects.toThrow('cannot upsert to built-in scope')
    })

    it('rejects invalid type definition', async () => {
      const invalid: OntologyTypeDefinition = {
        type: 'entity.test',
        label: '',
        description: 'Missing label',
        createdAt: '2026-05-10T00:00:00.000Z',
        status: 'active',
      }
      await expect(ontologyStore.upsertType('brain', invalid)).rejects.toThrow()
    })
  })

  describe('deleteType', () => {
    it('removes node type from scope', async () => {
      await ontologyStore.upsertType('brain', makeNodeDef())
      await ontologyStore.deleteType('brain', 'entity.invoice')
      const def = await ontologyStore.getType('brain', 'entity.invoice')
      expect(def).toBeUndefined()
    })

    it('removes edge type from scope', async () => {
      await ontologyStore.upsertType('brain', makeEdgeDef())
      await ontologyStore.deleteType('brain', 'manages')
      const def = await ontologyStore.getType('brain', 'manages')
      expect(def).toBeUndefined()
    })

    it('throws for nonexistent type', async () => {
      await expect(
        ontologyStore.deleteType('brain', 'entity.nonexistent'),
      ).rejects.toThrow('not found')
    })

    it('rejects delete from built-in scope', async () => {
      await expect(
        ontologyStore.deleteType('built-in', 'entity.customer'),
      ).rejects.toThrow('cannot delete from built-in scope')
    })
  })

  describe('listTypes', () => {
    it('lists all 59 built-in types', async () => {
      const types = await ontologyStore.listTypes('built-in')
      expect(types).toHaveLength(59)
    })

    it('filters built-in types by prefix', async () => {
      const types = await ontologyStore.listTypes('built-in', { prefix: 'entity.' })
      expect(types).toHaveLength(6)
      for (const td of types) {
        expect(td.type.startsWith('entity.')).toBe(true)
      }
    })

    it('filters by status', async () => {
      await ontologyStore.upsertType('brain', makeNodeDef({ status: 'active' }))
      await ontologyStore.upsertType('brain', makeNodeDef({
        type: 'entity.proposed_one',
        label: 'Proposed',
        description: 'A proposed type',
        status: 'proposed',
      }))
      const proposed = await ontologyStore.listTypes('brain', { status: 'proposed' })
      expect(proposed).toHaveLength(1)
      expect(proposed[0]!.type).toBe('entity.proposed_one')
    })

    it('returns empty for scope with no types', async () => {
      const types = await ontologyStore.listTypes('brain')
      expect(types).toHaveLength(0)
    })
  })

  describe('getResolvedOntology', () => {
    it('returns all built-in types when store is empty', async () => {
      const resolved = await ontologyStore.getResolvedOntology('brain-1', 'proj-1', 'org-1')
      expect(resolved.nodeTypes).toHaveLength(30)
      expect(resolved.edgeTypes).toHaveLength(29)
      expect(resolved.businessCategories).toHaveLength(8)
    })

    it('brain scope overrides built-in', async () => {
      await ontologyStore.upsertType('brain', {
        type: 'entity.customer',
        label: 'Custom Customer',
        description: 'Brain-level override',
        createdAt: '2026-05-10T00:00:00.000Z',
        status: 'active',
      })
      const resolved = await ontologyStore.getResolvedOntology('brain-1', 'proj-1', 'org-1')
      const customer = resolved.nodeTypes.find((t) => t.type === 'entity.customer')
      expect(customer).toBeDefined()
      expect(customer!.label).toBe('Custom Customer')
      expect(customer!.scope).toBe('brain')
    })

    it('brain overrides project overrides org', async () => {
      await ontologyStore.upsertType('organisation', {
        type: 'entity.tenant',
        label: 'Org Tenant',
        description: 'Org level',
        createdAt: '2026-05-10T00:00:00.000Z',
        status: 'active',
      })
      await ontologyStore.upsertType('project', {
        type: 'entity.tenant',
        label: 'Project Tenant',
        description: 'Project level',
        createdAt: '2026-05-10T00:00:00.000Z',
        status: 'active',
      })
      await ontologyStore.upsertType('brain', {
        type: 'entity.tenant',
        label: 'Brain Tenant',
        description: 'Brain level',
        createdAt: '2026-05-10T00:00:00.000Z',
        status: 'active',
      })
      const resolved = await ontologyStore.getResolvedOntology('brain-1', 'proj-1', 'org-1')
      const tenant = resolved.nodeTypes.find((t) => t.type === 'entity.tenant')
      expect(tenant).toBeDefined()
      expect(tenant!.label).toBe('Brain Tenant')
      expect(tenant!.scope).toBe('brain')
    })

    it('project overrides org when no brain override', async () => {
      await ontologyStore.upsertType('organisation', {
        type: 'entity.account',
        label: 'Org Account',
        description: 'Org level',
        createdAt: '2026-05-10T00:00:00.000Z',
        status: 'active',
      })
      await ontologyStore.upsertType('project', {
        type: 'entity.account',
        label: 'Project Account',
        description: 'Project level',
        createdAt: '2026-05-10T00:00:00.000Z',
        status: 'active',
      })
      const resolved = await ontologyStore.getResolvedOntology('brain-1', 'proj-1', 'org-1')
      const account = resolved.nodeTypes.find((t) => t.type === 'entity.account')
      expect(account).toBeDefined()
      expect(account!.label).toBe('Project Account')
      expect(account!.scope).toBe('project')
    })

    it('excludes deprecated types from resolution', async () => {
      await ontologyStore.upsertType('brain', {
        type: 'entity.legacy',
        label: 'Legacy',
        description: 'Deprecated type',
        createdAt: '2026-05-10T00:00:00.000Z',
        status: 'deprecated',
      })
      const resolved = await ontologyStore.getResolvedOntology('brain-1', 'proj-1', 'org-1')
      const legacy = resolved.nodeTypes.find((t) => t.type === 'entity.legacy')
      expect(legacy).toBeUndefined()
    })

    it('includes custom business categories from all scopes', async () => {
      const stored = {
        customNodeTypes: [],
        customEdgeTypes: [],
        customBusinessCategories: ['logistics', 'healthcare'],
      }
      const data = Buffer.from(JSON.stringify(stored), 'utf-8')
      await backingStore.write(toPath('ontology/org/org-1/types.json'), data)

      const resolved = await ontologyStore.getResolvedOntology('brain-1', 'proj-1', 'org-1')
      expect(resolved.businessCategories).toContain('logistics')
      expect(resolved.businessCategories).toContain('healthcare')
      expect(resolved.businessCategories).toContain('customer')
    })

    it('caches resolved ontology', async () => {
      const resolved1 = await ontologyStore.getResolvedOntology('brain-1', 'proj-1', 'org-1')
      const resolved2 = await ontologyStore.getResolvedOntology('brain-1', 'proj-1', 'org-1')
      expect(resolved1).toBe(resolved2)
    })

    it('invalidates cache on upsert', async () => {
      const resolved1 = await ontologyStore.getResolvedOntology('brain-1', 'proj-1', 'org-1')
      await ontologyStore.upsertType('brain', makeNodeDef())
      const resolved2 = await ontologyStore.getResolvedOntology('brain-1', 'proj-1', 'org-1')
      expect(resolved1).not.toBe(resolved2)
    })

    it('invalidates cache on delete', async () => {
      await ontologyStore.upsertType('brain', makeNodeDef())
      const resolved1 = await ontologyStore.getResolvedOntology('brain-1', 'proj-1', 'org-1')
      await ontologyStore.deleteType('brain', 'entity.invoice')
      const resolved2 = await ontologyStore.getResolvedOntology('brain-1', 'proj-1', 'org-1')
      expect(resolved1).not.toBe(resolved2)
    })
  })
})
