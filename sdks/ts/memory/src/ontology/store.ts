// SPDX-License-Identifier: Apache-2.0

/**
 * Ontology type registry with dual-backend persistence and 4-layer scope
 * resolution. Provides CRUD operations for custom type definitions and
 * resolves the complete ontology by merging built-in, organisation, project,
 * and brain scopes.
 *
 * Port of go/ontology/store.go and store_file.go.
 */

import type { Store } from '../store/index.js'
import type { Path } from '../store/path.js'
import type { OntologyScope, OntologyTypeDefinition, TypeStatus } from './types.js'

import { isNotFound } from '../store/errors.js'
import { toPath } from '../store/path.js'
import { BUILT_IN_EDGE_TYPES, BUILT_IN_NODE_TYPES, BUSINESS_CATEGORIES, NODE_TYPE_PREFIXES } from './types.js'
import { getBuiltInEdgeTypeDescription, getBuiltInNodeTypeDescription } from './descriptions.js'
import { formatEdgeTypeLabel, formatNodeTypeLabel } from './format.js'
import { validateTypeDefinition } from './validation.js'

export type ListTypesOpts = {
  readonly prefix?: string
  readonly status?: TypeStatus
}

export type ResolvedType = OntologyTypeDefinition & {
  readonly scope: OntologyScope
}

export type ResolvedOntology = {
  readonly nodeTypes: readonly ResolvedType[]
  readonly edgeTypes: readonly ResolvedType[]
  readonly businessCategories: readonly string[]
}

export type StoredOntology = {
  readonly customNodeTypes: readonly OntologyTypeDefinition[]
  readonly customEdgeTypes: readonly OntologyTypeDefinition[]
  readonly customBusinessCategories: readonly string[]
}

export type OntologyStore = {
  getType(scope: OntologyScope, typeId: string): Promise<OntologyTypeDefinition | undefined>
  listTypes(scope: OntologyScope, opts?: ListTypesOpts): Promise<readonly OntologyTypeDefinition[]>
  upsertType(scope: OntologyScope, def: OntologyTypeDefinition): Promise<void>
  deleteType(scope: OntologyScope, typeId: string): Promise<void>
  getResolvedOntology(
    brainId: string,
    projectId: string,
    orgId: string,
  ): Promise<ResolvedOntology>
  close(): Promise<void>
}

export type FileOntologyStoreConfig = {
  readonly brainId: string
  readonly projectId: string
  readonly orgId: string
}

const VALID_ID_PATTERN = /^[a-zA-Z0-9][a-zA-Z0-9_-]*$/

/**
 * Validates that an ID is safe for path interpolation. IDs must start with an
 * alphanumeric character and contain only alphanumeric, underscore, or hyphen
 * characters. This prevents path traversal attacks via ".." or "/" in IDs.
 */
function validateId(id: string): void {
  if (id === '') {
    throw new Error('ontology: ID must not be empty')
  }
  if (!VALID_ID_PATTERN.test(id)) {
    throw new Error(`ontology: invalid ID "${id}": must match ^[a-zA-Z0-9][a-zA-Z0-9_-]*$`)
  }
}

/**
 * Creates a FileOntologyStore backed by the given Store.
 * Types are stored as JSON files at well-known paths:
 *   - Organisation: ontology/org/{orgId}/types.json
 *   - Project: ontology/project/{projectId}/types.json
 *   - Brain: ontology/brain/{brainId}/types.json
 *
 * Built-in types are hardcoded and always available without I/O.
 * Throws if any configured ID fails validation (path traversal prevention).
 */
export function createFileOntologyStore(store: Store, config: FileOntologyStoreConfig): OntologyStore {
  if (config.brainId !== '') validateId(config.brainId)
  if (config.projectId !== '') validateId(config.projectId)
  if (config.orgId !== '') validateId(config.orgId)

  const cache = new Map<string, ResolvedOntology>()

  function idForScope(scope: OntologyScope): string {
    switch (scope) {
      case 'organisation':
        return config.orgId
      case 'project':
        return config.projectId
      case 'brain':
        return config.brainId
      case 'built-in':
        return ''
    }
  }

  function scopePath(scope: OntologyScope, id: string): Path {
    switch (scope) {
      case 'organisation':
        return toPath(`ontology/org/${id}/types.json`)
      case 'project':
        return toPath(`ontology/project/${id}/types.json`)
      case 'brain':
        return toPath(`ontology/brain/${id}/types.json`)
      case 'built-in':
        return toPath('ontology/types.json')
    }
  }

  async function readScopeById(scope: OntologyScope, id: string): Promise<StoredOntology> {
    const p = scopePath(scope, id)
    try {
      const data = await store.read(p)
      const parsed: unknown = JSON.parse(data.toString('utf-8'))
      return parseStoredOntology(parsed)
    } catch (err: unknown) {
      if (isNotFound(err)) {
        return { customNodeTypes: [], customEdgeTypes: [], customBusinessCategories: [] }
      }
      throw err
    }
  }

  async function writeScopeById(scope: OntologyScope, id: string, stored: StoredOntology): Promise<void> {
    const p = scopePath(scope, id)
    const data = Buffer.from(JSON.stringify(stored), 'utf-8')
    await store.write(p, data)
  }

  function invalidateCache(): void {
    cache.clear()
  }

  function hasNodeTypePrefix(typeId: string): boolean {
    for (const prefix of NODE_TYPE_PREFIXES) {
      if (typeId.startsWith(prefix)) {
        return true
      }
    }
    return false
  }

  function filterTypes(
    defs: readonly OntologyTypeDefinition[],
    opts?: ListTypesOpts,
  ): readonly OntologyTypeDefinition[] {
    if (!opts?.prefix && !opts?.status) {
      return defs
    }
    return defs.filter((d) => {
      if (opts?.prefix && !d.type.startsWith(opts.prefix)) return false
      if (opts?.status && d.status !== opts.status) return false
      return true
    })
  }

  async function getType(
    scope: OntologyScope,
    typeId: string,
  ): Promise<OntologyTypeDefinition | undefined> {
    if (scope === 'built-in') {
      return getBuiltInType(typeId)
    }
    const stored = await readScopeById(scope, idForScope(scope))
    for (const nt of stored.customNodeTypes) {
      if (nt.type === typeId) return nt
    }
    for (const et of stored.customEdgeTypes) {
      if (et.type === typeId) return et
    }
    return undefined
  }

  async function listTypes(
    scope: OntologyScope,
    opts?: ListTypesOpts,
  ): Promise<readonly OntologyTypeDefinition[]> {
    if (scope === 'built-in') {
      return filterTypes(listBuiltInTypes(), opts)
    }
    const stored = await readScopeById(scope, idForScope(scope))
    const combined = [...stored.customNodeTypes, ...stored.customEdgeTypes]
    return filterTypes(combined, opts)
  }

  async function upsertType(scope: OntologyScope, def: OntologyTypeDefinition): Promise<void> {
    if (scope === 'built-in') {
      throw new Error('ontology: cannot upsert to built-in scope')
    }
    const validationError = validateTypeDefinition(def)
    if (validationError !== undefined) {
      throw new Error(validationError)
    }
    const id = idForScope(scope)
    const stored = await readScopeById(scope, id)
    const mutableNodeTypes = [...stored.customNodeTypes]
    const mutableEdgeTypes = [...stored.customEdgeTypes]

    if (hasNodeTypePrefix(def.type)) {
      const existingIdx = mutableNodeTypes.findIndex((t) => t.type === def.type)
      if (existingIdx >= 0) {
        mutableNodeTypes[existingIdx] = def
      } else {
        mutableNodeTypes.push(def)
      }
    } else {
      const existingIdx = mutableEdgeTypes.findIndex((t) => t.type === def.type)
      if (existingIdx >= 0) {
        mutableEdgeTypes[existingIdx] = def
      } else {
        mutableEdgeTypes.push(def)
      }
    }

    await writeScopeById(scope, id, {
      customNodeTypes: mutableNodeTypes,
      customEdgeTypes: mutableEdgeTypes,
      customBusinessCategories: stored.customBusinessCategories,
    })
    invalidateCache()
  }

  async function deleteType(scope: OntologyScope, typeId: string): Promise<void> {
    if (scope === 'built-in') {
      throw new Error('ontology: cannot delete from built-in scope')
    }
    const id = idForScope(scope)
    const stored = await readScopeById(scope, id)
    const mutableNodeTypes = [...stored.customNodeTypes]
    const mutableEdgeTypes = [...stored.customEdgeTypes]

    const nodeIdx = mutableNodeTypes.findIndex((t) => t.type === typeId)
    if (nodeIdx >= 0) {
      mutableNodeTypes.splice(nodeIdx, 1)
      await writeScopeById(scope, id, {
        customNodeTypes: mutableNodeTypes,
        customEdgeTypes: mutableEdgeTypes,
        customBusinessCategories: stored.customBusinessCategories,
      })
      invalidateCache()
      return
    }

    const edgeIdx = mutableEdgeTypes.findIndex((t) => t.type === typeId)
    if (edgeIdx >= 0) {
      mutableEdgeTypes.splice(edgeIdx, 1)
      await writeScopeById(scope, id, {
        customNodeTypes: mutableNodeTypes,
        customEdgeTypes: mutableEdgeTypes,
        customBusinessCategories: stored.customBusinessCategories,
      })
      invalidateCache()
      return
    }

    throw new Error(`ontology: type "${typeId}" not found at scope ${scope}`)
  }

  async function getResolvedOntology(
    brainId: string,
    projectId: string,
    orgId: string,
  ): Promise<ResolvedOntology> {
    if (brainId !== '') validateId(brainId)
    if (projectId !== '') validateId(projectId)
    if (orgId !== '') validateId(orgId)

    const cacheKey = `org:${orgId}:proj:${projectId}:brain:${brainId}`
    const cached = cache.get(cacheKey)
    if (cached) {
      return cached
    }

    const nodeMap = new Map<string, ResolvedType>()
    const edgeMap = new Map<string, ResolvedType>()
    const categorySet = new Set<string>(BUSINESS_CATEGORIES)

    for (const nt of BUILT_IN_NODE_TYPES) {
      nodeMap.set(nt, {
        type: nt,
        label: formatNodeTypeLabel(nt),
        description: getBuiltInNodeTypeDescription(nt),
        createdAt: '1970-01-01T00:00:00.000Z',
        status: 'active',
        scope: 'built-in',
      })
    }
    for (const et of BUILT_IN_EDGE_TYPES) {
      edgeMap.set(et, {
        type: et,
        label: formatEdgeTypeLabel(et),
        description: getBuiltInEdgeTypeDescription(et),
        createdAt: '1970-01-01T00:00:00.000Z',
        status: 'active',
        scope: 'built-in',
      })
    }

    const layers: ReadonlyArray<{ scope: OntologyScope; id: string }> = [
      { scope: 'organisation', id: orgId },
      { scope: 'project', id: projectId },
      { scope: 'brain', id: brainId },
    ]

    for (const layer of layers) {
      if (!layer.id) continue
      const stored = await readScopeById(layer.scope, layer.id)
      for (const nt of stored.customNodeTypes) {
        nodeMap.set(nt.type, { ...nt, scope: layer.scope })
      }
      for (const et of stored.customEdgeTypes) {
        edgeMap.set(et.type, { ...et, scope: layer.scope })
      }
      for (const cat of stored.customBusinessCategories) {
        categorySet.add(cat)
      }
    }

    const activeNodes: ResolvedType[] = []
    for (const rt of nodeMap.values()) {
      if (rt.status === 'active') activeNodes.push(rt)
    }
    const activeEdges: ResolvedType[] = []
    for (const rt of edgeMap.values()) {
      if (rt.status === 'active') activeEdges.push(rt)
    }
    const sortedCategories = [...categorySet].sort()

    const resolved: ResolvedOntology = {
      nodeTypes: activeNodes,
      edgeTypes: activeEdges,
      businessCategories: sortedCategories,
    }

    cache.set(cacheKey, resolved)
    return resolved
  }

  async function close(): Promise<void> {
    cache.clear()
  }

  return {
    getType,
    listTypes,
    upsertType,
    deleteType,
    getResolvedOntology,
    close,
  }
}

function getBuiltInType(typeId: string): OntologyTypeDefinition | undefined {
  const builtInNodeSet: ReadonlySet<string> = new Set(BUILT_IN_NODE_TYPES)
  const builtInEdgeSet: ReadonlySet<string> = new Set(BUILT_IN_EDGE_TYPES)

  if (builtInNodeSet.has(typeId)) {
    return {
      type: typeId,
      label: formatNodeTypeLabel(typeId),
      description: getBuiltInNodeTypeDescription(typeId),
      createdAt: '1970-01-01T00:00:00.000Z',
      status: 'active',
    }
  }
  if (builtInEdgeSet.has(typeId)) {
    return {
      type: typeId,
      label: formatEdgeTypeLabel(typeId),
      description: getBuiltInEdgeTypeDescription(typeId),
      createdAt: '1970-01-01T00:00:00.000Z',
      status: 'active',
    }
  }
  return undefined
}

function listBuiltInTypes(): OntologyTypeDefinition[] {
  const defs: OntologyTypeDefinition[] = []
  for (const nt of BUILT_IN_NODE_TYPES) {
    defs.push({
      type: nt,
      label: formatNodeTypeLabel(nt),
      description: getBuiltInNodeTypeDescription(nt),
      createdAt: '1970-01-01T00:00:00.000Z',
      status: 'active',
    })
  }
  for (const et of BUILT_IN_EDGE_TYPES) {
    defs.push({
      type: et,
      label: formatEdgeTypeLabel(et),
      description: getBuiltInEdgeTypeDescription(et),
      createdAt: '1970-01-01T00:00:00.000Z',
      status: 'active',
    })
  }
  return defs
}

function isValidTypeDefinition(value: unknown): value is OntologyTypeDefinition {
  if (value === null || typeof value !== 'object') return false
  const obj = value as Record<string, unknown>
  return (
    typeof obj['type'] === 'string' &&
    typeof obj['label'] === 'string' &&
    typeof obj['status'] === 'string'
  )
}

function parseStoredOntology(raw: unknown): StoredOntology {
  if (raw === null || typeof raw !== 'object') {
    return { customNodeTypes: [], customEdgeTypes: [], customBusinessCategories: [] }
  }
  const obj = raw as Record<string, unknown>
  const nodeTypes = Array.isArray(obj['customNodeTypes'])
    ? (obj['customNodeTypes'] as unknown[]).filter(isValidTypeDefinition)
    : []
  const edgeTypes = Array.isArray(obj['customEdgeTypes'])
    ? (obj['customEdgeTypes'] as unknown[]).filter(isValidTypeDefinition)
    : []
  const categories = Array.isArray(obj['customBusinessCategories'])
    ? (obj['customBusinessCategories'] as unknown[]).filter((v): v is string => typeof v === 'string')
    : []
  return {
    customNodeTypes: nodeTypes,
    customEdgeTypes: edgeTypes,
    customBusinessCategories: categories,
  }
}
