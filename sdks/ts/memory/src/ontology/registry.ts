// SPDX-License-Identifier: Apache-2.0

/**
 * High-level ontology Registry that wraps an OntologyStore and provides
 * ProposeType (with Jaro-Winkler dedup guard >= 0.85), RegisterType,
 * DeprecateType, and Resolve.
 *
 * Port of go/ontology/registry.go.
 */

import type { OntologyScope, OntologyTypeDefinition, TypeStatus } from './types.js'
import type { OntologyStore, ResolvedOntology } from './store.js'
import { formatEdgeTypeLabel, formatNodeTypeLabel } from './format.js'

/**
 * Jaro-Winkler similarity threshold above which ProposeType considers
 * the candidate a duplicate and skips registration.
 */
export const PROPOSE_DEDUP_THRESHOLD = 0.85

export type RegistryOptions = {
  readonly store: OntologyStore
}

/**
 * Registry provides high-level ontology operations on top of an
 * OntologyStore. Adds ProposeType with fuzzy dedup guard, RegisterType,
 * DeprecateType, and Resolve.
 */
export class Registry {
  private readonly store: OntologyStore

  constructor(options: RegistryOptions) {
    this.store = options.store
  }

  /**
   * Adds or updates a type at the specified scope.
   */
  async registerType(scope: OntologyScope, def: OntologyTypeDefinition): Promise<void> {
    await this.store.upsertType(scope, def)
  }

  /**
   * Marks a type as deprecated at the specified scope.
   */
  async deprecateType(scope: OntologyScope, typeName: string): Promise<void> {
    const existing = await this.store.getType(scope, typeName)
    if (existing === undefined) {
      throw new Error(`ontology: type "${typeName}" not found at scope ${scope}`)
    }
    const deprecated: OntologyTypeDefinition = {
      ...existing,
      status: 'deprecated' as TypeStatus,
    }
    await this.store.upsertType(scope, deprecated)
  }

  /**
   * Creates a proposed type, skipping if a fuzzy-similar type already
   * exists (Jaro-Winkler >= 0.85). Returns without error when the type
   * is skipped as a duplicate.
   */
  async proposeType(
    scope: OntologyScope,
    category: 'nodeType' | 'edgeType',
    typeName: string,
    reason: string,
  ): Promise<void> {
    const resolved = await this.store.getResolvedOntology('', '', '')

    const existingLabels: string[] = []
    if (category === 'nodeType') {
      for (const rt of resolved.nodeTypes) {
        existingLabels.push(rt.label)
      }
    } else {
      for (const rt of resolved.edgeTypes) {
        existingLabels.push(rt.label)
      }
    }

    const proposedLabel =
      category === 'nodeType' ? formatNodeTypeLabel(typeName) : formatEdgeTypeLabel(typeName)

    for (const existing of existingLabels) {
      const sim = jaroWinkler(proposedLabel.toLowerCase(), existing.toLowerCase())
      if (sim >= PROPOSE_DEDUP_THRESHOLD) {
        return
      }
    }

    const now = new Date().toISOString()
    const def: OntologyTypeDefinition = {
      type: typeName,
      label: proposedLabel,
      description: reason,
      discoveredFrom: 'proposal',
      createdAt: now,
      status: 'proposed',
    }
    await this.store.upsertType(scope, def)
  }

  /**
   * Returns the fully merged ontology from the store.
   */
  async resolve(brainId: string, projectId: string, orgId: string): Promise<ResolvedOntology> {
    return this.store.getResolvedOntology(brainId, projectId, orgId)
  }
}

/**
 * Standalone Jaro-Winkler similarity for the propose dedup guard.
 * Avoids depending on the dedup package (P6-5).
 */
function jaroWinkler(s1: string, s2: string): number {
  if (s1 === s2) return 1.0
  if (s1.length === 0 || s2.length === 0) return 0.0

  const matchWindow = Math.max(0, Math.floor(Math.max(s1.length, s2.length) / 2) - 1)
  const s1Matches = new Array<boolean>(s1.length).fill(false)
  const s2Matches = new Array<boolean>(s2.length).fill(false)

  let matches = 0
  let transpositions = 0

  for (let i = 0; i < s1.length; i++) {
    const start = Math.max(0, i - matchWindow)
    const end = Math.min(i + matchWindow + 1, s2.length)
    for (let j = start; j < end; j++) {
      if (s2Matches[j] || s1[i] !== s2[j]) continue
      s1Matches[i] = true
      s2Matches[j] = true
      matches++
      break
    }
  }

  if (matches === 0) return 0.0

  let k = 0
  for (let i = 0; i < s1.length; i++) {
    if (!s1Matches[i]) continue
    while (!s2Matches[k]) k++
    if (s1[i] !== s2[k]) transpositions++
    k++
  }

  const jaro =
    (matches / s1.length + matches / s2.length + (matches - transpositions / 2) / matches) / 3

  let commonPrefix = 0
  const maxPrefix = Math.min(4, Math.min(s1.length, s2.length))
  for (let i = 0; i < maxPrefix; i++) {
    if (s1[i] !== s2[i]) break
    commonPrefix++
  }

  return jaro + commonPrefix * 0.1 * (1 - jaro)
}
