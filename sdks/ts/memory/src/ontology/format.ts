// SPDX-License-Identifier: Apache-2.0

/**
 * Label formatting functions for ontology type identifiers.
 * Converts machine-readable identifiers into human-readable labels.
 *
 * Port of apps/intelligence-service/src/services/ontology-built-in-descriptions.ts.
 */

/**
 * Converts a dotted node type identifier into a human-readable label.
 * "entity.customer" becomes "Customer (Entity)"
 * "process.approval_chain" becomes "Approval Chain (Process)"
 */
export function formatNodeTypeLabel(typ: string): string {
  const dotIndex = typ.indexOf('.')
  if (dotIndex === -1) {
    return typ
  }
  const prefix = typ.slice(0, dotIndex)
  const name = typ.slice(dotIndex + 1)
  const formattedName = name
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
  const formattedPrefix = prefix.charAt(0).toUpperCase() + prefix.slice(1)
  return `${formattedName} (${formattedPrefix})`
}

/**
 * Converts a snake_case edge type identifier into a human-readable label.
 * "requires_approval_from" becomes "Requires Approval From"
 * "triggers" becomes "Triggers"
 */
export function formatEdgeTypeLabel(typ: string): string {
  return typ
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}
