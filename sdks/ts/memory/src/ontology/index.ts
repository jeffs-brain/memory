// SPDX-License-Identifier: Apache-2.0

/**
 * Ontology module -- type definitions, validation, industry templates,
 * and deduplication for the memory ontology system.
 */

export type {
  BuiltInEdgeType,
  BuiltInNodeType,
  BusinessCategory,
  EdgeType,
  NodeType,
  NodeTypePrefix,
  OntologyScope,
  OntologyTypeDefinition,
  TypeStatus,
} from './types.js'

export {
  BUILT_IN_EDGE_TYPES,
  BUILT_IN_NODE_TYPES,
  BUSINESS_CATEGORIES,
  NODE_TYPE_PREFIXES,
  getNodeTypePrefixes,
  registerPrefix,
} from './types.js'

export {
  hasPrefix,
  isBuiltInBusinessCategory,
  isBuiltInEdgeType,
  isBuiltInNodeType,
  isValidBusinessCategory,
  isValidEdgeType,
  isValidNodeType,
  validateBusinessCategory,
  validateEdgeType,
  validateNodeType,
  validateTypeDefinition,
} from './validation.js'

export {
  getBuiltInEdgeTypeDescription,
  getBuiltInNodeTypeDescription,
} from './descriptions.js'

export {
  formatEdgeTypeLabel,
  formatNodeTypeLabel,
} from './format.js'

export type { TypeEntry, IndustryTemplate, TemplateName } from './templates.js'
export {
  INDUSTRY_TEMPLATES,
  listTemplates,
  getTemplate,
  registerTemplate,
} from './templates.js'

export type {
  FileOntologyStoreConfig,
  ListTypesOpts,
  OntologyStore,
  ResolvedOntology,
  ResolvedType,
  StoredOntology,
} from './store.js'

export { createFileOntologyStore } from './store.js'

export type { RegistryOptions } from './registry.js'
export { Registry, PROPOSE_DEDUP_THRESHOLD } from './registry.js'

export type { TypeDefinition } from './types.js'
export {
  Deduplicator,
  deduplicateType,
  EMBEDDING_AUTO_MERGE_THRESHOLD,
  EMBEDDING_REVIEW_THRESHOLD,
  FUZZY_LABEL_THRESHOLD,
} from './dedup.js'
export type {
  DeduplicationResult,
  DedupMethod,
  DedupResult,
  DedupResultKind,
  DeduplicatorConfig,
  MergedPair,
} from './dedup.js'
export { cosineSimilarity, jaroWinklerDistance } from './similarity.js'
