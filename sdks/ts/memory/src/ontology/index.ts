// SPDX-License-Identifier: Apache-2.0

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
