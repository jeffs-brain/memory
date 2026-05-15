// SPDX-License-Identifier: Apache-2.0

/**
 * Industry templates provide domain-specific starter packs of node types,
 * edge types, and business categories that seed an ontology for a given
 * industry vertical.
 *
 * Ported from packages/intelligence-database/src/templates/.
 */

export type TypeEntry = {
  readonly type: string
  readonly label: string
  readonly description: string
}

export type IndustryTemplate = {
  readonly label: string
  readonly description: string
  readonly nodeTypes: readonly TypeEntry[]
  readonly edgeTypes: readonly TypeEntry[]
  readonly businessCategories: readonly string[]
}

export type TemplateName =
  | 'server_hardware'
  | 'insurance'
  | 'logistics'
  | 'finance'
  | 'healthcare'
  | 'order_processing'

const NODE_TYPE_PATTERN = /^[a-z][a-z0-9]*\.[a-z][a-z0-9_]*$/
const EDGE_TYPE_PATTERN = /^[a-z][a-z0-9_]*$/
const BUSINESS_CATEGORY_PATTERN = /^[a-z][a-z0-9_]*$/

export function isValidNodeType(s: string): boolean {
  return NODE_TYPE_PATTERN.test(s)
}

export function isValidEdgeType(s: string): boolean {
  return EDGE_TYPE_PATTERN.test(s)
}

export function isValidBusinessCategory(s: string): boolean {
  return BUSINESS_CATEGORY_PATTERN.test(s)
}

const serverHardwareTemplate: IndustryTemplate = {
  label: 'Server Hardware',
  description: 'Server configuration, components, and compatibility rules',
  nodeTypes: [
    { type: 'entity.server_model', label: 'Server Model', description: 'A specific model of server hardware' },
    { type: 'entity.component_category', label: 'Component Category', description: 'A category of server components (CPU, Memory, Storage, etc.)' },
    { type: 'entity.component', label: 'Component', description: 'A specific hardware component' },
    { type: 'rule.hardware_compatibility', label: 'Hardware Compatibility', description: 'A compatibility constraint between components' },
    { type: 'rule.configuration_limit', label: 'Configuration Limit', description: 'A maximum or minimum limit on a configuration parameter' },
    { type: 'entity.server_generation', label: 'Server Generation', description: 'A server generation or family (e.g. 13th Gen, 14th Gen PowerEdge)' },
    { type: 'entity.configuration', label: 'Configuration', description: 'A specific hardware configuration combining a server model with selected components' },
    { type: 'rule.hardware_constraint', label: 'Hardware Constraint', description: 'A hard constraint that must be satisfied for a valid server configuration' },
    { type: 'rule.hardware_recommendation', label: 'Hardware Recommendation', description: 'A best-practice recommendation for optimal server configuration' },
  ],
  edgeTypes: [
    { type: 'compatible_with', label: 'Compatible With', description: 'Indicates hardware compatibility' },
    { type: 'installed_in', label: 'Installed In', description: 'Component installed in a server' },
    { type: 'requires', label: 'Requires', description: 'Component or configuration requires another component or condition' },
    { type: 'incompatible_with', label: 'Incompatible With', description: 'Two components or configurations cannot coexist in the same server' },
    { type: 'recommended_for', label: 'Recommended For', description: 'A component is recommended for a specific server model or configuration' },
    { type: 'applies_to', label: 'Applies To', description: 'A rule or constraint applies to a specific server model, generation, or component category' },
  ],
  businessCategories: ['server_hardware'],
} as const

const insuranceTemplate: IndustryTemplate = {
  label: 'Insurance',
  description: 'Policies, claims, underwriting rules, and risk assessment',
  nodeTypes: [
    { type: 'entity.policy', label: 'Policy', description: 'An insurance policy or cover' },
    { type: 'entity.claim', label: 'Claim', description: 'An insurance claim' },
    { type: 'entity.insured_party', label: 'Insured Party', description: 'A person or organisation covered by an insurance policy' },
    { type: 'entity.claimant', label: 'Claimant', description: 'A person or organisation making an insurance claim' },
    { type: 'entity.broker', label: 'Broker', description: 'An insurance broker or intermediary' },
    { type: 'entity.coverage', label: 'Coverage', description: 'A specific type of insurance coverage or protection' },
    { type: 'entity.peril', label: 'Peril', description: 'A specific risk or cause of loss that may be covered' },
    { type: 'entity.endorsement', label: 'Endorsement', description: 'A policy modification or amendment' },
    { type: 'entity.premium', label: 'Premium', description: 'The amount paid for insurance coverage' },
    { type: 'entity.deductible', label: 'Deductible', description: 'The amount the insured must pay before coverage applies' },
    { type: 'entity.risk_object', label: 'Risk Object', description: 'The property, person, or asset being insured' },
    { type: 'entity.underwriter', label: 'Underwriter', description: 'A person or system responsible for policy underwriting' },
    { type: 'entity.beneficiary', label: 'Beneficiary', description: 'A person or organisation entitled to policy benefits' },
    { type: 'rule.underwriting', label: 'Underwriting Rule', description: 'A rule governing policy underwriting decisions' },
    { type: 'rule.claims_assessment', label: 'Claims Assessment', description: 'A rule for assessing claim validity and settlement amount' },
    { type: 'rule.premium_calculation', label: 'Premium Calculation', description: 'A rule for calculating policy premiums based on risk factors' },
    { type: 'rule.coverage_determination', label: 'Coverage Determination', description: 'A rule determining what perils and losses are covered' },
    { type: 'rule.fraud_detection', label: 'Fraud Detection', description: 'A rule for identifying potentially fraudulent claims' },
    { type: 'rule.regulatory_compliance', label: 'Regulatory Compliance', description: 'A regulatory requirement that must be satisfied' },
    { type: 'rule.reserve_setting', label: 'Reserve Setting', description: 'A rule for setting financial reserves for claims' },
  ],
  edgeTypes: [
    { type: 'covers', label: 'Covers', description: 'Policy covers a risk object or peril' },
    { type: 'filed_against', label: 'Filed Against', description: 'A claim is filed against a specific policy' },
    { type: 'underwrites', label: 'Underwrites', description: 'An underwriter assesses and approves a policy' },
    { type: 'endorses', label: 'Endorses', description: 'An endorsement modifies a policy' },
    { type: 'cedes_to', label: 'Cedes To', description: 'Risk is transferred to a reinsurer' },
    { type: 'adjusts', label: 'Adjusts', description: 'An adjuster assesses a claim' },
    { type: 'pays_out', label: 'Pays Out', description: 'A policy pays benefits to a beneficiary' },
    { type: 'rated_for', label: 'Rated For', description: 'A premium is calculated for a specific risk classification' },
    { type: 'excludes', label: 'Excludes', description: 'A policy explicitly excludes coverage for a specific peril' },
    { type: 'subrogates_against', label: 'Subrogates Against', description: 'An insurer recovers costs from a third party' },
  ],
  businessCategories: [
    'underwriting',
    'claims_management',
    'policy_administration',
    'premium_rating',
    'reinsurance',
    'fraud_prevention',
    'regulatory_compliance',
  ],
} as const

const logisticsTemplate: IndustryTemplate = {
  label: 'Logistics & Supply Chain',
  description: 'Warehousing, transport, routing, and fulfilment rules',
  nodeTypes: [
    { type: 'entity.trade_item', label: 'Trade Item', description: 'A product or SKU that is traded or shipped' },
    { type: 'entity.logistic_unit', label: 'Logistic Unit', description: 'A packaging unit such as a pallet, carton, or container' },
    { type: 'entity.shipment', label: 'Shipment', description: 'A collection of goods being transported together' },
    { type: 'entity.consignment', label: 'Consignment', description: 'Goods entrusted to a carrier for delivery' },
    { type: 'entity.location', label: 'Location', description: 'A physical location such as a warehouse, depot, or store' },
    { type: 'entity.party', label: 'Party', description: 'An organisation or person involved in logistics operations' },
    { type: 'entity.transport_means', label: 'Transport Means', description: 'A vehicle, vessel, or aircraft used for transportation' },
    { type: 'entity.order', label: 'Order', description: 'A purchase order or sales order requiring fulfilment' },
    { type: 'entity.despatch_advice', label: 'Despatch Advice', description: 'Notification that goods have been dispatched' },
    { type: 'entity.receiving_advice', label: 'Receiving Advice', description: 'Notification that goods have been received' },
    { type: 'entity.bill_of_lading', label: 'Bill of Lading', description: 'A document acknowledging receipt of cargo for shipment' },
    { type: 'entity.customs_declaration', label: 'Customs Declaration', description: 'Documentation for customs clearance of international shipments' },
    { type: 'entity.inventory_position', label: 'Inventory Position', description: 'Stock level at a specific location and point in time' },
    { type: 'process.plan', label: 'Plan', description: 'Demand planning, supply planning, and capacity allocation' },
    { type: 'process.source', label: 'Source', description: 'Procurement and supplier management activities' },
    { type: 'process.make', label: 'Make', description: 'Manufacturing and production activities' },
    { type: 'process.deliver', label: 'Deliver', description: 'Order fulfilment, warehousing, and transportation activities' },
    { type: 'process.return_process', label: 'Return', description: 'Reverse logistics and returns processing' },
    { type: 'process.enable', label: 'Enable', description: 'Support processes such as data management and compliance' },
    { type: 'rule.routing', label: 'Routing Rule', description: 'A rule governing shipment routing and carrier selection' },
    { type: 'rule.capacity', label: 'Capacity Rule', description: 'A rule governing warehouse or vehicle capacity constraints' },
    { type: 'rule.lead_time', label: 'Lead Time', description: 'Expected time required for procurement, production, or delivery' },
    { type: 'rule.reorder_point', label: 'Reorder Point', description: 'Inventory threshold that triggers replenishment' },
    { type: 'rule.carrier_selection', label: 'Carrier Selection', description: 'A rule for selecting carriers based on cost, service level, or constraints' },
    { type: 'rule.hazmat_compliance', label: 'Hazmat Compliance', description: 'A rule governing handling and transport of hazardous materials' },
    { type: 'rule.temperature_control', label: 'Temperature Control', description: 'A rule for maintaining cold chain or controlled temperature' },
    { type: 'rule.customs_regulation', label: 'Customs Regulation', description: 'A customs requirement for international shipments' },
  ],
  edgeTypes: [
    { type: 'ships_to', label: 'Ships To', description: 'Origin location ships to destination location' },
    { type: 'stored_at', label: 'Stored At', description: 'Trade item or logistic unit stored at a location' },
    { type: 'carried_by', label: 'Carried By', description: 'Shipment or consignment carried by a transport means' },
    { type: 'ordered_by', label: 'Ordered By', description: 'An order is placed by a customer or party' },
    { type: 'supplies', label: 'Supplies', description: 'A supplier provides goods to a customer' },
    { type: 'received_at', label: 'Received At', description: 'Goods received at a location' },
    { type: 'returns_to', label: 'Returns To', description: 'Goods returned to a location or party' },
    { type: 'consolidates', label: 'Consolidates', description: 'Multiple shipments combined into a larger unit' },
    { type: 'deconsolidates', label: 'Deconsolidates', description: 'A shipment is broken down into smaller units' },
    { type: 'cross_docks', label: 'Cross Docks', description: 'Goods transferred directly from inbound to outbound transport' },
    { type: 'customs_cleared_by', label: 'Customs Cleared By', description: 'Shipment cleared through customs by an authority' },
  ],
  businessCategories: [
    'procurement',
    'warehousing',
    'transportation',
    'inventory_management',
    'order_fulfilment',
    'returns_processing',
    'demand_planning',
    'customs_compliance',
  ],
} as const

const financeTemplate: IndustryTemplate = {
  label: 'Finance & Banking',
  description: 'Accounts, transactions, approvals, and compliance rules',
  nodeTypes: [
    { type: 'entity.account', label: 'Account', description: 'A financial account for deposits, payments, or investments' },
    { type: 'entity.financial_instrument', label: 'Financial Instrument', description: 'A tradable asset such as a bond, equity, or derivative' },
    { type: 'entity.legal_entity', label: 'Legal Entity', description: 'An organisation with legal standing such as a company or trust' },
    { type: 'entity.natural_person', label: 'Natural Person', description: 'An individual human being who is a party to financial transactions' },
    { type: 'entity.currency', label: 'Currency', description: 'A monetary currency such as GBP, USD, or EUR' },
    { type: 'entity.collateral', label: 'Collateral', description: 'Assets pledged as security for a loan or credit facility' },
    { type: 'entity.portfolio', label: 'Portfolio', description: 'A collection of financial instruments held by an investor' },
    { type: 'entity.counterparty', label: 'Counterparty', description: 'The other party to a financial transaction or contract' },
    { type: 'entity.regulator', label: 'Regulator', description: 'A regulatory authority overseeing financial institutions' },
    { type: 'entity.market', label: 'Market', description: 'A financial market where instruments are traded' },
    { type: 'entity.payment', label: 'Payment', description: 'A transfer of funds between accounts or parties' },
    { type: 'entity.credit_facility', label: 'Credit Facility', description: 'A loan, line of credit, or other credit arrangement' },
    { type: 'rule.kyc', label: 'Know Your Customer', description: 'A rule requiring identity verification and due diligence' },
    { type: 'rule.aml', label: 'Anti-Money Laundering', description: 'A rule for detecting and preventing money laundering' },
    { type: 'rule.credit_limit', label: 'Credit Limit', description: 'A maximum amount of credit that may be extended' },
    { type: 'rule.compliance', label: 'Compliance Rule', description: 'A regulatory compliance requirement' },
    { type: 'rule.margin_requirement', label: 'Margin Requirement', description: 'Minimum collateral required to maintain a leveraged position' },
    { type: 'rule.risk_classification', label: 'Risk Classification', description: 'A rule categorising entities or instruments by risk level' },
    { type: 'rule.reporting_obligation', label: 'Reporting Obligation', description: 'A requirement to report transactions or positions to regulators' },
    { type: 'rule.sanctions_screening', label: 'Sanctions Screening', description: 'A rule checking parties against sanctions lists' },
    { type: 'rule.capital_adequacy', label: 'Capital Adequacy', description: 'Minimum capital requirements for financial institutions' },
    { type: 'rule.transaction_limit', label: 'Transaction Limit', description: 'A maximum amount or frequency for transactions' },
  ],
  edgeTypes: [
    { type: 'authorises', label: 'Authorises', description: 'An approver authorises a transaction or facility' },
    { type: 'subject_to', label: 'Subject To', description: 'An entity is subject to a regulation or requirement' },
    { type: 'issues', label: 'Issues', description: 'A legal entity issues a financial instrument' },
    { type: 'holds', label: 'Holds', description: 'A party holds an account or financial instrument' },
    { type: 'guarantees', label: 'Guarantees', description: 'A party guarantees payment or performance' },
    { type: 'settles', label: 'Settles', description: 'A transaction settles through a settlement system' },
    { type: 'denominates_in', label: 'Denominates In', description: 'An instrument or account is denominated in a currency' },
    { type: 'collateralised_by', label: 'Collateralised By', description: 'A facility is secured by collateral' },
    { type: 'trades_on', label: 'Trades On', description: 'An instrument trades on a specific market' },
    { type: 'regulated_by', label: 'Regulated By', description: 'An entity or market is regulated by an authority' },
    { type: 'beneficial_owner_of', label: 'Beneficial Owner Of', description: 'A natural person has beneficial ownership of an account or entity' },
    { type: 'counterparty_to', label: 'Counterparty To', description: 'A party is counterparty to a transaction or contract' },
  ],
  businessCategories: [
    'retail_banking',
    'commercial_banking',
    'investment_banking',
    'wealth_management',
    'payments',
    'regulatory_compliance',
    'risk_management',
    'trade_lifecycle',
    'credit_management',
  ],
} as const

const healthcareTemplate: IndustryTemplate = {
  label: 'Healthcare',
  description: 'Patients, procedures, medications, and clinical rules',
  nodeTypes: [
    { type: 'entity.patient', label: 'Patient', description: 'An individual receiving or registered to receive healthcare' },
    { type: 'entity.practitioner', label: 'Practitioner', description: 'A healthcare professional such as a doctor, nurse, or therapist' },
    { type: 'entity.organisation', label: 'Organisation', description: 'A healthcare provider organisation such as a hospital or clinic' },
    { type: 'entity.encounter', label: 'Encounter', description: 'An interaction between a patient and healthcare provider' },
    { type: 'entity.condition', label: 'Condition', description: 'A clinical condition, problem, or diagnosis' },
    { type: 'entity.procedure', label: 'Procedure', description: 'A medical procedure or intervention' },
    { type: 'entity.medication', label: 'Medication', description: 'A pharmaceutical product' },
    { type: 'entity.medication_request', label: 'Medication Request', description: 'A prescription or medication order' },
    { type: 'entity.observation', label: 'Observation', description: 'A clinical observation such as vital signs or lab results' },
    { type: 'entity.diagnostic_report', label: 'Diagnostic Report', description: 'Results of diagnostic investigations such as imaging or pathology' },
    { type: 'entity.allergy_intolerance', label: 'Allergy Intolerance', description: 'A documented allergy or intolerance to a substance' },
    { type: 'entity.immunisation', label: 'Immunisation', description: 'A vaccination record' },
    { type: 'entity.care_plan', label: 'Care Plan', description: 'A plan for delivering coordinated care to a patient' },
    { type: 'entity.care_team', label: 'Care Team', description: 'A group of practitioners responsible for a patient\'s care' },
    { type: 'entity.claim', label: 'Claim', description: 'A request for payment for healthcare services rendered' },
    { type: 'entity.coverage', label: 'Coverage', description: 'Health insurance coverage or benefit plan' },
    { type: 'entity.device', label: 'Device', description: 'A medical device or implant' },
    { type: 'entity.location', label: 'Location', description: 'A physical location such as a ward, clinic, or operating theatre' },
    { type: 'entity.service_request', label: 'Service Request', description: 'A request for a diagnostic service, referral, or procedure' },
    { type: 'entity.specimen', label: 'Specimen', description: 'A biological sample collected for testing' },
    { type: 'rule.clinical_protocol', label: 'Clinical Protocol', description: 'A rule governing clinical decision-making and treatment pathways' },
    { type: 'rule.contraindication', label: 'Contraindication', description: 'A rule about medication or procedure conflicts' },
    { type: 'rule.dosage_calculation', label: 'Dosage Calculation', description: 'A rule for calculating medication dosage based on patient factors' },
    { type: 'rule.eligibility', label: 'Eligibility', description: 'A rule determining patient eligibility for services or coverage' },
    { type: 'rule.coding_guideline', label: 'Coding Guideline', description: 'A guideline for coding diagnoses and procedures for billing' },
    { type: 'rule.prior_authorisation', label: 'Prior Authorisation', description: 'A requirement for approval before providing services' },
    { type: 'rule.clinical_pathway', label: 'Clinical Pathway', description: 'A standardised care pathway for a specific condition' },
    { type: 'rule.infection_control', label: 'Infection Control', description: 'A protocol for preventing and managing infections' },
    { type: 'rule.discharge_criteria', label: 'Discharge Criteria', description: 'Criteria that must be met before discharging a patient' },
    { type: 'rule.adverse_event', label: 'Adverse Event', description: 'A rule for reporting and managing adverse clinical events' },
  ],
  edgeTypes: [
    { type: 'treats', label: 'Treats', description: 'A procedure or medication treats a condition' },
    { type: 'diagnoses', label: 'Diagnoses', description: 'A practitioner diagnoses a patient with a condition' },
    { type: 'prescribes', label: 'Prescribes', description: 'A practitioner prescribes medication for a patient' },
    { type: 'contraindicates', label: 'Contraindicates', description: 'A substance or procedure is contraindicated for a condition' },
    { type: 'interacts_with', label: 'Interacts With', description: 'One medication interacts with another medication' },
    { type: 'finding_site', label: 'Finding Site', description: 'A condition or observation has a specific anatomical location' },
    { type: 'has_active_ingredient', label: 'Has Active Ingredient', description: 'A medication contains a specific active pharmaceutical ingredient' },
    { type: 'referred_to', label: 'Referred To', description: 'A patient is referred to a specialist or service' },
    { type: 'resulted_in', label: 'Resulted In', description: 'A procedure or intervention resulted in an outcome or condition' },
    { type: 'caused_by', label: 'Caused By', description: 'A condition is caused by another condition or exposure' },
    { type: 'covered_by', label: 'Covered By', description: 'A service or medication is covered by insurance' },
    { type: 'performed_at', label: 'Performed At', description: 'A procedure or service performed at a location' },
    { type: 'ordered_by', label: 'Ordered By', description: 'A service or medication ordered by a practitioner' },
    { type: 'monitors', label: 'Monitors', description: 'An observation monitors a condition or treatment' },
    { type: 'immunises_against', label: 'Immunises Against', description: 'A vaccination provides immunity against a disease' },
  ],
  businessCategories: [
    'patient_care',
    'clinical_documentation',
    'pharmacy',
    'laboratory',
    'radiology',
    'billing_coding',
    'referral_management',
    'public_health',
    'quality_measurement',
  ],
} as const

const orderProcessingTemplate: IndustryTemplate = {
  label: 'Order Processing',
  description: 'Email order intake, customer matching, product resolution, and ERP integration rules',
  nodeTypes: [
    { type: 'entity.order', label: 'Order', description: 'A sales order or purchase order being processed' },
    { type: 'entity.erp_system', label: 'ERP System', description: 'An enterprise resource planning system (Business Central, SAP, Zoho, etc.)' },
    { type: 'entity.email_source', label: 'Email Source', description: 'An email inbox or channel that receives order documents' },
    { type: 'rule.order_classification', label: 'Order Classification', description: 'A rule that classifies incoming emails or documents as specific order types' },
    { type: 'rule.customer_matching', label: 'Customer Matching', description: 'A rule for matching order sender to known customers using name, email, or identifiers' },
    { type: 'rule.product_resolution', label: 'Product Resolution', description: 'A rule for resolving product references to SKUs or catalogue items' },
    { type: 'rule.field_mapping', label: 'Field Mapping', description: 'A rule that maps extracted fields to ERP system fields' },
    { type: 'rule.erp_validation', label: 'ERP Validation', description: 'A validation gate that checks data readiness before ERP submission' },
  ],
  edgeTypes: [
    { type: 'maps_to', label: 'Maps To', description: 'One field or value maps to another in a different system' },
    { type: 'resolves_to', label: 'Resolves To', description: 'An extracted reference resolves to a specific system record' },
    { type: 'submits_to', label: 'Submits To', description: 'Processed data is submitted to an ERP or target system' },
  ],
  businessCategories: ['order_processing'],
} as const

/**
 * Mutable registry of industry templates. Starts with the 6 built-in
 * templates and can be extended at runtime via registerTemplate().
 */
const templateRegistry: Record<string, IndustryTemplate> = {
  server_hardware: serverHardwareTemplate,
  insurance: insuranceTemplate,
  logistics: logisticsTemplate,
  finance: financeTemplate,
  healthcare: healthcareTemplate,
  order_processing: orderProcessingTemplate,
}

/**
 * Read-only reference to the built-in templates for type narrowing.
 * Runtime lookups use templateRegistry which may include custom templates.
 */
export const INDUSTRY_TEMPLATES: Readonly<Record<TemplateName, IndustryTemplate>> = templateRegistry as Record<TemplateName, IndustryTemplate>

/**
 * Returns sorted template keys (includes any custom-registered templates).
 */
export function listTemplates(): string[] {
  return Object.keys(templateRegistry).sort()
}

/**
 * Returns a template by key, or undefined if not found.
 */
export function getTemplate(key: string): IndustryTemplate | undefined {
  return templateRegistry[key]
}

/**
 * Registers a custom industry template at runtime. Throws if the key
 * is empty, already registered, or the template contains invalid types.
 */
export function registerTemplate(key: string, template: IndustryTemplate): void {
  if (key === '') {
    throw new Error('ontology: template key must not be empty')
  }
  if (templateRegistry[key] !== undefined) {
    throw new Error(`ontology: template "${key}" is already registered`)
  }
  for (const nt of template.nodeTypes) {
    if (!isValidNodeType(nt.type)) {
      throw new Error(`ontology: template "${key}" contains invalid node type "${nt.type}"`)
    }
    if (!nt.label || !nt.description) {
      throw new Error(`ontology: template "${key}" node type "${nt.type}" has empty label or description`)
    }
  }
  for (const et of template.edgeTypes) {
    if (!isValidEdgeType(et.type)) {
      throw new Error(`ontology: template "${key}" contains invalid edge type "${et.type}"`)
    }
    if (!et.label || !et.description) {
      throw new Error(`ontology: template "${key}" edge type "${et.type}" has empty label or description`)
    }
  }
  for (const cat of template.businessCategories) {
    if (!isValidBusinessCategory(cat)) {
      throw new Error(`ontology: template "${key}" contains invalid business category "${cat}"`)
    }
  }
  templateRegistry[key] = template
}

/**
 * Resets the template registry to the 6 built-in templates (for testing).
 */
export function _resetTemplates(): void {
  for (const key of Object.keys(templateRegistry)) {
    if (!['server_hardware', 'insurance', 'logistics', 'finance', 'healthcare', 'order_processing'].includes(key)) {
      delete templateRegistry[key]
    }
  }
}
