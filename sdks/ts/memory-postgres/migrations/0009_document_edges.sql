-- SPDX-License-Identifier: Apache-2.0
-- Migration 0009: Document relationship graph edges (P4-2).
-- Stores weighted links between documents so graph APIs can expose
-- semantic, structural, and ontology relationships per brain/tenant.

CREATE TABLE IF NOT EXISTS memory.document_edges (
  brain_id       uuid NOT NULL,
  tenant_id      uuid NOT NULL,
  source_doc_id  uuid NOT NULL,
  target_doc_id  uuid NOT NULL,
  edge_type      text NOT NULL CHECK (edge_type IN (
    'semantic_similarity', 'shared_tag', 'shared_folder',
    'same_session', 'supersedes', 'session_episode',
    'episode_heuristic', 'document_ontology', 'wikilink'
  )),
  weight         real NOT NULL CHECK (weight >= 0 AND weight <= 1),
  label          text,
  created_at     timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (brain_id, source_doc_id, target_doc_id, edge_type),
  CONSTRAINT document_edges_brain_fk
    FOREIGN KEY (brain_id)
    REFERENCES memory.brains(brain_id)
    ON DELETE CASCADE,
  CONSTRAINT document_edges_tenant_fk
    FOREIGN KEY (tenant_id)
    REFERENCES platform.tenants(tenant_id)
    ON DELETE CASCADE,
  CONSTRAINT document_edges_source_doc_fk
    FOREIGN KEY (source_doc_id)
    REFERENCES memory.documents(document_id)
    ON DELETE CASCADE,
  CONSTRAINT document_edges_target_doc_fk
    FOREIGN KEY (target_doc_id)
    REFERENCES memory.documents(document_id)
    ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS document_edges_brain_idx
  ON memory.document_edges (brain_id);
CREATE INDEX IF NOT EXISTS document_edges_source_idx
  ON memory.document_edges (source_doc_id);
CREATE INDEX IF NOT EXISTS document_edges_target_idx
  ON memory.document_edges (target_doc_id);
CREATE INDEX IF NOT EXISTS document_edges_tenant_idx
  ON memory.document_edges (tenant_id);

ALTER TABLE memory.document_edges ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation ON memory.document_edges;
CREATE POLICY tenant_isolation ON memory.document_edges
  USING (tenant_id = current_setting('app.tenant_id', true)::uuid);
