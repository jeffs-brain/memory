-- 0006_ontology_proposals.sql
-- Ontology proposal workflow tables. Stores proposal batches and individual
-- proposals for the type discovery and review lifecycle. Supports filtering
-- by status and category for efficient retrieval.
--
-- Owned by ticket P6-7 (LLE-9795).

CREATE TABLE IF NOT EXISTS memory.ontology_proposal_batches (
  id              TEXT        PRIMARY KEY,
  domain          TEXT        NOT NULL DEFAULT '',
  source_document TEXT        NOT NULL DEFAULT '',
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS memory.ontology_proposals (
  id              TEXT        PRIMARY KEY,
  batch_id        TEXT        NOT NULL REFERENCES memory.ontology_proposal_batches(id) ON DELETE CASCADE,
  type_name       TEXT        NOT NULL,
  label           TEXT        NOT NULL,
  description     TEXT        NOT NULL DEFAULT '',
  category        TEXT        NOT NULL CHECK (category IN ('nodeType', 'edgeType')),
  discovered_from TEXT        NOT NULL DEFAULT '',
  status          TEXT        NOT NULL DEFAULT 'proposed'
                              CHECK (status IN ('proposed', 'accepted', 'merged', 'rejected')),
  reviewed_by     TEXT,
  reviewed_at     TIMESTAMPTZ,
  merged_into     TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ontology_proposals_batch_id
  ON memory.ontology_proposals(batch_id);

CREATE INDEX IF NOT EXISTS idx_ontology_proposals_status
  ON memory.ontology_proposals(status);

CREATE INDEX IF NOT EXISTS idx_ontology_proposals_category
  ON memory.ontology_proposals(category);
