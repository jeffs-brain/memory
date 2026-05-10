-- 0005_pipeline_state.sql
-- Pipeline state tracking for crash recovery. Stores the progress of each
-- document through the ingest pipeline so that incomplete documents can be
-- resumed from their last completed stage on restart.
--
-- Owned by ticket P1-1 (LLE-9779).

CREATE TABLE IF NOT EXISTS memory.pipeline_state (
  document_hash  TEXT        PRIMARY KEY,
  brain_id       TEXT        NOT NULL,
  stage          TEXT        NOT NULL DEFAULT 'received',
  retry_count    INTEGER     NOT NULL DEFAULT 0,
  last_error     TEXT,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at   TIMESTAMPTZ
);

-- Index for ListIncomplete queries: find all in-flight documents for a brain.
CREATE INDEX IF NOT EXISTS idx_pipeline_state_brain_incomplete
  ON memory.pipeline_state (brain_id, stage)
  WHERE stage NOT IN ('completed', 'failed');

-- Index for cleanup queries: find completed documents older than a threshold.
CREATE INDEX IF NOT EXISTS idx_pipeline_state_completed_at
  ON memory.pipeline_state (completed_at)
  WHERE completed_at IS NOT NULL;

COMMENT ON TABLE memory.pipeline_state IS
  'Tracks each document''s progress through the ingest pipeline for crash recovery.';
COMMENT ON COLUMN memory.pipeline_state.document_hash IS
  'Content-addressable hash of the raw document bytes (SHA-256 hex).';
COMMENT ON COLUMN memory.pipeline_state.brain_id IS
  'Identifies which brain the document belongs to.';
COMMENT ON COLUMN memory.pipeline_state.stage IS
  'Current pipeline stage: received, stored, chunked, embedded, indexed, completed, failed.';
COMMENT ON COLUMN memory.pipeline_state.retry_count IS
  'Number of times processing has been retried after failure.';
COMMENT ON COLUMN memory.pipeline_state.last_error IS
  'Human-readable description of the most recent failure, if any.';
COMMENT ON COLUMN memory.pipeline_state.completed_at IS
  'Timestamp when the document reached a terminal stage (completed or failed).';
