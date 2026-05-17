-- SPDX-License-Identifier: Apache-2.0
--
-- 0001_ingest_queue.sql
--
-- PostgreSQL ingest queue table for multi-worker job claiming.
-- Uses FOR UPDATE SKIP LOCKED for safe concurrent access and
-- LISTEN/NOTIFY for immediate wake on new jobs.

CREATE TABLE IF NOT EXISTS ingest_queue (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  brain_id        TEXT NOT NULL,
  status          TEXT NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'dead_letter')),
  payload         JSONB NOT NULL,
  retry_count     INTEGER NOT NULL DEFAULT 0,
  max_retries     INTEGER NOT NULL DEFAULT 3,
  error           TEXT,
  claimed_by      TEXT,
  claimed_at      TIMESTAMPTZ,
  last_heartbeat  TIMESTAMPTZ,
  next_retry_at   TIMESTAMPTZ,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  completed_at    TIMESTAMPTZ,
  metadata        JSONB,
  group_id        TEXT,
  idempotency_key TEXT
);

-- Claim index: find pending jobs ordered by creation time, filtering on retry eligibility.
CREATE INDEX IF NOT EXISTS idx_ingest_queue_claim
  ON ingest_queue (status, next_retry_at, created_at)
  WHERE status = 'pending';

-- Brain lookup index for per-brain queries.
CREATE INDEX IF NOT EXISTS idx_ingest_queue_brain
  ON ingest_queue (brain_id);

-- Group lookup index for batch/group queries.
CREATE INDEX IF NOT EXISTS idx_ingest_queue_group
  ON ingest_queue (group_id)
  WHERE group_id IS NOT NULL;

-- Stale heartbeat detection index for recovery of abandoned jobs.
CREATE INDEX IF NOT EXISTS idx_ingest_queue_stale
  ON ingest_queue (status, last_heartbeat)
  WHERE status = 'processing';

-- Idempotency constraint: prevent duplicate enqueue for the same logical
-- operation while the job is still active (not dead_letter).
CREATE UNIQUE INDEX IF NOT EXISTS idx_ingest_queue_idempotency
  ON ingest_queue (idempotency_key)
  WHERE idempotency_key IS NOT NULL AND status NOT IN ('dead_letter', 'completed', 'failed');

-- Trigger function for LISTEN/NOTIFY immediate dispatch.
CREATE OR REPLACE FUNCTION notify_ingest_queue() RETURNS trigger AS $$
BEGIN
  PERFORM pg_notify('ingest_queue_new_job', NEW.id::text);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER ingest_queue_notify
  AFTER INSERT ON ingest_queue
  FOR EACH ROW EXECUTE FUNCTION notify_ingest_queue();
