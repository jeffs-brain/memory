-- SPDX-License-Identifier: Apache-2.0
-- Migration 0007: Dead letter queue for permanent failure isolation (P3-3).
-- Jobs that exhaust their retry budget are moved here with full error
-- context so operators can inspect, retry, or purge them.

CREATE TABLE IF NOT EXISTS memory.ingest_dead_letter (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  original_job_id UUID NOT NULL,
  brain_id        TEXT NOT NULL,
  payload         JSONB NOT NULL,
  failure_reason  TEXT NOT NULL,
  last_error      TEXT,
  error_history   JSONB,
  retry_count     INTEGER NOT NULL DEFAULT 0,
  metadata        JSONB,
  group_id        TEXT,
  moved_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  resolved_at     TIMESTAMPTZ,
  resolved_by     TEXT
);

CREATE INDEX IF NOT EXISTS idx_dlq_brain
  ON memory.ingest_dead_letter (brain_id);

CREATE INDEX IF NOT EXISTS idx_dlq_moved
  ON memory.ingest_dead_letter (moved_at);

CREATE INDEX IF NOT EXISTS idx_dlq_unresolved
  ON memory.ingest_dead_letter (resolved_at)
  WHERE resolved_at IS NULL;
