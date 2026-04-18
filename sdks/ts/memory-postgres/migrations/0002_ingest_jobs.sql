-- 0002_ingest_jobs.sql
--
-- Async ingest queue for larger API uploads. This table intentionally does
-- not use tenant RLS because the worker claims jobs across all tenants.

CREATE TABLE platform.ingest_jobs (
  job_id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id      uuid NOT NULL REFERENCES platform.tenants(tenant_id) ON DELETE CASCADE,
  brain_id       uuid NOT NULL,
  document_id    uuid NOT NULL,
  path           text NOT NULL,
  mime           text NOT NULL,
  content_hash   bytea NOT NULL,
  status         text NOT NULL DEFAULT 'queued'
                 CHECK (status IN ('queued', 'running', 'completed', 'failed')),
  attempt_count  int NOT NULL DEFAULT 0,
  max_attempts   int NOT NULL DEFAULT 5,
  available_at   timestamptz NOT NULL DEFAULT now(),
  locked_at      timestamptz,
  last_error     text,
  result         jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at     timestamptz NOT NULL DEFAULT now(),
  updated_at     timestamptz NOT NULL DEFAULT now(),
  finished_at    timestamptz
);

CREATE INDEX ingest_jobs_status_available_idx
  ON platform.ingest_jobs (status, available_at, created_at);

CREATE INDEX ingest_jobs_tenant_brain_idx
  ON platform.ingest_jobs (tenant_id, brain_id, created_at DESC);
