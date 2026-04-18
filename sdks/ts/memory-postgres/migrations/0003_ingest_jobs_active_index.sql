-- 0003_ingest_jobs_active_index.sql
--
-- Prevent duplicate active queued or running jobs for the same document
-- content while still allowing historical completed and failed rows.

CREATE UNIQUE INDEX IF NOT EXISTS ingest_jobs_active_document_hash_idx
  ON platform.ingest_jobs (tenant_id, brain_id, document_id, content_hash)
  WHERE status IN ('queued', 'running');
