ALTER TABLE memory.documents
  ADD COLUMN IF NOT EXISTS metadata jsonb NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS documents_metadata_gin_idx
  ON memory.documents
  USING GIN (metadata jsonb_path_ops);
