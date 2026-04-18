-- 0001_init.sql
-- Initial schema for the Postgres-backed memory store.
-- Covers the platform.tenants + memory.* tables required by PostgresStore
-- and PostgresSearchIndex.
--
-- BM25 note: the plan calls for ParadeDB's pg_search (Tantivy-backed BM25).
-- That ships only in ParadeDB's custom Postgres build. For dev and the stock
-- postgres:17 image we fall back to the built-in tsvector + GIN plus pg_trgm
-- for fuzzy matching, which is plenty for the current traffic profile. The
-- hybrid retrieval SQL in plan.md §4.3 can be adapted to use `@@` + ts_rank_cd
-- against the tsvector column until we swap the image to paradedb/paradedb.
--
-- pgvectorscale (StreamingDiskANN) is likewise an extra build step; enable it
-- once we have the proper pg image in deploy. Until then HNSW on halfvec is
-- the retrieval index.
--
-- Note: no explicit BEGIN/COMMIT in this file. drizzle-kit wraps each
-- migration in a single transaction automatically, and for test harnesses
-- that want one we recommend running the file through `psql -1` or
-- `pg.query(fs.readFileSync(...))` inside a transaction block on the client
-- side. postgres.js's pooled mode refuses embedded BEGIN/COMMIT for safety.

-- Extensions
CREATE EXTENSION IF NOT EXISTS citext;
CREATE EXTENSION IF NOT EXISTS pgcrypto;           -- gen_random_uuid
CREATE EXTENSION IF NOT EXISTS vector;             -- pgvector 0.9+ for halfvec
CREATE EXTENSION IF NOT EXISTS pg_trgm;            -- fuzzy fallback for BM25-adjacent queries
-- CREATE EXTENSION IF NOT EXISTS pg_search;       -- enable when running paradedb/paradedb image
-- CREATE EXTENSION IF NOT EXISTS vectorscale;     -- enable when pg image supports it

-- Schemas
CREATE SCHEMA IF NOT EXISTS platform;
CREATE SCHEMA IF NOT EXISTS memory;

-- ------------------------------------------------------------
-- platform.tenants
-- ------------------------------------------------------------
CREATE TABLE platform.tenants (
  tenant_id   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  slug        text NOT NULL UNIQUE,
  name        text NOT NULL,
  plan        text NOT NULL DEFAULT 'free',
  region      text NOT NULL,
  status      text NOT NULL DEFAULT 'active',
  created_at  timestamptz NOT NULL DEFAULT now(),
  settings    jsonb NOT NULL DEFAULT '{}'::jsonb
);

ALTER TABLE platform.tenants ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenants_isolation ON platform.tenants
  USING (tenant_id = current_setting('app.tenant_id', true)::uuid);

-- ------------------------------------------------------------
-- platform.users
-- ------------------------------------------------------------
CREATE TABLE platform.users (
  user_id       uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id     uuid NOT NULL REFERENCES platform.tenants(tenant_id) ON DELETE CASCADE,
  external_id   text NOT NULL,
  email         citext NOT NULL,
  display_name  text,
  created_at    timestamptz NOT NULL DEFAULT now(),
  last_active_at timestamptz,
  UNIQUE (tenant_id, external_id),
  UNIQUE (tenant_id, email)
);
CREATE INDEX users_tenant_idx ON platform.users (tenant_id);

ALTER TABLE platform.users ENABLE ROW LEVEL SECURITY;
CREATE POLICY users_isolation ON platform.users
  USING (tenant_id = current_setting('app.tenant_id', true)::uuid);

-- ------------------------------------------------------------
-- platform.api_keys
-- ------------------------------------------------------------
CREATE TABLE platform.api_keys (
  key_id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id       uuid NOT NULL REFERENCES platform.tenants(tenant_id) ON DELETE CASCADE,
  key_prefix      text NOT NULL,
  hashed_secret   bytea NOT NULL,
  name            text NOT NULL,
  scopes          text[] NOT NULL DEFAULT ARRAY[]::text[],
  rate_limit_rps  int NOT NULL DEFAULT 10,
  expires_at      timestamptz,
  last_used_at    timestamptz,
  revoked_at      timestamptz
);
CREATE INDEX api_keys_tenant_idx ON platform.api_keys (tenant_id);
CREATE INDEX api_keys_prefix_idx ON platform.api_keys (key_prefix);

ALTER TABLE platform.api_keys ENABLE ROW LEVEL SECURITY;
CREATE POLICY api_keys_isolation ON platform.api_keys
  USING (tenant_id = current_setting('app.tenant_id', true)::uuid);

-- ------------------------------------------------------------
-- platform.audit_log
-- ------------------------------------------------------------
CREATE TABLE platform.audit_log (
  event_id    uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id   uuid NOT NULL REFERENCES platform.tenants(tenant_id) ON DELETE CASCADE,
  actor_id    text NOT NULL,
  action      text NOT NULL,
  resource    text NOT NULL,
  metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX audit_log_tenant_created_idx ON platform.audit_log (tenant_id, created_at DESC);

ALTER TABLE platform.audit_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY audit_log_isolation ON platform.audit_log
  USING (tenant_id = current_setting('app.tenant_id', true)::uuid);

-- ------------------------------------------------------------
-- platform.usage_hourly
-- ------------------------------------------------------------
CREATE TABLE platform.usage_hourly (
  tenant_id  uuid NOT NULL REFERENCES platform.tenants(tenant_id) ON DELETE CASCADE,
  metric     text NOT NULL,
  value      bigint NOT NULL,
  period     timestamptz NOT NULL,
  PRIMARY KEY (tenant_id, metric, period)
);

ALTER TABLE platform.usage_hourly ENABLE ROW LEVEL SECURITY;
CREATE POLICY usage_hourly_isolation ON platform.usage_hourly
  USING (tenant_id = current_setting('app.tenant_id', true)::uuid);

-- ------------------------------------------------------------
-- platform.acl_tuples
-- ------------------------------------------------------------
CREATE TABLE platform.acl_tuples (
  tenant_id      uuid NOT NULL REFERENCES platform.tenants(tenant_id) ON DELETE CASCADE,
  subject_kind   text NOT NULL,
  subject_id     text NOT NULL,
  relation       text NOT NULL,
  resource_type  text NOT NULL,
  resource_id    text NOT NULL,
  created_at     timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, subject_kind, subject_id, relation, resource_type, resource_id)
);
CREATE INDEX acl_tuples_tenant_subject_idx
  ON platform.acl_tuples (tenant_id, subject_kind, subject_id);
CREATE INDEX acl_tuples_tenant_resource_idx
  ON platform.acl_tuples (tenant_id, resource_type, resource_id);

ALTER TABLE platform.acl_tuples ENABLE ROW LEVEL SECURITY;
CREATE POLICY acl_tuples_isolation ON platform.acl_tuples
  USING (tenant_id = current_setting('app.tenant_id', true)::uuid);

-- ------------------------------------------------------------
-- memory.brains
-- ------------------------------------------------------------
CREATE TABLE memory.brains (
  brain_id      uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id     uuid NOT NULL REFERENCES platform.tenants(tenant_id) ON DELETE CASCADE,
  slug          text NOT NULL,
  name          text NOT NULL,
  visibility    text NOT NULL DEFAULT 'private',
  git_remote    text,
  storage_bytes bigint NOT NULL DEFAULT 0,
  doc_count     int NOT NULL DEFAULT 0,
  created_at    timestamptz NOT NULL DEFAULT now(),
  updated_at    timestamptz NOT NULL DEFAULT now(),
  deleted_at    timestamptz,
  UNIQUE (tenant_id, slug)
);
CREATE INDEX brains_tenant_idx ON memory.brains (tenant_id);

ALTER TABLE memory.brains ENABLE ROW LEVEL SECURITY;
CREATE POLICY brains_isolation ON memory.brains
  USING (tenant_id = current_setting('app.tenant_id', true)::uuid);

-- ------------------------------------------------------------
-- memory.documents
-- ------------------------------------------------------------
CREATE TABLE memory.documents (
  document_id   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  brain_id      uuid NOT NULL REFERENCES memory.brains(brain_id) ON DELETE CASCADE,
  tenant_id     uuid NOT NULL REFERENCES platform.tenants(tenant_id) ON DELETE CASCADE,
  path          text NOT NULL,
  content_hash  bytea NOT NULL,
  content       bytea NOT NULL DEFAULT ''::bytea,
  size          int NOT NULL,
  source        text NOT NULL,
  created_at    timestamptz NOT NULL DEFAULT now(),
  updated_at    timestamptz NOT NULL DEFAULT now(),
  UNIQUE (brain_id, path)
);
CREATE INDEX documents_tenant_idx ON memory.documents (tenant_id);
CREATE INDEX documents_brain_idx ON memory.documents (brain_id);

ALTER TABLE memory.documents ENABLE ROW LEVEL SECURITY;
CREATE POLICY documents_isolation ON memory.documents
  USING (tenant_id = current_setting('app.tenant_id', true)::uuid);

-- ------------------------------------------------------------
-- memory.chunks (hash partitioned on tenant_id, 64 partitions)
-- ------------------------------------------------------------
-- Drizzle has no partition DSL, so we emit the DDL directly and keep the
-- Drizzle table definition as a row-shape descriptor. PK must include the
-- partition key, so it is composite (chunk_id, tenant_id).
CREATE TABLE memory.chunks (
  chunk_id     bigint NOT NULL,
  tenant_id    uuid NOT NULL,
  document_id  uuid NOT NULL,
  ordinal      int NOT NULL,
  content      text NOT NULL,
  tokens       int NOT NULL,
  fts_vector   tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
  embedding    halfvec(1024),
  created_at   timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (chunk_id, tenant_id)
) PARTITION BY HASH (tenant_id);

-- RLS on the partitioned parent propagates to children.
ALTER TABLE memory.chunks ENABLE ROW LEVEL SECURITY;
CREATE POLICY chunks_isolation ON memory.chunks
  USING (tenant_id = current_setting('app.tenant_id', true)::uuid);

-- Create 64 hash partitions
DO $$
DECLARE
  i int;
BEGIN
  FOR i IN 0..63 LOOP
    EXECUTE format(
      'CREATE TABLE memory.chunks_p%1$s PARTITION OF memory.chunks FOR VALUES WITH (modulus 64, remainder %1$s);',
      i
    );
  END LOOP;
END $$;

-- Indexes: declared on the partitioned parent so Postgres creates matching
-- children on each partition (Postgres 11+ does this automatically for btree,
-- GIN, and BRIN; HNSW is created per-partition via a loop because partitioned
-- CREATE INDEX does not yet dispatch access methods from pgvector reliably).
CREATE INDEX chunks_document_idx ON memory.chunks (document_id);
CREATE INDEX chunks_tenant_document_idx ON memory.chunks (tenant_id, document_id, ordinal);
CREATE INDEX chunks_fts_idx ON memory.chunks USING GIN (fts_vector);
CREATE INDEX chunks_content_trgm_idx ON memory.chunks USING GIN (content gin_trgm_ops);

-- HNSW per partition for halfvec cosine distance.
DO $$
DECLARE
  i int;
BEGIN
  FOR i IN 0..63 LOOP
    EXECUTE format(
      'CREATE INDEX chunks_p%1$s_embedding_hnsw ON memory.chunks_p%1$s USING hnsw (embedding halfvec_cosine_ops);',
      i
    );
  END LOOP;
END $$;

-- ------------------------------------------------------------
-- memory.episodes
-- ------------------------------------------------------------
CREATE TABLE memory.episodes (
  episode_id   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  brain_id     uuid NOT NULL REFERENCES memory.brains(brain_id) ON DELETE CASCADE,
  tenant_id    uuid NOT NULL REFERENCES platform.tenants(tenant_id) ON DELETE CASCADE,
  actor_id     text NOT NULL,
  started_at   timestamptz NOT NULL DEFAULT now(),
  ended_at     timestamptz,
  summary      text,
  outcome      text
);
CREATE INDEX episodes_tenant_idx ON memory.episodes (tenant_id);
CREATE INDEX episodes_brain_idx ON memory.episodes (brain_id);

ALTER TABLE memory.episodes ENABLE ROW LEVEL SECURITY;
CREATE POLICY episodes_isolation ON memory.episodes
  USING (tenant_id = current_setting('app.tenant_id', true)::uuid);

-- ------------------------------------------------------------
-- memory.extraction_cursors
-- ------------------------------------------------------------
CREATE TABLE memory.extraction_cursors (
  cursor_id       uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  brain_id        uuid NOT NULL REFERENCES memory.brains(brain_id) ON DELETE CASCADE,
  tenant_id       uuid NOT NULL REFERENCES platform.tenants(tenant_id) ON DELETE CASCADE,
  actor_id        text NOT NULL,
  last_message_id text,
  updated_at      timestamptz NOT NULL DEFAULT now(),
  UNIQUE (brain_id, actor_id)
);
CREATE INDEX extraction_cursors_tenant_idx ON memory.extraction_cursors (tenant_id);

ALTER TABLE memory.extraction_cursors ENABLE ROW LEVEL SECURITY;
CREATE POLICY extraction_cursors_isolation ON memory.extraction_cursors
  USING (tenant_id = current_setting('app.tenant_id', true)::uuid);
