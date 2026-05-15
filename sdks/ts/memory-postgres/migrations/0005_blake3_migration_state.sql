-- 0005_blake3_migration_state.sql
--
-- Tracks BLAKE3 hash migration progress for PostgresStore backends.
-- File-based FsStore uses a JSON sidecar at raw/.pipeline-state/.blake3-migration.json.
-- PostgresStore uses this table instead, allowing atomic state updates and
-- concurrent migration workers without file-locking concerns.

CREATE TABLE IF NOT EXISTS memory.blake3_migration_state (
  id            int PRIMARY KEY DEFAULT 1 CHECK (id = 1),
  cursor        text NOT NULL DEFAULT '',
  migrated      int NOT NULL DEFAULT 0,
  total         int NOT NULL DEFAULT 0,
  started_at    timestamptz NOT NULL DEFAULT now(),
  updated_at    timestamptz NOT NULL DEFAULT now()
);

-- Seed the singleton row so UPSERT works cleanly.
INSERT INTO memory.blake3_migration_state (id) VALUES (1)
  ON CONFLICT (id) DO NOTHING;
