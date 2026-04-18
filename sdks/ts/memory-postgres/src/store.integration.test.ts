import { execSync } from 'node:child_process'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import postgres from 'postgres'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { toPath } from '@jeffs-brain/memory/store'
import { createPostgresStore } from './store.js'
import type { PgSql } from './store.js'

const hasDocker = (() => {
  try {
    execSync('docker info', { stdio: 'ignore' })
    return true
  } catch {
    return false
  }
})()

const maybe = hasDocker ? describe : describe.skip

maybe('PostgresStore (testcontainers)', () => {
  type RawSql = ReturnType<typeof postgres>
  let superSql: RawSql
  let appSql: RawSql
  let stop: () => Promise<void>
  const tenantA = '11111111-1111-1111-1111-111111111111'
  const tenantB = '22222222-2222-2222-2222-222222222222'
  const brainA = 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'
  const brainB = 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb'

  beforeAll(async () => {
    const { PostgreSqlContainer } = await import('@testcontainers/postgresql')
    const container = await new PostgreSqlContainer('pgvector/pgvector:pg17')
      .withDatabase('jeffs_brain')
      .withUsername('postgres')
      .withPassword('postgres')
      .start()

    const uri = container.getConnectionUri()
    superSql = postgres(uri, { max: 2, prepare: false })

    const here = path.dirname(fileURLToPath(import.meta.url))
    const migration = readFileSync(
      path.join(here, '..', 'migrations', '0001_init.sql'),
      'utf8',
    )
    await superSql.unsafe(migration).simple()

    // Seed tenants + brains as superuser.
    await superSql.unsafe(
      `insert into platform.tenants (tenant_id, slug, name, region) values
         ('${tenantA}', 'tenant-a', 'Tenant A', 'eu-fsn1'),
         ('${tenantB}', 'tenant-b', 'Tenant B', 'eu-fsn1')`,
    )
    await superSql.unsafe(
      `insert into memory.brains (brain_id, tenant_id, slug, name) values
         ('${brainA}', '${tenantA}', 'brain-a', 'Brain A'),
         ('${brainB}', '${tenantB}', 'brain-b', 'Brain B')`,
    )

    // Create a non-superuser role for RLS-enforced tests.
    await superSql.unsafe(`
      do $$ begin
        if not exists (select 1 from pg_roles where rolname = 'app') then
          create role app login password 'app';
        end if;
      end $$;
      grant usage on schema platform, memory to app;
      grant select, insert, update, delete on all tables in schema platform to app;
      grant select, insert, update, delete on all tables in schema memory to app;
    `)

    // Swap the connection user to `app`. Passwords are the local one we set.
    const appUri = uri.replace('postgres://postgres:postgres@', 'postgres://app:app@')
    appSql = postgres(appUri, { max: 2, prepare: false })

    stop = async () => {
      await appSql.end({ timeout: 5 })
      await superSql.end({ timeout: 5 })
      await container.stop()
    }
  }, 120_000)

  afterAll(async () => {
    if (stop) await stop()
  })

  it('put + get roundtrip', async () => {
    const store = await createPostgresStore({
      sql: superSql as unknown as PgSql,
      tenantId: tenantA,
      brainId: brainA,
    })
    const p = toPath('notes/hello.md')
    await store.write(p, Buffer.from('hello world'))
    const got = await store.read(p)
    expect(got.toString('utf8')).toBe('hello world')

    expect(await store.exists(p)).toBe(true)
    const info = await store.stat(p)
    expect(info.path).toBe(p)
    expect(info.size).toBe(11)
    expect(info.isDir).toBe(false)

    const top = await store.list('', { recursive: false })
    expect(top.map((f) => f.path)).toContain('notes')

    const all = await store.list('', { recursive: true })
    expect(all.map((f) => f.path)).toContain('notes/hello.md')

    await store.delete(p)
    expect(await store.exists(p)).toBe(false)

    await store.close()
  }, 60_000)

  it('batch wraps multiple ops in one transaction', async () => {
    const store = await createPostgresStore({
      sql: superSql as unknown as PgSql,
      tenantId: tenantA,
      brainId: brainA,
    })
    const a = toPath('batch/a.txt')
    const b = toPath('batch/b.txt')
    await store.batch({ reason: 'init' }, async (batch) => {
      await batch.write(a, Buffer.from('alpha'))
      await batch.write(b, Buffer.from('beta'))
    })
    expect((await store.read(a)).toString()).toBe('alpha')
    expect((await store.read(b)).toString()).toBe('beta')

    const c = toPath('batch/c.txt')
    await expect(
      store.batch({ reason: 'boom' }, async (batch) => {
        await batch.write(c, Buffer.from('gamma'))
        throw new Error('boom')
      }),
    ).rejects.toThrow('boom')
    expect(await store.exists(c)).toBe(false)

    await store.close()
  }, 60_000)

  it('enforces cross-tenant isolation via RLS', async () => {
    // Seed as superuser (RLS bypassed for the owner) so we know the row exists.
    const superStore = await createPostgresStore({
      sql: superSql as unknown as PgSql,
      tenantId: tenantA,
      brainId: brainA,
    })
    const secret = toPath('private/secret.md')
    await superStore.write(secret, Buffer.from('tenant-A-only'))
    await superStore.close()

    // Run a PostgresStore as tenant B on the RLS-constrained `app` pool.
    // The store already sets app.tenant_id via set_config inside every
    // transaction, so the policy on memory.documents (tenant_id =
    // current_setting('app.tenant_id')) scopes reads to tenant B.
    const storeB = await createPostgresStore({
      sql: appSql as unknown as PgSql,
      tenantId: tenantB,
      brainId: brainB,
      // The content column was already added by the superuser init; skip
      // the ALTER (app role cannot alter the table).
      initContentColumn: false,
    })
    expect(await storeB.exists(secret)).toBe(false)
    await expect(storeB.read(secret)).rejects.toThrow(/not found/)
    const listing = await storeB.list('', { recursive: true })
    expect(listing.map((f) => f.path)).not.toContain('private/secret.md')
    await storeB.close()

    // The same document IS visible when we use an RLS-scoped store for tenant A.
    const storeA = await createPostgresStore({
      sql: appSql as unknown as PgSql,
      tenantId: tenantA,
      brainId: brainA,
      initContentColumn: false,
    })
    expect(await storeA.exists(secret)).toBe(true)
    expect((await storeA.read(secret)).toString()).toBe('tenant-A-only')
    await storeA.close()
  }, 60_000)
})
