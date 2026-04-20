// SPDX-License-Identifier: Apache-2.0

/**
 * ACL operations. Uses the in-memory RBAC adapter as the reference
 * implementation; tuples are held in a JSON sidecar at
 * `<brain>/acl/tuples.json` so `grant` / `revoke` / `list` actually persist
 * between runs.
 */

import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import { defineCommand } from 'citty'
import type {
  ReadTuplesQuery,
  Resource,
  ResourceType,
  Subject,
  SubjectKind,
  Tuple,
} from '../../acl/index.js'
import { resourceKey, subjectKey, tupleKey } from '../../acl/index.js'
import {
  type RbacRole,
  type RbacTupleStore,
  createInMemoryTupleStore,
  createRbacProvider,
} from '../../acl/rbac.js'
import { CliUsageError, resolveBrainDir } from '../config.js'

const ACL_PATH = 'acl/tuples.json'

const isResourceType = (v: string): v is ResourceType =>
  v === 'workspace' || v === 'brain' || v === 'collection' || v === 'document'

const isSubjectKind = (v: string): v is SubjectKind =>
  v === 'user' || v === 'api_key' || v === 'service'

const isRole = (v: string): v is RbacRole => v === 'admin' || v === 'writer' || v === 'reader'

const parseRef = (
  raw: string,
  validate: (kind: string) => boolean,
  what: string,
): { kind: string; id: string } => {
  const idx = raw.indexOf(':')
  if (idx === -1) {
    throw new CliUsageError(`${what} must be in the form <kind>:<id>, got '${raw}'`)
  }
  const kind = raw.slice(0, idx)
  const id = raw.slice(idx + 1)
  if (!validate(kind) || id === '') {
    throw new CliUsageError(`${what} has unsupported kind '${kind}' or empty id`)
  }
  return { kind, id }
}

const parseSubject = (raw: string): Subject => {
  const { kind, id } = parseRef(raw, isSubjectKind, 'subject')
  return { kind: kind as SubjectKind, id }
}

const parseResource = (raw: string): Resource => {
  const { kind, id } = parseRef(raw, isResourceType, 'resource')
  return { type: kind as ResourceType, id }
}

const loadStore = async (
  brainDir: string,
): Promise<{
  store: RbacTupleStore
  save: () => Promise<void>
  all: () => Tuple[]
}> => {
  const full = join(brainDir, ACL_PATH)
  const store = createInMemoryTupleStore()
  const all: Tuple[] = []
  try {
    const raw = await readFile(full, 'utf8')
    const parsed = JSON.parse(raw) as unknown
    if (Array.isArray(parsed)) {
      for (const row of parsed) {
        if (isTuple(row)) {
          store.add(row)
          all.push(row)
        }
      }
    }
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== 'ENOENT') throw err
  }
  return {
    store: {
      add: (t) => {
        store.add(t)
        const key = tupleKey(t)
        const idx = all.findIndex((x) => tupleKey(x) === key)
        if (idx === -1) all.push(t)
        else all[idx] = t
      },
      remove: (t) => {
        store.remove(t)
        const key = tupleKey(t)
        const idx = all.findIndex((x) => tupleKey(x) === key)
        if (idx !== -1) all.splice(idx, 1)
      },
      list: (q: ReadTuplesQuery) => store.list(q),
    },
    save: async () => {
      await mkdir(dirname(full), { recursive: true })
      await writeFile(full, `${JSON.stringify(all, null, 2)}\n`, 'utf8')
    },
    all: () => [...all],
  }
}

const isTuple = (row: unknown): row is Tuple => {
  if (row === null || typeof row !== 'object') return false
  const r = row as Record<string, unknown>
  const sub = r.subject
  const res = r.resource
  return (
    typeof r.relation === 'string' &&
    sub !== null &&
    typeof sub === 'object' &&
    res !== null &&
    typeof res === 'object'
  )
}

const grantCommand = defineCommand({
  meta: { name: 'grant', description: 'Grant a role to a subject on a resource' },
  args: {
    brain: { type: 'string', description: 'Brain directory' },
    subject: {
      type: 'string',
      description: 'Subject in <kind>:<id> form (user|api_key|service)',
      required: true,
    },
    role: { type: 'string', description: 'admin|writer|reader', required: true },
    resource: {
      type: 'string',
      description: 'Resource in <kind>:<id> form (workspace|brain|collection|document)',
      required: true,
    },
  },
  run: async ({ args }) => {
    if (!isRole(String(args.role))) {
      throw new CliUsageError(`acl grant: invalid role '${String(args.role)}'`)
    }
    const subject = parseSubject(String(args.subject))
    const resource = parseResource(String(args.resource))
    const brainDir = resolveBrainDir(typeof args.brain === 'string' ? args.brain : undefined)
    const loaded = await loadStore(brainDir)
    const provider = createRbacProvider({ store: loaded.store })
    await provider.write?.({
      writes: [{ subject, relation: String(args.role), resource }],
    })
    await loaded.save()
    process.stdout.write(
      `${JSON.stringify({
        granted: {
          subject: subjectKey(subject),
          role: args.role,
          resource: resourceKey(resource),
        },
      })}\n`,
    )
  },
})

const revokeCommand = defineCommand({
  meta: { name: 'revoke', description: 'Revoke a role from a subject on a resource' },
  args: {
    brain: { type: 'string', description: 'Brain directory' },
    subject: { type: 'string', required: true },
    role: { type: 'string', required: true },
    resource: { type: 'string', required: true },
  },
  run: async ({ args }) => {
    if (!isRole(String(args.role))) {
      throw new CliUsageError(`acl revoke: invalid role '${String(args.role)}'`)
    }
    const subject = parseSubject(String(args.subject))
    const resource = parseResource(String(args.resource))
    const brainDir = resolveBrainDir(typeof args.brain === 'string' ? args.brain : undefined)
    const loaded = await loadStore(brainDir)
    const provider = createRbacProvider({ store: loaded.store })
    await provider.write?.({
      deletes: [{ subject, relation: String(args.role), resource }],
    })
    await loaded.save()
    process.stdout.write(`${JSON.stringify({ revoked: true })}\n`)
  },
})

const listAclCommand = defineCommand({
  meta: { name: 'list', description: 'List ACL tuples' },
  args: {
    brain: { type: 'string', description: 'Brain directory' },
  },
  run: async ({ args }) => {
    const brainDir = resolveBrainDir(typeof args.brain === 'string' ? args.brain : undefined)
    const loaded = await loadStore(brainDir)
    process.stdout.write(`${JSON.stringify({ tuples: loaded.all() })}\n`)
  },
})

export const aclCommand = defineCommand({
  meta: { name: 'acl', description: 'Manage ACL tuples on a brain' },
  subCommands: {
    grant: grantCommand,
    revoke: revokeCommand,
    list: listAclCommand,
  },
})
