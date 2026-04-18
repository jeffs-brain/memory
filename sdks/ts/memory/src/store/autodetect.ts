import { access } from 'node:fs/promises'
import { join } from 'node:path'
import { createFsStore } from './fsstore.js'
import { createGitStore } from './gitstore.js'
import type { Store } from './index.js'

export type AutodetectOptions = {
  readonly root: string
}

// autodetectStore picks a backend for the given root. If `.git` is present
// the git-backed store is used so mutations produce commits; otherwise the
// filesystem store runs on bare files.
export const autodetectStore = async (opts: AutodetectOptions): Promise<Store> => {
  if (await hasGit(opts.root)) {
    return createGitStore({ dir: opts.root })
  }
  return createFsStore({ root: opts.root })
}

export const hasGit = async (root: string): Promise<boolean> => {
  try {
    await access(join(root, '.git'))
    return true
  } catch {
    return false
  }
}
