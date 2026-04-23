import { type Path, joinPath, validatePathSegment } from '../store/index.js'

export type Scope = 'global' | 'project' | 'agent'

export const MEMORY_GLOBAL_PREFIX: Path = 'memory/global' as Path
export const MEMORY_PROJECTS_PREFIX: Path = 'memory/project' as Path
export const MEMORY_AGENT_PREFIX: Path = 'memory/agent' as Path
export const REFLECTIONS_PREFIX: Path = 'reflections' as Path

const actorSegment = (actorId: string): string => {
  validatePathSegment(actorId)
  return actorId
}

export const scopePrefix = (scope: Scope, actorId: string): Path => {
  switch (scope) {
    case 'global':
      return MEMORY_GLOBAL_PREFIX
    case 'project':
      return joinPath(MEMORY_PROJECTS_PREFIX, actorSegment(actorId))
    case 'agent':
      return joinPath(MEMORY_AGENT_PREFIX, actorSegment(actorId))
  }
}

export const scopeTopic = (scope: Scope, actorId: string, filename: string): Path => {
  return joinPath(scopePrefix(scope, actorId), ensureMarkdown(filename))
}

export const scopeIndex = (scope: Scope, actorId: string): Path =>
  joinPath(scopePrefix(scope, actorId), 'MEMORY.md')

export const reflectionPath = (sessionId: string): Path =>
  joinPath(REFLECTIONS_PREFIX, ensureMarkdown(sessionId))

export const sanitiseFilename = (name: string): string => {
  const last = name.split(/[\\/]/).filter(Boolean).pop() ?? name
  const cleaned = last.replace(/[^A-Za-z0-9._-]/g, '_')
  if (cleaned === '' || cleaned.startsWith('.')) {
    return `note_${Date.now()}.md`
  }
  return cleaned
}

export const ensureMarkdown = (name: string): string => {
  const safe = sanitiseFilename(name)
  return safe.endsWith('.md') ? safe : `${safe}.md`
}
