// SPDX-License-Identifier: Apache-2.0

/**
 * String and vector similarity functions for the ontology deduplication
 * system. Jaro-Winkler is implemented inline (~30 lines) to avoid adding
 * a dependency for a single algorithm.
 */

/**
 * Compute the Jaro similarity between two strings.
 *
 * Time: O(max(len(s1), len(s2)))
 * Space: O(max(len(s1), len(s2)))
 */
function jaroSimilarity(s1: string, s2: string): number {
  if (s1.length === 0 && s2.length === 0) return 1.0
  if (s1.length === 0 || s2.length === 0) return 0.0

  const matchWindow = Math.max(0, Math.floor(Math.max(s1.length, s2.length) / 2) - 1)
  const s1Matches = new Array<boolean>(s1.length).fill(false)
  const s2Matches = new Array<boolean>(s2.length).fill(false)

  let matches = 0
  let transpositions = 0

  for (let i = 0; i < s1.length; i++) {
    const start = Math.max(0, i - matchWindow)
    const end = Math.min(i + matchWindow + 1, s2.length)
    for (let j = start; j < end; j++) {
      if (s2Matches[j] || s1[i] !== s2[j]) continue
      s1Matches[i] = true
      s2Matches[j] = true
      matches++
      break
    }
  }

  if (matches === 0) return 0.0

  let k = 0
  for (let i = 0; i < s1.length; i++) {
    if (!s1Matches[i]) continue
    while (!s2Matches[k]) k++
    if (s1[i] !== s2[k]) transpositions++
    k++
  }

  return (
    (matches / s1.length + matches / s2.length + (matches - transpositions / 2) / matches) / 3
  )
}

/**
 * Compute the Jaro-Winkler similarity between two strings.
 * Case-insensitive and whitespace-trimmed. Returns a value in [0, 1]
 * where 1 indicates identical strings.
 *
 * Time: O(max(len(s1), len(s2)))
 * Space: O(max(len(s1), len(s2)))
 */
export function jaroWinklerDistance(s1: string, s2: string): number {
  const a = s1.trim().toLowerCase()
  const b = s2.trim().toLowerCase()
  if (a.length === 0 && b.length === 0) return 1.0
  if (a.length === 0 || b.length === 0) return 0.0

  const jaro = jaroSimilarity(a, b)

  let commonPrefix = 0
  const maxPrefix = Math.min(4, Math.min(a.length, b.length))
  for (let i = 0; i < maxPrefix; i++) {
    if (a[i] !== b[i]) break
    commonPrefix++
  }

  return jaro + commonPrefix * 0.1 * (1 - jaro)
}

/**
 * Compute the cosine similarity between two number arrays. Returns a
 * value in [-1, 1] where 1 indicates identical direction. Returns 0
 * when either vector has zero magnitude.
 *
 * Throws if the vectors have different lengths.
 *
 * Time: O(n) where n = a.length
 * Space: O(1)
 */
export function cosineSimilarity(a: readonly number[], b: readonly number[]): number {
  if (a.length !== b.length) {
    throw new Error(`ontology: cosine similarity requires equal-length vectors (${a.length} vs ${b.length})`)
  }
  if (a.length === 0) return 0

  let dot = 0
  let normA = 0
  let normB = 0

  for (let i = 0; i < a.length; i++) {
    const ai = a[i]!
    const bi = b[i]!
    dot += ai * bi
    normA += ai * ai
    normB += bi * bi
  }

  const magA = Math.sqrt(normA)
  const magB = Math.sqrt(normB)
  if (magA === 0 || magB === 0) return 0

  return dot / (magA * magB)
}
