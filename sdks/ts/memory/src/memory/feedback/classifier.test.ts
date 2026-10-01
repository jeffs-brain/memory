// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'vitest'
import { toPath } from '../../store/path.js'
import { createFeedbackClassifier } from './classifier.js'

const testPath = toPath('memory/global/test.md')

describe('FeedbackClassifier', () => {
  describe('classify', () => {
    it('detects positive signal', () => {
      const c = createFeedbackClassifier()
      const result = c.classify('perfect, thanks for that', [testPath])

      expect(result.events).toHaveLength(1)
      const ev = result.events[0]
      expect(ev.reaction).toBe('reinforced')
      expect(ev.confidence).toBeGreaterThan(0)
    })

    it('detects negative signal', () => {
      const c = createFeedbackClassifier()
      const result = c.classify("that's wrong, it was updated last week", [testPath])

      expect(result.events).toHaveLength(1)
      const ev = result.events[0]
      expect(ev.reaction).toBe('corrected')
    })

    it('returns neutral when no patterns match', () => {
      const c = createFeedbackClassifier()
      const result = c.classify('can you show me the deployment logs', [testPath])

      expect(result.events).toHaveLength(1)
      const ev = result.events[0]
      expect(ev.reaction).toBe('neutral')
      expect(ev.confidence).toBe(0.0)
    })

    it('positive wins when more positive matches', () => {
      const c = createFeedbackClassifier()
      const result = c.classify("no but perfect, that's helpful, spot on", [testPath])

      expect(result.events).toHaveLength(1)
      const ev = result.events[0]
      expect(ev.reaction).toBe('reinforced')
    })

    it('negative wins when more negative matches', () => {
      const c = createFeedbackClassifier()
      const result = c.classify("yes but that's wrong, incorrect, try again", [testPath])

      expect(result.events).toHaveLength(1)
      const ev = result.events[0]
      expect(ev.reaction).toBe('corrected')
    })

    it('tie goes to neutral with 0.2 confidence', () => {
      const c = createFeedbackClassifier()
      const result = c.classify('yes and no', [testPath])

      expect(result.events).toHaveLength(1)
      const ev = result.events[0]
      expect(ev.reaction).toBe('neutral')
      expect(ev.confidence).toBe(0.2)
    })

    it('returns no events for empty input', () => {
      const c = createFeedbackClassifier()
      const result = c.classify('', [testPath])

      expect(result.events).toHaveLength(0)
    })

    it('returns no events for empty surfaced paths', () => {
      const c = createFeedbackClassifier()
      const result = c.classify('perfect, thanks', [])

      expect(result.events).toHaveLength(0)
    })

    it('produces events for each surfaced path', () => {
      const c = createFeedbackClassifier()
      const paths = [
        toPath('memory/global/topic-a.md'),
        toPath('memory/global/topic-b.md'),
        toPath('memory/project/foo/bar.md'),
      ]
      const result = c.classify('great, thanks', paths)

      expect(result.events).toHaveLength(3)
      for (let i = 0; i < result.events.length; i++) {
        expect(result.events[i].memoryPath).toBe(paths[i])
        expect(result.events[i].reaction).toBe('reinforced')
      }
    })

    it('confidence scales with match count', () => {
      const c = createFeedbackClassifier()

      const r1 = c.classify('thanks', [testPath])
      const r2 = c.classify("thanks, that's helpful, you remembered, that helps, spot on", [
        testPath,
      ])

      expect(r1.events).toHaveLength(1)
      expect(r2.events).toHaveLength(1)

      const c1 = r1.events[0].confidence
      const c2 = r2.events[0].confidence

      expect(c1).toBeLessThan(c2)
      expect(Math.abs(c1 - 0.3)).toBeLessThan(0.001)
      expect(c2).toBeLessThanOrEqual(1.0)
    })

    it('captures the matched pattern text', () => {
      const c = createFeedbackClassifier()
      const result = c.classify('you remembered that well', [testPath])

      expect(result.events).toHaveLength(1)
      const ev = result.events[0]
      expect(ev.pattern).not.toBe('')
      expect(ev.reaction).toBe('reinforced')
    })

    it('truncates snippet to 200 characters', () => {
      const c = createFeedbackClassifier()
      const longInput = `perfect ${'x'.repeat(300)}`
      const result = c.classify(longInput, [testPath])

      expect(result.events).toHaveLength(1)
      const ev = result.events[0]
      expect(ev.snippet.length).toBeLessThanOrEqual(203)
      expect(ev.snippet).toMatch(/\.\.\.$/)
    })

    it('truncates turnContent to 500 characters', () => {
      const c = createFeedbackClassifier()
      const longInput = `perfect ${'x'.repeat(600)}`
      const result = c.classify(longInput, [testPath])

      expect(result.turnContent.length).toBeLessThanOrEqual(503)
    })

    // Additional test cases from task requirements

    it('classifies "that\'s correct" as reinforced', () => {
      const c = createFeedbackClassifier()
      const result = c.classify("that's correct", [testPath])

      expect(result.events).toHaveLength(1)
      expect(result.events[0].reaction).toBe('reinforced')
    })

    it('classifies "good job" as neutral (no memory-specific pattern)', () => {
      const c = createFeedbackClassifier()
      // "good job" does not match "good memory/recall/find" pattern,
      // but "good" is not in the positive word list alone, so neutral.
      // However, looking at the Go patterns, there is no standalone "good" match.
      // Only "good memory", "good recall", "good find".
      // "job" is not in the positive patterns.
      const result = c.classify('good job', [testPath])

      expect(result.events).toHaveLength(1)
      // "good job" does not match any pattern — neutral.
      expect(result.events[0].reaction).toBe('neutral')
    })

    it('classifies "that\'s wrong" as corrected', () => {
      const c = createFeedbackClassifier()
      const result = c.classify("that's wrong", [testPath])

      expect(result.events).toHaveLength(1)
      expect(result.events[0].reaction).toBe('corrected')
    })

    it('classifies "no, incorrect" as corrected', () => {
      const c = createFeedbackClassifier()
      const result = c.classify('no, incorrect', [testPath])

      expect(result.events).toHaveLength(1)
      expect(result.events[0].reaction).toBe('corrected')
    })

    it('classifies "actually it\'s X not Y" as corrected', () => {
      const c = createFeedbackClassifier()
      const result = c.classify("actually it's TypeScript not JavaScript", [testPath])

      expect(result.events).toHaveLength(1)
      expect(result.events[0].reaction).toBe('corrected')
    })

    it('classifies normal conversational text as neutral', () => {
      const c = createFeedbackClassifier()
      const result = c.classify('can you explain how the build system works', [testPath])

      expect(result.events).toHaveLength(1)
      expect(result.events[0].reaction).toBe('neutral')
      expect(result.events[0].confidence).toBe(0.0)
    })

    it('returns empty events for whitespace-only input', () => {
      const c = createFeedbackClassifier()
      const result = c.classify('   ', [testPath])

      expect(result.events).toHaveLength(0)
    })
  })
})
