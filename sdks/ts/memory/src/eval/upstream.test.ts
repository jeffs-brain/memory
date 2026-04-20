// SPDX-License-Identifier: Apache-2.0

import path from 'node:path'
import { describe, expect, it } from 'vitest'
import {
  LONG_MEM_EVAL_OFFICIAL_REPO_REF,
  LONG_MEM_EVAL_OFFICIAL_REPO_URL,
  LONG_MEM_EVAL_UPSTREAM_DATASETS,
  resolveUpstreamDatasetPath,
} from './upstream.js'

describe('upstream dataset metadata', () => {
  it('pins both legacy and cleaned bundles', () => {
    expect(LONG_MEM_EVAL_UPSTREAM_DATASETS.cleaned.files.oracle.filename).toBe(
      'longmemeval_oracle.json',
    )
    expect(LONG_MEM_EVAL_UPSTREAM_DATASETS.legacy.files.s.filename).toBe('longmemeval_s')
    expect(LONG_MEM_EVAL_UPSTREAM_DATASETS.cleaned.files.m.sha256).toMatch(/^[0-9a-f]{64}$/)
    expect(LONG_MEM_EVAL_OFFICIAL_REPO_URL).toBe('https://github.com/xiaowu0162/LongMemEval')
    expect(LONG_MEM_EVAL_OFFICIAL_REPO_REF).toBe('982fbd7045c9977e9119b5424cab0d7790d19413')
  })

  it('resolves deterministic local dataset paths', () => {
    const filePath = resolveUpstreamDatasetPath('cleaned', 's', '/tmp/lme')
    expect(filePath).toBe(path.join('/tmp/lme', 'longmemeval_s_cleaned.json'))
  })
})
