// SPDX-License-Identifier: Apache-2.0

// Package lme is the LongMemEval benchmark runner for the Jeffs Brain Go
// SDK. It ingests the LongMemEval dataset into an isolated brain, runs a
// direct-search pipeline for every question, scores the answers with an
// LLM judge (with a deterministic exact-match fallback) and aggregates a
// full report with latency percentiles, bootstrap confidence intervals,
// and per-stage cost accounting.
//
// The canonical dataset is longmemeval_s.json (500 questions). The judge
// model defaults to claude-haiku-4-5 with an env override via
// JB_LME_JUDGE_MODEL; the actor/reader model defaults to gpt-4o with an
// env override via JB_LME_ACTOR_MODEL.
//
// This port omits a handful of jeff-specific components that depend on
// an agentic-loop harness we have not ported to the SDK yet; see the
// TODO comments in agent.go, ingest_replay.go and review.go.
package lme
