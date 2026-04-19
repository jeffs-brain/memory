// SPDX-License-Identifier: Apache-2.0

package main

import (
	"log/slog"
	"testing"

	"github.com/jeffs-brain/memory/go/llm"
	"github.com/jeffs-brain/memory/go/retrieval"
)

func TestBuildReranker_AutoPrefersHTTP(t *testing.T) {
	t.Setenv("JB_RERANK_PROVIDER", "auto")
	t.Setenv("JB_RERANK_URL", "http://127.0.0.1:8012")

	rr := buildReranker(&Daemon{LLM: llm.NewFake([]string{"ok"})}, slog.Default())
	if _, ok := rr.(*retrieval.AutoReranker); !ok {
		t.Fatalf("reranker type = %T, want *retrieval.AutoReranker", rr)
	}
}

func TestBuildReranker_AutoFallsBackToLLM(t *testing.T) {
	t.Setenv("JB_RERANK_PROVIDER", "auto")
	t.Setenv("JB_RERANK_URL", "")

	rr := buildReranker(&Daemon{LLM: llm.NewFake([]string{"ok"})}, slog.Default())
	if _, ok := rr.(*retrieval.LLMReranker); !ok {
		t.Fatalf("reranker type = %T, want *retrieval.LLMReranker", rr)
	}
}

func TestBuildReranker_HTTPAliasUsesHTTPReranker(t *testing.T) {
	t.Setenv("JB_RERANK_PROVIDER", "tei")
	t.Setenv("JB_RERANK_URL", "http://127.0.0.1:8012")

	rr := buildReranker(&Daemon{LLM: llm.NewFake([]string{"ok"})}, slog.Default())
	if _, ok := rr.(*retrieval.HTTPReranker); !ok {
		t.Fatalf("reranker type = %T, want *retrieval.HTTPReranker", rr)
	}
}
