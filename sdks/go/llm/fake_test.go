// SPDX-License-Identifier: Apache-2.0

package llm_test

import (
	"context"
	"math"
	"strings"
	"testing"

	"github.com/jeffs-brain/memory/go/llm"
)

func TestFakeRoundRobin(t *testing.T) {
	t.Parallel()
	p := llm.NewFake([]string{"one", "two", "three"})
	defer func() { _ = p.Close() }()

	ctx := context.Background()
	req := llm.CompleteRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	}

	want := []string{"one", "two", "three", "one", "two"}
	for _, w := range want {
		got, err := p.Complete(ctx, req)
		if err != nil {
			t.Fatalf("complete: %v", err)
		}
		if got.Text != w {
			t.Fatalf("want %q, got %q", w, got.Text)
		}
		if got.Stop != llm.StopEndTurn {
			t.Fatalf("unexpected stop %q", got.Stop)
		}
	}
}

func TestFakeStreaming(t *testing.T) {
	t.Parallel()
	p := llm.NewFake([]string{"abc"})
	defer func() { _ = p.Close() }()

	ctx := context.Background()
	ch, err := p.CompleteStream(ctx, llm.CompleteRequest{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	var got strings.Builder
	var stop llm.StopReason
	for chunk := range ch {
		got.WriteString(chunk.DeltaText)
		if chunk.Stop != "" {
			stop = chunk.Stop
		}
	}
	if got.String() != "abc" {
		t.Fatalf("want abc got %q", got.String())
	}
	if stop != llm.StopEndTurn {
		t.Fatalf("want end_turn got %q", stop)
	}
}

func TestFakeRejectsEmpty(t *testing.T) {
	t.Parallel()
	p := llm.NewFake([]string{"x"})
	if _, err := p.Complete(context.Background(), llm.CompleteRequest{}); err == nil {
		t.Fatalf("expected error on empty messages")
	}
}

func TestFakeEmbedderDeterministic(t *testing.T) {
	t.Parallel()
	e := llm.NewFakeEmbedder(16)
	defer func() { _ = e.Close() }()

	a, err := e.Embed(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	b, err := e.Embed(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	if len(a) != 2 || len(b) != 2 {
		t.Fatalf("unexpected vector count")
	}
	for i := range a {
		if len(a[i]) != 16 {
			t.Fatalf("dim mismatch at %d: %d", i, len(a[i]))
		}
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				t.Fatalf("non-deterministic at [%d][%d]: %v vs %v", i, j, a[i][j], b[i][j])
			}
		}
	}
	if e.Dimensions() != 16 {
		t.Fatalf("want dims 16 got %d", e.Dimensions())
	}
}

func TestFakeEmbedderUnitLength(t *testing.T) {
	t.Parallel()
	e := llm.NewFakeEmbedder(32)
	vecs, err := e.Embed(context.Background(), []string{"alpha", "beta"})
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	for i, v := range vecs {
		var sum float64
		for _, x := range v {
			sum += float64(x) * float64(x)
		}
		if math.Abs(math.Sqrt(sum)-1) > 1e-5 {
			t.Fatalf("vec %d not unit length: %v", i, math.Sqrt(sum))
		}
	}
}

func TestFakeEmbedderDifferentInputs(t *testing.T) {
	t.Parallel()
	e := llm.NewFakeEmbedder(8)
	vecs, err := e.Embed(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatalf("embed: %v", err)
	}
	equal := true
	for i := range vecs[0] {
		if vecs[0][i] != vecs[1][i] {
			equal = false
			break
		}
	}
	if equal {
		t.Fatalf("distinct inputs produced identical vectors")
	}
}
