// SPDX-License-Identifier: Apache-2.0

package memory

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"github.com/jeffs-brain/memory/go/llm"
)

// capturingProvider records the request it received and replies with a
// fixed payload.
type capturingProvider struct {
	mu    sync.Mutex
	reply string
	calls int
	req   llm.CompleteRequest
}

func (p *capturingProvider) Complete(ctx context.Context, req llm.CompleteRequest) (llm.CompleteResponse, error) {
	if err := ctx.Err(); err != nil {
		return llm.CompleteResponse{}, err
	}
	p.mu.Lock()
	p.calls++
	p.req = req
	p.mu.Unlock()
	return llm.CompleteResponse{Text: p.reply}, nil
}

func (p *capturingProvider) CompleteStream(_ context.Context, _ llm.CompleteRequest) (<-chan llm.StreamChunk, error) {
	return nil, nil
}

func (p *capturingProvider) Close() error { return nil }

func (p *capturingProvider) systemPrompt() string {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, m := range p.req.Messages {
		if m.Role == RoleSystem {
			return m.Content
		}
	}
	return ""
}

func (p *capturingProvider) callCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.calls
}

const priorsExtractReply = `{"memories":[{"action":"create","filename":"project-note.md","name":"Note","description":"a note","type":"project","scope":"project","content":"A durable project fact.","index_entry":"- project-note.md: note"}]}`

func priorsMessages() []Message {
	return []Message{
		{Role: RoleUser, Content: "msg 0"},
		{Role: RoleAssistant, Content: "msg 1"},
		{Role: RoleUser, Content: "msg 2"},
		{Role: RoleAssistant, Content: "msg 3"},
	}
}

func TestExtractFromMessagesWithPriors_InjectsBlock(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &capturingProvider{reply: priorsExtractReply}
	priors := &CodecPriors{
		Entities:    []string{"RoyalAWare", "Sprint"},
		Relations:   []string{"dependsOn"},
		DomainTerms: []string{"deployment"},
	}

	_, err := ExtractFromMessagesWithPriors(context.Background(), provider, "test-model", mem, "/project", priorsMessages(), "", "", priors)
	if err != nil {
		t.Fatalf("ExtractFromMessagesWithPriors: %v", err)
	}

	system := provider.systemPrompt()
	if !strings.HasPrefix(system, extractionPrompt) {
		t.Fatalf("system prompt does not start with base prompt")
	}
	for _, want := range []string{"## Project codec priors", "- RoyalAWare", "- dependsOn", "- deployment"} {
		if !strings.Contains(system, want) {
			t.Fatalf("system prompt missing %q", want)
		}
	}
}

func TestExtractFromMessagesWithPriors_NilIsBaseline(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &capturingProvider{reply: priorsExtractReply}

	_, err := ExtractFromMessagesWithPriors(context.Background(), provider, "test-model", mem, "/project", priorsMessages(), "", "", nil)
	if err != nil {
		t.Fatalf("ExtractFromMessagesWithPriors: %v", err)
	}
	if got := provider.systemPrompt(); got != extractionPrompt {
		t.Fatalf("expected byte-identical base prompt with nil priors")
	}
}

func TestExtractFromMessagesWithPriors_EmptyIsBaseline(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &capturingProvider{reply: priorsExtractReply}

	_, err := ExtractFromMessagesWithPriors(context.Background(), provider, "test-model", mem, "/project", priorsMessages(), "", "", &CodecPriors{Entities: []string{}, Relations: []string{}})
	if err != nil {
		t.Fatalf("ExtractFromMessagesWithPriors: %v", err)
	}
	if got := provider.systemPrompt(); got != extractionPrompt {
		t.Fatalf("expected byte-identical base prompt with empty priors")
	}
}

func TestExtractFromMessagesWithPriors_MalformedReturnsTypedErrorNoCall(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &capturingProvider{reply: priorsExtractReply}
	malformed := &CodecPriors{Entities: []string{"ok", "bad\ninjection"}}

	_, err := ExtractFromMessagesWithPriors(context.Background(), provider, "test-model", mem, "/project", priorsMessages(), "", "", malformed)
	if !errors.Is(err, ErrInvalidCodecPriors) {
		t.Fatalf("err = %v, want ErrInvalidCodecPriors", err)
	}
	if provider.callCount() != 0 {
		t.Fatalf("provider was called %d times for malformed priors", provider.callCount())
	}
}

func TestExtractFromMessagesWithPriors_HonoursCancelledContext(t *testing.T) {
	mem, _ := newTestMemory(t)
	provider := &capturingProvider{reply: priorsExtractReply}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := ExtractFromMessagesWithPriors(ctx, provider, "test-model", mem, "/project", priorsMessages(), "", "", &CodecPriors{Entities: []string{"Person"}})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if provider.callCount() != 0 {
		t.Fatalf("provider was called despite cancelled context")
	}
}

func TestExtractFromMessagesWithPriors_ConcurrentCallsIndependent(t *testing.T) {
	memA, _ := newTestMemory(t)
	memB, _ := newTestMemory(t)
	provA := &capturingProvider{reply: priorsExtractReply}
	provB := &capturingProvider{reply: priorsExtractReply}

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		_, _ = ExtractFromMessagesWithPriors(context.Background(), provA, "test-model", memA, "/a", priorsMessages(), "", "", &CodecPriors{Entities: []string{"Alpha"}})
	}()
	go func() {
		defer wg.Done()
		_, _ = ExtractFromMessagesWithPriors(context.Background(), provB, "test-model", memB, "/b", priorsMessages(), "", "", &CodecPriors{Entities: []string{"Beta"}})
	}()
	wg.Wait()

	sysA := provA.systemPrompt()
	sysB := provB.systemPrompt()
	if !strings.Contains(sysA, "- Alpha") || strings.Contains(sysA, "- Beta") {
		t.Fatalf("provider A bled state: %q", sysA)
	}
	if !strings.Contains(sysB, "- Beta") || strings.Contains(sysB, "- Alpha") {
		t.Fatalf("provider B bled state: %q", sysB)
	}
}
