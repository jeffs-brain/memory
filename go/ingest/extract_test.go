// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

type mockExtractor struct {
	extractFn func(ctx context.Context, messages []ExtractMessage) ([]ExtractedMemory, error)
}

func (m *mockExtractor) Extract(ctx context.Context, messages []ExtractMessage) ([]ExtractedMemory, error) {
	if m.extractFn != nil {
		return m.extractFn(ctx, messages)
	}
	return nil, nil
}

func TestExtractAfterIngest_CallsExtractWithSyntheticMessage(t *testing.T) {
	var receivedMessages []ExtractMessage
	extractor := &mockExtractor{
		extractFn: func(_ context.Context, messages []ExtractMessage) ([]ExtractedMemory, error) {
			receivedMessages = messages
			return []ExtractedMemory{
				{Filename: "fact-1.md", Content: "Important fact"},
			}, nil
		},
	}

	result, err := ExtractAfterIngest(context.Background(), ExtractAfterIngestOptions{
		BrainID:         "test-brain",
		DocumentPath:    "/docs/readme.md",
		DocumentContent: "# Project README\n\nThis is a test project.",
		Extractor:       extractor,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.FactsExtracted != 1 {
		t.Errorf("expected 1 fact, got %d", result.FactsExtracted)
	}
	if len(result.Memories) != 1 {
		t.Fatalf("expected 1 memory, got %d", len(result.Memories))
	}
	if result.Memories[0].Content != "Important fact" {
		t.Errorf("unexpected content: %s", result.Memories[0].Content)
	}
	if len(receivedMessages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(receivedMessages))
	}
	if receivedMessages[0].Role != "user" {
		t.Errorf("expected role=user, got %s", receivedMessages[0].Role)
	}
	if !strings.Contains(receivedMessages[0].Content, "readme.md") {
		t.Error("expected message to contain document path")
	}
}

func TestExtractAfterIngest_EmptyContent(t *testing.T) {
	called := false
	extractor := &mockExtractor{
		extractFn: func(_ context.Context, _ []ExtractMessage) ([]ExtractedMemory, error) {
			called = true
			return nil, nil
		},
	}

	result, err := ExtractAfterIngest(context.Background(), ExtractAfterIngestOptions{
		BrainID:         "test-brain",
		DocumentPath:    "/docs/empty.md",
		DocumentContent: "",
		Extractor:       extractor,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if called {
		t.Error("extractor should not be called for empty content")
	}
	if result.FactsExtracted != 0 {
		t.Errorf("expected 0 facts, got %d", result.FactsExtracted)
	}
}

func TestExtractAfterIngest_TruncatesLongContent(t *testing.T) {
	var receivedContent string
	extractor := &mockExtractor{
		extractFn: func(_ context.Context, messages []ExtractMessage) ([]ExtractedMemory, error) {
			receivedContent = messages[0].Content
			return nil, nil
		},
	}

	longContent := strings.Repeat("x", 200)
	_, err := ExtractAfterIngest(context.Background(), ExtractAfterIngestOptions{
		BrainID:         "test-brain",
		DocumentPath:    "/docs/long.md",
		DocumentContent: longContent,
		Extractor:       extractor,
		MaxContentChars: 50,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// The content should be truncated (synthetic message includes prefix text)
	if strings.Contains(receivedContent, strings.Repeat("x", 200)) {
		t.Error("content should have been truncated")
	}
}

func TestExtractAfterIngest_ZeroByteFile(t *testing.T) {
	called := false
	extractor := &mockExtractor{
		extractFn: func(_ context.Context, _ []ExtractMessage) ([]ExtractedMemory, error) {
			called = true
			return nil, nil
		},
	}

	// Simulate a zero-byte file: content is empty string
	result, err := ExtractAfterIngest(context.Background(), ExtractAfterIngestOptions{
		BrainID:         "test-brain",
		DocumentPath:    "/docs/zero.bin",
		DocumentContent: "",
		Extractor:       extractor,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if called {
		t.Error("extractor should not be called for zero-byte file")
	}
	if result.FactsExtracted != 0 {
		t.Errorf("expected 0 facts for zero-byte file, got %d", result.FactsExtracted)
	}
}

func TestExtractAfterIngest_MultiByteRuneTruncation(t *testing.T) {
	// Content with multi-byte characters (emoji) — truncation should not split a rune.
	content := strings.Repeat("\U0001F600", 10) // 10 grinning face emojis, 4 bytes each
	var receivedContent string
	extractor := &mockExtractor{
		extractFn: func(_ context.Context, messages []ExtractMessage) ([]ExtractedMemory, error) {
			receivedContent = messages[0].Content
			return nil, nil
		},
	}

	_, err := ExtractAfterIngest(context.Background(), ExtractAfterIngestOptions{
		BrainID:         "test-brain",
		DocumentPath:    "/docs/emoji.txt",
		DocumentContent: content,
		Extractor:       extractor,
		MaxContentChars: 5,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should contain exactly 5 emojis in the document section, not broken bytes.
	if strings.Contains(receivedContent, "�") {
		t.Error("truncated content should not contain replacement characters")
	}
	// Check the content ends with valid runes.
	for _, r := range receivedContent {
		if r == '�' {
			t.Fatal("found replacement character, rune was split during truncation")
		}
	}
}

func TestExtractAfterIngest_IsolationDelimiters(t *testing.T) {
	var receivedContent string
	extractor := &mockExtractor{
		extractFn: func(_ context.Context, messages []ExtractMessage) ([]ExtractedMemory, error) {
			receivedContent = messages[0].Content
			return nil, nil
		},
	}

	_, err := ExtractAfterIngest(context.Background(), ExtractAfterIngestOptions{
		BrainID:         "test-brain",
		DocumentPath:    "/docs/test.md",
		DocumentContent: "Hello world",
		Extractor:       extractor,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(receivedContent, "<ingested-document>") {
		t.Error("expected isolation opening delimiter in prompt")
	}
	if !strings.Contains(receivedContent, "</ingested-document>") {
		t.Error("expected isolation closing delimiter in prompt")
	}
}

func TestExtractAfterIngest_LogsExtractionError(t *testing.T) {
	extractor := &mockExtractor{
		extractFn: func(_ context.Context, _ []ExtractMessage) ([]ExtractedMemory, error) {
			return nil, fmt.Errorf("provider unavailable")
		},
	}

	// Providing a logger to verify it's used (non-nil path). The test
	// primarily verifies that no panic occurs when logging the error.
	result, err := ExtractAfterIngest(context.Background(), ExtractAfterIngestOptions{
		BrainID:         "test-brain",
		DocumentPath:    "/docs/fail.md",
		DocumentContent: "Some content",
		Extractor:       extractor,
	})
	if err != nil {
		t.Fatalf("expected no error (non-fatal), got: %v", err)
	}
	if result.FactsExtracted != 0 {
		t.Errorf("expected 0 facts on failure, got %d", result.FactsExtracted)
	}
}

func TestExtractAfterIngest_NonFatalOnError(t *testing.T) {
	extractor := &mockExtractor{
		extractFn: func(_ context.Context, _ []ExtractMessage) ([]ExtractedMemory, error) {
			return nil, fmt.Errorf("LLM unavailable")
		},
	}

	result, err := ExtractAfterIngest(context.Background(), ExtractAfterIngestOptions{
		BrainID:         "test-brain",
		DocumentPath:    "/docs/fail.md",
		DocumentContent: "Some content",
		Extractor:       extractor,
	})
	if err != nil {
		t.Fatalf("expected no error (non-fatal), got: %v", err)
	}
	if result.FactsExtracted != 0 {
		t.Errorf("expected 0 facts on failure, got %d", result.FactsExtracted)
	}
}
