// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestAudioExtractor_Name(t *testing.T) {
	t.Parallel()
	ext := NewAudioExtractor(AudioExtractorConfig{})
	if ext.Name() != "audio-whisper" {
		t.Fatalf("Name() = %q, want %q", ext.Name(), "audio-whisper")
	}
}

func TestAudioExtractor_ContentTypes(t *testing.T) {
	t.Parallel()
	ext := NewAudioExtractor(AudioExtractorConfig{})
	types := ext.ContentTypes()

	expected := []string{
		"audio/mpeg",
		"audio/wav",
		"audio/x-wav",
		"audio/flac",
		"audio/mp4",
		"audio/ogg",
		"audio/aac",
		"audio/x-ms-wma",
		"audio/opus",
	}

	if len(types) != len(expected) {
		t.Fatalf("ContentTypes() length = %d, want %d", len(types), len(expected))
	}
	for i, ct := range types {
		if ct != expected[i] {
			t.Fatalf("ContentTypes()[%d] = %q, want %q", i, ct, expected[i])
		}
	}
}

func TestAudioExtractor_Capability(t *testing.T) {
	t.Parallel()
	ext := NewAudioExtractor(AudioExtractorConfig{})
	cap := ext.Capability()

	if !cap.RequiresBinary {
		t.Fatal("expected RequiresBinary = true")
	}

	extensionSet := make(map[string]struct{}, len(cap.Extensions))
	for _, e := range cap.Extensions {
		extensionSet[e] = struct{}{}
	}
	for _, want := range []string{".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"} {
		if _, ok := extensionSet[want]; !ok {
			t.Fatalf("expected extension %q in capability", want)
		}
	}

	mimeSet := make(map[string]struct{}, len(cap.MIMETypes))
	for _, m := range cap.MIMETypes {
		mimeSet[m] = struct{}{}
	}
	for _, want := range []string{"audio/mpeg", "audio/wav", "audio/mp4", "audio/ogg"} {
		if _, ok := mimeSet[want]; !ok {
			t.Fatalf("expected MIME type %q in capability", want)
		}
	}

	if len(cap.MagicBytes) == 0 {
		t.Fatal("expected non-empty MagicBytes")
	}
}

func TestAudioExtractor_Available_NoPython(t *testing.T) {
	t.Parallel()
	ext := NewAudioExtractor(AudioExtractorConfig{
		PythonBinary: "/nonexistent/python3",
	})

	avail, err := ext.Available(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if avail {
		t.Fatal("expected available=false when python binary does not exist")
	}
}

func TestAudioExtractor_Available_CachesResult(t *testing.T) {
	t.Parallel()
	ext := NewAudioExtractor(AudioExtractorConfig{
		PythonBinary: "/nonexistent/python3",
	})

	// First call.
	avail1, _ := ext.Available(context.Background())
	// Second call should use cache.
	avail2, _ := ext.Available(context.Background())

	if avail1 != avail2 {
		t.Fatal("expected cached result to match")
	}
	if avail1 {
		t.Fatal("expected false for nonexistent binary")
	}
}

func TestAudioExtractor_Extract_EmptyInput(t *testing.T) {
	t.Parallel()
	ext := NewAudioExtractor(AudioExtractorConfig{})
	result, err := ext.Extract(context.Background(), []byte{}, ExtractOptions{
		ContentType: "audio/wav",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !result.Skipped {
		t.Fatal("expected skipped=true for empty input")
	}
	if result.Reason != "empty audio input" {
		t.Fatalf("unexpected reason: %q", result.Reason)
	}
}

func TestAudioExtractor_Extract_FileTooLarge(t *testing.T) {
	t.Parallel()
	ext := NewAudioExtractor(AudioExtractorConfig{
		MaxFileSize: 100, // 100 bytes max
	})

	largeInput := make([]byte, 200)
	_, err := ext.Extract(context.Background(), largeInput, ExtractOptions{
		ContentType: "audio/wav",
	})
	if err == nil {
		t.Fatal("expected error for oversized file")
	}
	if !strings.Contains(err.Error(), "exceeds maximum") {
		t.Fatalf("unexpected error message: %v", err)
	}
}

func TestAudioExtractor_ComputeTimeout(t *testing.T) {
	t.Parallel()
	ext := NewAudioExtractor(AudioExtractorConfig{
		TimeoutPerMin: 30 * time.Second,
		MinTimeout:    120 * time.Second,
	})

	cases := []struct {
		name     string
		fileSize int
		wantMin  time.Duration
	}{
		{"small file uses minimum", 1024, 120 * time.Second},
		{"1MB file", 1024 * 1024, 120 * time.Second},          // 30s < 120s min
		{"10MB file", 10 * 1024 * 1024, 300 * time.Second},    // 10 * 30s = 300s
		{"100MB file", 100 * 1024 * 1024, 3000 * time.Second}, // 100 * 30s
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			timeout := ext.computeTimeout(tc.fileSize)
			if timeout < tc.wantMin {
				t.Fatalf("timeout %s < expected minimum %s", timeout, tc.wantMin)
			}
		})
	}
}

func TestAudioExtractor_DefaultConfig(t *testing.T) {
	t.Parallel()
	ext := NewAudioExtractor(AudioExtractorConfig{})

	if ext.pythonBinary != defaultPythonBinary {
		t.Fatalf("pythonBinary = %q, want %q", ext.pythonBinary, defaultPythonBinary)
	}
	if ext.modelSize != defaultWhisperModel {
		t.Fatalf("modelSize = %q, want %q", ext.modelSize, defaultWhisperModel)
	}
	if ext.maxFileSize != defaultMaxAudioSize {
		t.Fatalf("maxFileSize = %d, want %d", ext.maxFileSize, defaultMaxAudioSize)
	}
	if ext.minTimeout != defaultMinAudioTimeout {
		t.Fatalf("minTimeout = %s, want %s", ext.minTimeout, defaultMinAudioTimeout)
	}
}

func TestAudioExtractor_CustomConfig(t *testing.T) {
	t.Parallel()
	ext := NewAudioExtractor(AudioExtractorConfig{
		PythonBinary:    "/usr/local/bin/python3.11",
		ModelSize:       "large-v3",
		DefaultLanguage: "fr",
		MaxFileSize:     1024 * 1024,
		TimeoutPerMin:   60 * time.Second,
		MinTimeout:      60 * time.Second,
	})

	if ext.pythonBinary != "/usr/local/bin/python3.11" {
		t.Fatalf("pythonBinary = %q", ext.pythonBinary)
	}
	if ext.modelSize != "large-v3" {
		t.Fatalf("modelSize = %q", ext.modelSize)
	}
	if ext.defaultLang != "fr" {
		t.Fatalf("defaultLang = %q", ext.defaultLang)
	}
}

func TestFormatTranscription_EmptySegments(t *testing.T) {
	t.Parallel()
	result := formatTranscription(nil)
	if result != "" {
		t.Fatalf("expected empty string, got %q", result)
	}
}

func TestFormatTranscription_SingleSegment(t *testing.T) {
	t.Parallel()
	segments := []TranscriptionSegment{
		{Start: 0, End: 5, Text: "Hello world"},
	}
	result := formatTranscription(segments)
	if !strings.Contains(result, "[00:00") {
		t.Fatalf("expected timestamp marker, got %q", result)
	}
	if !strings.Contains(result, "Hello world") {
		t.Fatalf("expected text content, got %q", result)
	}
}

func TestFormatTranscription_ParagraphBreaks(t *testing.T) {
	t.Parallel()
	segments := []TranscriptionSegment{
		{Start: 0, End: 10, Text: "First segment."},
		{Start: 10, End: 20, Text: "Second segment."},
		{Start: 35, End: 45, Text: "Third segment after break."},
		{Start: 45, End: 55, Text: "Fourth segment."},
	}
	result := formatTranscription(segments)

	// Should have paragraph break between second and third segments.
	parts := strings.Split(result, "\n\n")
	if len(parts) < 2 {
		t.Fatalf("expected at least 2 paragraphs, got %d: %q", len(parts), result)
	}
	if !strings.Contains(parts[0], "First segment.") {
		t.Fatalf("first paragraph missing first segment text")
	}
	if !strings.Contains(result, "Third segment after break.") {
		t.Fatalf("missing third segment text")
	}
}

func TestFormatTranscription_TimestampFormat(t *testing.T) {
	t.Parallel()
	segments := []TranscriptionSegment{
		{Start: 65, End: 90, Text: "One minute five."},
	}
	result := formatTranscription(segments)
	if !strings.Contains(result, "[01:05") {
		t.Fatalf("expected [01:05 timestamp, got %q", result)
	}
}

func TestFormatTimestamp(t *testing.T) {
	t.Parallel()
	cases := []struct {
		seconds float64
		want    string
	}{
		{0, "[00:00"},
		{30, "[00:30"},
		{60, "[01:00"},
		{65, "[01:05"},
		{3600, "[60:00"},
		{3661, "[61:01"},
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("%.0fs", tc.seconds), func(t *testing.T) {
			t.Parallel()
			got := formatTimestamp(tc.seconds)
			if got != tc.want {
				t.Fatalf("formatTimestamp(%f) = %q, want %q", tc.seconds, got, tc.want)
			}
		})
	}
}

func TestExtensionFromContentType(t *testing.T) {
	t.Parallel()
	cases := []struct {
		contentType string
		fileName    string
		want        string
	}{
		{"audio/mpeg", "", ".mp3"},
		{"audio/wav", "", ".wav"},
		{"audio/x-wav", "", ".wav"},
		{"audio/flac", "", ".flac"},
		{"audio/mp4", "", ".m4a"},
		{"audio/ogg", "", ".ogg"},
		{"audio/aac", "", ".aac"},
		{"audio/opus", "", ".opus"},
		{"audio/x-ms-wma", "", ".wma"},
		{"unknown/type", "", ".wav"},
		{"audio/mpeg", "recording.mp3", ".mp3"},
		{"audio/mpeg", "recording.m4a", ".m4a"},
	}
	for _, tc := range cases {
		t.Run(tc.contentType+"_"+tc.fileName, func(t *testing.T) {
			t.Parallel()
			got := extensionFromContentType(tc.contentType, tc.fileName)
			if got != tc.want {
				t.Fatalf("extensionFromContentType(%q, %q) = %q, want %q",
					tc.contentType, tc.fileName, got, tc.want)
			}
		})
	}
}

func TestAudioExtractor_Extract_SilentAudio(t *testing.T) {
	t.Parallel()

	segments := []TranscriptionSegment{}
	text := formatTranscription(segments)
	if text != "" {
		t.Fatalf("expected empty text for silent audio, got %q", text)
	}
}

func TestAudioExtractor_InterfaceCompliance(t *testing.T) {
	t.Parallel()
	var _ Extractor = (*AudioExtractor)(nil)
}

func TestAudioExtractor_TempFileCleanup(t *testing.T) {
	t.Parallel()

	ext := NewAudioExtractor(AudioExtractorConfig{
		PythonBinary: "/nonexistent/python3",
		MinTimeout:   5 * time.Second,
	})

	input := []byte("fake audio data")
	_, _ = ext.Extract(context.Background(), input, ExtractOptions{
		ContentType: "audio/wav",
	})

	// Verify no leaked temp files by checking for our pattern.
	matches, err := filepath.Glob(filepath.Join(os.TempDir(), "memory-audio-*"))
	if err != nil {
		t.Fatalf("glob error: %v", err)
	}
	for _, m := range matches {
		info, err := os.Stat(m)
		if err != nil {
			continue
		}
		if time.Since(info.ModTime()) < 5*time.Second {
			t.Errorf("leaked temp file detected: %s", m)
		}
	}
}

func TestAudioExtractor_ExtractStream_FileTooLarge(t *testing.T) {
	t.Parallel()
	ext := NewAudioExtractor(AudioExtractorConfig{
		MaxFileSize: 50,
	})

	largeData := strings.NewReader(strings.Repeat("x", 100))
	_, err := ext.ExtractStream(context.Background(), largeData, ExtractOptions{
		ContentType: "audio/wav",
	})
	if err == nil {
		t.Fatal("expected error for oversized stream")
	}
	if !strings.Contains(err.Error(), "exceeds maximum") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestWhisperOutputParsing(t *testing.T) {
	t.Parallel()

	jsonStr := `{
		"language": "fr",
		"language_probability": 0.95,
		"duration": 65.3,
		"segments": [
			{"start": 0.0, "end": 30.0, "text": "Bonjour le monde"},
			{"start": 30.0, "end": 65.3, "text": "Comment allez-vous"}
		]
	}`

	var output whisperOutput
	if err := json.Unmarshal([]byte(jsonStr), &output); err != nil {
		t.Fatalf("parsing JSON: %v", err)
	}

	if output.Language != "fr" {
		t.Fatalf("language = %q, want %q", output.Language, "fr")
	}
	if output.Duration != 65.3 {
		t.Fatalf("duration = %f, want 65.3", output.Duration)
	}
	if len(output.Segments) != 2 {
		t.Fatalf("segments count = %d, want 2", len(output.Segments))
	}

	text := formatTranscription(output.Segments)
	if !strings.Contains(text, "Bonjour le monde") {
		t.Fatalf("missing first segment text in %q", text)
	}
	if !strings.Contains(text, "Comment allez-vous") {
		t.Fatalf("missing second segment text in %q", text)
	}
}

func TestWhisperOutputParsing_WithWordTimestamps(t *testing.T) {
	t.Parallel()

	jsonStr := `{
		"language": "en",
		"language_probability": 0.99,
		"duration": 5.0,
		"segments": [
			{
				"start": 0.0,
				"end": 5.0,
				"text": "Hello world",
				"words": [
					{"start": 0.0, "end": 0.5, "word": "Hello", "probability": 0.99},
					{"start": 0.6, "end": 1.0, "word": "world", "probability": 0.97}
				]
			}
		]
	}`

	var output whisperOutput
	if err := json.Unmarshal([]byte(jsonStr), &output); err != nil {
		t.Fatalf("parsing JSON: %v", err)
	}

	if len(output.Segments[0].Words) != 2 {
		t.Fatalf("words count = %d, want 2", len(output.Segments[0].Words))
	}
	if output.Segments[0].Words[0].Word != "Hello" {
		t.Fatalf("first word = %q, want %q", output.Segments[0].Words[0].Word, "Hello")
	}
	if output.Segments[0].Words[1].Probability != 0.97 {
		t.Fatalf("second word probability = %f, want 0.97", output.Segments[0].Words[1].Probability)
	}
}

func TestAudioExtractor_RegisterWithRegistry(t *testing.T) {
	t.Parallel()
	registry := NewExtractorRegistry()
	ext := NewAudioExtractor(AudioExtractorConfig{})
	registry.Register(ext)

	result, err := registry.Extract(context.Background(), []byte{}, ExtractOptions{
		ContentType: "audio/mpeg",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Reason != "empty audio input" {
		t.Fatalf("expected 'empty audio input' reason, got %q", result.Reason)
	}
}
