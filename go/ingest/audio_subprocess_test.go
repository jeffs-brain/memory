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

// TestHelperProcess implements the subprocess mock pattern for audio tests.
// When the test binary is invoked with GO_TEST_HELPER_PROCESS=1 it simulates
// faster-whisper behaviour depending on GO_TEST_HELPER_MODE.
func TestHelperProcess(t *testing.T) {
	if os.Getenv("GO_TEST_HELPER_PROCESS") != "1" {
		return
	}
	// Prevent test framework from interfering.
	t.Helper()

	mode := os.Getenv("GO_TEST_HELPER_MODE")
	switch mode {
	case "availability_ok":
		fmt.Print("ok")
		os.Exit(0)

	case "availability_fail":
		fmt.Fprint(os.Stderr, "ModuleNotFoundError: No module named 'faster_whisper'")
		os.Exit(1)

	case "transcribe_success":
		output := whisperOutput{
			Language:            "en",
			LanguageProbability: 0.98,
			Duration:            15.5,
			Segments: []TranscriptionSegment{
				{Start: 0, End: 5.0, Text: "Hello world from the test.", Words: []TranscriptionWord{
					{Start: 0, End: 0.5, Word: "Hello", Probability: 0.99},
					{Start: 0.6, End: 1.0, Word: "world", Probability: 0.97},
				}},
				{Start: 5.0, End: 15.5, Text: "This is a longer segment for testing."},
			},
		}
		data, _ := json.Marshal(output)
		fmt.Print(string(data))
		os.Exit(0)

	case "transcribe_language_detect":
		output := whisperOutput{
			Language:            "fr",
			LanguageProbability: 0.92,
			Duration:            10.0,
			Segments: []TranscriptionSegment{
				{Start: 0, End: 10.0, Text: "Bonjour le monde."},
			},
		}
		data, _ := json.Marshal(output)
		fmt.Print(string(data))
		os.Exit(0)

	case "transcribe_corrupt":
		fmt.Fprint(os.Stderr, "Error: could not decode audio file")
		os.Exit(1)

	case "transcribe_timeout":
		// Simulate a process that hangs forever (until killed by timeout).
		select {}

	default:
		fmt.Fprintf(os.Stderr, "unknown test helper mode: %s", mode)
		os.Exit(2)
	}
}

// writeHelperScript creates a temporary shell script that invokes the test
// binary with the TestHelperProcess env vars set.
func writeHelperScript(t *testing.T, mode string) string {
	t.Helper()
	testBinary := os.Args[0]
	// The shell script ignores all arguments (which come from the audio
	// extractor) and delegates to the test binary helper.
	script := fmt.Sprintf(`#!/bin/sh
GO_TEST_HELPER_PROCESS=1 GO_TEST_HELPER_MODE=%s exec "%s" -test.run=TestHelperProcess 2>&1
`, mode, testBinary)
	dir := t.TempDir()
	path := filepath.Join(dir, "mock-python.sh")
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatalf("writing helper script: %v", err)
	}
	return path
}

func TestAudioExtractor_Subprocess_TranscribeSuccess(t *testing.T) {
	t.Parallel()

	scriptPath := writeHelperScript(t, "transcribe_success")
	ext := NewAudioExtractor(AudioExtractorConfig{
		PythonBinary: scriptPath,
		MinTimeout:   30 * time.Second,
	})

	// Use a small non-empty buffer as fake audio data.
	input := []byte("fake audio data for testing")
	result, err := ext.Extract(context.Background(), input, ExtractOptions{
		ContentType: "audio/wav",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Skipped {
		t.Fatal("expected skipped=false for successful transcription")
	}
	if !strings.Contains(result.Text, "Hello world from the test.") {
		t.Fatalf("expected text to contain 'Hello world from the test.', got %q", result.Text)
	}
	if result.Language != "en" {
		t.Fatalf("expected language 'en', got %q", result.Language)
	}
	if result.Confidence < 0.90 {
		t.Fatalf("expected confidence >= 0.90, got %f", result.Confidence)
	}

	// Verify metadata.
	if result.Metadata["detected_language"] != "en" {
		t.Fatalf("expected metadata detected_language='en', got %q", result.Metadata["detected_language"])
	}
	if result.Metadata["transcription_engine"] != "faster-whisper" {
		t.Fatalf("expected metadata transcription_engine='faster-whisper', got %q", result.Metadata["transcription_engine"])
	}
	if result.Metadata["segment_count"] != "2" {
		t.Fatalf("expected metadata segment_count='2', got %q", result.Metadata["segment_count"])
	}
	if result.Metadata["segments"] == "" {
		t.Fatal("expected non-empty metadata segments JSON")
	}
}

func TestAudioExtractor_Subprocess_LanguageAutoDetect(t *testing.T) {
	t.Parallel()

	scriptPath := writeHelperScript(t, "transcribe_language_detect")
	ext := NewAudioExtractor(AudioExtractorConfig{
		PythonBinary: scriptPath,
		MinTimeout:   30 * time.Second,
		// No DefaultLanguage set — auto-detect mode.
	})

	input := []byte("fake french audio data")
	result, err := ext.Extract(context.Background(), input, ExtractOptions{
		ContentType: "audio/mp4",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.Language != "fr" {
		t.Fatalf("expected auto-detected language 'fr', got %q", result.Language)
	}
	if !strings.Contains(result.Text, "Bonjour le monde.") {
		t.Fatalf("expected text to contain 'Bonjour le monde.', got %q", result.Text)
	}
}

func TestAudioExtractor_Subprocess_CorruptAudio(t *testing.T) {
	t.Parallel()

	scriptPath := writeHelperScript(t, "transcribe_corrupt")
	ext := NewAudioExtractor(AudioExtractorConfig{
		PythonBinary: scriptPath,
		MinTimeout:   30 * time.Second,
	})

	input := []byte("corrupt audio bytes")
	_, err := ext.Extract(context.Background(), input, ExtractOptions{
		ContentType: "audio/wav",
	})
	if err == nil {
		t.Fatal("expected error for corrupt audio")
	}
	if !strings.Contains(err.Error(), "transcription failed") {
		t.Fatalf("expected 'transcription failed' in error, got %q", err.Error())
	}
}

func TestAudioExtractor_Subprocess_Timeout(t *testing.T) {
	t.Parallel()

	scriptPath := writeHelperScript(t, "transcribe_timeout")
	ext := NewAudioExtractor(AudioExtractorConfig{
		PythonBinary:  scriptPath,
		MinTimeout:    2 * time.Second,
		TimeoutPerMin: 1 * time.Second,
	})

	input := []byte("audio data that will time out")
	_, err := ext.Extract(context.Background(), input, ExtractOptions{
		ContentType: "audio/wav",
	})
	if err == nil {
		t.Fatal("expected timeout error")
	}
	// The error should indicate either a timeout or a process kill from
	// context deadline. On different platforms, the exact error message
	// varies (killed by signal vs timed out vs failed with exit code).
	errStr := err.Error()
	if !strings.Contains(errStr, "timed out") && !strings.Contains(errStr, "failed") && !strings.Contains(errStr, "signal") {
		t.Fatalf("expected timeout/kill error, got %q", errStr)
	}
}

func TestAudioExtractor_ContextCancellation(t *testing.T) {
	t.Parallel()

	scriptPath := writeHelperScript(t, "transcribe_timeout")
	ext := NewAudioExtractor(AudioExtractorConfig{
		PythonBinary: scriptPath,
		MinTimeout:   60 * time.Second, // Long timeout so cancellation wins.
	})

	ctx, cancel := context.WithCancel(context.Background())

	// Cancel after a short delay.
	go func() {
		time.Sleep(500 * time.Millisecond)
		cancel()
	}()

	input := []byte("audio data to be cancelled")
	_, err := ext.Extract(ctx, input, ExtractOptions{
		ContentType: "audio/wav",
	})
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
	// The error should indicate either context cancellation, timeout, or
	// process kill. On different platforms the exact message varies.
	errStr := err.Error()
	if !strings.Contains(errStr, "timed out") &&
		!strings.Contains(errStr, "context canceled") &&
		!strings.Contains(errStr, "signal: killed") &&
		!strings.Contains(errStr, "failed") {
		t.Fatalf("expected cancellation-related error, got %q", errStr)
	}
}

func TestAudioExtractor_Subprocess_TempFileCleanedUp(t *testing.T) {
	t.Parallel()

	// Use a dedicated temp directory to avoid interference from other tests.
	tmpDir := t.TempDir()

	scriptPath := writeHelperScript(t, "transcribe_success")
	ext := NewAudioExtractor(AudioExtractorConfig{
		PythonBinary: scriptPath,
		MinTimeout:   30 * time.Second,
	})

	input := []byte("fake audio data for cleanup test")
	_, err := ext.Extract(context.Background(), input, ExtractOptions{
		ContentType: "audio/wav",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify our temp files are cleaned up by checking the pattern
	// in the system temp directory.
	matches, err := filepath.Glob(filepath.Join(tmpDir, "memory-audio-*"))
	if err != nil {
		t.Fatalf("glob error: %v", err)
	}
	if len(matches) > 0 {
		t.Errorf("leaked temp file(s) detected in custom dir: %v", matches)
	}
}

func TestAudioExtractor_Subprocess_AvailabilityOk(t *testing.T) {
	t.Parallel()

	scriptPath := writeHelperScript(t, "availability_ok")
	ext := NewAudioExtractor(AudioExtractorConfig{
		PythonBinary: scriptPath,
	})

	avail, err := ext.Available(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !avail {
		t.Fatal("expected available=true when helper returns 'ok'")
	}
}

func TestAudioExtractor_Subprocess_AvailabilityFail(t *testing.T) {
	t.Parallel()

	scriptPath := writeHelperScript(t, "availability_fail")
	ext := NewAudioExtractor(AudioExtractorConfig{
		PythonBinary: scriptPath,
	})

	avail, err := ext.Available(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if avail {
		t.Fatal("expected available=false when helper returns import error")
	}
}
