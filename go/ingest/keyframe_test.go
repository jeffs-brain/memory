// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestExtractKeyframes_NilOCRExtractor(t *testing.T) {
	t.Parallel()
	cfg := KeyframeConfig{
		OCRExtractor: nil,
	}
	results, err := extractKeyframes(context.Background(), "/fake/video.mp4", t.TempDir(), cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if results != nil {
		t.Error("expected nil results when OCRExtractor is nil")
	}
}

func TestExtractKeyframes_UnavailableOCRExtractor(t *testing.T) {
	t.Parallel()
	ResetBinaryCache()
	defer ResetBinaryCache()

	ocrExt := &BaseExtractor{
		AvailableFn: func(ctx context.Context) (bool, error) {
			return false, nil
		},
	}

	cfg := KeyframeConfig{
		OCRExtractor: ocrExt,
	}
	results, err := extractKeyframes(context.Background(), "/fake/video.mp4", t.TempDir(), cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if results != nil {
		t.Error("expected nil results when OCR extractor is unavailable")
	}
}

func TestApplyKeyframeDefaults(t *testing.T) {
	t.Parallel()
	cfg := KeyframeConfig{}
	applyKeyframeDefaults(&cfg)

	if cfg.FFmpegBinary != "ffmpeg" {
		t.Errorf("expected default FFmpegBinary %q, got %q", "ffmpeg", cfg.FFmpegBinary)
	}
	if cfg.IntervalSecs != defaultKeyframeInterval {
		t.Errorf("expected default IntervalSecs %d, got %d", defaultKeyframeInterval, cfg.IntervalSecs)
	}
	if cfg.MaxKeyframes != defaultMaxKeyframes {
		t.Errorf("expected default MaxKeyframes %d, got %d", defaultMaxKeyframes, cfg.MaxKeyframes)
	}
	if cfg.Timeout != 60*time.Second {
		t.Errorf("expected default Timeout 60s, got %v", cfg.Timeout)
	}
}

func TestApplyKeyframeDefaults_EnvOverride(t *testing.T) {
	t.Setenv("MEMORY_KEYFRAME_INTERVAL_SECS", "60")
	t.Setenv("MEMORY_FFMPEG_PATH", "/custom/ffmpeg")

	cfg := KeyframeConfig{}
	applyKeyframeDefaults(&cfg)

	if cfg.IntervalSecs != 60 {
		t.Errorf("expected IntervalSecs 60 from env, got %d", cfg.IntervalSecs)
	}
	if cfg.FFmpegBinary != "/custom/ffmpeg" {
		t.Errorf("expected FFmpegBinary from env, got %q", cfg.FFmpegBinary)
	}
}

func TestApplyKeyframeDefaults_ConfigOverridesEnv(t *testing.T) {
	t.Setenv("MEMORY_KEYFRAME_INTERVAL_SECS", "60")

	cfg := KeyframeConfig{
		IntervalSecs: 15,
		MaxKeyframes: 10,
	}
	applyKeyframeDefaults(&cfg)

	if cfg.IntervalSecs != 15 {
		t.Errorf("expected configured IntervalSecs 15, got %d", cfg.IntervalSecs)
	}
	if cfg.MaxKeyframes != 10 {
		t.Errorf("expected configured MaxKeyframes 10, got %d", cfg.MaxKeyframes)
	}
}

func TestFormatKeyframeText_Empty(t *testing.T) {
	t.Parallel()
	result := formatKeyframeText(nil)
	if result != "" {
		t.Errorf("expected empty string for nil results, got %q", result)
	}
	result = formatKeyframeText([]KeyframeResult{})
	if result != "" {
		t.Errorf("expected empty string for empty results, got %q", result)
	}
}

func TestFormatKeyframeText_WithResults(t *testing.T) {
	t.Parallel()
	results := []KeyframeResult{
		{TimestampSecs: 30, Text: "First frame text", Confidence: 0.9},
		{TimestampSecs: 60, Text: "Second frame text", Confidence: 0.85},
		{TimestampSecs: 90, Text: "Third frame text", Confidence: 0.92},
	}

	text := formatKeyframeText(results)

	if !strings.Contains(text, "--- Keyframe Text ---") {
		t.Error("expected header in keyframe text")
	}
	if !strings.Contains(text, "[00:30]") {
		t.Error("expected timestamp [00:30] in keyframe text")
	}
	if !strings.Contains(text, "[01:00]") {
		t.Error("expected timestamp [01:00] in keyframe text")
	}
	if !strings.Contains(text, "[01:30]") {
		t.Error("expected timestamp [01:30] in keyframe text")
	}
	if !strings.Contains(text, "First frame text") {
		t.Error("expected first frame text content")
	}
	if !strings.Contains(text, "Second frame text") {
		t.Error("expected second frame text content")
	}
	if !strings.Contains(text, "Third frame text") {
		t.Error("expected third frame text content")
	}
}

func TestFormatKeyframeTimestamp(t *testing.T) {
	t.Parallel()
	cases := []struct {
		input    float64
		expected string
	}{
		{0, "00:00"},
		{30, "00:30"},
		{60, "01:00"},
		{90, "01:30"},
		{3600, "60:00"},
		{125, "02:05"},
	}
	for _, tc := range cases {
		t.Run(tc.expected, func(t *testing.T) {
			t.Parallel()
			result := formatKeyframeTimestamp(tc.input)
			if result != tc.expected {
				t.Errorf("formatKeyframeTimestamp(%f) = %q, want %q", tc.input, result, tc.expected)
			}
		})
	}
}

func TestMergeKeyframeMetadata_Empty(t *testing.T) {
	t.Parallel()
	metadata := map[string]string{"existing": "value"}
	mergeKeyframeMetadata(metadata, nil)

	if metadata["keyframe_count"] != "0" {
		t.Errorf("expected keyframe_count=0, got %q", metadata["keyframe_count"])
	}
	if _, ok := metadata["keyframe_avg_confidence"]; ok {
		t.Error("expected no keyframe_avg_confidence for empty results")
	}
	if metadata["existing"] != "value" {
		t.Error("existing metadata should be preserved")
	}
}

func TestMergeKeyframeMetadata_WithResults(t *testing.T) {
	t.Parallel()
	metadata := map[string]string{}
	results := []KeyframeResult{
		{Confidence: 0.9},
		{Confidence: 0.8},
	}
	mergeKeyframeMetadata(metadata, results)

	if metadata["keyframe_count"] != "2" {
		t.Errorf("expected keyframe_count=2, got %q", metadata["keyframe_count"])
	}
	if metadata["keyframe_avg_confidence"] != "0.8500" {
		t.Errorf("expected avg confidence 0.8500, got %q", metadata["keyframe_avg_confidence"])
	}
}

func TestFormatConfidence_Keyframe(t *testing.T) {
	t.Parallel()
	cases := []struct {
		input    float64
		expected string
	}{
		{0.95, "0.9500"},
		{1.0, "1.0000"},
		{0.0, "0.0000"},
		{0.12345, "0.1235"},
	}
	for _, tc := range cases {
		t.Run(tc.expected, func(t *testing.T) {
			t.Parallel()
			if got := formatConfidence(tc.input); got != tc.expected {
				t.Errorf("formatConfidence(%f) = %q, want %q", tc.input, got, tc.expected)
			}
		})
	}
}

// --- Mock-based extraction test ---

const mockFFmpegKeyframeScript = `#!/bin/sh
# Creates fake PNG frame files to simulate keyframe extraction.
# Parse output pattern from args - it's the last argument.
output_pattern=""
for arg; do
  output_pattern="$arg"
done
# The pattern is like /tmp/xxx/keyframes/frame-%04d.png
# We just need the directory and prefix.
dir=$(dirname "$output_pattern")
mkdir -p "$dir"
printf '\x89PNG' > "${dir}/frame-0001.png"
printf '\x89PNG' > "${dir}/frame-0002.png"
`

func TestExtractKeyframes_MockedFFmpeg(t *testing.T) {
	t.Parallel()
	ResetBinaryCache()
	defer ResetBinaryCache()

	tmpDir := t.TempDir()
	ffmpegPath := writeTestScript(t, tmpDir, "ffmpeg", mockFFmpegKeyframeScript)

	// Create a mock OCR extractor that returns text for any image.
	ocrExt := &BaseExtractor{
		AvailableFn: func(ctx context.Context) (bool, error) {
			return true, nil
		},
		ExtractFn: func(ctx context.Context, raw []byte, opts ExtractOptions) (ExtractResult, error) {
			return ExtractResult{
				Text:       "OCR text from keyframe",
				Confidence: 0.88,
			}, nil
		},
		ContentTypesFn: func() []string {
			return []string{"image/png"}
		},
		NameFn: func() string {
			return "mock-ocr"
		},
	}

	// Create a dummy video file.
	videoDir := t.TempDir()
	videoPath := filepath.Join(videoDir, "test.mp4")
	if err := os.WriteFile(videoPath, []byte("fake video data"), 0o600); err != nil {
		t.Fatal(err)
	}

	workDir := t.TempDir()
	cfg := KeyframeConfig{
		FFmpegBinary: ffmpegPath,
		IntervalSecs: 30,
		MaxKeyframes: 5,
		OCRExtractor: ocrExt,
		Timeout:      10 * time.Second,
	}

	results, err := extractKeyframes(context.Background(), videoPath, workDir, cfg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 keyframe results, got %d", len(results))
	}
	if results[0].Text != "OCR text from keyframe" {
		t.Errorf("expected OCR text, got %q", results[0].Text)
	}
	if results[0].TimestampSecs != 30 {
		t.Errorf("expected first timestamp 30s, got %f", results[0].TimestampSecs)
	}
	if results[1].TimestampSecs != 60 {
		t.Errorf("expected second timestamp 60s, got %f", results[1].TimestampSecs)
	}
}

// writeTestScript writes a shell script to a temporary directory and
// makes it executable.
func writeTestScript(t *testing.T, dir, name, content string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0o755); err != nil {
		t.Fatalf("writing test script %s: %v", name, err)
	}
	return path
}

func TestVideoExtractor_KeyframeExtractionIntegration(t *testing.T) {
	t.Parallel()
	cfg := VideoExtractorConfig{
		KeyframeExtraction: true,
		MaxKeyframes:       5,
	}
	cfg.applyDefaults()

	if !cfg.KeyframeExtraction {
		t.Error("KeyframeExtraction should be true when explicitly enabled")
	}
	if cfg.MaxKeyframes != 5 {
		t.Errorf("MaxKeyframes = %d, want 5", cfg.MaxKeyframes)
	}
}

func TestVideoExtractor_KeyframeExtractionDisabledByDefault_Explicit(t *testing.T) {
	t.Parallel()
	cfg := VideoExtractorConfig{}
	cfg.applyDefaults()
	if cfg.KeyframeExtraction {
		t.Error("KeyframeExtraction should be disabled by default")
	}
}
