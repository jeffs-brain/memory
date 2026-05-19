// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestVideoExtractor_Name(t *testing.T) {
	e := NewVideoExtractor(VideoExtractorConfig{})
	if got := e.Name(); got != "video-ffmpeg" {
		t.Errorf("Name() = %q, want %q", got, "video-ffmpeg")
	}
}

func TestVideoExtractor_ContentTypes(t *testing.T) {
	e := NewVideoExtractor(VideoExtractorConfig{})
	types := e.ContentTypes()
	if len(types) == 0 {
		t.Fatal("ContentTypes() returned empty slice")
	}
	wantTypes := map[string]bool{
		"video/mp4":        true,
		"video/x-msvideo":  true,
		"video/x-matroska": true,
		"video/quicktime":  true,
		"video/webm":       true,
		"video/x-ms-wmv":   true,
		"video/x-flv":      true,
	}
	for _, ct := range types {
		if !wantTypes[ct] {
			t.Errorf("unexpected content type %q", ct)
		}
		delete(wantTypes, ct)
	}
	for ct := range wantTypes {
		t.Errorf("missing expected content type %q", ct)
	}
}

func TestVideoExtractor_Capability(t *testing.T) {
	e := NewVideoExtractor(VideoExtractorConfig{})
	cap := e.Capability()

	wantExts := map[string]bool{
		".mp4": true, ".avi": true, ".mkv": true, ".mov": true,
		".webm": true, ".wmv": true, ".flv": true, ".m4v": true,
	}
	for _, ext := range cap.Extensions {
		if !wantExts[ext] {
			t.Errorf("unexpected extension %q in capability", ext)
		}
		delete(wantExts, ext)
	}
	for ext := range wantExts {
		t.Errorf("missing expected extension %q in capability", ext)
	}

	if !cap.RequiresBinary {
		t.Error("RequiresBinary should be true")
	}

	if len(cap.MagicBytes) == 0 {
		t.Error("expected at least one magic byte signature")
	}
}

func TestVideoExtractor_Available_NoFFmpeg(t *testing.T) {
	ResetBinaryCache()
	defer ResetBinaryCache()

	e := NewVideoExtractor(VideoExtractorConfig{
		FFmpegBinary: "/nonexistent/ffmpeg",
	})
	ok, err := e.Available(t.Context())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ok {
		t.Error("Available() should return false when ffmpeg is absent")
	}
}

func TestVideoExtractor_Available_NoFFprobe(t *testing.T) {
	ResetBinaryCache()
	defer ResetBinaryCache()

	e := NewVideoExtractor(VideoExtractorConfig{
		FFprobeBinary: "/nonexistent/ffprobe",
	})
	ok, err := e.Available(t.Context())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ok {
		t.Error("Available() should return false when ffprobe is absent")
	}
}

func TestVideoExtractor_Available_NoAudioExtractor(t *testing.T) {
	e := NewVideoExtractor(VideoExtractorConfig{
		AudioExtractor: nil,
	})
	ok, err := e.Available(t.Context())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ok {
		t.Error("Available() should return false when AudioExtractor is nil")
	}
}

func TestVideoExtractor_Extract_FileTooLarge(t *testing.T) {
	e := NewVideoExtractor(VideoExtractorConfig{
		MaxFileSize: 1024,
	})

	input := make([]byte, 2048)
	_, err := e.Extract(t.Context(), input, ExtractOptions{})
	if err == nil {
		t.Fatal("expected error for oversized file")
	}
	if !strings.Contains(err.Error(), "exceeds maximum") {
		t.Errorf("error should mention size limit: %v", err)
	}
}

func TestVideoExtractor_Extract_NoAudioExtractorDependency(t *testing.T) {
	e := NewVideoExtractor(VideoExtractorConfig{
		AudioExtractor: nil,
	})
	_, err := e.Extract(t.Context(), []byte("dummy"), ExtractOptions{})
	if err == nil {
		t.Fatal("expected error when AudioExtractor is nil")
	}
	if !strings.Contains(err.Error(), "AudioExtractor") {
		t.Errorf("error should mention AudioExtractor: %v", err)
	}
}

func TestVideoExtractor_DefaultConfig(t *testing.T) {
	cfg := VideoExtractorConfig{}
	cfg.applyDefaults()

	if cfg.FFmpegBinary != "ffmpeg" {
		t.Errorf("default FFmpegBinary = %q, want %q", cfg.FFmpegBinary, "ffmpeg")
	}
	if cfg.FFprobeBinary != "ffprobe" {
		t.Errorf("default FFprobeBinary = %q, want %q", cfg.FFprobeBinary, "ffprobe")
	}
	if cfg.MaxFileSize != defaultMaxVideoSize {
		t.Errorf("default MaxFileSize = %d, want %d", cfg.MaxFileSize, defaultMaxVideoSize)
	}
	if cfg.ExtractionTimeout != defaultVideoExtractionTimeout {
		t.Errorf("default ExtractionTimeout = %v, want %v", cfg.ExtractionTimeout, defaultVideoExtractionTimeout)
	}
	if cfg.MaxKeyframes != defaultMaxKeyframes {
		t.Errorf("default MaxKeyframes = %d, want %d", cfg.MaxKeyframes, defaultMaxKeyframes)
	}
}

func TestVideoExtractor_EnvVarOverride(t *testing.T) {
	t.Setenv("MEMORY_FFMPEG_PATH", "/custom/ffmpeg")
	t.Setenv("MEMORY_FFPROBE_PATH", "/custom/ffprobe")

	cfg := VideoExtractorConfig{}
	cfg.applyDefaults()

	if cfg.FFmpegBinary != "/custom/ffmpeg" {
		t.Errorf("FFmpegBinary = %q, want %q", cfg.FFmpegBinary, "/custom/ffmpeg")
	}
	if cfg.FFprobeBinary != "/custom/ffprobe" {
		t.Errorf("FFprobeBinary = %q, want %q", cfg.FFprobeBinary, "/custom/ffprobe")
	}
}

func TestBuildVideoMetadata(t *testing.T) {
	probe := ffprobeOutput{
		Format: ffprobeFormat{Duration: "125.4"},
		Streams: []ffprobeStream{
			{
				CodecType:  "video",
				CodecName:  "h264",
				Width:      1920,
				Height:     1080,
				RFrameRate: "30/1",
			},
			{
				CodecType: "audio",
				CodecName: "aac",
			},
		},
	}

	meta := buildVideoMetadata(probe)

	if meta.DurationSeconds != 125.4 {
		t.Errorf("DurationSeconds = %f, want 125.4", meta.DurationSeconds)
	}
	if meta.Width != 1920 {
		t.Errorf("Width = %d, want 1920", meta.Width)
	}
	if meta.Height != 1080 {
		t.Errorf("Height = %d, want 1080", meta.Height)
	}
	if meta.Codec != "h264" {
		t.Errorf("Codec = %q, want %q", meta.Codec, "h264")
	}
	if !meta.HasAudio {
		t.Error("HasAudio should be true")
	}
	if meta.FrameRate != 30.0 {
		t.Errorf("FrameRate = %f, want 30.0", meta.FrameRate)
	}
}

func TestBuildVideoMetadata_NoAudio(t *testing.T) {
	probe := ffprobeOutput{
		Format: ffprobeFormat{Duration: "60.0"},
		Streams: []ffprobeStream{
			{
				CodecType:  "video",
				CodecName:  "vp9",
				Width:      1280,
				Height:     720,
				RFrameRate: "24000/1001",
			},
		},
	}

	meta := buildVideoMetadata(probe)
	if meta.HasAudio {
		t.Error("HasAudio should be false when no audio stream present")
	}
	if meta.FrameRate < 23.9 || meta.FrameRate > 24.0 {
		t.Errorf("FrameRate = %f, want ~23.976", meta.FrameRate)
	}
}

func TestBuildVideoMetadata_EmptyDuration(t *testing.T) {
	probe := ffprobeOutput{
		Format:  ffprobeFormat{Duration: ""},
		Streams: []ffprobeStream{},
	}
	meta := buildVideoMetadata(probe)
	if meta.DurationSeconds != 0 {
		t.Errorf("DurationSeconds = %f, want 0", meta.DurationSeconds)
	}
}

func TestParseFrameRate(t *testing.T) {
	tests := []struct {
		input string
		want  float64
	}{
		{"30/1", 30.0},
		{"24000/1001", 23.976},
		{"0/0", 0},
		{"invalid", 0},
		{"abc/def", 0},
		{"", 0},
	}

	for _, tt := range tests {
		got := parseFrameRate(tt.input)
		if tt.want == 0 && got != 0 {
			t.Errorf("parseFrameRate(%q) = %f, want 0", tt.input, got)
		}
		if tt.want != 0 {
			diff := got - tt.want
			if diff < 0 {
				diff = -diff
			}
			if diff > 0.01 {
				t.Errorf("parseFrameRate(%q) = %f, want ~%f", tt.input, got, tt.want)
			}
		}
	}
}

func TestComputeExtractionTimeout(t *testing.T) {
	e := NewVideoExtractor(VideoExtractorConfig{})

	// Short video: should get at least 60s.
	timeout := e.computeExtractionTimeout(30)
	if timeout < 60*time.Second {
		t.Errorf("short video timeout = %v, want >= 60s", timeout)
	}

	// 20-minute video: ~2 blocks of 10 min = 240s.
	timeout = e.computeExtractionTimeout(1200)
	if timeout < 200*time.Second {
		t.Errorf("20-min video timeout = %v, want >= 200s", timeout)
	}
}

func TestMergeVideoAudioMetadata(t *testing.T) {
	video := VideoMetadata{
		DurationSeconds: 120,
		Width:           1920,
		Height:          1080,
		HasAudio:        true,
	}
	audio := map[string]string{
		"detected_language": "en",
		"segment_count":     "5",
	}

	merged := mergeVideoAudioMetadata(video, audio)

	if merged["detected_language"] != "en" {
		t.Error("audio metadata not preserved")
	}
	if merged["video_duration_seconds"] != "120.00" {
		t.Errorf("video_duration_seconds = %q, want %q", merged["video_duration_seconds"], "120.00")
	}
	if merged["video_width"] != "1920" {
		t.Errorf("video_width = %q, want %q", merged["video_width"], "1920")
	}
	if merged["video_height"] != "1080" {
		t.Errorf("video_height = %q, want %q", merged["video_height"], "1080")
	}
	if merged["video_has_audio"] != "true" {
		t.Errorf("video_has_audio = %q, want %q", merged["video_has_audio"], "true")
	}
}

func TestWriteVideoToTempFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.bin")
	data := []byte("hello world")

	n, err := writeVideoToTempFile(path, bytes.NewReader(data), 1024)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if n != int64(len(data)) {
		t.Errorf("written = %d, want %d", n, len(data))
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, data) {
		t.Error("file content mismatch")
	}
}

func TestWriteVideoToTempFile_ExceedsLimit(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.bin")
	data := make([]byte, 100)

	_, err := writeVideoToTempFile(path, bytes.NewReader(data), 50)
	if err == nil {
		t.Fatal("expected error for oversized input")
	}
	if !strings.Contains(err.Error(), "exceeds maximum") {
		t.Errorf("error should mention limit: %v", err)
	}
}

func TestWriteVideoToTempFile_Permissions(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "perms.bin")
	data := []byte("secret video data")

	_, err := writeVideoToTempFile(path, bytes.NewReader(data), 1024)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	perm := info.Mode().Perm()
	if perm != 0o600 {
		t.Errorf("file permissions = %o, want 0600", perm)
	}
}

func TestVideoMetadata_JSONRoundTrip(t *testing.T) {
	meta := VideoMetadata{
		DurationSeconds: 123.45,
		Width:           1920,
		Height:          1080,
		Codec:           "h264",
		HasAudio:        true,
		FrameRate:       29.97,
	}

	data, err := json.Marshal(meta)
	if err != nil {
		t.Fatal(err)
	}

	var decoded VideoMetadata
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}

	if decoded != meta {
		t.Errorf("round-trip mismatch: got %+v, want %+v", decoded, meta)
	}
}

func TestVideoExtractor_KeyframeExtractionDisabledByDefault(t *testing.T) {
	cfg := VideoExtractorConfig{}
	cfg.applyDefaults()
	if cfg.KeyframeExtraction {
		t.Error("KeyframeExtraction should be disabled by default")
	}
}

func TestVideoExtractor_OverrideTimeout(t *testing.T) {
	e := NewVideoExtractor(VideoExtractorConfig{})

	if got := e.overrideTimeout(); got != 0 {
		t.Errorf("overrideTimeout() without env = %v, want 0", got)
	}

	t.Setenv("MEMORY_EXTRACTOR_TIMEOUT_MS", "5000")
	if got := e.overrideTimeout(); got != 5*time.Second {
		t.Errorf("overrideTimeout() with env = %v, want 5s", got)
	}

	t.Setenv("MEMORY_EXTRACTOR_TIMEOUT_MS", "invalid")
	if got := e.overrideTimeout(); got != 0 {
		t.Errorf("overrideTimeout() with invalid env = %v, want 0", got)
	}
}

func TestEnvOrDefault(t *testing.T) {
	key := fmt.Sprintf("MEMORY_TEST_%d", time.Now().UnixNano())

	if got := envOrDefault(key, "fallback"); got != "fallback" {
		t.Errorf("envOrDefault unset = %q, want %q", got, "fallback")
	}

	t.Setenv(key, "custom")
	if got := envOrDefault(key, "fallback"); got != "custom" {
		t.Errorf("envOrDefault set = %q, want %q", got, "custom")
	}
}

// Compile-time verification that VideoExtractor implements Extractor.
var _ Extractor = (*VideoExtractor)(nil)
