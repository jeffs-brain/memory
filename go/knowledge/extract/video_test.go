// SPDX-License-Identifier: Apache-2.0

package extract

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// skipWithoutFFmpeg skips a test when ffmpeg is not available.
func skipWithoutFFmpeg(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not available")
	}
}

// skipWithoutFFprobe skips a test when ffprobe is not available.
func skipWithoutFFprobe(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("ffprobe"); err != nil {
		t.Skip("ffprobe not available")
	}
}

func TestVideoExtractor_Name(t *testing.T) {
	e := NewVideoExtractor(VideoExtractorConfig{})
	if got := e.Name(); got != "video-ffmpeg" {
		t.Errorf("Name() = %q, want %q", got, "video-ffmpeg")
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

	wantMimes := map[string]bool{
		"video/mp4": true, "video/x-msvideo": true, "video/x-matroska": true,
		"video/quicktime": true, "video/webm": true, "video/x-ms-wmv": true,
		"video/x-flv": true,
	}
	for _, mime := range cap.MIMETypes {
		if !wantMimes[mime] {
			t.Errorf("unexpected MIME type %q in capability", mime)
		}
		delete(wantMimes, mime)
	}
	for mime := range wantMimes {
		t.Errorf("missing expected MIME type %q in capability", mime)
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
		FFmpegBinary:   "/nonexistent/ffmpeg",
		AudioExtractor: NewAudioExtractor(AudioExtractorConfig{}),
	})
	ok, err := e.Available(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ok {
		t.Error("Available() should return false when ffmpeg is absent")
	}
}

func TestVideoExtractor_Available_NoFFprobe(t *testing.T) {
	skipWithoutFFmpeg(t)
	ResetBinaryCache()
	defer ResetBinaryCache()

	e := NewVideoExtractor(VideoExtractorConfig{
		FFprobeBinary:  "/nonexistent/ffprobe",
		AudioExtractor: NewAudioExtractor(AudioExtractorConfig{}),
	})
	ok, err := e.Available(context.Background())
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
	ok, err := e.Available(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ok {
		t.Error("Available() should return false when AudioExtractor is nil")
	}
}

func TestVideoExtractor_Extract_FileTooLarge(t *testing.T) {
	e := NewVideoExtractor(VideoExtractorConfig{
		MaxFileSize:    1024,
		AudioExtractor: NewAudioExtractor(AudioExtractorConfig{}),
	})

	input := make([]byte, 2048)
	_, err := e.Extract(context.Background(), input, ExtractOptions{})
	if err == nil {
		t.Fatal("expected error for oversized file")
	}
	if !strings.Contains(err.Error(), "exceeds maximum") {
		t.Errorf("error should mention size limit: %v", err)
	}
}

func TestVideoExtractor_ExtractStream_FileTooLarge(t *testing.T) {
	e := NewVideoExtractor(VideoExtractorConfig{
		MaxFileSize:    1024,
		AudioExtractor: NewAudioExtractor(AudioExtractorConfig{}),
	})

	reader := bytes.NewReader(make([]byte, 2048))
	_, err := e.ExtractStream(context.Background(), reader, 2048, ExtractOptions{})
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
	_, err := e.Extract(context.Background(), []byte("dummy"), ExtractOptions{})
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
	if cfg.ExtractionTimeout != defaultExtractionTimeout {
		t.Errorf("default ExtractionTimeout = %v, want %v", cfg.ExtractionTimeout, defaultExtractionTimeout)
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
	// 24000/1001 ~ 23.976
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

func TestMergeMetadata(t *testing.T) {
	video := VideoMetadata{
		DurationSeconds: 120,
		Width:           1920,
		Height:          1080,
		HasAudio:        true,
	}
	audio := map[string]any{
		"detected_language": "en",
		"segments":          []TranscriptionSegment{{Start: 0, End: 5, Text: "hello"}},
	}

	merged := mergeMetadata(video, audio)

	if merged["detected_language"] != "en" {
		t.Error("audio metadata not preserved")
	}
	if _, ok := merged["video_info"]; !ok {
		t.Error("video_info not set in merged metadata")
	}
	info, ok := merged["video_info"].(VideoMetadata)
	if !ok {
		t.Fatal("video_info has wrong type")
	}
	if info.Width != 1920 {
		t.Errorf("video_info.Width = %d, want 1920", info.Width)
	}
}

func TestWriteToTempFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.bin")
	data := []byte("hello world")

	n, err := writeToTempFile(path, bytes.NewReader(data), 1024)
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

func TestWriteToTempFile_ExceedsLimit(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.bin")
	data := make([]byte, 100)

	_, err := writeToTempFile(path, bytes.NewReader(data), 50)
	if err == nil {
		t.Fatal("expected error for oversized input")
	}
	if !strings.Contains(err.Error(), "exceeds maximum") {
		t.Errorf("error should mention limit: %v", err)
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

func TestVideoMetadata_JSONOmitsZeroFields(t *testing.T) {
	meta := VideoMetadata{
		DurationSeconds: 60,
		HasAudio:        false,
	}
	data, err := json.Marshal(meta)
	if err != nil {
		t.Fatal(err)
	}
	s := string(data)
	if strings.Contains(s, "width") {
		t.Error("zero-value width should be omitted from JSON")
	}
	if strings.Contains(s, "height") {
		t.Error("zero-value height should be omitted from JSON")
	}
	if strings.Contains(s, "codec") {
		t.Error("zero-value codec should be omitted from JSON")
	}
}

// TestProbeMetadata_Integration requires ffprobe and a real video file.
// It creates a minimal test video using ffmpeg and then probes it.
func TestProbeMetadata_Integration(t *testing.T) {
	skipWithoutFFmpeg(t)
	skipWithoutFFprobe(t)

	// Generate a minimal 1-second test video.
	dir := t.TempDir()
	videoPath := filepath.Join(dir, "test.mp4")

	genCmd := exec.Command("ffmpeg",
		"-y",
		"-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=25",
		"-f", "lavfi", "-i", "sine=frequency=440:duration=1",
		"-c:v", "libx264", "-preset", "ultrafast",
		"-c:a", "aac",
		"-shortest",
		videoPath,
	)
	genCmd.Stderr = os.Stderr
	if err := genCmd.Run(); err != nil {
		t.Skipf("could not generate test video: %v", err)
	}

	e := NewVideoExtractor(VideoExtractorConfig{})
	meta, err := e.probeMetadata(context.Background(), videoPath)
	if err != nil {
		t.Fatalf("probeMetadata failed: %v", err)
	}

	if meta.DurationSeconds < 0.5 || meta.DurationSeconds > 2.0 {
		t.Errorf("DurationSeconds = %f, want ~1.0", meta.DurationSeconds)
	}
	if meta.Width != 320 {
		t.Errorf("Width = %d, want 320", meta.Width)
	}
	if meta.Height != 240 {
		t.Errorf("Height = %d, want 240", meta.Height)
	}
	if !meta.HasAudio {
		t.Error("HasAudio should be true for test video with sine wave")
	}
	if meta.Codec == "" {
		t.Error("Codec should not be empty")
	}
}

// TestExtractAudio_Integration requires ffmpeg. Creates a test video
// and extracts the audio track.
func TestExtractAudio_Integration(t *testing.T) {
	skipWithoutFFmpeg(t)

	dir := t.TempDir()
	videoPath := filepath.Join(dir, "test.mp4")

	genCmd := exec.Command("ffmpeg",
		"-y",
		"-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=10",
		"-f", "lavfi", "-i", "sine=frequency=440:duration=1",
		"-c:v", "libx264", "-preset", "ultrafast",
		"-c:a", "aac",
		"-shortest",
		videoPath,
	)
	if err := genCmd.Run(); err != nil {
		t.Skipf("could not generate test video: %v", err)
	}

	e := NewVideoExtractor(VideoExtractorConfig{})
	wavData, err := e.extractAudio(context.Background(), videoPath, 1.0)
	if err != nil {
		t.Fatalf("extractAudio failed: %v", err)
	}

	// WAV files start with RIFF header.
	if len(wavData) < 44 {
		t.Fatalf("WAV output too small: %d bytes", len(wavData))
	}
	if string(wavData[:4]) != "RIFF" {
		t.Errorf("output does not start with RIFF header: %x", wavData[:4])
	}
}

// TestExtractAudio_NoAudioTrack verifies that extracting audio from a
// video-only file (no audio stream) produces an error from FFmpeg.
func TestExtractAudio_NoAudioTrack(t *testing.T) {
	skipWithoutFFmpeg(t)
	skipWithoutFFprobe(t)

	dir := t.TempDir()
	videoPath := filepath.Join(dir, "silent.mp4")

	genCmd := exec.Command("ffmpeg",
		"-y",
		"-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=10",
		"-c:v", "libx264", "-preset", "ultrafast",
		"-an",
		videoPath,
	)
	if err := genCmd.Run(); err != nil {
		t.Skipf("could not generate silent test video: %v", err)
	}

	// The video extractor should detect no audio via probe and return
	// the graceful message.
	videoData, err := os.ReadFile(videoPath)
	if err != nil {
		t.Fatal(err)
	}

	e := NewVideoExtractor(VideoExtractorConfig{
		AudioExtractor: NewAudioExtractor(AudioExtractorConfig{}),
	})
	result, err := e.Extract(context.Background(), videoData, ExtractOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(result.Content, "no audio track") {
		t.Errorf("expected 'no audio track' message, got: %q", result.Content)
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

	// Without env var.
	if got := e.overrideTimeout(); got != 0 {
		t.Errorf("overrideTimeout() without env = %v, want 0", got)
	}

	// With env var.
	t.Setenv("MEMORY_EXTRACTOR_TIMEOUT_MS", "5000")
	if got := e.overrideTimeout(); got != 5*time.Second {
		t.Errorf("overrideTimeout() with env = %v, want 5s", got)
	}

	// Invalid env var.
	t.Setenv("MEMORY_EXTRACTOR_TIMEOUT_MS", "invalid")
	if got := e.overrideTimeout(); got != 0 {
		t.Errorf("overrideTimeout() with invalid env = %v, want 0", got)
	}
}

func TestCheckBinaryAvailable(t *testing.T) {
	ResetBinaryCache()
	defer ResetBinaryCache()

	// "ls" should be available on all platforms.
	if !CheckBinaryAvailable("ls") {
		t.Error("ls should be available")
	}

	// Non-existent binary.
	if CheckBinaryAvailable("memory-nonexistent-binary-12345") {
		t.Error("nonexistent binary should not be available")
	}
}

func TestCheckBinaryAvailable_Cache(t *testing.T) {
	ResetBinaryCache()
	defer ResetBinaryCache()

	// First call populates cache.
	result1 := CheckBinaryAvailable("ls")
	// Second call reads from cache.
	result2 := CheckBinaryAvailable("ls")

	if result1 != result2 {
		t.Error("cached result should match first call")
	}
}

func TestFormatTranscription(t *testing.T) {
	segments := []TranscriptionSegment{
		{Start: 0, End: 10, Text: "Hello world."},
		{Start: 10, End: 20, Text: "This is a test."},
		{Start: 35, End: 45, Text: "Second paragraph."},
		{Start: 45, End: 55, Text: "More content."},
	}

	got := formatTranscription(segments)

	if !strings.Contains(got, "[00:00 - 00:35]") {
		t.Errorf("expected first paragraph marker, got:\n%s", got)
	}
	if !strings.Contains(got, "Hello world.") {
		t.Error("expected 'Hello world.' in output")
	}
	if !strings.Contains(got, "Second paragraph.") {
		t.Error("expected 'Second paragraph.' in output")
	}
}

func TestFormatTranscription_Empty(t *testing.T) {
	got := formatTranscription(nil)
	if got != "" {
		t.Errorf("expected empty string for nil segments, got: %q", got)
	}
}

func TestFormatTimestamp(t *testing.T) {
	tests := []struct {
		seconds float64
		want    string
	}{
		{0, "00:00"},
		{30, "00:30"},
		{60, "01:00"},
		{90, "01:30"},
		{3661, "61:01"},
	}
	for _, tt := range tests {
		got := formatTimestamp(tt.seconds)
		if got != tt.want {
			t.Errorf("formatTimestamp(%f) = %q, want %q", tt.seconds, got, tt.want)
		}
	}
}

func TestEnvOrDefault(t *testing.T) {
	key := fmt.Sprintf("MEMORY_TEST_%d", time.Now().UnixNano())

	// Unset: should return fallback.
	if got := envOrDefault(key, "fallback"); got != "fallback" {
		t.Errorf("envOrDefault unset = %q, want %q", got, "fallback")
	}

	// Set: should return env value.
	t.Setenv(key, "custom")
	if got := envOrDefault(key, "fallback"); got != "custom" {
		t.Errorf("envOrDefault set = %q, want %q", got, "custom")
	}
}

func TestSubprocessResult(t *testing.T) {
	ctx := context.Background()
	result, err := RunSubprocess(ctx, "echo", []string{"hello"}, nil, 5*time.Second)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.ExitCode != 0 {
		t.Errorf("ExitCode = %d, want 0", result.ExitCode)
	}
	if !strings.Contains(string(result.Stdout), "hello") {
		t.Errorf("Stdout = %q, want 'hello'", result.Stdout)
	}
}

func TestSubprocessResult_NonZeroExit(t *testing.T) {
	ctx := context.Background()
	result, err := RunSubprocess(ctx, "sh", []string{"-c", "exit 42"}, nil, 5*time.Second)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.ExitCode != 42 {
		t.Errorf("ExitCode = %d, want 42", result.ExitCode)
	}
}

func TestSubprocessResult_NonExistentBinary(t *testing.T) {
	ctx := context.Background()
	_, err := RunSubprocess(ctx, "/nonexistent/binary", nil, nil, 5*time.Second)
	if err == nil {
		t.Fatal("expected error for nonexistent binary")
	}
}
