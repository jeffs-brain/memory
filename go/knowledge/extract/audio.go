// SPDX-License-Identifier: Apache-2.0

package extract

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// AudioExtractorConfig configures the audio transcription extractor.
type AudioExtractorConfig struct {
	PythonBinary    string        // default: "python3"
	ModelSize       string        // default: "base"
	DefaultLanguage string        // default: "" (auto-detect)
	MaxFileSize     int64         // default: 500 * 1024 * 1024
	TimeoutPerMin   time.Duration // default: 30s per estimated minute
	MinTimeout      time.Duration // default: 120s
}

func (c *AudioExtractorConfig) applyDefaults() {
	if c.PythonBinary == "" {
		c.PythonBinary = envOrDefault("MEMORY_PYTHON_PATH", "python3")
	}
	if c.ModelSize == "" {
		c.ModelSize = "base"
	}
	if c.MaxFileSize == 0 {
		c.MaxFileSize = 500 * 1024 * 1024
	}
	if c.TimeoutPerMin == 0 {
		c.TimeoutPerMin = 30 * time.Second
	}
	if c.MinTimeout == 0 {
		c.MinTimeout = 120 * time.Second
	}
}

// TranscriptionSegment represents a single timestamped segment of
// transcribed audio.
type TranscriptionSegment struct {
	Start float64 `json:"start"`
	End   float64 `json:"end"`
	Text  string  `json:"text"`
}

// whisperOutput is the JSON structure returned by the embedded
// faster-whisper Python script.
type whisperOutput struct {
	Language string                 `json:"language"`
	Segments []TranscriptionSegment `json:"segments"`
}

// whisperScript is the Python script used to invoke faster-whisper.
// Embedded as a constant to avoid shipping a separate .py file.
const whisperScript = `
import sys, json
from faster_whisper import WhisperModel
model = WhisperModel(sys.argv[2], device="cpu", compute_type="int8")
segments, info = model.transcribe(sys.argv[1], language=sys.argv[3] if len(sys.argv) > 3 else None)
result = {"language": info.language, "segments": []}
for s in segments:
    result["segments"].append({"start": s.start, "end": s.end, "text": s.text})
json.dump(result, sys.stdout)
`

// AudioExtractor transcribes audio files using faster-whisper via a
// Python subprocess. Both Go and TS use the same subprocess approach.
type AudioExtractor struct {
	cfg AudioExtractorConfig
}

// NewAudioExtractor creates an AudioExtractor with the given config.
// Zero-value fields are replaced with sensible defaults.
func NewAudioExtractor(cfg AudioExtractorConfig) *AudioExtractor {
	cfg.applyDefaults()
	return &AudioExtractor{cfg: cfg}
}

// Name returns the unique name for this extractor.
func (e *AudioExtractor) Name() string { return "audio-whisper" }

// Capability describes the audio formats this extractor handles.
func (e *AudioExtractor) Capability() ExtractorCapability {
	return ExtractorCapability{
		Extensions: []string{".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma", ".opus"},
		MIMETypes: []string{
			"audio/mpeg", "audio/wav", "audio/x-wav", "audio/flac",
			"audio/mp4", "audio/ogg", "audio/aac", "audio/x-ms-wma", "audio/opus",
		},
		MagicBytes: []MagicSignature{
			{Offset: 0, Bytes: []byte{0x52, 0x49, 0x46, 0x46}}, // RIFF (WAV)
			{Offset: 0, Bytes: []byte{0x66, 0x4C, 0x61, 0x43}}, // fLaC
			{Offset: 0, Bytes: []byte{0x4F, 0x67, 0x67, 0x53}}, // OggS
			{Offset: 0, Bytes: []byte{0x49, 0x44, 0x33}},       // ID3 (MP3)
			{Offset: 0, Bytes: []byte{0xFF, 0xFB}},              // MP3 sync
		},
		RequiresBinary: true,
	}
}

// Available reports whether Python 3 and faster-whisper are installed.
func (e *AudioExtractor) Available(ctx context.Context) (bool, error) {
	if !CheckBinaryAvailable(e.cfg.PythonBinary) {
		return false, nil
	}
	result, err := RunSubprocess(ctx, e.cfg.PythonBinary, []string{"-c", "import faster_whisper"}, nil, 10*time.Second)
	if err != nil {
		return false, nil
	}
	return result.ExitCode == 0, nil
}

// Extract transcribes audio bytes into timestamped text. The input is
// written to a temporary file because faster-whisper requires a file
// path. The temp file is removed after transcription completes.
func (e *AudioExtractor) Extract(ctx context.Context, input []byte, opts ExtractOptions) (ExtractionResult, error) {
	if int64(len(input)) > e.cfg.MaxFileSize {
		return ExtractionResult{}, fmt.Errorf("extract: audio file size %d exceeds maximum %d bytes", len(input), e.cfg.MaxFileSize)
	}

	tmpDir, err := os.MkdirTemp("", "memory-audio-*")
	if err != nil {
		return ExtractionResult{}, fmt.Errorf("extract: creating temp dir: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	ext := ".wav"
	if opts.Filename != "" {
		if e := filepath.Ext(opts.Filename); e != "" {
			ext = e
		}
	}
	tmpFile := filepath.Join(tmpDir, "input"+ext)
	if err := os.WriteFile(tmpFile, input, 0o600); err != nil {
		return ExtractionResult{}, fmt.Errorf("extract: writing temp file: %w", err)
	}

	timeout := e.computeTimeout(input)
	args := []string{"-c", whisperScript, tmpFile, e.cfg.ModelSize}
	if e.cfg.DefaultLanguage != "" {
		args = append(args, e.cfg.DefaultLanguage)
	}
	if opts.Language != "" {
		args = append(args[:4], opts.Language)
	}

	result, err := RunSubprocess(ctx, e.cfg.PythonBinary, args, nil, timeout)
	if err != nil {
		return ExtractionResult{}, fmt.Errorf("extract: running whisper: %w", err)
	}
	if result.ExitCode != 0 {
		return ExtractionResult{}, fmt.Errorf("extract: whisper exited %d: %s", result.ExitCode, result.Stderr)
	}

	var output whisperOutput
	if err := json.Unmarshal(result.Stdout, &output); err != nil {
		return ExtractionResult{}, fmt.Errorf("extract: parsing whisper output: %w", err)
	}

	content := formatTranscription(output.Segments)
	metadata := map[string]any{
		"detected_language": output.Language,
		"segments":          output.Segments,
	}
	if len(output.Segments) > 0 {
		last := output.Segments[len(output.Segments)-1]
		metadata["duration_seconds"] = last.End
	}

	return ExtractionResult{
		Content:  content,
		MIME:     "text/plain",
		Metadata: metadata,
		Language: output.Language,
	}, nil
}

// ExtractFromBuffer is a convenience alias used by VideoExtractor when
// delegating extracted audio data for transcription.
func (e *AudioExtractor) ExtractFromBuffer(ctx context.Context, wavData []byte, opts ExtractOptions) (ExtractionResult, error) {
	opts.Filename = "extracted.wav"
	return e.Extract(ctx, wavData, opts)
}

// computeTimeout estimates an appropriate timeout based on file size.
// Assumes roughly 1 MB/min for compressed formats and 10 MB/min for
// WAV, with a minimum floor.
func (e *AudioExtractor) computeTimeout(input []byte) time.Duration {
	// Conservative estimate: assume compressed (~1 MB/min)
	estimatedMinutes := float64(len(input)) / (1024 * 1024)
	timeout := time.Duration(estimatedMinutes) * e.cfg.TimeoutPerMin
	if timeout < e.cfg.MinTimeout {
		timeout = e.cfg.MinTimeout
	}
	return timeout
}

// formatTranscription converts timestamped segments into readable
// paragraphs with time markers every 30 seconds.
func formatTranscription(segments []TranscriptionSegment) string {
	if len(segments) == 0 {
		return ""
	}

	var b strings.Builder
	var paragraphStart float64
	var paragraphTexts []string

	for _, seg := range segments {
		if seg.Start-paragraphStart >= 30 && len(paragraphTexts) > 0 {
			writeParagraph(&b, paragraphStart, seg.Start, paragraphTexts)
			paragraphStart = seg.Start
			paragraphTexts = nil
		}
		text := strings.TrimSpace(seg.Text)
		if text != "" {
			paragraphTexts = append(paragraphTexts, text)
		}
	}

	if len(paragraphTexts) > 0 {
		lastEnd := segments[len(segments)-1].End
		writeParagraph(&b, paragraphStart, lastEnd, paragraphTexts)
	}

	return strings.TrimSpace(b.String())
}

func writeParagraph(b *strings.Builder, start, end float64, texts []string) {
	if b.Len() > 0 {
		b.WriteString("\n\n")
	}
	b.WriteString(fmt.Sprintf("[%s - %s]\n", formatTimestamp(start), formatTimestamp(end)))
	b.WriteString(strings.Join(texts, " "))
}

func formatTimestamp(seconds float64) string {
	mins := int(seconds) / 60
	secs := int(seconds) % 60
	return fmt.Sprintf("%02d:%02d", mins, secs)
}

// envOrDefault reads an environment variable, returning the fallback
// when the variable is unset or empty.
func envOrDefault(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
