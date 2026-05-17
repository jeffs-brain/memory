// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Default configuration values for the AudioExtractor.
const (
	defaultPythonBinary    = "python3"
	defaultWhisperModel    = "base"
	defaultMaxAudioSize    = 500 * 1024 * 1024 // 500 MB
	defaultTimeoutPerMin   = 30 * time.Second
	defaultMinAudioTimeout = 120 * time.Second
	paragraphBreakSeconds  = 30.0
)

// whisperScript is the embedded Python script invoked via subprocess to
// transcribe audio using faster-whisper. It outputs JSON to stdout.
const whisperScript = `
import sys, json
from faster_whisper import WhisperModel

audio_path = sys.argv[1]
model_size = sys.argv[2]
language = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "" else None

model = WhisperModel(model_size, device="cpu", compute_type="int8")
segments, info = model.transcribe(audio_path, language=language, word_timestamps=True)

result = {"language": info.language, "language_probability": info.language_probability, "duration": info.duration, "segments": []}
for s in segments:
    seg = {"start": s.start, "end": s.end, "text": s.text}
    if s.words:
        seg["words"] = [{"start": w.start, "end": w.end, "word": w.word, "probability": w.probability} for w in s.words]
    result["segments"].append(seg)

json.dump(result, sys.stdout)
`

// availabilityCheckScript verifies that python3 can import faster_whisper.
const availabilityCheckScript = `import faster_whisper; print("ok")`

// AudioExtractorConfig holds configuration for the AudioExtractor.
type AudioExtractorConfig struct {
	// PythonBinary is the path to the Python 3 interpreter.
	// Default: "python3". Override via MEMORY_WHISPER_PATH env var.
	PythonBinary string

	// ModelSize is the faster-whisper model to use.
	// Default: "base".
	ModelSize string

	// DefaultLanguage is the language hint for transcription.
	// Empty string means auto-detect.
	DefaultLanguage string

	// MaxFileSize is the maximum audio file size in bytes.
	// Default: 500 MB.
	MaxFileSize int64

	// TimeoutPerMin is the timeout budget per minute of estimated audio.
	// Default: 30 seconds.
	TimeoutPerMin time.Duration

	// MinTimeout is the minimum timeout regardless of file size.
	// Default: 120 seconds.
	MinTimeout time.Duration

	// Logger is the structured logger. Defaults to slog.Default().
	Logger *slog.Logger
}

// TranscriptionSegment represents a single timed segment of transcription.
type TranscriptionSegment struct {
	Start float64            `json:"start"`
	End   float64            `json:"end"`
	Text  string             `json:"text"`
	Words []TranscriptionWord `json:"words,omitempty"`
}

// TranscriptionWord represents a word-level timestamp.
type TranscriptionWord struct {
	Start       float64 `json:"start"`
	End         float64 `json:"end"`
	Word        string  `json:"word"`
	Probability float64 `json:"probability"`
}

// whisperOutput is the JSON structure returned by the Python subprocess.
type whisperOutput struct {
	Language            string                 `json:"language"`
	LanguageProbability float64                `json:"language_probability"`
	Duration            float64                `json:"duration"`
	Segments            []TranscriptionSegment `json:"segments"`
}

// AudioExtractor transcribes audio files using faster-whisper via a
// Python subprocess. It implements the Extractor interface.
type AudioExtractor struct {
	pythonBinary  string
	modelSize     string
	defaultLang   string
	maxFileSize   int64
	timeoutPerMin time.Duration
	minTimeout    time.Duration
	logger        *slog.Logger

	availMu     sync.Mutex
	availCached *bool
	availAt     time.Time
}

// Compile-time interface check.
var _ Extractor = (*AudioExtractor)(nil)

// audioContentTypes lists the MIME types handled by AudioExtractor.
var audioContentTypes = []string{
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

// audioExtensions lists file extensions handled by AudioExtractor.
var audioExtensions = []string{
	".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma", ".opus",
}

// audioMagicSignatures contains magic byte patterns for audio format detection.
var audioMagicSignatures = []MagicSignature{
	{Offset: 0, Bytes: []byte{0x52, 0x49, 0x46, 0x46}},  // WAV (RIFF)
	{Offset: 8, Bytes: []byte{0x57, 0x41, 0x56, 0x45}},   // WAV (WAVE marker)
	{Offset: 0, Bytes: []byte{0x66, 0x4C, 0x61, 0x43}},   // FLAC
	{Offset: 0, Bytes: []byte{0x4F, 0x67, 0x67, 0x53}},   // OGG
	{Offset: 0, Bytes: []byte{0x49, 0x44, 0x33}},          // MP3 ID3
	{Offset: 0, Bytes: []byte{0xFF, 0xFB}},                // MP3 sync
	{Offset: 0, Bytes: []byte{0xFF, 0xF3}},                // MP3 sync
	{Offset: 0, Bytes: []byte{0xFF, 0xF2}},                // MP3 sync
}

// NewAudioExtractor creates a new AudioExtractor with the given config.
// Zero-value fields use sensible defaults.
func NewAudioExtractor(cfg AudioExtractorConfig) *AudioExtractor {
	pythonBin := cfg.PythonBinary
	if pythonBin == "" {
		pythonBin = os.Getenv("MEMORY_WHISPER_PATH")
	}
	if pythonBin == "" {
		pythonBin = defaultPythonBinary
	}

	modelSize := cfg.ModelSize
	if modelSize == "" {
		modelSize = defaultWhisperModel
	}

	maxSize := cfg.MaxFileSize
	if maxSize <= 0 {
		maxSize = defaultMaxAudioSize
	}

	timeoutPerMin := cfg.TimeoutPerMin
	if timeoutPerMin <= 0 {
		timeoutPerMin = defaultTimeoutPerMin
	}

	minTimeout := cfg.MinTimeout
	if minTimeout <= 0 {
		envTimeout := os.Getenv("MEMORY_EXTRACTOR_TIMEOUT_MS")
		if envTimeout != "" {
			if ms, err := strconv.ParseInt(envTimeout, 10, 64); err == nil && ms > 0 {
				minTimeout = time.Duration(ms) * time.Millisecond
			}
		}
	}
	if minTimeout <= 0 {
		minTimeout = defaultMinAudioTimeout
	}

	logger := cfg.Logger
	if logger == nil {
		logger = slog.Default()
	}

	return &AudioExtractor{
		pythonBinary:  pythonBin,
		modelSize:     modelSize,
		defaultLang:   cfg.DefaultLanguage,
		maxFileSize:   maxSize,
		timeoutPerMin: timeoutPerMin,
		minTimeout:    minTimeout,
		logger:        logger,
	}
}

// Name returns the extractor identifier.
func (e *AudioExtractor) Name() string {
	return "audio-whisper"
}

// ContentTypes returns the MIME types this extractor handles.
func (e *AudioExtractor) ContentTypes() []string {
	return audioContentTypes
}

// Capability returns the capability descriptor for audio extraction.
func (e *AudioExtractor) Capability() ExtractorCapability {
	return ExtractorCapability{
		Extensions:     audioExtensions,
		MIMETypes:      audioContentTypes,
		MagicBytes:     audioMagicSignatures,
		RequiresBinary: true,
	}
}

// Available checks whether python3 and faster_whisper are installed.
// Results are cached for 5 minutes.
func (e *AudioExtractor) Available(ctx context.Context) (bool, error) {
	e.availMu.Lock()
	defer e.availMu.Unlock()

	if e.availCached != nil && time.Since(e.availAt) < 5*time.Minute {
		return *e.availCached, nil
	}

	checkCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	cmd := exec.CommandContext(checkCtx, e.pythonBinary, "-c", availabilityCheckScript)
	output, err := cmd.Output()

	available := err == nil && strings.TrimSpace(string(output)) == "ok"
	e.availCached = &available
	e.availAt = time.Now()

	if !available {
		e.logger.Warn("audio-whisper: faster-whisper not available",
			"python", e.pythonBinary,
			"error", err,
		)
	}

	return available, nil
}

// Extract transcribes buffered audio bytes using faster-whisper.
func (e *AudioExtractor) Extract(ctx context.Context, raw []byte, opts ExtractOptions) (ExtractResult, error) {
	if int64(len(raw)) > e.maxFileSize {
		return ExtractResult{}, fmt.Errorf(
			"ingest: audio file size %d bytes exceeds maximum %d bytes",
			len(raw), e.maxFileSize,
		)
	}

	if len(raw) == 0 {
		return ExtractResult{
			Text:        "",
			ContentType: opts.ContentType,
			Encoding:    "UTF-8",
			Metadata:    map[string]string{},
			Skipped:     true,
			Reason:      "empty audio input",
		}, nil
	}

	// Write audio to a temp file for faster-whisper.
	ext := extensionFromContentType(opts.ContentType, opts.FileName)
	tmpFile, err := os.CreateTemp("", "memory-audio-*"+ext)
	if err != nil {
		return ExtractResult{}, fmt.Errorf("ingest: creating temp audio file: %w", err)
	}
	tmpPath := tmpFile.Name()
	defer func() {
		_ = os.Remove(tmpPath)
	}()

	if _, err := tmpFile.Write(raw); err != nil {
		_ = tmpFile.Close()
		return ExtractResult{}, fmt.Errorf("ingest: writing temp audio file: %w", err)
	}
	if err := tmpFile.Close(); err != nil {
		return ExtractResult{}, fmt.Errorf("ingest: closing temp audio file: %w", err)
	}

	return e.transcribe(ctx, tmpPath, len(raw), opts)
}

// ExtractStream reads audio from a stream and delegates to Extract.
func (e *AudioExtractor) ExtractStream(ctx context.Context, reader io.Reader, opts ExtractOptions) (ExtractResult, error) {
	var limitReader io.Reader = reader
	if e.maxFileSize > 0 {
		limitReader = io.LimitReader(reader, e.maxFileSize+1)
	}

	raw, err := io.ReadAll(limitReader)
	if err != nil {
		return ExtractResult{}, fmt.Errorf("ingest: reading audio stream: %w", err)
	}

	return e.Extract(ctx, raw, opts)
}

// transcribe runs the faster-whisper subprocess and formats the result.
func (e *AudioExtractor) transcribe(ctx context.Context, audioPath string, fileSize int, opts ExtractOptions) (ExtractResult, error) {
	timeout := e.computeTimeout(fileSize)
	subCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	language := e.defaultLang
	if opts.Encoding != "" {
		// Allow language override via the encoding field as a convention.
		// Primary use: opts can carry language hint this way.
	}

	args := []string{"-c", whisperScript, audioPath, e.modelSize}
	if language != "" {
		args = append(args, language)
	}

	cmd := exec.CommandContext(subCtx, e.pythonBinary, args...)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		stderrStr := strings.TrimSpace(stderr.String())
		if subCtx.Err() == context.DeadlineExceeded {
			return ExtractResult{}, fmt.Errorf(
				"ingest: audio transcription timed out after %s: %s",
				timeout, stderrStr,
			)
		}
		return ExtractResult{}, fmt.Errorf(
			"ingest: audio transcription failed (exit %d): %s",
			cmd.ProcessState.ExitCode(), stderrStr,
		)
	}

	var output whisperOutput
	if err := json.Unmarshal(stdout.Bytes(), &output); err != nil {
		return ExtractResult{}, fmt.Errorf("ingest: parsing whisper output: %w", err)
	}

	text := formatTranscription(output.Segments)

	segmentsJSON, err := json.Marshal(output.Segments)
	if err != nil {
		return ExtractResult{}, fmt.Errorf("ingest: serialising segments: %w", err)
	}

	metadata := map[string]string{
		"detected_language":    output.Language,
		"language_probability": fmt.Sprintf("%.4f", output.LanguageProbability),
		"duration_seconds":     fmt.Sprintf("%.2f", output.Duration),
		"segment_count":        strconv.Itoa(len(output.Segments)),
		"segments":             string(segmentsJSON),
		"model_size":           e.modelSize,
		"transcription_engine": "faster-whisper",
	}

	return ExtractResult{
		Text:        text,
		ContentType: opts.ContentType,
		Encoding:    "UTF-8",
		Metadata:    metadata,
		Language:    output.Language,
		Confidence:  output.LanguageProbability,
	}, nil
}

// computeTimeout estimates an appropriate timeout based on file size.
// Estimates audio duration from file size using conservative ratios,
// then applies the per-minute timeout budget with a minimum floor.
func (e *AudioExtractor) computeTimeout(fileSize int) time.Duration {
	// Conservative estimate: 1 MB/min for compressed audio (MP3/AAC/OGG),
	// 10 MB/min for uncompressed (WAV). Use the lower ratio (compressed)
	// for safety, meaning more generous timeout.
	estimatedMinutes := float64(fileSize) / (1024 * 1024) // 1 MB per minute estimate
	timeout := time.Duration(estimatedMinutes * float64(e.timeoutPerMin))
	if timeout < e.minTimeout {
		timeout = e.minTimeout
	}
	return timeout
}

// formatTranscription concatenates segments into readable text with
// timestamp markers. A paragraph break is inserted every 30 seconds.
func formatTranscription(segments []TranscriptionSegment) string {
	if len(segments) == 0 {
		return ""
	}

	var builder strings.Builder
	var currentParagraphStart float64
	paragraphStarted := false

	for _, seg := range segments {
		if !paragraphStarted || (seg.Start-currentParagraphStart) >= paragraphBreakSeconds {
			if paragraphStarted {
				builder.WriteString("\n\n")
			}
			builder.WriteString(formatTimestamp(seg.Start))
			builder.WriteString(" - ")

			// Find the end of this paragraph block.
			paragraphEnd := seg.End
			builder.WriteString(formatTimestamp(paragraphEnd))
			builder.WriteString("]\n")

			currentParagraphStart = seg.Start
			paragraphStarted = true
		}
		builder.WriteString(strings.TrimSpace(seg.Text))
		builder.WriteString(" ")
	}

	return strings.TrimSpace(builder.String())
}

// formatTimestamp converts seconds to [MM:SS] format.
func formatTimestamp(seconds float64) string {
	totalSeconds := int(seconds)
	minutes := totalSeconds / 60
	secs := totalSeconds % 60
	return fmt.Sprintf("[%02d:%02d", minutes, secs)
}

// extensionFromContentType derives a file extension from content type or
// filename. Falls back to .wav when neither provides a hint.
func extensionFromContentType(contentType, fileName string) string {
	if fileName != "" {
		ext := filepath.Ext(fileName)
		if ext != "" {
			return ext
		}
	}

	contentTypeExtMap := map[string]string{
		"audio/mpeg":     ".mp3",
		"audio/wav":      ".wav",
		"audio/x-wav":    ".wav",
		"audio/flac":     ".flac",
		"audio/mp4":      ".m4a",
		"audio/ogg":      ".ogg",
		"audio/aac":      ".aac",
		"audio/x-ms-wma": ".wma",
		"audio/opus":     ".opus",
	}

	normalised := normaliseContentType(contentType)
	if ext, ok := contentTypeExtMap[normalised]; ok {
		return ext
	}

	return ".wav"
}
