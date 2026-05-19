// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// defaultMaxVideoSize is the maximum video file size accepted for
// extraction. Videos larger than this are rejected before any
// processing begins.
const defaultMaxVideoSize = 2 * 1024 * 1024 * 1024 // 2 GiB

// defaultVideoExtractionTimeout is the per-ten-minutes-of-video
// wall-clock budget for the FFmpeg audio-extraction step.
const defaultVideoExtractionTimeout = 120 * time.Second

// defaultMaxKeyframes caps the number of keyframes sampled when
// keyframe extraction is enabled.
const defaultMaxKeyframes = 20

// VideoExtractorConfig configures the video extraction pipeline.
type VideoExtractorConfig struct {
	// FFmpegBinary is the path or name of the FFmpeg binary.
	// Default: "ffmpeg". Override via MEMORY_FFMPEG_PATH env var.
	FFmpegBinary string

	// FFprobeBinary is the path or name of the FFprobe binary.
	// Default: "ffprobe". Override via MEMORY_FFPROBE_PATH env var.
	FFprobeBinary string

	// MaxFileSize is the maximum video file size in bytes.
	// Default: 2 GiB.
	MaxFileSize int64

	// ExtractionTimeout is the timeout per 10 minutes of video.
	// Default: 120 seconds.
	ExtractionTimeout time.Duration

	// KeyframeExtraction enables keyframe sampling.
	// Default: false.
	KeyframeExtraction bool

	// MaxKeyframes caps the number of keyframes sampled.
	// Default: 20.
	MaxKeyframes int

	// AudioExtractor is the delegate used for transcribing extracted
	// audio tracks. Must implement the ingest.Extractor interface.
	AudioExtractor Extractor
}

func (c *VideoExtractorConfig) applyDefaults() {
	if c.FFmpegBinary == "" {
		c.FFmpegBinary = envOrDefault("MEMORY_FFMPEG_PATH", "ffmpeg")
	}
	if c.FFprobeBinary == "" {
		c.FFprobeBinary = envOrDefault("MEMORY_FFPROBE_PATH", "ffprobe")
	}
	if c.MaxFileSize == 0 {
		c.MaxFileSize = defaultMaxVideoSize
	}
	if c.ExtractionTimeout == 0 {
		c.ExtractionTimeout = defaultVideoExtractionTimeout
	}
	if c.MaxKeyframes == 0 {
		c.MaxKeyframes = defaultMaxKeyframes
	}
}

// envOrDefault returns the value of the named environment variable, or
// the fallback when the variable is empty or unset.
func envOrDefault(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// VideoMetadata holds the structural properties of a video file as
// reported by FFprobe.
type VideoMetadata struct {
	DurationSeconds float64 `json:"duration_seconds"`
	Width           int     `json:"width,omitempty"`
	Height          int     `json:"height,omitempty"`
	Codec           string  `json:"codec,omitempty"`
	HasAudio        bool    `json:"has_audio"`
	FrameRate       float64 `json:"frame_rate,omitempty"`
}

// ffprobeOutput mirrors the JSON structure returned by
// `ffprobe -print_format json -show_format -show_streams`.
type ffprobeOutput struct {
	Format  ffprobeFormat   `json:"format"`
	Streams []ffprobeStream `json:"streams"`
}

type ffprobeFormat struct {
	Duration string `json:"duration"`
}

type ffprobeStream struct {
	CodecType  string `json:"codec_type"`
	CodecName  string `json:"codec_name"`
	Width      int    `json:"width"`
	Height     int    `json:"height"`
	RFrameRate string `json:"r_frame_rate"`
}

// videoContentTypes lists all MIME types handled by the video extractor.
var videoContentTypes = []string{
	"video/mp4",
	"video/x-msvideo",
	"video/x-matroska",
	"video/quicktime",
	"video/webm",
	"video/x-ms-wmv",
	"video/x-flv",
}

// videoExtensions lists all file extensions handled by the video extractor.
var videoExtensions = []string{
	".mp4", ".avi", ".mkv", ".mov", ".webm", ".wmv", ".flv", ".m4v",
}

// videoMagicBytes contains magic byte signatures for supported video formats.
var videoMagicBytes = []MagicSignature{
	{Offset: 4, Bytes: []byte{0x66, 0x74, 0x79, 0x70}},   // MP4/MOV ftyp
	{Offset: 0, Bytes: []byte{0x1A, 0x45, 0xDF, 0xA3}},   // MKV/WebM (EBML)
	{Offset: 0, Bytes: []byte{0x52, 0x49, 0x46, 0x46}},   // RIFF (AVI header)
}

// VideoExtractor extracts content from video files by separating the
// audio track via FFmpeg and delegating transcription to an
// AudioExtractor. Implements the canonical ingest.Extractor interface.
type VideoExtractor struct {
	cfg VideoExtractorConfig
}

// Compile-time interface check.
var _ Extractor = (*VideoExtractor)(nil)

// NewVideoExtractor creates a VideoExtractor with the supplied config.
// Zero-value fields receive sensible defaults.
func NewVideoExtractor(cfg VideoExtractorConfig) *VideoExtractor {
	cfg.applyDefaults()
	return &VideoExtractor{cfg: cfg}
}

// Name returns the unique extractor identifier.
func (e *VideoExtractor) Name() string { return "video-ffmpeg" }

// ContentTypes returns the MIME types this extractor handles.
func (e *VideoExtractor) ContentTypes() []string {
	return videoContentTypes
}

// Capability describes which video formats this extractor supports.
func (e *VideoExtractor) Capability() ExtractorCapability {
	return ExtractorCapability{
		Extensions:     videoExtensions,
		MIMETypes:      videoContentTypes,
		MagicBytes:     videoMagicBytes,
		RequiresBinary: true,
	}
}

// Available reports whether FFmpeg, FFprobe, and the AudioExtractor are
// all present and usable.
func (e *VideoExtractor) Available(ctx context.Context) (bool, error) {
	if !CheckBinaryAvailable(ctx, e.cfg.FFmpegBinary) {
		return false, nil
	}
	if !CheckBinaryAvailable(ctx, e.cfg.FFprobeBinary) {
		return false, nil
	}
	if e.cfg.AudioExtractor == nil {
		return false, nil
	}
	return e.cfg.AudioExtractor.Available(ctx)
}

// Extract processes a video provided as a byte slice.
func (e *VideoExtractor) Extract(ctx context.Context, raw []byte, opts ExtractOptions) (ExtractResult, error) {
	if int64(len(raw)) > e.cfg.MaxFileSize {
		return ExtractResult{}, fmt.Errorf(
			"ingest: video file size %d exceeds maximum %d bytes",
			len(raw), e.cfg.MaxFileSize,
		)
	}
	return e.extractFromReader(ctx, bytes.NewReader(raw), int64(len(raw)), opts)
}

// ExtractStream buffers the reader and delegates to Extract.
func (e *VideoExtractor) ExtractStream(ctx context.Context, reader io.Reader, opts ExtractOptions) (ExtractResult, error) {
	var limitReader io.Reader = reader
	if opts.MaxBytes > 0 {
		limitReader = io.LimitReader(reader, opts.MaxBytes)
	}
	raw, err := io.ReadAll(limitReader)
	if err != nil {
		return ExtractResult{}, fmt.Errorf("ingest: reading video stream: %w", err)
	}
	return e.Extract(ctx, raw, opts)
}

// extractFromReader is the unified implementation that handles both
// buffered and streaming input paths.
func (e *VideoExtractor) extractFromReader(ctx context.Context, reader io.Reader, sizeBound int64, opts ExtractOptions) (ExtractResult, error) {
	if e.cfg.AudioExtractor == nil {
		return ExtractResult{}, fmt.Errorf("ingest: video extractor requires an AudioExtractor dependency")
	}

	// FFprobe requires seekable input, so we write the source to a temp
	// file. Audio extraction then streams the temp file to FFmpeg via
	// stdin (pipe:0).
	tmpDir, err := os.MkdirTemp("", "memory-video-*")
	if err != nil {
		return ExtractResult{}, fmt.Errorf("ingest: creating temp dir: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	tmpVideoPath := tmpDir + "/input.video"
	written, err := writeVideoToTempFile(tmpVideoPath, reader, e.cfg.MaxFileSize)
	if err != nil {
		return ExtractResult{}, err
	}

	// Probe metadata.
	videoMeta, err := e.probeMetadata(ctx, tmpVideoPath)
	if err != nil {
		return ExtractResult{}, fmt.Errorf("ingest: probing video metadata: %w", err)
	}

	if !videoMeta.HasAudio {
		return ExtractResult{
			Text:        "Video contains no audio track.",
			ContentType: "text/plain",
			Metadata: map[string]string{
				"video_duration_seconds": fmt.Sprintf("%.2f", videoMeta.DurationSeconds),
				"video_width":           strconv.Itoa(videoMeta.Width),
				"video_height":          strconv.Itoa(videoMeta.Height),
				"video_codec":           videoMeta.Codec,
				"video_has_audio":       "false",
			},
		}, nil
	}

	// Extract audio via FFmpeg to a temp WAV file instead of buffering
	// the entire WAV payload in memory (~230 MB for a 2-hour video).
	wavPath, err := e.extractAudioToFile(ctx, tmpVideoPath, tmpDir, videoMeta.DurationSeconds)
	if err != nil {
		return ExtractResult{}, fmt.Errorf("ingest: extracting audio track: %w", err)
	}

	// Read the WAV file for the AudioExtractor. The file is already on
	// disk so peak memory is bounded by one copy rather than two.
	wavData, err := os.ReadFile(wavPath)
	if err != nil {
		return ExtractResult{}, fmt.Errorf("ingest: reading extracted audio: %w", err)
	}

	// Delegate transcription to the AudioExtractor.
	audioOpts := ExtractOptions{
		ContentType: "audio/wav",
		FileName:    "extracted.wav",
		Language:    opts.Language,
	}
	audioResult, err := e.cfg.AudioExtractor.Extract(ctx, wavData, audioOpts)
	if err != nil {
		return ExtractResult{}, fmt.Errorf("ingest: transcribing audio: %w", err)
	}

	// Merge video metadata into the audio result metadata.
	metadata := mergeVideoAudioMetadata(videoMeta, audioResult.Metadata)
	metadata["source_bytes"] = strconv.FormatInt(written, 10)

	resultText := audioResult.Text

	// Extract keyframes if enabled.
	if e.cfg.KeyframeExtraction {
		keyframeCfg := KeyframeConfig{
			FFmpegBinary: e.cfg.FFmpegBinary,
			MaxKeyframes: e.cfg.MaxKeyframes,
			Timeout:      e.cfg.ExtractionTimeout,
		}
		keyframes, kfErr := extractKeyframes(ctx, tmpVideoPath, tmpDir, keyframeCfg)
		if kfErr != nil {
			metadata["keyframe_error"] = kfErr.Error()
		} else {
			kfText := formatKeyframeText(keyframes)
			if kfText != "" {
				resultText += kfText
			}
			mergeKeyframeMetadata(metadata, keyframes)
		}
	}

	return ExtractResult{
		Text:        resultText,
		ContentType: "text/plain",
		Metadata:    metadata,
		Language:    audioResult.Language,
		Confidence:  audioResult.Confidence,
	}, nil
}

// writeVideoToTempFile streams reader contents to a temp file,
// enforcing a size limit. Returns the number of bytes written. The file
// is created with restrictive permissions (0600).
func writeVideoToTempFile(path string, reader io.Reader, maxSize int64) (int64, error) {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return 0, fmt.Errorf("ingest: creating temp video file: %w", err)
	}
	defer func() { _ = f.Close() }()

	limited := io.LimitReader(reader, maxSize+1)
	n, err := io.Copy(f, limited)
	if err != nil {
		return 0, fmt.Errorf("ingest: writing video to temp file: %w", err)
	}
	if n > maxSize {
		return 0, fmt.Errorf("ingest: video stream exceeds maximum %d bytes", maxSize)
	}
	return n, nil
}

// probeMetadata runs FFprobe against the video file and parses the
// JSON output into a VideoMetadata struct.
func (e *VideoExtractor) probeMetadata(ctx context.Context, videoPath string) (VideoMetadata, error) {
	timeout := 30 * time.Second
	if t := e.overrideTimeout(); t > 0 {
		timeout = t
	}

	probeCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(probeCtx, e.cfg.FFprobeBinary,
		"-v", "quiet",
		"-print_format", "json",
		"-show_format",
		"-show_streams",
		videoPath,
	)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return VideoMetadata{}, fmt.Errorf("ffprobe failed: %w (stderr: %s)", err, stderr.String())
	}

	var probe ffprobeOutput
	if err := json.Unmarshal(stdout.Bytes(), &probe); err != nil {
		return VideoMetadata{}, fmt.Errorf("parsing ffprobe output: %w", err)
	}

	return buildVideoMetadata(probe), nil
}

// buildVideoMetadata converts raw FFprobe output into the structured
// VideoMetadata type.
func buildVideoMetadata(probe ffprobeOutput) VideoMetadata {
	meta := VideoMetadata{}

	if probe.Format.Duration != "" {
		if d, err := strconv.ParseFloat(probe.Format.Duration, 64); err == nil {
			meta.DurationSeconds = d
		}
	}

	for _, stream := range probe.Streams {
		switch stream.CodecType {
		case "video":
			meta.Width = stream.Width
			meta.Height = stream.Height
			meta.Codec = stream.CodecName
			meta.FrameRate = parseFrameRate(stream.RFrameRate)
		case "audio":
			meta.HasAudio = true
		}
	}

	return meta
}

// parseFrameRate converts FFprobe's fractional frame rate (e.g. "30/1"
// or "24000/1001") into a float.
func parseFrameRate(raw string) float64 {
	parts := strings.SplitN(raw, "/", 2)
	if len(parts) != 2 {
		return 0
	}
	num, err1 := strconv.ParseFloat(parts[0], 64)
	den, err2 := strconv.ParseFloat(parts[1], 64)
	if err1 != nil || err2 != nil || den == 0 {
		return 0
	}
	return num / den
}

// extractAudioToFile runs FFmpeg to extract the audio track from a
// video file as 16-bit PCM WAV at 16 kHz mono. The WAV output is
// written directly to a file in tmpDir instead of being buffered in
// memory, which is critical for long videos where WAV data can exceed
// 200 MB.
func (e *VideoExtractor) extractAudioToFile(ctx context.Context, videoPath, tmpDir string, durationSeconds float64) (string, error) {
	timeout := e.computeExtractionTimeout(durationSeconds)

	extractCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	wavPath := tmpDir + "/extracted.wav"

	cmd := exec.CommandContext(extractCtx, e.cfg.FFmpegBinary,
		"-i", videoPath,
		"-vn",
		"-acodec", "pcm_s16le",
		"-ar", "16000",
		"-ac", "1",
		"-y",
		wavPath,
	)

	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("ffmpeg audio extraction failed: %w (stderr: %s)", err, stderr.String())
	}

	return wavPath, nil
}

// computeExtractionTimeout calculates a wall-clock timeout proportional
// to the video duration.
func (e *VideoExtractor) computeExtractionTimeout(durationSeconds float64) time.Duration {
	tenMinBlocks := durationSeconds / 600
	if tenMinBlocks < 1 {
		tenMinBlocks = 1
	}
	timeout := time.Duration(tenMinBlocks) * e.cfg.ExtractionTimeout
	if timeout < 60*time.Second {
		timeout = 60 * time.Second
	}
	return timeout
}

// overrideTimeout reads an optional environment-variable override for
// the extractor timeout.
func (e *VideoExtractor) overrideTimeout() time.Duration {
	raw := os.Getenv("MEMORY_EXTRACTOR_TIMEOUT_MS")
	if raw == "" {
		return 0
	}
	ms, err := strconv.ParseInt(raw, 10, 64)
	if err != nil {
		return 0
	}
	return time.Duration(ms) * time.Millisecond
}

// mergeVideoAudioMetadata combines video probe data with audio
// transcription metadata into a single map[string]string.
func mergeVideoAudioMetadata(video VideoMetadata, audio map[string]string) map[string]string {
	merged := make(map[string]string, len(audio)+6)
	for k, v := range audio {
		merged[k] = v
	}
	merged["video_duration_seconds"] = fmt.Sprintf("%.2f", video.DurationSeconds)
	merged["video_width"] = strconv.Itoa(video.Width)
	merged["video_height"] = strconv.Itoa(video.Height)
	merged["video_codec"] = video.Codec
	merged["video_has_audio"] = strconv.FormatBool(video.HasAudio)
	if video.FrameRate > 0 {
		merged["video_frame_rate"] = fmt.Sprintf("%.3f", video.FrameRate)
	}
	return merged
}
