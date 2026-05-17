// SPDX-License-Identifier: Apache-2.0

package extract

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

// defaultExtractionTimeout is the per-ten-minutes-of-video wall-clock
// budget for the FFmpeg audio-extraction step.
const defaultExtractionTimeout = 120 * time.Second

// defaultMaxKeyframes caps the number of keyframes sampled when
// keyframe extraction is enabled.
const defaultMaxKeyframes = 20

// VideoExtractorConfig configures the video extraction pipeline.
type VideoExtractorConfig struct {
	FFmpegBinary       string        // default: "ffmpeg"
	FFprobeBinary      string        // default: "ffprobe"
	MaxFileSize        int64         // default: 2 GiB
	ExtractionTimeout  time.Duration // default: 120s per 10 min of video
	KeyframeExtraction bool          // default: false
	MaxKeyframes       int           // default: 20
	AudioExtractor     *AudioExtractor
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
		c.ExtractionTimeout = defaultExtractionTimeout
	}
	if c.MaxKeyframes == 0 {
		c.MaxKeyframes = defaultMaxKeyframes
	}
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
	CodecType string `json:"codec_type"`
	CodecName string `json:"codec_name"`
	Width     int    `json:"width"`
	Height    int    `json:"height"`
	RFrameRate string `json:"r_frame_rate"`
}

// VideoExtractor extracts content from video files by separating the
// audio track via FFmpeg and delegating transcription to an
// AudioExtractor. The video itself is never fully buffered in memory;
// FFmpeg reads from stdin via an io.Pipe.
type VideoExtractor struct {
	cfg VideoExtractorConfig
}

// NewVideoExtractor creates a VideoExtractor with the supplied config.
// Zero-value fields receive sensible defaults.
func NewVideoExtractor(cfg VideoExtractorConfig) *VideoExtractor {
	cfg.applyDefaults()
	return &VideoExtractor{cfg: cfg}
}

// Name returns the unique extractor identifier.
func (e *VideoExtractor) Name() string { return "video-ffmpeg" }

// Capability describes which video formats this extractor supports.
func (e *VideoExtractor) Capability() ExtractorCapability {
	return ExtractorCapability{
		Extensions: []string{
			".mp4", ".avi", ".mkv", ".mov", ".webm", ".wmv", ".flv", ".m4v",
		},
		MIMETypes: []string{
			"video/mp4", "video/x-msvideo", "video/x-matroska",
			"video/quicktime", "video/webm", "video/x-ms-wmv",
			"video/x-flv",
		},
		MagicBytes: []MagicSignature{
			{Offset: 4, Bytes: []byte{0x66, 0x74, 0x79, 0x70}},             // MP4/MOV ftyp
			{Offset: 0, Bytes: []byte{0x1A, 0x45, 0xDF, 0xA3}},             // MKV/WebM (EBML)
			{Offset: 0, Bytes: []byte{0x52, 0x49, 0x46, 0x46}},             // RIFF (AVI header)
		},
		RequiresBinary: true,
	}
}

// Available reports whether FFmpeg, FFprobe, and the AudioExtractor are
// all present and usable.
func (e *VideoExtractor) Available(ctx context.Context) (bool, error) {
	if !CheckBinaryAvailable(e.cfg.FFmpegBinary) {
		return false, nil
	}
	if !CheckBinaryAvailable(e.cfg.FFprobeBinary) {
		return false, nil
	}
	if e.cfg.AudioExtractor == nil {
		return false, nil
	}
	return e.cfg.AudioExtractor.Available(ctx)
}

// Extract processes a video provided as a byte slice. For videos that
// exceed memory-friendly sizes, callers should prefer ExtractStream.
func (e *VideoExtractor) Extract(ctx context.Context, input []byte, opts ExtractOptions) (ExtractionResult, error) {
	if int64(len(input)) > e.cfg.MaxFileSize {
		return ExtractionResult{}, fmt.Errorf(
			"extract: video file size %d exceeds maximum %d bytes",
			len(input), e.cfg.MaxFileSize,
		)
	}
	return e.extractFromReader(ctx, bytes.NewReader(input), int64(len(input)), opts)
}

// ExtractStream processes a video from an io.Reader without buffering
// the full file in memory. The sizeBound argument is used for the file
// size check; pass -1 when the size is unknown (the extractor will
// enforce the limit during the streaming copy).
func (e *VideoExtractor) ExtractStream(ctx context.Context, reader io.Reader, sizeBound int64, opts ExtractOptions) (ExtractionResult, error) {
	if sizeBound > e.cfg.MaxFileSize {
		return ExtractionResult{}, fmt.Errorf(
			"extract: video file size %d exceeds maximum %d bytes",
			sizeBound, e.cfg.MaxFileSize,
		)
	}
	return e.extractFromReader(ctx, reader, sizeBound, opts)
}

// extractFromReader is the unified implementation that handles both
// buffered and streaming input paths.
func (e *VideoExtractor) extractFromReader(ctx context.Context, reader io.Reader, sizeBound int64, opts ExtractOptions) (ExtractionResult, error) {
	if e.cfg.AudioExtractor == nil {
		return ExtractionResult{}, fmt.Errorf("extract: video extractor requires an AudioExtractor dependency")
	}

	// Probe video metadata first. FFprobe requires seekable input, so we
	// write the source to a temp file. The temp file also serves the
	// FFmpeg audio-extraction step, ensuring we never hold the full video
	// in a Go byte slice.
	tmpDir, err := os.MkdirTemp("", "memory-video-*")
	if err != nil {
		return ExtractionResult{}, fmt.Errorf("extract: creating temp dir: %w", err)
	}
	defer os.RemoveAll(tmpDir)

	tmpVideoPath := tmpDir + "/input.video"
	written, err := writeToTempFile(tmpVideoPath, reader, e.cfg.MaxFileSize)
	if err != nil {
		return ExtractionResult{}, err
	}

	// Probe metadata.
	videoMeta, err := e.probeMetadata(ctx, tmpVideoPath)
	if err != nil {
		return ExtractionResult{}, fmt.Errorf("extract: probing video metadata: %w", err)
	}

	if !videoMeta.HasAudio {
		return ExtractionResult{
			Content: "Video contains no audio track.",
			MIME:    "text/plain",
			Metadata: map[string]any{
				"video_info": videoMeta,
			},
		}, nil
	}

	// Extract audio via FFmpeg using streaming pipes.
	wavData, err := e.extractAudio(ctx, tmpVideoPath, videoMeta.DurationSeconds)
	if err != nil {
		return ExtractionResult{}, fmt.Errorf("extract: extracting audio track: %w", err)
	}

	// Delegate transcription to the AudioExtractor.
	audioResult, err := e.cfg.AudioExtractor.ExtractFromBuffer(ctx, wavData, opts)
	if err != nil {
		return ExtractionResult{}, fmt.Errorf("extract: transcribing audio: %w", err)
	}

	// Merge metadata from video probe and audio transcription.
	metadata := mergeMetadata(videoMeta, audioResult.Metadata)
	metadata["source_bytes"] = written

	return ExtractionResult{
		Content:    audioResult.Content,
		MIME:       "text/plain",
		Metadata:   metadata,
		Language:   audioResult.Language,
		Confidence: audioResult.Confidence,
	}, nil
}

// writeToTempFile streams reader contents to a temp file, enforcing a
// size limit. Returns the number of bytes written.
func writeToTempFile(path string, reader io.Reader, maxSize int64) (int64, error) {
	f, err := os.Create(path)
	if err != nil {
		return 0, fmt.Errorf("extract: creating temp video file: %w", err)
	}
	defer f.Close()

	limited := io.LimitReader(reader, maxSize+1)
	n, err := io.Copy(f, limited)
	if err != nil {
		return 0, fmt.Errorf("extract: writing video to temp file: %w", err)
	}
	if n > maxSize {
		return 0, fmt.Errorf("extract: video stream exceeds maximum %d bytes", maxSize)
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

// extractAudio runs FFmpeg to extract the audio track from a video file
// as 16-bit PCM WAV at 16 kHz mono. The extracted audio is captured in
// memory as a byte slice (WAV files for typical video lengths are
// manageable; a 2-hour video yields ~230 MB of 16 kHz mono PCM).
func (e *VideoExtractor) extractAudio(ctx context.Context, videoPath string, durationSeconds float64) ([]byte, error) {
	timeout := e.computeExtractionTimeout(durationSeconds)

	extractCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(extractCtx, e.cfg.FFmpegBinary,
		"-i", videoPath,
		"-vn",
		"-acodec", "pcm_s16le",
		"-ar", "16000",
		"-ac", "1",
		"-f", "wav",
		"pipe:1",
	)

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("ffmpeg audio extraction failed: %w (stderr: %s)", err, stderr.String())
	}

	return stdout.Bytes(), nil
}

// computeExtractionTimeout calculates a wall-clock timeout proportional
// to the video duration.
func (e *VideoExtractor) computeExtractionTimeout(durationSeconds float64) time.Duration {
	// 120 seconds per 10 minutes of video, with a minimum of 60 seconds.
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

// mergeMetadata combines video probe data with audio transcription
// metadata into a single map.
func mergeMetadata(video VideoMetadata, audio map[string]any) map[string]any {
	merged := make(map[string]any, len(audio)+1)
	for k, v := range audio {
		merged[k] = v
	}
	merged["video_info"] = video
	return merged
}
