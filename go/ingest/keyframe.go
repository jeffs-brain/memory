// SPDX-License-Identifier: Apache-2.0

package ingest

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// defaultKeyframeInterval is the default time between keyframe
// captures in seconds. Override via MEMORY_KEYFRAME_INTERVAL_SECS.
const defaultKeyframeInterval = 30

// KeyframeConfig holds the resolved keyframe extraction settings.
type KeyframeConfig struct {
	FFmpegBinary string
	IntervalSecs int
	MaxKeyframes int
	OCRExtractor Extractor
	Timeout      time.Duration
}

// KeyframeResult holds the text extracted from a single keyframe.
type KeyframeResult struct {
	TimestampSecs float64
	Text          string
	Confidence    float64
}

// applyKeyframeDefaults fills zero-value fields with defaults.
func applyKeyframeDefaults(cfg *KeyframeConfig) {
	if cfg.FFmpegBinary == "" {
		cfg.FFmpegBinary = envOrDefault("MEMORY_FFMPEG_PATH", "ffmpeg")
	}
	if cfg.IntervalSecs <= 0 {
		if envVal := os.Getenv("MEMORY_KEYFRAME_INTERVAL_SECS"); envVal != "" {
			if v, err := strconv.Atoi(envVal); err == nil && v > 0 {
				cfg.IntervalSecs = v
			}
		}
		if cfg.IntervalSecs <= 0 {
			cfg.IntervalSecs = defaultKeyframeInterval
		}
	}
	if cfg.MaxKeyframes <= 0 {
		cfg.MaxKeyframes = defaultMaxKeyframes
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 60 * time.Second
	}
}

// extractKeyframes extracts keyframe images from a video file at the
// configured interval, runs OCR on each frame, and returns the
// collected text results. The caller is responsible for the tmpDir
// lifecycle.
func extractKeyframes(ctx context.Context, videoPath, tmpDir string, cfg KeyframeConfig) ([]KeyframeResult, error) {
	applyKeyframeDefaults(&cfg)

	if cfg.OCRExtractor == nil {
		return nil, nil
	}

	available, err := cfg.OCRExtractor.Available(ctx)
	if err != nil || !available {
		return nil, nil
	}

	framesDir := filepath.Join(tmpDir, "keyframes")
	if err := os.MkdirAll(framesDir, 0o700); err != nil {
		return nil, fmt.Errorf("ingest: creating keyframes dir: %w", err)
	}

	// Extract keyframes using FFmpeg fps filter.
	fpsFilter := fmt.Sprintf("fps=1/%d", cfg.IntervalSecs)
	outputPattern := filepath.Join(framesDir, "frame-%04d.png")

	args := []string{
		"-i", videoPath,
		"-vf", fpsFilter,
		"-frames:v", strconv.Itoa(cfg.MaxKeyframes),
		"-vsync", "vfr",
		outputPattern,
	}

	result, err := RunSubprocess(ctx, cfg.FFmpegBinary, args, nil, SubprocessOptions{
		Timeout: cfg.Timeout,
	})
	if err != nil {
		return nil, fmt.Errorf("ingest: keyframe extraction: %w", err)
	}
	if result.ExitCode != 0 {
		return nil, fmt.Errorf("ingest: ffmpeg keyframe extraction exited with code %d: %s",
			result.ExitCode, result.Stderr)
	}

	// Collect generated frame images.
	frameFiles, err := filepath.Glob(filepath.Join(framesDir, "frame-*.png"))
	if err != nil {
		return nil, fmt.Errorf("ingest: listing keyframe images: %w", err)
	}

	sort.Strings(frameFiles)

	// Cap at MaxKeyframes.
	if len(frameFiles) > cfg.MaxKeyframes {
		frameFiles = frameFiles[:cfg.MaxKeyframes]
	}

	var keyframeResults []KeyframeResult

	for i, framePath := range frameFiles {
		frameData, readErr := os.ReadFile(framePath)
		if readErr != nil {
			continue
		}

		ocrOpts := ExtractOptions{
			ContentType: "image/png",
			FileName:    filepath.Base(framePath),
		}

		ocrResult, ocrErr := cfg.OCRExtractor.Extract(ctx, frameData, ocrOpts)
		if ocrErr != nil || ocrResult.Skipped || ocrResult.Text == "" {
			continue
		}

		timestampSecs := float64(i+1) * float64(cfg.IntervalSecs)

		keyframeResults = append(keyframeResults, KeyframeResult{
			TimestampSecs: timestampSecs,
			Text:          ocrResult.Text,
			Confidence:    ocrResult.Confidence,
		})
	}

	return keyframeResults, nil
}

// formatKeyframeText formats keyframe OCR results into a structured
// text section suitable for appending to the video extraction result.
func formatKeyframeText(results []KeyframeResult) string {
	if len(results) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("\n\n--- Keyframe Text ---\n")

	for _, kf := range results {
		timestamp := formatKeyframeTimestamp(kf.TimestampSecs)
		sb.WriteString(fmt.Sprintf("\n[%s]\n%s\n", timestamp, kf.Text))
	}

	return sb.String()
}

// formatKeyframeTimestamp converts seconds to MM:SS format.
func formatKeyframeTimestamp(seconds float64) string {
	mins := int(seconds) / 60
	secs := int(seconds) % 60
	return fmt.Sprintf("%02d:%02d", mins, secs)
}

// mergeKeyframeMetadata adds keyframe-related metadata entries to the
// provided metadata map. Mutates the map in place.
func mergeKeyframeMetadata(metadata map[string]string, results []KeyframeResult) {
	metadata["keyframe_count"] = strconv.Itoa(len(results))

	if len(results) == 0 {
		return
	}

	var totalConf float64
	for _, kf := range results {
		totalConf += kf.Confidence
	}
	avgConf := totalConf / float64(len(results))
	metadata["keyframe_avg_confidence"] = formatConfidence(avgConf)
}
