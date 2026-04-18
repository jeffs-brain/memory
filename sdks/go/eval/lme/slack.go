// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// SlackReport formats an LMEResult for Slack and posts it via an
// Incoming Webhook URL. The Go SDK port uses a plain HTTP POST rather
// than the upstream jeff Slack client so the package stays dependency
// free.
//
// webhookURL is the full Slack Incoming Webhook endpoint
// (https://hooks.slack.com/services/...). When empty the function
// returns an error.
func SlackReport(ctx context.Context, webhookURL string, result *LMEResult, manifest *RunManifest) error {
	if webhookURL == "" {
		return fmt.Errorf("lme slack: webhook URL is required")
	}
	if result == nil {
		return fmt.Errorf("lme slack: nil result")
	}

	blocks := buildSlackBlocks(result, manifest)
	payload := map[string]any{
		"text":   fallbackText(result),
		"blocks": blocks,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("lme slack: marshal payload: %w", err)
	}

	reqCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodPost, webhookURL, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("lme slack: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("lme slack: post: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("lme slack: server returned %d: %s", resp.StatusCode, string(respBody))
	}
	return nil
}

// buildSlackBlocks renders the Slack block-kit payload for a result.
func buildSlackBlocks(result *LMEResult, manifest *RunManifest) []map[string]any {
	blocks := make([]map[string]any, 0, 6)

	blocks = append(blocks, map[string]any{
		"type": "header",
		"text": map[string]any{
			"type":  "plain_text",
			"text":  "LME Benchmark Results",
			"emoji": true,
		},
	})

	var summary strings.Builder
	fmt.Fprintf(&summary, "*Overall score:* %.1f%%\n", result.OverallScore*100)
	fmt.Fprintf(&summary, "*Exact match:* %.1f%%\n", result.ExactMatchScore*100)
	fmt.Fprintf(&summary, "*Questions run:* %d\n", result.QuestionsRun)
	fmt.Fprintf(&summary, "*Ingest mode:* %s\n", result.IngestMode)
	if result.JudgeModel != "" {
		fmt.Fprintf(&summary, "*Judge:* %s\n", result.JudgeModel)
	}

	blocks = append(blocks, map[string]any{
		"type": "section",
		"text": map[string]any{
			"type": "mrkdwn",
			"text": summary.String(),
		},
	})

	if len(result.ByCategory) > 0 {
		blocks = append(blocks, map[string]any{"type": "divider"})
		var cats strings.Builder
		cats.WriteString("*Per category:*\n")
		for cat, c := range result.ByCategory {
			fmt.Fprintf(&cats, "  %s: %d/%d (%.1f%%)\n", cat, c.Correct, c.Run, c.Score*100)
		}
		blocks = append(blocks, map[string]any{
			"type": "section",
			"text": map[string]any{
				"type": "mrkdwn",
				"text": cats.String(),
			},
		})
	}

	if result.CostAccounting.TotalUSD > 0 {
		blocks = append(blocks, map[string]any{"type": "divider"})
		cost := fmt.Sprintf("*Cost:* $%.2f (ingest: $%.2f, agent: $%.2f, judge: $%.2f)",
			result.CostAccounting.TotalUSD,
			result.CostAccounting.IngestUSD,
			result.CostAccounting.AgentUSD,
			result.CostAccounting.JudgeUSD,
		)
		blocks = append(blocks, map[string]any{
			"type": "section",
			"text": map[string]any{
				"type": "mrkdwn",
				"text": cost,
			},
		})
	}

	if manifest != nil {
		blocks = append(blocks, map[string]any{
			"type": "context",
			"elements": []any{
				map[string]any{
					"type": "mrkdwn",
					"text": fmt.Sprintf("Dataset: `%.12s` | Seed: %d | Prompt v%d",
						manifest.DatasetSHA, manifest.RunSeed, manifest.JudgePromptVersion),
				},
			},
		})
	}

	return blocks
}

func fallbackText(result *LMEResult) string {
	return fmt.Sprintf("LME Benchmark: %.1f%% overall (%d questions, exact match %.1f%%)",
		result.OverallScore*100, result.QuestionsRun, result.ExactMatchScore*100)
}
