// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/jeffs-brain/memory/go/eval/lme"
	"github.com/jeffs-brain/memory/go/llm"
	"github.com/jeffs-brain/memory/go/memory"
	"github.com/jeffs-brain/memory/go/retrieval"
)

func evalCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "eval",
		Short: "Run SDK evaluation harnesses",
	}
	cmd.AddCommand(evalLmeCmd())
	return cmd
}

func evalLmeCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "lme",
		Short: "LongMemEval benchmark harness",
	}
	cmd.AddCommand(evalLmeRunCmd())
	cmd.AddCommand(evalLmeReviewCmd())
	return cmd
}

// evalLmeRunCmd drives a full LongMemEval benchmark.
//
//	memory eval lme run --dataset longmemeval_s.json --concurrency 8 \
//	                    --judge claude-haiku-4-5
func evalLmeRunCmd() *cobra.Command {
	var (
		datasetPath        string
		sampleSize         int
		seed               int64
		concurrency        int
		ingestMode         string
		expectedSHA        string
		maxCostUSD         float64
		judgeModel         string
		actorModel         string
		extractModel       string
		replayConcurrency  int
		maxIterations      int
		questionTimeout    time.Duration
		outputPath         string
		manifestPath       string
		slackWebhook       string
		disableReader      bool
		judgeTimeout       time.Duration
		extractOnly        bool
		brainCache         string
		contextualise      bool
		contextualCacheDir string
		actorEndpoint      string
		actorBrain         string
		actorEndpointStyle string
		actorTopK          int
		actorCandidateK    int
		actorRerankTopN    int
		actorScope         string
		actorProject       string
		actorPathPrefix    string
	)

	cmd := &cobra.Command{
		Use:   "run",
		Short: "Run a LongMemEval benchmark against a dataset",
		Long: "Loads the dataset, ingests every haystack session into an " +
			"isolated in-memory brain, answers each question via direct " +
			"search, and scores the outcomes with an LLM judge.\n\n" +
			"The judge model defaults to claude-haiku-4-5 with an env " +
			"override via JB_LME_JUDGE_MODEL. The actor model defaults " +
			"to gpt-4o with an env override via JB_LME_ACTOR_MODEL.",
		RunE: func(cmd *cobra.Command, args []string) error {
			if datasetPath == "" {
				return errors.New("--dataset is required")
			}

			ctx := cmd.Context()
			if ctx == nil {
				ctx = context.Background()
			}

			judgeModel = resolveJudgeModel(judgeModel)
			actorModel = resolveActorModel(actorModel)

			// Normalise ingest mode aliases. "agentic" is not a real
			// ingest mode (it still needs a populated corpus) so we
			// leave the ingest mode at replay by default and flip the
			// agentic flag on the RunConfig.
			agenticMode := false
			normalIngest := ingestMode
			switch ingestMode {
			case "agentic":
				agenticMode = true
				normalIngest = "replay"
			case "":
				normalIngest = "bulk"
			}

			if extractOnly && brainCache == "" {
				return errors.New("--extract-only requires --brain-cache <path>")
			}

			cfg := lme.RunConfig{
				DatasetPath:        datasetPath,
				SampleSize:         sampleSize,
				Seed:               seed,
				Concurrency:        concurrency,
				IngestMode:         normalIngest,
				ExpectedSHA:        expectedSHA,
				MaxCostUSD:         maxCostUSD,
				ReplayConcurrency:  replayConcurrency,
				ReplayExtractModel: extractModel,
				AgenticMode:        agenticMode,
				MaxIterations:      maxIterations,
				QuestionTimeout:    questionTimeout,
				ExtractOnly:        extractOnly,
				BrainCache:         brainCache,
				ActorEndpoint:      actorEndpoint,
				ActorBrainID:       actorBrain,
				ActorEndpointStyle: actorEndpointStyle,
				ActorTopK:          actorTopK,
				ActorCandidateK:    actorCandidateK,
				ActorRerankTopN:    actorRerankTopN,
				ActorFilters: retrieval.Filters{
					Scope:      actorScope,
					Project:    actorProject,
					PathPrefix: actorPathPrefix,
				},
			}

			// Wire the judge unless the caller explicitly passed
			// `--judge ""` or the run is extract-only (no questions
			// to score in that mode).
			var judgeProvider llm.Provider
			if judgeModel != "" && !extractOnly {
				p, err := llm.ProviderFromEnv(providerEnvFor(judgeModel))
				if err != nil {
					return fmt.Errorf("judge provider for %q: %w", judgeModel, err)
				}
				judgeProvider = p
				cfg.Judge = &lme.JudgeConfig{
					Provider:   p,
					Model:      judgeModel,
					MaxRetries: 3,
					Timeout:    judgeTimeout,
				}
			}

			// Wire the reader / actor when we'll run it in-process:
			// default (no actor-endpoint), or actor-endpoint with
			// style=retrieve-only (daemon retrieves only; we read +
			// judge locally). Full-style actor-endpoint mode lets the
			// daemon do its own reading.
			needsReader := !disableReader && actorModel != "" && !extractOnly &&
				(actorEndpoint == "" || strings.EqualFold(actorEndpointStyle, "retrieve-only"))
			var readerProvider llm.Provider
			if needsReader {
				p, err := llm.ProviderFromEnv(providerEnvFor(actorModel))
				if err != nil {
					return fmt.Errorf("actor provider for %q: %w", actorModel, err)
				}
				readerProvider = p
				cfg.Reader = &lme.ReaderConfig{
					Provider: p,
					Model:    actorModel,
				}
			}

			// Wire the replay extraction provider when the caller asked
			// for replay or agentic mode. Agentic mode still needs a
			// populated corpus, so replay fires first then the agent
			// loop drives retrieval. Uses the env-configured provider
			// pinned to the extraction model.
			var extractProvider llm.Provider
			if cfg.IngestMode == "replay" {
				em := extractModel
				if em == "" {
					em = lme.DefaultReplayExtractModel
				}
				p, err := llm.ProviderFromEnv(providerEnvFor(em))
				if err != nil {
					return fmt.Errorf("replay extract provider for %q: %w", em, err)
				}
				extractProvider = p
				cfg.Provider = p
				if contextualise {
					cacheDir := contextualCacheDir
					if cacheDir == "" {
						cacheDir = defaultEvalContextualiseCacheDir()
					}
					cfg.Contextualiser = memory.NewContextualiser(memory.ContextualiserConfig{
						Provider: extractProvider,
						Model:    em,
						CacheDir: cacheDir,
					})
					if cfg.Contextualiser == nil {
						return fmt.Errorf("build contextualiser: provider unavailable")
					}
					fmt.Fprintf(cmd.ErrOrStderr(), "memory contextualiser enabled (model=%s)\n", cfg.Contextualiser.ModelName())
				}
			}

			result, runErr := lme.Run(ctx, cfg)

			// Close providers best-effort regardless of run outcome.
			if judgeProvider != nil {
				_ = judgeProvider.Close()
			}
			if readerProvider != nil {
				_ = readerProvider.Close()
			}
			if extractProvider != nil {
				_ = extractProvider.Close()
			}

			if runErr != nil {
				return runErr
			}

			if cfg.MaxCostUSD > 0 && result.CostAccounting.TotalUSD > cfg.MaxCostUSD {
				fmt.Fprintf(cmd.ErrOrStderr(), "warning: run cost $%.2f exceeded cap $%.2f\n",
					result.CostAccounting.TotalUSD, cfg.MaxCostUSD)
			}

			if err := writeResult(cmd.OutOrStdout(), outputPath, result); err != nil {
				return err
			}

			if manifestPath != "" {
				manifest := lme.BuildRunManifest(result, cfg, judgeModel)
				if err := lme.SaveManifest(manifest, manifestPath); err != nil {
					return fmt.Errorf("save manifest: %w", err)
				}
				if slackWebhook != "" {
					if err := lme.SlackReport(ctx, slackWebhook, result, &manifest); err != nil {
						fmt.Fprintf(cmd.ErrOrStderr(), "slack report: %v\n", err)
					}
				}
			} else if slackWebhook != "" {
				if err := lme.SlackReport(ctx, slackWebhook, result, nil); err != nil {
					fmt.Fprintf(cmd.ErrOrStderr(), "slack report: %v\n", err)
				}
			}

			printSummary(cmd.OutOrStdout(), result)
			return nil
		},
	}

	cmd.Flags().StringVar(&datasetPath, "dataset", "", "Path to the LME dataset JSON (e.g. longmemeval_s.json)")
	cmd.Flags().IntVar(&sampleSize, "sample-size", 0, "Stratified subsample size (0 = full dataset)")
	cmd.Flags().Int64Var(&seed, "seed", 0, "Deterministic seed for sampling and bootstrap CI")
	cmd.Flags().IntVar(&concurrency, "concurrency", 16, "Parallel question workers (clamped to [1,64]). 16 is the local default for the tri-SDK orchestrator.")
	cmd.Flags().StringVar(&ingestMode, "ingest-mode", "replay", "Ingest mode (bulk|replay|none|agentic). 'agentic' implies replay + agent loop. 'none' assumes the brain is already populated out-of-band.")
	cmd.Flags().StringVar(&expectedSHA, "expected-sha", "", "Optional SHA256 of the dataset file")
	cmd.Flags().Float64Var(&maxCostUSD, "max-cost-usd", 20.0, "Soft cap on total run cost")
	cmd.Flags().StringVar(&judgeModel, "judge", "claude-haiku-4-5", "LLM judge model (override via JB_LME_JUDGE_MODEL)")
	cmd.Flags().StringVar(&actorModel, "actor", "gpt-4o", "LLM actor/reader model (override via JB_LME_ACTOR_MODEL). Ignored when --actor-endpoint is set.")
	cmd.Flags().StringVar(&extractModel, "extract-model", lme.DefaultReplayExtractModel, "LLM model used for replay extraction")
	cmd.Flags().IntVar(&replayConcurrency, "replay-concurrency", 16, "Parallel extraction workers during replay ingest. Tuned for cheap extract models with rate limits above 16 RPS.")
	cmd.Flags().IntVar(&maxIterations, "max-iterations", 0, "Per-question tool-call budget in agentic mode (0 = package default)")
	cmd.Flags().DurationVar(&questionTimeout, "question-timeout", 0, "Per-question wall-clock cap in agentic mode (0 = package default)")
	cmd.Flags().StringVar(&outputPath, "output", "", "Write the full LMEResult to this path as JSON (default: stdout)")
	cmd.Flags().StringVar(&manifestPath, "manifest", "", "Write a RunManifest alongside the result")
	cmd.Flags().StringVar(&slackWebhook, "slack-webhook", "", "Slack Incoming Webhook URL to post the summary to")
	cmd.Flags().BoolVar(&disableReader, "no-reader", false, "Skip the LLM reader/actor step and feed raw retrieval to the judge")
	cmd.Flags().DurationVar(&judgeTimeout, "judge-timeout", 0, "Per-question judge timeout (0 = no cap)")
	cmd.Flags().BoolVar(&extractOnly, "extract-only", false, "Run the extraction phase only (seed bulk + replay), write the manifest, then exit. Requires --brain-cache.")
	cmd.Flags().StringVar(&brainCache, "brain-cache", "", "Persistent brain cache directory. Required when --extract-only is set. Shared by downstream daemons.")
	cmd.Flags().BoolVar(&contextualise, "contextualise", false, "Enable replay contextualisation so extracted facts carry a situating prefix.")
	cmd.Flags().StringVar(&contextualCacheDir, "contextualise-cache-dir", "", "Override the replay contextualiser cache directory.")
	cmd.Flags().StringVar(&actorEndpoint, "actor-endpoint", "", "HTTP endpoint of a spec-compliant memory daemon. When set, the runner skips in-process retrieval and routes through the daemon. See --actor-endpoint-style for read vs retrieve-only mode.")
	cmd.Flags().StringVar(&actorBrain, "actor-brain", "", "Brain id the actor endpoint should query (defaults to 'eval-lme').")
	cmd.Flags().StringVar(&actorEndpointStyle, "actor-endpoint-style", "retrieve-only", "How to use --actor-endpoint: 'full' posts each question to /ask (daemon retrieves + reads), 'retrieve-only' posts to /search and runs the augmented CoT reader + judge in-process (recommended for apples-to-apples cross-SDK benchmarking).")
	cmd.Flags().IntVar(&actorTopK, "actor-topk", 20, "Chunks to request from the actor daemon's /search endpoint per question (retrieve-only mode only). LongMemEval multi-session questions reference 2-6 sessions so top-5 BM25 frequently misses supporting evidence.")
	cmd.Flags().IntVar(&actorCandidateK, "actor-candidatek", 0, "Per-leg candidate slate size to request from the actor daemon during retrieve-only runs. Zero defers to the daemon default.")
	cmd.Flags().IntVar(&actorRerankTopN, "actor-rerank-topn", 0, "Post-fusion head width to rerank on the actor daemon during retrieve-only runs. Zero defers to the daemon default.")
	cmd.Flags().StringVar(&actorScope, "actor-scope", "", "Optional retrieve-only daemon scope filter, for example 'project' to search replay memory facts only.")
	cmd.Flags().StringVar(&actorProject, "actor-project", "", "Optional retrieve-only daemon project slug filter, for example 'eval-lme' for memory/project/eval-lme.")
	cmd.Flags().StringVar(&actorPathPrefix, "actor-path-prefix", "", "Optional retrieve-only daemon path prefix filter.")

	return cmd
}

func defaultEvalContextualiseCacheDir() string {
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return ""
	}
	return filepath.Join(home, ".local", "state", "jeffs-brain", "evals", "contextualise-cache")
}

func resolveJudgeModel(flagValue string) string {
	if v := strings.TrimSpace(os.Getenv("JB_LME_JUDGE_MODEL")); v != "" {
		return v
	}
	return flagValue
}

func resolveActorModel(flagValue string) string {
	if v := strings.TrimSpace(os.Getenv("JB_LME_ACTOR_MODEL")); v != "" {
		return v
	}
	return flagValue
}

// providerEnvFor returns a [llm.Getenv] that surfaces the requested model
// name via JB_LLM_MODEL while otherwise delegating to the process
// environment. The llm package reads JB_LLM_MODEL to seed the default
// model on auto-detected providers (OpenAI, Anthropic, Ollama), so this
// wrapper lets the CLI pass a per-stage model without mutating the
// process environment.
func providerEnvFor(model string) llm.Getenv {
	return func(key string) string {
		if key == llm.EnvModel {
			return model
		}
		return os.Getenv(key)
	}
}

// writeResult serialises the run result as indented JSON to outputPath
// or stdout when outputPath is empty.
func writeResult(stdout io.Writer, outputPath string, result *lme.LMEResult) error {
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal result: %w", err)
	}
	if outputPath == "" {
		_, err := stdout.Write(data)
		if err != nil {
			return fmt.Errorf("write result: %w", err)
		}
		_, _ = stdout.Write([]byte("\n"))
		return nil
	}
	if err := os.WriteFile(outputPath, data, 0o644); err != nil {
		return fmt.Errorf("write result: %w", err)
	}
	return nil
}

func printSummary(w io.Writer, result *lme.LMEResult) {
	fmt.Fprintf(w, "\nLME summary:\n")
	fmt.Fprintf(w, "  Questions run:   %d\n", result.QuestionsRun)
	fmt.Fprintf(w, "  Overall score:   %.3f\n", result.OverallScore)
	fmt.Fprintf(w, "  Task-avg score:  %.3f\n", result.TaskAvgScore)
	fmt.Fprintf(w, "  Exact-match:     %.3f\n", result.ExactMatchScore)
	fmt.Fprintf(w, "  Latency p50/p95: %dms / %dms\n", result.LatencyP50Ms, result.LatencyP95Ms)
	fmt.Fprintf(w, "  Cost USD total:  $%.4f (ingest $%.4f / agent $%.4f / judge $%.4f)\n",
		result.CostAccounting.TotalUSD,
		result.CostAccounting.IngestUSD,
		result.CostAccounting.AgentUSD,
		result.CostAccounting.JudgeUSD,
	)
}
