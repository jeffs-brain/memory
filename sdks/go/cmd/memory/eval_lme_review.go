// SPDX-License-Identifier: Apache-2.0

package main

import (
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/spf13/cobra"

	"github.com/jeffs-brain/memory/go/eval/lme"
)

// evalLmeReviewCmd drives the before/after diff tool. Given two LME
// report JSONs it emits a markdown summary highlighting regressions,
// improvements, and verdict churn. The output is ideal for PR review.
//
//	memory eval lme review --before runs/before.json --after runs/after.json --output diff.md
func evalLmeReviewCmd() *cobra.Command {
	var (
		beforePath string
		afterPath  string
		outputPath string
	)

	cmd := &cobra.Command{
		Use:   "review",
		Short: "Diff two LME reports as markdown",
		Long: "Compares two LME run reports (report.json files) and emits a " +
			"markdown summary highlighting regressions, improvements, and " +
			"verdict churn. Useful as a PR review artefact after a benchmark " +
			"sweep on both sides of a change.",
		RunE: func(cmd *cobra.Command, args []string) error {
			if beforePath == "" {
				return errors.New("--before is required")
			}
			if afterPath == "" {
				return errors.New("--after is required")
			}

			before, err := lme.LoadReport(beforePath)
			if err != nil {
				return fmt.Errorf("load before: %w", err)
			}
			after, err := lme.LoadReport(afterPath)
			if err != nil {
				return fmt.Errorf("load after: %w", err)
			}

			md, err := lme.DiffReports(before, after)
			if err != nil {
				return fmt.Errorf("diff reports: %w", err)
			}

			return writeDiffOutput(cmd.OutOrStdout(), outputPath, md)
		},
	}

	cmd.Flags().StringVar(&beforePath, "before", "", "Path to the baseline report.json")
	cmd.Flags().StringVar(&afterPath, "after", "", "Path to the new report.json")
	cmd.Flags().StringVar(&outputPath, "output", "", "Write the markdown diff to this path (default: stdout)")

	return cmd
}

func writeDiffOutput(stdout io.Writer, outputPath, md string) error {
	if outputPath == "" {
		if _, err := io.WriteString(stdout, md); err != nil {
			return fmt.Errorf("write diff: %w", err)
		}
		if !hasTrailingNewline(md) {
			_, _ = io.WriteString(stdout, "\n")
		}
		return nil
	}
	if err := os.WriteFile(outputPath, []byte(md), 0o644); err != nil {
		return fmt.Errorf("write diff: %w", err)
	}
	return nil
}

func hasTrailingNewline(s string) bool {
	return len(s) > 0 && s[len(s)-1] == '\n'
}
