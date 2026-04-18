// SPDX-License-Identifier: Apache-2.0

// Command memory is the reference CLI + daemon for the Jeffs Brain Go SDK.
package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

// rootCmd is the top-level cobra command.
func rootCmd() *cobra.Command {
	root := &cobra.Command{
		Use:   "memory",
		Short: "Jeffs Brain memory CLI + daemon",
		Long: "memory is the reference Go CLI for interacting with a Jeffs " +
			"Brain instance. Subcommands mirror the capabilities exposed " +
			"by the HTTP protocol and MCP tools.",
		SilenceUsage: true,
	}

	root.AddCommand(
		versionCmd(),
		serveCmd(),
		initCmd(),
		ingestCmd(),
		searchCmd(),
		askCmd(),
		rememberCmd(),
		recallCmd(),
		reflectCmd(),
		consolidateCmd(),
		createBrainCmd(),
		listBrainsCmd(),
	)

	return root
}

func main() {
	if err := rootCmd().Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// stubCmd returns a cobra command whose Run prints "not yet implemented".
// Used for every subcommand except version and serve.
func stubCmd(use, short string) *cobra.Command {
	return &cobra.Command{
		Use:   use,
		Short: short,
		RunE: func(cmd *cobra.Command, args []string) error {
			fmt.Fprintf(cmd.OutOrStdout(), "%s: not yet implemented\n", use)
			return nil
		},
	}
}
