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
		Long: "memory is the reference Go CLI for the implemented Go " +
			"daemon and native eval commands. Planned local workflow " +
			"commands remain callable as explicit not-yet-implemented " +
			"placeholders, but are hidden from help until they ship.",
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
		evalCmd(),
	)

	return root
}

func main() {
	if err := rootCmd().Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// stubCmd returns a hidden cobra command whose Run prints "not yet implemented".
// Used for planned subcommands that are not part of the production CLI surface.
func stubCmd(use, short string) *cobra.Command {
	return &cobra.Command{
		Use:    use,
		Short:  short,
		Hidden: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			fmt.Fprintf(cmd.OutOrStdout(), "%s: not yet implemented\n", use)
			return nil
		},
	}
}
