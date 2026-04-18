// SPDX-License-Identifier: Apache-2.0

package main

import "github.com/spf13/cobra"

func initCmd() *cobra.Command {
	return stubCmd("init", "Initialise a new brain at the configured root")
}

func createBrainCmd() *cobra.Command {
	return stubCmd("create-brain", "Create a new brain via the HTTP daemon")
}

func listBrainsCmd() *cobra.Command {
	return stubCmd("list-brains", "List brains known to the HTTP daemon")
}
