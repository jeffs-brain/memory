// SPDX-License-Identifier: Apache-2.0

package main

import "github.com/spf13/cobra"

func searchCmd() *cobra.Command {
	return stubCmd("search", "Run a hybrid search against a brain")
}
