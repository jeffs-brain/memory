// SPDX-License-Identifier: Apache-2.0

package main

import "github.com/spf13/cobra"

func recallCmd() *cobra.Command {
	return stubCmd("recall", "Recall memories matching a query")
}
