// SPDX-License-Identifier: Apache-2.0

package main

import "github.com/spf13/cobra"

func askCmd() *cobra.Command {
	return stubCmd("ask", "Ask a question backed by hybrid retrieval")
}
