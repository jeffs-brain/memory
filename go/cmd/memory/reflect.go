// SPDX-License-Identifier: Apache-2.0

package main

import "github.com/spf13/cobra"

func reflectCmd() *cobra.Command {
	return stubCmd("reflect", "Run the memory reflection pass")
}
