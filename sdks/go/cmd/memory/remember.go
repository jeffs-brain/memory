// SPDX-License-Identifier: Apache-2.0

package main

import "github.com/spf13/cobra"

func rememberCmd() *cobra.Command {
	return stubCmd("remember", "Record a new memory observation")
}
