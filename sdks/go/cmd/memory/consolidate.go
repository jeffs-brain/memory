// SPDX-License-Identifier: Apache-2.0

package main

import "github.com/spf13/cobra"

func consolidateCmd() *cobra.Command {
	return stubCmd("consolidate", "Consolidate memories, collapsing superseded entries")
}
