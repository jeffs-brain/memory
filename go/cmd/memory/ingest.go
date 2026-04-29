// SPDX-License-Identifier: Apache-2.0

package main

import "github.com/spf13/cobra"

func ingestCmd() *cobra.Command {
	return stubCmd("ingest", "Ingest raw content into a brain")
}
