// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/spf13/cobra"

	"github.com/jeffs-brain/memory/go/internal/httpd"
)

// Environment variable names recognised by the daemon. Mirror the
// JB_ namespace from the rollout plan so callers can feed identical
// env vars to both this binary and the TS SDK.
const (
	envHome                  = "JB_HOME"
	envAddr                  = "JB_ADDR"
	envAuthToken             = "JB_AUTH_TOKEN"
	envIngestRoot            = "JB_INGEST_ROOT"
	envContextualise         = "JB_CONTEXTUALISE"
	envContextualiseCacheDir = "JB_CONTEXTUALISE_CACHE_DIR"
)

func serveCmd() *cobra.Command {
	var (
		addr               string
		root               string
		token              string
		ingestRoot         string
		contextualise      bool
		contextualiseCache string
	)
	cmd := &cobra.Command{
		Use:   "serve",
		Short: "Run the memory HTTP daemon",
		Long: "serve starts the HTTP daemon that speaks the Jeffs Brain " +
			"wire protocol documented at spec/PROTOCOL.md.",
		RunE: func(cmd *cobra.Command, args []string) error {
			if root == "" {
				root = os.Getenv(envHome)
			}
			if token == "" {
				token = os.Getenv(envAuthToken)
			}
			if ingestRoot == "" {
				ingestRoot = os.Getenv(envIngestRoot)
			}
			bindAddr, err := httpd.ResolveBindAddr(addr, token)
			if err != nil {
				return err
			}
			resolvedIngestRoot, err := resolveIngestRoot(ingestRoot)
			if err != nil {
				return err
			}
			if cmd.Flags().Changed("contextualise") {
				if contextualise {
					_ = os.Setenv(envContextualise, "1")
				} else {
					_ = os.Unsetenv(envContextualise)
				}
			}
			if cmd.Flags().Changed("contextualise-cache-dir") {
				if strings.TrimSpace(contextualiseCache) == "" {
					_ = os.Unsetenv(envContextualiseCacheDir)
				} else {
					_ = os.Setenv(envContextualiseCacheDir, contextualiseCache)
				}
			}
			log := slog.New(slog.NewJSONHandler(os.Stdout, nil))
			slog.SetDefault(log)

			ctx, cancel := signal.NotifyContext(cmd.Context(), os.Interrupt, syscall.SIGTERM)
			defer cancel()

			daemon, err := NewDaemon(ctx, root, token, log)
			if err != nil {
				return err
			}
			defer func() { _ = daemon.Close() }()
			daemon.IngestRoot = resolvedIngestRoot

			srv := httpd.NewServer(bindAddr, log)
			if token != "" {
				srv.SetAuthToken(token)
			}
			srv.AddRouteRegistrar(daemon.RegisterRoutes)
			srv.AddShutdownHook(func(_ context.Context) error {
				return daemon.Close()
			})

			return srv.Run(ctx)
		},
	}
	defaultAddr := os.Getenv(envAddr)
	if defaultAddr == "" {
		defaultAddr = httpd.DefaultAddr
	}
	cmd.Flags().StringVar(&addr, "addr", defaultAddr, "address to bind (host:port); a non-loopback host requires an auth token")
	cmd.Flags().StringVar(&root, "root", "", "JB_HOME directory (default $JB_HOME or ~/.jeffs-brain)")
	cmd.Flags().StringVar(&token, "auth-token", "", "shared bearer token (default $JB_AUTH_TOKEN, optional)")
	cmd.Flags().StringVar(&ingestRoot, "ingest-root", "", "directory ingest/file may read server-side paths from (default $JB_INGEST_ROOT; unset disables path ingest)")
	cmd.Flags().BoolVar(&contextualise, "contextualise", false, "Enable live extraction contextualisation so extracted facts carry a situating prefix.")
	cmd.Flags().StringVar(&contextualiseCache, "contextualise-cache-dir", "", "Optional cache directory for live extraction contextualisation.")
	return cmd
}

// RegisterRoutes wires every endpoint defined by the daemon onto mux.
// The shape mirrors spec/PROTOCOL.md plus the brain-management endpoints
// in spec/MCP-TOOLS.md.
func (d *Daemon) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /v1/brains", d.handleListBrains)
	mux.HandleFunc("POST /v1/brains", d.handleCreateBrain)
	mux.HandleFunc("GET /v1/brains/{brainId}", d.handleGetBrain)
	mux.HandleFunc("DELETE /v1/brains/{brainId}", d.handleDeleteBrain)

	mux.HandleFunc("GET /v1/brains/{brainId}/documents/read", d.handleDocRead)
	mux.HandleFunc("GET /v1/brains/{brainId}/documents/stat", d.handleDocStat)
	mux.HandleFunc("GET /v1/brains/{brainId}/documents", d.handleDocListOrHead)
	mux.HandleFunc("HEAD /v1/brains/{brainId}/documents", d.handleDocHead)
	mux.HandleFunc("PUT /v1/brains/{brainId}/documents", d.handleDocPut)
	mux.HandleFunc("POST /v1/brains/{brainId}/documents/append", d.handleDocAppend)
	mux.HandleFunc("DELETE /v1/brains/{brainId}/documents", d.handleDocDelete)
	mux.HandleFunc("POST /v1/brains/{brainId}/documents/rename", d.handleDocRename)
	mux.HandleFunc("POST /v1/brains/{brainId}/documents/batch-ops", d.handleDocBatch)

	mux.HandleFunc("POST /v1/brains/{brainId}/search", d.handleSearch)
	mux.HandleFunc("POST /v1/brains/{brainId}/ask", d.handleAsk)

	mux.HandleFunc("POST /v1/brains/{brainId}/ingest/file", d.handleIngestFile)
	mux.HandleFunc("POST /v1/brains/{brainId}/ingest/url", d.handleIngestURL)

	mux.HandleFunc("POST /v1/brains/{brainId}/remember", d.handleRemember)
	mux.HandleFunc("POST /v1/brains/{brainId}/recall", d.handleRecall)
	mux.HandleFunc("POST /v1/brains/{brainId}/extract", d.handleExtract)
	mux.HandleFunc("POST /v1/brains/{brainId}/reflect", d.handleReflect)
	mux.HandleFunc("POST /v1/brains/{brainId}/consolidate", d.handleConsolidate)

	mux.HandleFunc("GET /v1/brains/{brainId}/events", d.handleEvents)
}
