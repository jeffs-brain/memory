// SPDX-License-Identifier: Apache-2.0

// Command memory-mcp is the Go implementation of the Jeffs Brain
// memory MCP stdio wrapper. It exposes the eleven canonical memory_*
// tools defined in spec/MCP-TOOLS.md over MCP stdio, dispatching each
// call to either an in-process memory pipeline (local mode, no
// JB_TOKEN) or a remote memory serve (hosted mode, JB_TOKEN set).
package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/jeffs-brain/memory/go/cmd/memory-mcp/tools"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

const (
	serverName    = "@jeffs-brain/memory-mcp"
	serverVersion = "0.0.1"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "memory-mcp: %v\n", err)
		os.Exit(1)
	}
}

// run bootstraps the server. Extracted from main so tests can exercise
// the same code path without spawning a subprocess.
func run() error {
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))
	cfg, err := ResolveConfig(os.Getenv)
	if err != nil {
		return fmt.Errorf("resolving config: %w", err)
	}
	client, err := NewMemoryClient(cfg, logger)
	if err != nil {
		return fmt.Errorf("constructing memory client: %w", err)
	}
	defer func() { _ = client.Close() }()

	server := mcp.NewServer(&mcp.Implementation{Name: serverName, Version: serverVersion}, nil)
	tools.Register(server, toolAdapter{client: client})

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := server.Run(ctx, &mcp.StdioTransport{}); err != nil {
		return fmt.Errorf("server run: %w", err)
	}
	return nil
}

// toolAdapter adapts the package-local [MemoryClient] to the
// [tools.MemoryClient] interface. The local package types are a
// superset of the tool types (e.g. they carry hosted-mode wiring) so
// each method translates between the two argument shapes.
type toolAdapter struct {
	client MemoryClient
}

func (a toolAdapter) Remember(ctx context.Context, args tools.RememberArgs) (map[string]any, error) {
	return a.client.Remember(ctx, RememberArgs{
		Content: args.Content,
		Title:   args.Title,
		Brain:   args.Brain,
		Tags:    args.Tags,
		Path:    args.Path,
	})
}

func (a toolAdapter) Search(ctx context.Context, args tools.SearchArgs) (map[string]any, error) {
	return a.client.Search(ctx, SearchArgs{
		Query: args.Query,
		Brain: args.Brain,
		TopK:  args.TopK,
		Scope: args.Scope,
		Sort:  args.Sort,
	})
}

func (a toolAdapter) Recall(ctx context.Context, args tools.RecallArgs) (map[string]any, error) {
	return a.client.Recall(ctx, RecallArgs{
		Query:     args.Query,
		Brain:     args.Brain,
		Scope:     args.Scope,
		SessionID: args.SessionID,
		TopK:      args.TopK,
	})
}

func (a toolAdapter) Ask(ctx context.Context, args tools.AskArgs, progress tools.ProgressEmitter) (map[string]any, error) {
	return a.client.Ask(ctx, AskArgs{
		Query: args.Query,
		Brain: args.Brain,
		TopK:  args.TopK,
	}, ProgressEmitter(progress))
}

func (a toolAdapter) IngestFile(ctx context.Context, args tools.IngestFileArgs, progress tools.ProgressEmitter) (map[string]any, error) {
	return a.client.IngestFile(ctx, IngestFileArgs{
		Path:  args.Path,
		Brain: args.Brain,
		As:    args.As,
	}, ProgressEmitter(progress))
}

func (a toolAdapter) IngestURL(ctx context.Context, args tools.IngestURLArgs, progress tools.ProgressEmitter) (map[string]any, error) {
	return a.client.IngestURL(ctx, IngestURLArgs{
		URL:   args.URL,
		Brain: args.Brain,
	}, ProgressEmitter(progress))
}

func (a toolAdapter) Extract(ctx context.Context, args tools.ExtractArgs, progress tools.ProgressEmitter) (map[string]any, error) {
	messages := make([]ExtractMessage, 0, len(args.Messages))
	for _, m := range args.Messages {
		messages = append(messages, ExtractMessage{Role: m.Role, Content: m.Content})
	}
	return a.client.Extract(ctx, ExtractArgs{
		Messages:  messages,
		Brain:     args.Brain,
		ActorID:   args.ActorID,
		SessionID: args.SessionID,
	}, ProgressEmitter(progress))
}

func (a toolAdapter) Reflect(ctx context.Context, args tools.ReflectArgs, progress tools.ProgressEmitter) (map[string]any, error) {
	return a.client.Reflect(ctx, ReflectArgs{
		SessionID: args.SessionID,
		Brain:     args.Brain,
	}, ProgressEmitter(progress))
}

func (a toolAdapter) Consolidate(ctx context.Context, args tools.ConsolidateArgs, progress tools.ProgressEmitter) (map[string]any, error) {
	return a.client.Consolidate(ctx, ConsolidateArgs{Brain: args.Brain}, ProgressEmitter(progress))
}

func (a toolAdapter) CreateBrain(ctx context.Context, args tools.CreateBrainArgs) (map[string]any, error) {
	return a.client.CreateBrain(ctx, CreateBrainArgs{
		Name:       args.Name,
		Slug:       args.Slug,
		Visibility: args.Visibility,
	})
}

func (a toolAdapter) ListBrains(ctx context.Context) (map[string]any, error) {
	return a.client.ListBrains(ctx)
}
