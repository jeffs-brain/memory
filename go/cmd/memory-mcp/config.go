// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Mode enumerates the two dispatch modes the wrapper supports.
type Mode string

const (
	// ModeLocal drives an in-process memory pipeline rooted at
	// [Config.BrainRoot]. Selected when JB_TOKEN is unset.
	ModeLocal Mode = "local"
	// ModeHosted dispatches every tool call to a remote memory serve.
	// Selected when JB_TOKEN is set.
	ModeHosted Mode = "hosted"
)

// Config captures the resolved runtime configuration for the MCP wrapper.
//
// Local mode fields (BrainRoot) and hosted mode fields (Endpoint, Token)
// are mutually exclusive at runtime but modelled on the same struct so
// the dispatcher can inspect Mode and pick the right branch.
type Config struct {
	Mode         Mode
	BrainRoot    string
	Endpoint     string
	Token        string
	DefaultBrain string
}

const (
	defaultEndpoint = "https://api.jeffsbrain.com"
	defaultHomeDir  = ".jeffs-brain"
)

// ResolveConfig reads environment variables and returns a populated
// [Config]. The env argument is a [os.Getenv]-shaped lookup so callers
// can inject a controlled environment in tests.
func ResolveConfig(env func(string) string) (Config, error) {
	if env == nil {
		env = os.Getenv
	}
	defaultBrain := strings.TrimSpace(env("JB_BRAIN"))
	token := strings.TrimSpace(env("JB_TOKEN"))
	if token != "" {
		endpoint := strings.TrimSpace(env("JB_ENDPOINT"))
		if endpoint == "" {
			endpoint = defaultEndpoint
		}
		return Config{
			Mode:         ModeHosted,
			Endpoint:     endpoint,
			Token:        token,
			DefaultBrain: defaultBrain,
		}, nil
	}
	root := strings.TrimSpace(env("JB_HOME"))
	if root == "" {
		home, err := userHomeDir(env)
		if err != nil {
			return Config{}, fmt.Errorf("memory-mcp: resolving JB_HOME: %w", err)
		}
		root = filepath.Join(home, defaultHomeDir)
	}
	abs, err := filepath.Abs(root)
	if err != nil {
		return Config{}, fmt.Errorf("memory-mcp: resolving JB_HOME path: %w", err)
	}
	return Config{
		Mode:         ModeLocal,
		BrainRoot:    abs,
		DefaultBrain: defaultBrain,
	}, nil
}

// userHomeDir returns $HOME via the injected env lookup, falling back
// to [os.UserHomeDir] when the env var is unset.
func userHomeDir(env func(string) string) (string, error) {
	if h := strings.TrimSpace(env("HOME")); h != "" {
		return h, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return home, nil
}
