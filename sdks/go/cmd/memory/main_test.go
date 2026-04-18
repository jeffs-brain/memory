// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"testing"
	"time"
)

// TestSmokeVersion verifies `memory version` runs and prints the version.
func TestSmokeVersion(t *testing.T) {
	var stdout bytes.Buffer
	cmd := rootCmd()
	cmd.SetArgs([]string{"version"})
	cmd.SetOut(&stdout)
	cmd.SetErr(&stdout)
	if err := cmd.Execute(); err != nil {
		t.Fatalf("version execute: %v", err)
	}
	if got := strings.TrimSpace(stdout.String()); got != version {
		t.Fatalf("version output = %q, want %q", got, version)
	}
}

// TestSmokeServe verifies `memory serve` binds a port and serves /healthz
// then shuts down cleanly on context cancel.
func TestSmokeServe(t *testing.T) {
	addr, err := findFreePort()
	if err != nil {
		t.Fatalf("find free port: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	root := t.TempDir()
	cmd := rootCmd()
	cmd.SetArgs([]string{"serve", "--addr", addr, "--root", root})
	cmd.SetOut(io.Discard)
	cmd.SetErr(io.Discard)
	cmd.SetContext(ctx)

	runErr := make(chan error, 1)
	go func() {
		runErr <- cmd.Execute()
	}()

	// Poll /healthz until the listener is accepting.
	base := "http://" + addr
	client := &http.Client{Timeout: 500 * time.Millisecond}
	deadline := time.Now().Add(3 * time.Second)
	var resp *http.Response
	for time.Now().Before(deadline) {
		resp, err = client.Get(base + "/healthz")
		if err == nil {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if err != nil {
		cancel()
		t.Fatalf("healthz: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("healthz status = %d, want 200", resp.StatusCode)
	}

	var body map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		t.Fatalf("decode healthz body: %v", err)
	}
	if ok, _ := body["ok"].(bool); !ok {
		t.Fatalf("healthz body = %v, want ok=true", body)
	}

	// Probe a protocol endpoint against an unprovisioned brain to
	// verify it returns 404 Problem+JSON now that real handlers are
	// wired in.
	protoResp, err := client.Get(base + "/v1/brains/demo/documents/read?path=a.md")
	if err != nil {
		cancel()
		t.Fatalf("protocol probe: %v", err)
	}
	defer protoResp.Body.Close()
	if protoResp.StatusCode != http.StatusNotFound {
		t.Fatalf("protocol endpoint status = %d, want 404", protoResp.StatusCode)
	}
	if ct := protoResp.Header.Get("Content-Type"); !strings.Contains(ct, "problem+json") {
		t.Fatalf("protocol content-type = %q, want problem+json", ct)
	}

	cancel()
	select {
	case err := <-runErr:
		if err != nil {
			t.Fatalf("serve exit: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("serve did not exit after cancel")
	}
}

// findFreePort returns "127.0.0.1:<port>" with an OS-chosen free port.
func findFreePort() (string, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return "", err
	}
	defer l.Close()
	return fmt.Sprintf("127.0.0.1:%d", l.Addr().(*net.TCPAddr).Port), nil
}
