// SPDX-License-Identifier: Apache-2.0

package httpd

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"golang.org/x/sync/errgroup"
)

// RouteRegistrar attaches handlers to the daemon's shared mux.
type RouteRegistrar func(mux *http.ServeMux)

// ShutdownHook is a cleanup callback invoked during Server.Run teardown.
type ShutdownHook func(ctx context.Context) error

// Server is the HTTP daemon behind `memory serve`. It ports the structure
// of jeff's server.Server pared down to the HTTP-only surface.
type Server struct {
	addr      string
	authToken string
	log       *slog.Logger
	mu        sync.Mutex
	routers   []RouteRegistrar
	shutdowns []ShutdownHook
}

// NewServer returns a Server bound to addr.
func NewServer(addr string, log *slog.Logger) *Server {
	if addr == "" {
		addr = ":8080"
	}
	if log == nil {
		log = slog.Default()
	}
	return &Server{addr: addr, log: log}
}

// SetAuthToken installs a shared-secret bearer token. When non-empty
// every request except /healthz must carry "Authorization: Bearer
// <token>". Pass an empty string to disable auth.
func (s *Server) SetAuthToken(token string) {
	s.mu.Lock()
	s.authToken = token
	s.mu.Unlock()
}

// AddRouteRegistrar queues a registrar. Registrars run in registration
// order against the shared mux when Run starts.
func (s *Server) AddRouteRegistrar(r RouteRegistrar) {
	if r == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.routers = append(s.routers, r)
}

// AddShutdownHook queues a cleanup callback. Hooks run in LIFO order when
// Run returns.
func (s *Server) AddShutdownHook(h ShutdownHook) {
	if h == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.shutdowns = append(s.shutdowns, h)
}

// Addr returns the server's bind address.
func (s *Server) Addr() string { return s.addr }

// Run starts the HTTP listener and blocks until ctx is cancelled.
func (s *Server) Run(ctx context.Context) error {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": true})
	})

	s.mu.Lock()
	routers := append([]RouteRegistrar(nil), s.routers...)
	hooks := append([]ShutdownHook(nil), s.shutdowns...)
	token := s.authToken
	s.mu.Unlock()

	for _, r := range routers {
		r(mux)
	}

	srv := &http.Server{
		Addr:              s.addr,
		Handler:           LogMiddleware(s.log, AuthMiddleware(token, mux)),
		ReadHeaderTimeout: 10 * time.Second,
	}

	g, gctx := errgroup.WithContext(ctx)

	g.Go(func() error {
		s.log.Info("http server listening", "addr", s.addr, "routes", len(routers)+1)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			return err
		}
		return nil
	})

	g.Go(func() error {
		<-gctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		return srv.Shutdown(shutdownCtx)
	})

	err := g.Wait()
	s.runShutdownHooks(hooks)
	return err
}

func (s *Server) runShutdownHooks(hooks []ShutdownHook) {
	for i := len(hooks) - 1; i >= 0; i-- {
		hook := hooks[i]
		hookCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		if err := hook(hookCtx); err != nil {
			s.log.Warn("shutdown hook failed", "err", err)
		}
		cancel()
	}
}
