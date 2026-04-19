// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"errors"
	"net/http"
	"sort"
	"time"

	"github.com/jeffs-brain/memory/go/internal/httpd"
)

type brainSummary struct {
	BrainID     string    `json:"brainId"`
	Description string    `json:"description,omitempty"`
	Created     time.Time `json:"created"`
}

type createBrainRequest struct {
	BrainID     string `json:"brainId"`
	Description string `json:"description,omitempty"`
}

func (d *Daemon) handleListBrains(w http.ResponseWriter, r *http.Request) {
	ids, err := d.Brains.List()
	if err != nil {
		httpd.InternalError(w, err.Error())
		return
	}
	sort.Strings(ids)
	out := make([]brainSummary, 0, len(ids))
	for _, id := range ids {
		out = append(out, brainSummary{BrainID: id})
	}
	writeJSON(w, http.StatusOK, map[string]any{"items": out})
}

func (d *Daemon) handleGetBrain(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("brainId")
	if id == "" {
		httpd.ValidationError(w, "missing brainId")
		return
	}
	if !d.brainExists(id) {
		httpd.NotFound(w, "brain not found: "+id)
		return
	}
	writeJSON(w, http.StatusOK, brainSummary{BrainID: id})
}

func (d *Daemon) handleCreateBrain(w http.ResponseWriter, r *http.Request) {
	var req createBrainRequest
	if err := decodeJSONBody(r, &req, 64*1024); err != nil {
		httpd.ValidationError(w, err.Error())
		return
	}
	if req.BrainID == "" {
		httpd.ValidationError(w, "brainId required")
		return
	}
	if _, err := d.Brains.Create(r.Context(), req.BrainID); err != nil {
		if httpd.WriteStoreError(w, err) {
			return
		}
		httpd.InternalError(w, err.Error())
		return
	}
	writeJSON(w, http.StatusCreated, brainSummary{
		BrainID:     req.BrainID,
		Description: req.Description,
	})
}

// handleDeleteBrain removes a brain. Requires X-Confirm-Delete: yes.
func (d *Daemon) handleDeleteBrain(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("brainId")
	if id == "" {
		httpd.ValidationError(w, "missing brainId")
		return
	}
	if r.Header.Get("X-Confirm-Delete") != "yes" {
		httpd.WriteProblem(w, httpd.Problem{
			Status: http.StatusPreconditionRequired,
			Title:  "Confirmation Required",
			Detail: "delete brain requires X-Confirm-Delete: yes header",
			Code:   "confirmation_required",
		})
		return
	}
	if err := d.Brains.Delete(id); err != nil {
		if errors.Is(err, ErrBrainNotFound) {
			httpd.NotFound(w, "brain not found: "+id)
			return
		}
		httpd.InternalError(w, err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}
