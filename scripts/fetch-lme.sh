#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# LongMemEval dataset fetcher for the jeffs-brain memory eval harness.
# Downloads longmemeval_s.json from the official Hugging Face mirror and
# verifies its SHA256 against the pinned expected value. Idempotent: a
# second run is a no-op when the file is already present and the SHA
# matches.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATASET_DIR="${LME_DATASET_DIR:-${REPO_ROOT}/eval/datasets}"
DATASET_NAME="longmemeval_s.json"
DEST="${DATASET_DIR}/${DATASET_NAME}"

# Canonical source: the authors' Hugging Face dataset repo. The file is
# named longmemeval_s on HF (the original .json extension was removed
# when Hugging Face migrated the file to Git LFS / xet storage). We
# persist it locally with the .json suffix for clarity.
HF_URL="https://huggingface.co/datasets/xiaowu0162/LongMemEval/resolve/main/longmemeval_s"

# Pinned SHA256 for longmemeval_s.json. Update when intentionally
# upgrading the dataset. Keep in sync with ExpectedLMESmallSHA256 in
# go/eval/lme/lme_dataset.go.
EXPECTED_SHA256="08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894"

log() { printf '[fetch-lme] %s\n' "$*"; }
err() { printf '[fetch-lme] ERROR: %s\n' "$*" >&2; exit 1; }

compute_sha256() {
  sha256sum "$1" | cut -d' ' -f1
}

mkdir -p "${DATASET_DIR}"

if [[ -f "${DEST}" ]]; then
  actual_sha="$(compute_sha256 "${DEST}")"
  if [[ "${actual_sha}" == "${EXPECTED_SHA256}" ]]; then
    log "${DATASET_NAME}: already present at ${DEST}, SHA verified."
    exit 0
  fi
  log "${DATASET_NAME}: SHA mismatch (got ${actual_sha}, want ${EXPECTED_SHA256}), re-downloading."
fi

log "Downloading ${DATASET_NAME} from ${HF_URL}"
log "Target: ${DEST}"

if command -v curl >/dev/null 2>&1; then
  curl -fSL --progress-bar -o "${DEST}.tmp" "${HF_URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -q --show-progress -O "${DEST}.tmp" "${HF_URL}"
else
  err "Neither curl nor wget is available. Install one and retry."
fi

actual_sha="$(compute_sha256 "${DEST}.tmp")"
if [[ "${actual_sha}" != "${EXPECTED_SHA256}" ]]; then
  rm -f "${DEST}.tmp"
  err "${DATASET_NAME}: SHA mismatch after download (got ${actual_sha}, want ${EXPECTED_SHA256}). The upstream dataset may have changed; inspect and update EXPECTED_SHA256 if intentional."
fi

mv "${DEST}.tmp" "${DEST}"
log "${DATASET_NAME}: downloaded and verified (SHA: ${actual_sha})."
log "Dataset ready at ${DEST}"
