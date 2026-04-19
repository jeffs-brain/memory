// SPDX-License-Identifier: Apache-2.0

package lme

// ExpectedLMESmallSHA256 is the canonical SHA256 for longmemeval_s.json.
// Fetched from https://huggingface.co/datasets/xiaowu0162/LongMemEval/resolve/main/longmemeval_s
// on 2026-04-18. A SHA mismatch on LoadDataset means the dataset has
// drifted from the paper's official release. Keep this in sync with
// EXPECTED_SHA256 in scripts/fetch-lme.sh.
const ExpectedLMESmallSHA256 = "08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894"
