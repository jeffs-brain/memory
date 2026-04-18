// SPDX-License-Identifier: Apache-2.0

package lme

import (
	"encoding/binary"
	"math/rand/v2"
	"sort"
)

const (
	defaultBootstrapResamples = 1000
	bootstrapLowPercentile    = 2.5
	bootstrapHighPercentile   = 97.5
)

// BootstrapCI returns a 95% bootstrap confidence interval over a slice of
// binary outcomes. Pass resamples <= 0 to use the default of 1000.
func BootstrapCI(outcomes []bool, seed int64, resamples int) [2]float64 {
	if len(outcomes) == 0 {
		return [2]float64{0, 0}
	}
	if resamples <= 0 {
		resamples = defaultBootstrapResamples
	}

	rng := newSeededRand(seed)
	n := len(outcomes)
	means := make([]float64, resamples)

	for i := range resamples {
		correct := 0
		for range n {
			idx := rng.IntN(n)
			if outcomes[idx] {
				correct++
			}
		}
		means[i] = float64(correct) / float64(n)
	}

	sort.Float64s(means)
	return [2]float64{
		percentileOfSorted(means, bootstrapLowPercentile),
		percentileOfSorted(means, bootstrapHighPercentile),
	}
}

// BootstrapCategoryCI computes a 95% bootstrap CI per category.
func BootstrapCategoryCI(
	outcomes []QuestionOutcome,
	categoryKey func(QuestionOutcome) string,
	correctKey func(QuestionOutcome) bool,
	seed int64,
	resamples int,
) map[string][2]float64 {
	if len(outcomes) == 0 {
		return map[string][2]float64{}
	}

	buckets := make(map[string][]bool)
	order := make([]string, 0)
	for _, o := range outcomes {
		key := categoryKey(o)
		if _, seen := buckets[key]; !seen {
			order = append(order, key)
		}
		buckets[key] = append(buckets[key], correctKey(o))
	}

	result := make(map[string][2]float64, len(buckets))
	for i, key := range order {
		subSeed := seed + int64(i+1)
		result[key] = BootstrapCI(buckets[key], subSeed, resamples)
	}
	return result
}

// LatencyPercentile returns the pct-th percentile latency (milliseconds)
// using the nearest-rank method on a sorted copy.
func LatencyPercentile(latencies []int, pct float64) int {
	if len(latencies) == 0 {
		return 0
	}
	if pct < 0 {
		pct = 0
	}
	if pct > 100 {
		pct = 100
	}

	sorted := make([]int, len(latencies))
	copy(sorted, latencies)
	sort.Ints(sorted)

	n := len(sorted)
	rank := int((pct/100.0)*float64(n) + 0.999999999)
	if rank < 1 {
		rank = 1
	}
	if rank > n {
		rank = n
	}
	return sorted[rank-1]
}

// percentileOfSorted returns the pct-th percentile of an already-sorted
// slice using the nearest-rank method.
func percentileOfSorted(sorted []float64, pct float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	n := len(sorted)
	rank := int((pct/100.0)*float64(n) + 0.999999999)
	if rank < 1 {
		rank = 1
	}
	if rank > n {
		rank = n
	}
	return sorted[rank-1]
}

// newSeededRand constructs a deterministic rand.Rand from a 64-bit seed.
func newSeededRand(seed int64) *rand.Rand {
	var key [32]byte
	u := uint64(seed)
	for i := 0; i < 4; i++ {
		binary.LittleEndian.PutUint64(key[i*8:(i+1)*8], u+uint64(i)*0x9E3779B97F4A7C15)
	}
	return rand.New(rand.NewChaCha8(key))
}
