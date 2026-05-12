// SPDX-License-Identifier: Apache-2.0

package memory

import "time"

// Age-based confidence thresholds ported from the TS consolidation
// stage (sdks/ts/memory/src/memory/consolidate.ts). These constants
// control when a heuristic's confidence is demoted due to staleness or
// promoted due to reinforcement.
const (
	// staleAfterDays is the number of days since last observation after
	// which a heuristic is considered stale. If the current confidence
	// is "high" it is demoted to "medium"; otherwise it becomes "low".
	staleAfterDays = 90

	// deepStaleAfterDays is the number of days since last observation
	// after which a heuristic's confidence is forced to "low" regardless
	// of other signals.
	deepStaleAfterDays = 180

	// heuristicReinforcementDays is the minimum span (in days) between
	// creation and last observation for a heuristic to be considered
	// reinforced. If the current confidence is "high" it stays "high";
	// otherwise it is promoted to "medium".
	heuristicReinforcementDays = 14

	// heuristicStrongReinforcementDays is the span (in days) between
	// creation and last observation that unconditionally promotes a
	// heuristic to "high" confidence.
	heuristicStrongReinforcementDays = 45
)

// DeriveHeuristicConfidence computes the confidence level for a
// heuristic memory based on its age and reinforcement span. It mirrors
// the TS function `deriveHeuristicConfidence` in consolidate.ts.
//
// Parameters:
//   - current: the existing confidence level ("low", "medium", or "high")
//   - observedAt: the most recent observation time (typically from the
//     frontmatter "modified" field)
//   - createdAt: the creation time of the heuristic
//   - now: the reference time for computing age
//
// The function returns the derived confidence string.
func DeriveHeuristicConfidence(current string, observedAt, createdAt, now time.Time) string {
	normCurrent := normaliseConfidenceLevel(current)
	ageDays := diffDaysTruncated(observedAt, now)
	reinforcementDays := diffDaysTruncated(createdAt, observedAt)
	if reinforcementDays < 0 {
		reinforcementDays = 0
	}

	if ageDays >= deepStaleAfterDays {
		return "low"
	}
	if ageDays >= staleAfterDays {
		if normCurrent == "high" {
			return "medium"
		}
		return "low"
	}
	if reinforcementDays >= heuristicStrongReinforcementDays {
		return "high"
	}
	if reinforcementDays >= heuristicReinforcementDays {
		if normCurrent == "high" {
			return "high"
		}
		return "medium"
	}
	return normCurrent
}

// normaliseConfidenceLevel normalises a confidence string to one of
// "low", "medium", or "high". Unrecognised values default to "low".
func normaliseConfidenceLevel(value string) string {
	switch value {
	case "high", "medium", "low":
		return value
	default:
		return "low"
	}
}

// diffDaysTruncated returns the number of whole days between start and
// end, truncating toward zero. Matches the TS `diffDays` which uses
// Math.floor on (end - start) / millisPerDay.
func diffDaysTruncated(start, end time.Time) int {
	duration := end.Sub(start)
	return int(duration.Hours() / 24)
}
