#!/usr/bin/env bash
set -euo pipefail

package_dir="${1:?package directory is required}"

cd "$package_dir"

package_name="$(node -p "JSON.parse(require('fs').readFileSync('package.json', 'utf8')).name")"
local_version="$(node -p "JSON.parse(require('fs').readFileSync('package.json', 'utf8')).version")"
published_version="$(npm view "$package_name" version 2>/dev/null || true)"

if [[ "$published_version" == "$local_version" ]]; then
  echo "Skipping $package_name@$local_version because it is already published."
  exit 0
fi

echo "Publishing $package_name@$local_version"

# Detect prerelease version (contains hyphen like 0.4.0-rc.1)
tag_flag=""
if [[ "$local_version" == *-* ]]; then
  tag_flag="--tag rc"
fi

# Provenance requires the tag to be on the default branch (main).
# Prerelease tags may sit on short-lived branches, so provenance is skipped
# for any version containing a prerelease suffix.
provenance_flag="--provenance"
if [[ "$local_version" == *-* ]]; then
  provenance_flag=""
fi

# --ignore-scripts skips prepublishOnly (typecheck + test + build already
# ran in the CI workflow steps before this script is called).
npm publish --access public $provenance_flag --ignore-scripts $tag_flag
