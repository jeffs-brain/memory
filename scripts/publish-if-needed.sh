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
npm publish --access public --provenance
