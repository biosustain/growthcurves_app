#!/usr/bin/env bash
# Write AutoGrowth/VERSION as <YY.M.D>+<short-hash> of the last commit (e.g. 26.11.23+bf5b7e2).
# Usage: scripts/update_version.sh            # from the last commit
#        scripts/update_version.sh 26.11.23+abc1234   # manual override
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
if [[ $# -ge 1 ]]; then
  version="$1"
else
  date="$(git log -1 --format=%cd --date=format:%y.%-m.%-d)"
  version="${date}+$(git log -1 --format=%h)"
fi
printf '%s\n' "$version" > AutoGrowth/VERSION
echo "AutoGrowth/VERSION -> $version"
