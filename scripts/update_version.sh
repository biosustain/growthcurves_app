#!/usr/bin/env bash
# Write AutoGrowth/VERSION as <YY.M.D>+<short-hash> of the last commit (e.g. 26.11.23+bf5b7e2).
# Usage: scripts/update_version.sh            # from the last commit
#        scripts/update_version.sh 26.11.23+abc1234   # manual override
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
if [[ $# -ge 1 ]]; then
  version="$1"
else
  # %-m / %-d are not supported by git's strftime on Windows, so format with
  # zero-padded fields and strip the leading zeros in bash (base-10 forced).
  read -r yy mm dd <<< "$(git log -1 --format=%cd --date=format:'%y %m %d')"
  date="${yy}.$((10#$mm)).$((10#$dd))"
  version="${date}+$(git log -1 --format=%h)"
fi
printf '%s\n' "$version" > AutoGrowth/VERSION
echo "AutoGrowth/VERSION -> $version"
