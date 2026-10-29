#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 || ! -d "$1" ]]; then
  echo "Usage: bash scripts/create_macos_dmg.sh <source-directory> <output.dmg>" >&2
  exit 2
fi

source_directory="$1"
output="$2"
log_file="$(mktemp "${TMPDIR:-/tmp}/bhe-hdiutil.XXXXXX")"
trap 'rm -f "$log_file"' EXIT
mkdir -p "$(dirname "$output")"

# DiskImages can race with Spotlight on hosted macOS runners. Retry only
# that transient error, preserving the real status for all other failures.
# https://github.com/actions/runner-images/issues/7522
for attempt in 1 2 3 4 5; do
  # Do not pipe hdiutil: its helper process can keep the pipe open on failure.
  if hdiutil create -volname "BHE" -fs HFS+ -srcfolder "$source_directory" \
    -ov -format UDZO "$output" > "$log_file" 2>&1; then
    cat "$log_file"
    test -s "$output"
    hdiutil verify "$output"
    exit 0
  else
    status=$?
    cat "$log_file" >&2
    if [[ "$attempt" == 5 ]] || ! grep -q 'Resource busy' "$log_file"; then
      exit "$status"
    fi
    delay=$((2 ** attempt))
    echo "DiskImages resource busy; retry $attempt/4 in ${delay}s" >&2
    sleep "$delay"
  fi
done
