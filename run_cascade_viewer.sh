#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
manifest="$repo_dir/data/visualization/fsri30-best-flow-display/manifest.json"
full_flow="$repo_dir/data/visualization/fsri30-full-flow/manifest.json"
binary="${BATTERY_VIEWER_BINARY:-$repo_dir/build/ui/battery_sim_viewer}"
if [[ -f "$full_flow" ]] && python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); sys.exit(m.get("partial",True))' "$full_flow"; then
  manifest="$full_flow"
elif [[ ! -f "$manifest" ]]; then
  python3 "$repo_dir/scripts/export_visual_replay.py" \
    "$repo_dir/data/calibration/fsri30-explosion-priority-20261026-r1/trial-0023" \
    "$repo_dir/data/visualization/fsri30-best-flow-display"
fi
if [[ ! -x "$binary" ]]; then
  exec "$repo_dir/run_viewer.sh" --replay "$manifest" --at 2480 "$@"
fi
exec "$binary" --replay "$manifest" --at 2480 "$@"
