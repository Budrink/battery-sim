#!/usr/bin/env bash
set -eu
record_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec /home/konstantin/work/battery-sim/build/ui/battery_sim_viewer-cylinders-20261027-r1 \
    --replay "$record_dir/visual/manifest.json" --at 1700 --play "$@"
