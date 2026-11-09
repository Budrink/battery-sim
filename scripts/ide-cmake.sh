#!/usr/bin/env bash
# GNOME-launched VS Code may not inherit ~/.local/bin from the login shell.
set -eu
for candidate in "${HOME}/.local/bin/cmake" /usr/local/bin/cmake "$(command -v cmake || true)"; do
    [[ -x "$candidate" ]] || continue
    version="$("$candidate" --version | head -n 1)"
    if [[ "$version" =~ ([0-9]+)\.([0-9]+) ]]; then
        major="${BASH_REMATCH[1]}"
        minor="${BASH_REMATCH[2]}"
        if (( major > 3 || (major == 3 && minor >= 24) )); then
            exec "$candidate" "$@"
        fi
    fi
done
echo "BatterySim needs CMake >= 3.24. Install it in PATH or ~/.local/bin." >&2
exit 1
