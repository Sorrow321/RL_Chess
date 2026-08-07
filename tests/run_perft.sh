#!/usr/bin/env bash
# Full engine validation: build, then the perft suite and the rules tests.
#
# This is the gate for Phase 0 (docs/00-overview.md). If it is not green, the
# movegen cannot be trusted and nothing downstream of it means anything —
# stop, fix, re-run all.
#
#   tests/run_perft.sh                    build + full suite
#   tests/run_perft.sh --threads 1        honest single-threaded nodes/s
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="$root/build"
python="${PYTHON:-$(command -v python3)}"

cmake -S "$root" -B "$build" -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$python" >/dev/null
cmake --build "$build" --target perft_test rules_test -j "$(nproc)" >/dev/null

echo "=== perft suite ==="
"$build/perft_test" "$@"
echo
echo "=== rules ==="
"$build/rules_test"
