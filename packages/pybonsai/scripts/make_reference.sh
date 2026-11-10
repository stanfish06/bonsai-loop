#!/usr/bin/env bash
# Build penjing's own command-line tools and run them on a Bonsai output folder.
# tests/test_reference.py compares the binding against what this writes.
# Usage: nix develop -c scripts/make_reference.sh <final-bonsai-folder>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- options -------------------------------------------------------------
TREE="${1:?usage: make_reference.sh <final-bonsai-folder>}"   # Bonsai output folder to score
OUT="${OUT:-$ROOT/results/reference}"                          # where the reference files go
CLI_BUILD="${CLI_BUILD:-$ROOT/build/penjing-cli}"              # build tree for the tools
THREADS="${THREADS:-8}"
TOP="${TOP:-50}"                                               # convergence candidates to keep
MAX_PAIRS="${MAX_PAIRS:-20000000}"                             # pair limit for detect_convergence
# --------------------------------------------------------------------------

TOOLS=(calculate_tree_distances calculate_posterior_distances detect_convergence)

cmake -S "$ROOT/third_party/penjing" -B "$CLI_BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF -DBONSAI_PORT_SOURCE_DIR="$ROOT/third_party/bonsai_port" >"$CLI_BUILD.configure.log" 2>&1 ||
    { tail -20 "$CLI_BUILD.configure.log"; exit 1; }
cmake --build "$CLI_BUILD" --target "${TOOLS[@]}" >"$CLI_BUILD.build.log" 2>&1 ||
    { tail -20 "$CLI_BUILD.build.log"; exit 1; }

mkdir -p "$OUT"
echo "tree=$TREE threads=$THREADS start=$(date -Is)" | tee "$OUT/run.log"

"$CLI_BUILD/calculate_tree_distances" --tree "$TREE" --out "$OUT/tree" --leaves-only \
    --threads "$THREADS" 2>&1 | tee -a "$OUT/run.log"
"$CLI_BUILD/calculate_posterior_distances" --tree "$TREE" --out "$OUT/posterior" --leaves-only \
    --with-uncertainty --threads "$THREADS" 2>&1 | tee -a "$OUT/run.log"
"$CLI_BUILD/detect_convergence" --tree "$TREE" --out "$OUT/convergence" --leaves-only \
    --top "$TOP" --max-pairs "$MAX_PAIRS" 2>&1 | tee -a "$OUT/run.log"

echo "end=$(date -Is)" | tee -a "$OUT/run.log"
echo "PYBONSAI_TREE_DIR=$TREE PYBONSAI_REFERENCE_DIR=$OUT"
