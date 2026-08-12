#!/usr/bin/env bash
# init.sh — reproduce the pinned llama.cpp build from clean, in one command.
#
#   ./init.sh            # incremental (reuses build/ if present)
#   ./init.sh --clean    # wipe build/ and rebuild from scratch
#
# Everything about the trunk this project diffs against is pinned here.
set -euo pipefail

WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA="$WORK/llama.cpp"

# ---- pinned trunk -----------------------------------------------------------
PIN_REPO="https://github.com/ggml-org/llama.cpp.git"
PIN_SHA="0b1bad14ff204627636aeb1de22ddcd5acb859d4"

# ---- clone if missing -------------------------------------------------------
if [[ ! -d "$LLAMA/.git" ]]; then
    echo "[init] cloning $PIN_REPO -> $LLAMA"
    git clone --filter=blob:none "$PIN_REPO" "$LLAMA"
fi

cd "$LLAMA"
CUR="$(git rev-parse HEAD)"
if [[ "$CUR" != "$PIN_SHA" ]]; then
    echo "[init] checking out pinned $PIN_SHA (was $CUR)"
    git fetch --quiet origin "$PIN_SHA" 2>/dev/null || git fetch --quiet origin
    git checkout --quiet "$PIN_SHA"
fi

# ---- clean if asked ---------------------------------------------------------
if [[ "${1:-}" == "--clean" ]]; then
    echo "[init] removing $LLAMA/build"
    rm -rf "$LLAMA/build"
fi

# ---- configure --------------------------------------------------------------
# Frozen CMake configuration. Any change here invalidates every number in
# BASELINE.md — bump BUILD.md if you touch it.
cmake -S "$LLAMA" -B "$LLAMA/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DGGML_BLAS=OFF \
    -DGGML_ACCELERATE=ON \
    -DLLAMA_CURL=ON \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DBUILD_SHARED_LIBS=OFF

# ---- build ------------------------------------------------------------------
cmake --build "$LLAMA/build" --config Release -j "$(sysctl -n hw.ncpu)"

# ---- verify the binaries we depend on exist ---------------------------------
BIN="$LLAMA/build/bin"
MISSING=0
for b in llama-bench llama-cli llama-server llama-perplexity test-backend-ops; do
    if [[ -x "$BIN/$b" ]]; then
        echo "[init] ok  $b"
    else
        echo "[init] MISSING  $b" >&2
        MISSING=1
    fi
done
[[ $MISSING -eq 0 ]] || { echo "[init] FAIL: required binaries missing" >&2; exit 1; }

# ---- bandwidth probe --------------------------------------------------------
if [[ -f "$WORK/bench/bwprobe/bwprobe.m" ]]; then
    echo "[init] building bwprobe"
    make -s -C "$WORK/bench/bwprobe"
fi

echo "[init] done. trunk = $PIN_SHA"
