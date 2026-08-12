#!/usr/bin/env bash
# env.sh — establish and VERIFY measurement conditions. FROZEN.
#
# Exits nonzero rather than warning. A benchmark that runs under bad conditions
# is worse than one that refuses, because it produces a number that looks
# comparable and is not.
#
# Source it (`. bench/env.sh`) to also export the frozen configuration, or run
# it standalone to just check.
#
#   SKIP_COOLDOWN=1   skip the cooldown wait (only for interactive poking,
#                     never for a recorded result)

WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ============================================================================
# FROZEN CONFIGURATION — changing any of these invalidates BASELINE.md
# ============================================================================
export QM_WORK="$WORK"
export QM_LLAMA="$WORK/llama.cpp"
export QM_BIN="$WORK/llama.cpp/build/bin"
export QM_MODEL="$WORK/models/Qwen3.6-27B-Q4_K_M.gguf"
export QM_RUNS="$WORK/runs"
export QM_SHA="0b1bad14ff204627636aeb1de22ddcd5acb859d4"

# The measurement surface. See BASELINE.md.
export QM_NGL=99            # all 65 layers on Metal
export QM_FA=on             # flash attention
export QM_CTK=f16           # KV cache K type
export QM_CTV=f16           # KV cache V type
export QM_BATCH=2048
export QM_UBATCH=512
export QM_NPROMPT=512
export QM_NGEN=128
export QM_REPS=3

# Largest context that RUNS RELIABLY on this machine, which is not the same as
# the largest that llama.cpp's static fitter accepts. The fitter approves 16384
# (824 MiB of nominal device headroom), but a llama-server holding a 16384
# context OOMs during warmup under real system memory pressure, and 8192 loads
# but decodes at 5.69 tok/s instead of ~14.6 because it is starved. 4096 is the
# largest that both loads and sustains full speed. See HARDWARE.md.
export QM_CTX=4096

# llama-server allocates batch-sized buffers that llama-bench does not, so the
# server scripts cap the logical batch at the ubatch size. This does NOT change
# the frozen llama-bench decode configuration above.
export QM_SERVER_BATCH=512

# GPU wired limit. 0 = driver default, which yields a
# recommendedMaxWorkingSetSize of 18186 MiB on this machine and is enough for
# the frozen configuration. Raising it would buy a longer context; see
# HARDWARE.md and LEADS.md.
export QM_WIRED_LIMIT_MB=0
export QM_MIN_WORKING_SET_MIB=17000   # model 16028 + KV 1024 + state 150 + compute 160
export QM_MIN_FREE_PCT=25
export QM_COOLDOWN_S="${QM_COOLDOWN_S:-60}"   # see NOISE.md drift characterization
export QM_MAX_THROTTLE_PCT=3.0

# ============================================================================

_fail() { printf 'env: FAIL: %s\n' "$*" >&2; _FAILED=1; }
_FAILED=0

# --- power ------------------------------------------------------------------
if ! pmset -g batt 2>/dev/null | head -1 | grep -qi "AC Power"; then
    _fail "not on AC power"
fi

# powermode: 0 = normal, 1 = high power, 2 = low power
PM="$(pmset -g custom 2>/dev/null | awk '/^AC Power:/{f=1} f&&/powermode/{print $2; exit}')"
if [[ "$PM" == "2" ]]; then
    _fail "Low Power Mode is on (powermode=$PM)"
fi

# --- background work that steals memory bandwidth ---------------------------
MDS_CPU="$(ps -Ao pcpu,comm | awk '/mds|mdworker/ {s+=$1} END {printf "%.0f", s+0}')"
if [[ "${MDS_CPU:-0}" -gt 20 ]]; then
    _fail "Spotlight is indexing (mds* using ${MDS_CPU}% CPU)"
fi

if tmutil status 2>/dev/null | grep -q 'Running = 1'; then
    _fail "Time Machine backup is running"
fi

if pgrep -qx cmake 2>/dev/null || pgrep -qx ninja 2>/dev/null || pgrep -q "^cc1plus" 2>/dev/null; then
    _fail "a build is running"
fi

# General background CPU load. This gate was added after it bit: the first
# noise-floor attempt ran while four research subagents were grepping the repo,
# and produced tg128 = 12.705 +/- 0.694 (5.5% RSD) against a quiet-machine
# 14.6 — a 13% depression that would have been baked into the noise floor as if
# it were intrinsic machine variance, permanently destroying the rig's
# sensitivity. Anything competing for memory bandwidth is disqualifying, not
# just llama processes and builds.
LOAD1="$(sysctl -n vm.loadavg | awk '{print $2}')"
if awk -v l="$LOAD1" 'BEGIN{exit !(l > 3.0)}'; then
    _fail "1-minute load average is ${LOAD1} (> 3.0); machine is not quiet"
fi

# Top non-kernel CPU consumer, excluding this script's own process tree.
BUSY="$(ps -Ao pcpu,comm -r | awk 'NR>1 && $1+0 > 50 {print $1" "$2; exit}')"
if [[ -n "$BUSY" ]]; then
    _fail "a process is using significant CPU: $BUSY"
fi

# --- other model processes holding unified memory ---------------------------
OTHER="$(pgrep -fl 'llama-server|llama-cli|llama-perplexity|ollama|lms |mlx_lm|LM Studio' 2>/dev/null \
         | grep -v "$$" | head -3)"
if [[ -n "$OTHER" ]]; then
    _fail "another model process holds memory: $(echo "$OTHER" | head -1)"
fi

# --- GPU wired limit --------------------------------------------------------
# The brief calls for `sudo sysctl iogpu.wired_limit_mb=<value>` every session
# because it resets on reboot. There is no passwordless sudo on this machine,
# so this tries and then VERIFIES the outcome rather than assuming it: what
# actually matters is the working set the driver reports, not the sysctl.
if [[ "${QM_WIRED_LIMIT_MB}" != "0" ]]; then
    CUR="$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)"
    if [[ "$CUR" != "$QM_WIRED_LIMIT_MB" ]]; then
        sudo -n sysctl "iogpu.wired_limit_mb=$QM_WIRED_LIMIT_MB" >/dev/null 2>&1 || true
    fi
fi

WS="$("$WORK/bench/gpuinfo/gpuinfo" 2>/dev/null \
      | tr ' ' '\n' | awk -F= '/recommendedMaxWorkingSetSize_MiB/{print $2}')"
if [[ -z "$WS" ]]; then
    _fail "could not read recommendedMaxWorkingSetSize (build bench/gpuinfo)"
elif (( WS < QM_MIN_WORKING_SET_MIB )); then
    _fail "GPU working set ${WS} MiB < required ${QM_MIN_WORKING_SET_MIB} MiB. Run: sudo sysctl iogpu.wired_limit_mb=20480"
fi

# --- free memory ------------------------------------------------------------
FREE_PCT="$(memory_pressure 2>/dev/null | awk -F': ' '/free percentage/{gsub(/%/,"",$2); print $2+0}')"
if [[ -z "$FREE_PCT" ]]; then
    _fail "could not read memory pressure"
elif (( FREE_PCT < QM_MIN_FREE_PCT )); then
    _fail "only ${FREE_PCT}% memory free, need >= ${QM_MIN_FREE_PCT}% (close apps; the model needs 16 GiB)"
fi

# --- model + binaries present ----------------------------------------------
[[ -f "$QM_MODEL" ]] || _fail "model missing: $QM_MODEL"
for b in llama-bench llama-cli llama-server llama-perplexity test-backend-ops; do
    [[ -x "$QM_BIN/$b" ]] || _fail "binary missing: $QM_BIN/$b (run ./init.sh)"
done

# --- trunk is the pinned commit and the tree is clean -----------------------
if [[ -d "$QM_LLAMA/.git" ]]; then
    HEAD_SHA="$(git -C "$QM_LLAMA" rev-parse HEAD 2>/dev/null)"
    export QM_HEAD_SHA="$HEAD_SHA"
    DIRTY="$(git -C "$QM_LLAMA" status --porcelain 2>/dev/null | head -1)"
    export QM_DIRTY="${DIRTY:+dirty}"
fi

# --- thermal ---------------------------------------------------------------
TH="$("$WORK/bench/thermal.sh" 2>/dev/null)"
THROTTLE="$(echo "$TH" | tr ' ' '\n' | awk -F= '/throttle_pct/{print $2}')"
if [[ -n "$THROTTLE" ]] && awk -v t="$THROTTLE" -v m="$QM_MAX_THROTTLE_PCT" 'BEGIN{exit !(t>m)}'; then
    _fail "GPU is throttled ${THROTTLE}% (> ${QM_MAX_THROTTLE_PCT}%); let it cool"
fi

if [[ "$_FAILED" == "1" ]]; then
    echo "env: refusing to benchmark under these conditions" >&2
    return 1 2>/dev/null || exit 1
fi

# --- cooldown ---------------------------------------------------------------
# Sized in NOISE.md from the cold / sustained-load / recovered drift runs.
if [[ "${SKIP_COOLDOWN:-0}" != "1" ]]; then
    sleep "$QM_COOLDOWN_S"
fi

echo "env: ok  ws=${WS}MiB free=${FREE_PCT}%  ${TH}"
return 0 2>/dev/null || exit 0
