#!/usr/bin/env bash
# decode.sh — the primary measurement loop. FROZEN.
#
#   bench/decode.sh [--label NAME] [--bin DIR] [--no-env] [--tag TAG]
#
# Wraps llama-bench at the frozen configuration in bench/env.sh, brackets it
# with memory and thermal readings, and DISCARDS the run if swap moved.
#
# Emits one human-readable line to stdout and appends a full JSON record to
# runs/decode/<date>.jsonl. Exit codes:
#   0  ok
#   2  discarded (swap moved during the run)
#   1  environment refused, or llama-bench failed
#
# --no-env skips the env gate and the cooldown. Only for the null/positive
# control loops in ab.sh, which run their own gate once for the whole batch;
# never for a standalone recorded number.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LABEL="trunk"; BINDIR=""; RUN_ENV=1; TAG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --label) LABEL="$2"; shift 2 ;;
        --bin)   BINDIR="$2"; shift 2 ;;
        --tag)   TAG="$2";    shift 2 ;;
        --no-env) RUN_ENV=0;  shift ;;
        *) echo "decode.sh: unknown arg $1" >&2; exit 1 ;;
    esac
done

if [[ "$RUN_ENV" == "1" ]]; then
    . "$HERE/env.sh" || exit 1
else
    SKIP_COOLDOWN=1 . "$HERE/env.sh" >/dev/null 2>&1 || {
        # still need the frozen vars even when not gating
        :
    }
fi
: "${QM_MODEL:?env.sh did not export the frozen configuration}"

BIN="${BINDIR:-$QM_BIN}"
OUTDIR="$QM_RUNS/decode"
mkdir -p "$OUTDIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DAY="$(date -u +%Y%m%d)"

MEM_BEFORE="$("$HERE/mem.sh")"
TH_BEFORE="$("$HERE/thermal.sh")"
SWAPIN_BEFORE="$(echo "$MEM_BEFORE" | tr ' ' '\n' | awk -F= '/swapins_total/{print $2}')"

T0="$(date +%s)"
JSON="$("$BIN/llama-bench" \
    -m  "$QM_MODEL" \
    -ngl "$QM_NGL" \
    -fa  "$QM_FA" \
    -ctk "$QM_CTK" -ctv "$QM_CTV" \
    -b   "$QM_BATCH" -ub "$QM_UBATCH" \
    -p   "$QM_NPROMPT" -n "$QM_NGEN" \
    -r   "$QM_REPS" \
    -o   json 2>/dev/null)"
RC=$?
T1="$(date +%s)"
WALL=$((T1 - T0))

if [[ $RC -ne 0 || -z "$JSON" ]]; then
    echo "decode: llama-bench failed (rc=$RC)" >&2
    exit 1
fi

MEM_AFTER="$("$HERE/mem.sh")"
TH_AFTER="$("$HERE/thermal.sh")"
SWAPIN_AFTER="$(echo "$MEM_AFTER" | tr ' ' '\n' | awk -F= '/swapins_total/{print $2}')"
SWAP_DELTA=$(( SWAPIN_AFTER - SWAPIN_BEFORE ))

# Parse the two results out of llama-bench JSON.
read -r PP_AVG PP_SD TG_AVG TG_SD <<<"$(python3 - "$JSON" <<'PY'
import json, sys
rows = json.loads(sys.argv[1])
out = {}
for r in rows:
    kind = "pp" if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0 else "tg"
    out[kind] = (r.get("avg_ts"), r.get("stddev_ts"))
pp = out.get("pp", (0, 0)); tg = out.get("tg", (0, 0))
print(pp[0] or 0, pp[1] or 0, tg[0] or 0, tg[1] or 0)
PY
)"

# --- validity gates ---------------------------------------------------------
# SWAP. The brief's rule is "any run whose swap-in count moved is discarded".
# vm.compressor.swapper.swapins_total is a SYSTEM-WIDE counter, and this
# machine swaps continuously from other applications: five idle 40 s windows
# measured 639 / 1290 / 2080 / 3071 / 21259 pages with no benchmark running at
# all (SETUP-LOG.md). A strict != 0 gate therefore discards every run on a
# machine with a browser open, and would have discarded the smoke test.
#
# The failure mode the rule exists to catch is the model itself being evicted
# and re-read per token. That is not subtle: the observed thrashing load moved
# 289,342 pages, an order of magnitude above the idle ceiling. So the gate is
# set just above the measured idle maximum, and the raw delta is recorded on
# every run regardless so a suspicious result can always be re-examined.
QM_MAX_SWAPIN_PAGES="${QM_MAX_SWAPIN_PAGES:-25000}"

# PREFILL HEALTH. Added after the Phase F positive control exposed the gap.
# Prefill is compute-bound; decode-side changes cannot legitimately move it. So a
# pp512 far below the frozen baseline distribution means the machine was
# disturbed during the run, whatever the cause. In the positive control four runs
# passed both the swap and variance gates while sitting 8-9% low on prefill
# (281-284 vs a 306.75 baseline) -- internally consistent, just uniformly slow,
# which is exactly the signature a sustained external disturbance produces and
# the other two gates are blind to. Threshold is 5% below the noise-floor mean,
# ~2.5 sd on a 1.99% rsd.
#
# NOTE: this gate was added AFTER the null and positive controls were run, so
# both controls were measured WITHOUT it. That is the conservative direction for
# the positive control (more noise, harder to detect, and it detected anyway) but
# the LENIENT direction for the null test. Re-run the null test under this gate
# before relying on it. Recorded in NOISE.md.
QM_PP_BASELINE="${QM_PP_BASELINE:-306.75}"
QM_MAX_PP_DROP_PCT="${QM_MAX_PP_DROP_PCT:-5.0}"

# VARIANCE. A run that thrashed has huge spread between repetitions even when
# its mean looks plausible. This catches what the swap counter cannot attribute.
QM_MAX_RSD_PCT="${QM_MAX_RSD_PCT:-5.0}"
TG_RSD="$(awk -v a="$TG_AVG" -v s="$TG_SD" 'BEGIN{ printf "%.3f", (a>0? 100*s/a : 99) }')"

STATUS="ok"; EXIT=0
if [[ "$SWAP_DELTA" -gt "$QM_MAX_SWAPIN_PAGES" ]]; then
    STATUS="DISCARDED_SWAP"; EXIT=2
elif awk -v r="$TG_RSD" -v m="$QM_MAX_RSD_PCT" 'BEGIN{exit !(r>m)}'; then
    STATUS="DISCARDED_VARIANCE"; EXIT=2
elif awk -v p="$PP_AVG" -v b="$QM_PP_BASELINE" -v d="$QM_MAX_PP_DROP_PCT" \
        'BEGIN{exit !(p < b*(1-d/100))}'; then
    STATUS="DISCARDED_PREFILL"; EXIT=2
fi

python3 - >>"$OUTDIR/$DAY.jsonl" <<PY
import json
print(json.dumps({
  "stamp": "$STAMP", "label": "$LABEL", "tag": "$TAG", "status": "$STATUS",
  "commit": "${QM_HEAD_SHA:-unknown}", "dirty": "${QM_DIRTY:-clean}",
  "bin": "$BIN",
  "config": {"ngl": $QM_NGL, "fa": "$QM_FA", "ctk": "$QM_CTK", "ctv": "$QM_CTV",
             "b": $QM_BATCH, "ub": $QM_UBATCH, "p": $QM_NPROMPT, "n": $QM_NGEN,
             "r": $QM_REPS},
  "pp_tps": $PP_AVG, "pp_sd": $PP_SD, "tg_tps": $TG_AVG, "tg_sd": $TG_SD,
  "wall_s": $WALL, "swapin_delta": $SWAP_DELTA, "tg_rsd_pct": $TG_RSD,
  "mem_before": "$MEM_BEFORE", "mem_after": "$MEM_AFTER",
  "thermal_before": "$TH_BEFORE", "thermal_after": "$TH_AFTER",
  "raw": json.loads(r'''$JSON'''),
}))
PY

printf '%-14s pp512=%7.2f±%.2f  tg128=%6.3f±%.3f (rsd %.2f%%)  wall=%ss  swapin=%s  %s\n' \
    "$LABEL" "$PP_AVG" "$PP_SD" "$TG_AVG" "$TG_SD" "$TG_RSD" "$WALL" "$SWAP_DELTA" "$STATUS"

[[ "$STATUS" == "ok" ]] || echo "decode: run DISCARDED, swap moved by $SWAP_DELTA pages" >&2
exit $EXIT
