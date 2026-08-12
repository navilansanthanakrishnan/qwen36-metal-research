#!/usr/bin/env bash
# phase-f.sh — run the Phase F controls. NOT part of the frozen measurement
# surface; this is the calibration harness that produced NOISE.md.
#
#   bench/phase-f.sh null        null test: identical binaries, must say NO DIFFERENCE
#   bench/phase-f.sh positive    injected ~4% regression, must be DETECTED
#   bench/phase-f.sh drift       cold / sustained-load / recovered
#
# The null and positive tests deliberately share one binary and differ only by
# the QM_SABOTAGE_US environment variable, which the sabotage patch reads. That
# removes "the compiler laid the two builds out differently" as a confound: in
# the null test the two arms are byte-identical code paths, so anything the
# procedure reports is by construction noise.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_COOLDOWN=1 . "$HERE/env.sh" >/dev/null 2>&1 || :
: "${QM_BIN:?env.sh did not export the frozen configuration}"

SHIMDIR="${TMPDIR:-/tmp}/qm-shims"
mkdir -p "$SHIMDIR/A" "$SHIMDIR/B"

mkshim() {   # $1 = dir, $2 = env assignment or empty
    cat >"$SHIMDIR/$1/llama-bench" <<EOF
#!/usr/bin/env bash
${2:+export $2}
exec "$QM_BIN/llama-bench" "\$@"
EOF
    chmod +x "$SHIMDIR/$1/llama-bench"
}

case "${1:-}" in
  null)
    echo "=== PHASE F.2 — NULL TEST (negative control) ==="
    echo "Identical binary, identical environment, labelled A and B."
    echo "The procedure MUST report NO DIFFERENCE. If it reports a win, it is broken."
    mkshim A ""
    mkshim B ""
    "$HERE/ab.sh" --a "$SHIMDIR/A" --b "$SHIMDIR/B" --la "A-ident" --lb "B-ident" \
                  --pairs "${PAIRS:-8}"
    rc=$?
    echo
    case $rc in
      0) echo "NULL TEST: PASS (no difference reported)" ;;
      3) echo "NULL TEST: *** FAIL *** — procedure reported a DIFFERENCE between identical binaries" ;;
      *) echo "NULL TEST: DID NOT RUN (env refused or error, rc=$rc). This is not a pass and not a fail." ;;
    esac
    exit $rc
    ;;

  positive)
    # 4% of the measured baseline decode step. Recomputed from the argument so
    # it is always tied to a real measured latency, never to a guess.
    TG="${TG_BASELINE:-14.0}"
    US="$(awk -v t="$TG" 'BEGIN{printf "%d", 0.04*1e6/t}')"
    echo "=== PHASE F.3 — POSITIVE CONTROL (injected regression) ==="
    echo "baseline decode ${TG} tok/s -> step $(awk -v t="$TG" 'BEGIN{printf "%.2f", 1000/t}') ms"
    echo "injecting ${US} us per decoded token = 4.0% slowdown"
    echo "The procedure MUST report DIFFERENT, in the SLOWER direction, near 4%."
    mkshim A ""
    mkshim B "QM_SABOTAGE_US=$US"
    "$HERE/ab.sh" --a "$SHIMDIR/A" --b "$SHIMDIR/B" --la "clean" --lb "sabotaged" \
                  --pairs "${PAIRS:-8}"
    rc=$?
    echo
    case $rc in
      3) echo "POSITIVE CONTROL: DETECTED (check direction and size above)" ;;
      0) echo "POSITIVE CONTROL: *** FAIL *** — a real 4% regression was not detected" ;;
      *) echo "POSITIVE CONTROL: DID NOT RUN (env refused or error, rc=$rc)." ;;
    esac
    exit $rc
    ;;

  drift)
    echo "=== PHASE F.4 — THERMAL DRIFT ==="
    echo "--- cold (machine idle beforehand) ---"
    "$HERE/thermal.sh"
    "$HERE/decode.sh" --label drift-cold --tag "drift cold"

    echo "--- 20 minutes of sustained load ---"
    END=$(( $(date +%s) + 1200 ))
    n=0
    while [[ $(date +%s) -lt $END ]]; do
        "$HERE/decode.sh" --no-env --label drift-load --tag "sustained $n" 2>/dev/null
        n=$((n+1))
    done
    echo "--- hot (immediately after sustained load) ---"
    "$HERE/thermal.sh"
    "$HERE/decode.sh" --no-env --label drift-hot --tag "drift hot"

    echo "--- cooldown ${QM_COOLDOWN_S}s, then recovered ---"
    sleep "$QM_COOLDOWN_S"
    "$HERE/thermal.sh"
    "$HERE/decode.sh" --no-env --label drift-recovered --tag "drift recovered"
    ;;

  *)
    echo "usage: bench/phase-f.sh {null|positive|drift}" >&2; exit 1 ;;
esac
