#!/usr/bin/env bash
# Block until bench/env.sh passes, instead of failing the run.
#
# env.sh is frozen and correct: it refuses to measure on battery, under load,
# in Low Power Mode, or without enough working set. But a *refusal* ends a
# session, and a long-running agent that ends on a refusal has stopped for a
# reason that will clear on its own in ten minutes.
#
# This wrapper turns the refusal into a wait. It never relaxes a gate.
#
#   bash bench/wait-for-env.sh            # wait up to 60 min, poll every 60 s
#   bash bench/wait-for-env.sh 7200 120   # wait up to 2 h, poll every 2 min
#
# Exit 0 once conditions are met, 1 on timeout. On timeout the caller should
# go do unmeasured work, not stop.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TIMEOUT="${1:-3600}"
INTERVAL="${2:-60}"

start=$(date +%s)
attempt=0
last_reason=""

while :; do
    attempt=$((attempt + 1))
    if out="$(bash "$HERE/env.sh" 2>&1)"; then
        [ "$attempt" -gt 1 ] && echo "wait-for-env: clear after ${attempt} checks ($(( $(date +%s) - start ))s)"
        exit 0
    fi

    reason="$(printf '%s\n' "$out" | grep -iE '^\s*(FAIL|refus|not on|Low Power|load|working set)' | head -1)"
    [ -z "$reason" ] && reason="$(printf '%s\n' "$out" | tail -1)"

    if [ "$reason" != "$last_reason" ]; then
        echo "wait-for-env: blocked — ${reason}"
        last_reason="$reason"
    fi

    elapsed=$(( $(date +%s) - start ))
    if [ "$elapsed" -ge "$TIMEOUT" ]; then
        echo "wait-for-env: still blocked after ${elapsed}s — ${reason}"
        echo "wait-for-env: go do unmeasured work; do not stop."
        exit 1
    fi
    sleep "$INTERVAL"
done
