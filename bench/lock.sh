#!/usr/bin/env bash
# lock.sh — exclusive measurement lock. NEW tooling; does not alter what any
# frozen script measures.
#
#   bench/lock.sh acquire [tag]   take the lock (blocks until free or stale)
#   bench/lock.sh release         release it
#   bench/lock.sh status          print holder
#   bench/lock.sh with <cmd...>   acquire, run, release even on failure
#
# WHY THIS EXISTS. There is one GPU, 18186 MiB of working set, and a model that
# takes 16 GiB of it. Two benchmark processes at once means both numbers are
# garbage — and not obviously garbage: you get a plausible-looking number that is
# 20% low. bench/env.sh refuses above load 3.0 or with any process over 50% CPU,
# which catches the common case, but a second llama-bench that has not yet
# ramped up passes that check.
#
# HOLD IT ACROSS AN ENTIRE A/B, never per invocation. A lock released between
# arms lets another run land inside yours, which produces a believable number
# instead of an obviously broken one — the worst possible failure mode.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCKDIR="${QM_LOCKDIR:-/tmp/qm-measure.lock}"
STALE_S="${QM_LOCK_STALE_S:-5400}"     # 90 min: longer than any single A/B
WAIT_S="${QM_LOCK_WAIT_S:-7200}"       # how long to block before giving up

_holder() { [[ -f "$LOCKDIR/info" ]] && cat "$LOCKDIR/info" 2>/dev/null; }

_is_stale() {
    [[ -f "$LOCKDIR/pid" ]] || return 0
    local pid age now start
    pid="$(cat "$LOCKDIR/pid" 2>/dev/null || echo 0)"
    # holder process gone -> stale
    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then return 0; fi
    now="$(date +%s)"; start="$(cat "$LOCKDIR/start" 2>/dev/null || echo "$now")"
    age=$(( now - start ))
    (( age > STALE_S ))
}

acquire() {
    local tag="${1:-unnamed}" waited=0
    while true; do
        # mkdir is atomic on POSIX filesystems; this is the whole mutual exclusion
        if mkdir "$LOCKDIR" 2>/dev/null; then
            echo $$ > "$LOCKDIR/pid"
            date +%s > "$LOCKDIR/start"
            printf 'pid=%s tag=%s host=%s started=%s\n' \
                "$$" "$tag" "$(hostname -s)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOCKDIR/info"
            return 0
        fi
        if _is_stale; then
            echo "lock: breaking stale lock held by: $(_holder)" >&2
            rm -rf "$LOCKDIR"
            continue
        fi
        if (( waited == 0 )); then
            echo "lock: waiting for: $(_holder)" >&2
        fi
        sleep 10; waited=$((waited + 10))
        if (( waited > WAIT_S )); then
            echo "lock: gave up after ${WAIT_S}s; holder: $(_holder)" >&2
            return 1
        fi
    done
}

release() {
    if [[ -f "$LOCKDIR/pid" ]]; then
        local pid; pid="$(cat "$LOCKDIR/pid" 2>/dev/null || echo 0)"
        if [[ "$pid" != "$$" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "lock: refusing to release a lock held by pid $pid (we are $$)" >&2
            return 1
        fi
    fi
    rm -rf "$LOCKDIR"
}

case "${1:-status}" in
    acquire) shift; acquire "${1:-unnamed}" ;;
    release) release ;;
    status)  if [[ -d "$LOCKDIR" ]]; then echo "HELD: $(_holder)"; else echo "free"; fi ;;
    with)
        shift
        acquire "${QM_LOCK_TAG:-with}" || exit 1
        trap 'release' EXIT INT TERM
        "$@"
        exit $?
        ;;
    *) echo "usage: lock.sh {acquire [tag]|release|status|with <cmd...>}" >&2; exit 1 ;;
esac
