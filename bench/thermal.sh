#!/usr/bin/env bash
# thermal.sh — one line describing the machine's thermal condition. FROZEN.
#
# WHY THIS AND NOT powermetrics: `sudo powermetrics --samplers thermal` is the
# obvious source and it is unusable here — there is no passwordless sudo on
# this machine, so it cannot run inside an unattended benchmark. `pmset -g
# therm` runs without sudo but on macOS 26.5 it reports only a warning *level*,
# which stays unset right up until the machine is already in trouble. Neither
# gives a continuous signal.
#
# So the primary signal is measured, not reported: bench/gpuinfo --clock runs a
# register-resident FMA loop with zero memory traffic, whose throughput is
# proportional to GPU clock x active cores. That is precisely the quantity
# thermal throttling and power capping reduce. It reads 6258 GFLOPs on a cold
# idle machine (see HARDWARE.md) and drops when the chip is hot.
#
# throttle_pct is the shortfall against that frozen cold reference. A benchmark
# taken at throttle_pct > 3 is not comparable to one taken at 0.
#
# Cost: ~60 ms.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUINFO="$HERE/gpuinfo/gpuinfo"

# Frozen cold-idle reference, measured on this machine in Phase A.
# Do not re-fit this to a later measurement — that is how thermal drift gets
# laundered into a clean-looking baseline.
REF_GFLOPS=6258.0

if [[ -x "$GPUINFO" ]]; then
    GF="$($GPUINFO --clock 2>/dev/null || echo 0)"
else
    GF=0
fi

THROTTLE="$(awk -v g="$GF" -v r="$REF_GFLOPS" 'BEGIN{ if(r>0) printf "%.1f", (1-g/r)*100; else print "NA" }')"

# pmset warning levels, kept as a secondary flag: they are rarely set, but when
# they are set they are unambiguous.
THERM="$(pmset -g therm 2>/dev/null | awk -F'= ' '/CPU_Speed_Limit/{print "cpu_speed_limit="$2}' | tr '\n' ' ')"
[[ -n "$THERM" ]] || THERM="cpu_speed_limit=unreported"

PWR="$(pmset -g batt 2>/dev/null | head -1 | grep -qi "AC Power" && echo AC || echo battery)"

printf 'thermal gpu_gflops=%s ref_gflops=%s throttle_pct=%s power=%s %s\n' \
    "$GF" "$REF_GFLOPS" "$THROTTLE" "$PWR" "$THERM"
