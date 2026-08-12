#!/usr/bin/env bash
# ab.sh — the frozen A/B comparison procedure. FROZEN.
#
#   bench/ab.sh --a <bindir> --b <bindir> [--pairs N] [--la NAME] [--lb NAME]
#
# Runs A and B interleaved in ABBA order, so a linear drift in machine state
# over the session cancels to first order instead of being attributed to
# whichever variant happened to run later. Naive "run all of A, then all of B"
# is how thermal drift gets published as a speedup.
#
# The verdict is deliberately conservative and has TWO independent conditions.
# A difference is reported only if BOTH hold:
#
#   1. the 95% CI of the mean paired relative difference excludes zero, and
#   2. |mean relative difference| >= QM_MDE_PCT, the minimum detectable effect
#      established in NOISE.md.
#
# Condition 2 is what stops the procedure from certifying a 0.4% "win" just
# because that particular batch happened to have low variance. Without it, a
# long enough run of any A/B test eventually reports significance on noise.
#
# PAIR COUNT IS NOT ARBITRARY. The exact sign-flip permutation test below can
# return at best p = 2/2^n, so n = 4 pairs bottoms out at p = 0.125 and n = 5 at
# p = 0.0625 — neither can EVER reach significance at the 5% level no matter how
# large the true effect is. n = 8 bottoms out at p = 0.0078. The default is 8.
# Phase F measured the paired spread at ~3.1% of the mean, which at n = 8 gives a
# 95% CI half-width of ~2.6% — enough to resolve the 4% positive control, and not
# much better than enough. Do not lower it.
#
# Exit codes: 0 = NO DIFFERENCE, 3 = DIFFERENT, 1 = error/refused.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

A_BIN=""; B_BIN=""; PAIRS=8; LA="A"; LB="B"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --a) A_BIN="$2"; shift 2 ;;
        --b) B_BIN="$2"; shift 2 ;;
        --pairs) PAIRS="$2"; shift 2 ;;
        --la) LA="$2"; shift 2 ;;
        --lb) LB="$2"; shift 2 ;;
        *) echo "ab.sh: unknown arg $1" >&2; exit 1 ;;
    esac
done
[[ -n "$A_BIN" && -n "$B_BIN" ]] || { echo "ab.sh: --a and --b are required" >&2; exit 1; }

# Gate once for the whole batch, with the cooldown, then manage spacing here.
. "$HERE/env.sh" || exit 1

# Minimum detectable effect, from NOISE.md. Overridable only for the Phase F
# calibration itself.
QM_MDE_PCT="${QM_MDE_PCT:-3.0}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
: >"$TMP/a"; : >"$TMP/b"

run_one() {   # $1=bindir $2=label $3=outfile $4=tag
    local line tg
    line="$("$HERE/decode.sh" --no-env --bin "$1" --label "$2" --tag "$4" 2>/dev/null)"
    local rc=$?
    tg="$(echo "$line" | sed -nE 's/.*tg128=[[:space:]]*([0-9.]+).*/\1/p')"
    echo "    $line"
    if [[ $rc -eq 0 && -n "$tg" ]]; then
        echo "$tg" >>"$3"
    else
        echo "    ^ excluded from the comparison (not an ok run)" >&2
        echo "" >>"$3"     # placeholder keeps pairing aligned
    fi
    sleep "$QM_COOLDOWN_S"
}

echo "ab: $LA=$A_BIN"
echo "ab: $LB=$B_BIN"
echo "ab: $PAIRS pairs, ABBA interleaved, cooldown ${QM_COOLDOWN_S}s, MDE ${QM_MDE_PCT}%"

for ((i = 1; i <= PAIRS; i++)); do
    echo "  pair $i:"
    if (( i % 2 == 1 )); then
        run_one "$A_BIN" "$LA" "$TMP/a" "ab-p$i-A"
        run_one "$B_BIN" "$LB" "$TMP/b" "ab-p$i-B"
    else
        run_one "$B_BIN" "$LB" "$TMP/b" "ab-p$i-B"
        run_one "$A_BIN" "$LA" "$TMP/a" "ab-p$i-A"
    fi
done

python3 - "$TMP/a" "$TMP/b" "$LA" "$LB" "$QM_MDE_PCT" <<'PY'
import sys, statistics as st, itertools, math

fa, fb, la, lb, mde = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
A = [l.strip() for l in open(fa)]
B = [l.strip() for l in open(fb)]
pairs = [(float(a), float(b)) for a, b in zip(A, B) if a and b]

if len(pairs) < 3:
    print(f"\nab: VERDICT = INCONCLUSIVE ({len(pairs)} usable pairs, need >= 3)")
    sys.exit(1)

a = [p[0] for p in pairs]
b = [p[1] for p in pairs]
# per-pair relative difference, B against A, in percent
d = [100.0 * (bb - aa) / aa for aa, bb in pairs]

n     = len(d)
mean  = st.mean(d)
sd    = st.stdev(d) if n > 1 else 0.0
sem   = sd / math.sqrt(n)
# t critical, 95%, two-sided
tcrit = {2:4.303, 3:3.182, 4:2.776, 5:2.571, 6:2.447, 7:2.365, 8:2.306,
         9:2.262, 10:2.228, 11:2.201, 12:2.179}.get(n - 1, 1.96)
lo, hi = mean - tcrit * sem, mean + tcrit * sem

# Exact paired sign-flip permutation test: under the null, the sign of each
# pair's difference is arbitrary. With n<=12 this enumerates every assignment
# rather than sampling, so the p-value is exact.
count = 0
for signs in itertools.product((1, -1), repeat=n):
    if abs(st.mean([s * x for s, x in zip(signs, d)])) >= abs(mean) - 1e-12:
        count += 1
p = count / (2 ** n)

ci_excludes_zero = (lo > 0) or (hi < 0)
exceeds_mde      = abs(mean) >= mde
different        = ci_excludes_zero and exceeds_mde

print()
print(f"ab: pairs used        {n}")
print(f"ab: {la:<12} mean {st.mean(a):8.4f} tok/s   (runs: {', '.join(f'{x:.3f}' for x in a)})")
print(f"ab: {lb:<12} mean {st.mean(b):8.4f} tok/s   (runs: {', '.join(f'{x:.3f}' for x in b)})")
print(f"ab: paired delta      {mean:+.3f}%  sd {sd:.3f}  95% CI [{lo:+.3f}%, {hi:+.3f}%]")
print(f"ab: permutation p     {p:.4f}  (exact, 2^{n} sign flips)")
print(f"ab: CI excludes zero  {ci_excludes_zero}")
print(f"ab: |delta| >= MDE    {exceeds_mde}  (MDE {mde}%)")
print()
if different:
    direction = "FASTER" if mean > 0 else "SLOWER"
    print(f"ab: VERDICT = DIFFERENT — {lb} is {direction} than {la} by {abs(mean):.2f}%")
    sys.exit(3)
else:
    print(f"ab: VERDICT = NO DIFFERENCE")
    sys.exit(0)
PY
exit $?
