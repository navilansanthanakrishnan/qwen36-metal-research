#!/usr/bin/env bash
# quality.sh — the quality oracle. One command, pass/fail, failure named. FROZEN.
#
#   bench/quality.sh --generate          build the trunk references (run once)
#   bench/quality.sh                     verify against them
#   bench/quality.sh --ctk q4_0 --ctv q4_0     verify a degraded config (must FAIL)
#   bench/quality.sh --only tokens|ppl|kld|ops
#
# The downstream goal is maximum decode speed AT UNCHANGED OUTPUT QUALITY. That
# constraint is worthless unless "unchanged" is mechanically checkable, so this
# checks it four independent ways, from strictest to loosest:
#
#   1. TOKEN-EXACT   greedy (temp 0, fixed seed) output for every frozen prompt
#                    must match the stored trunk reference token for token.
#                    Speculative decoding is lossless BY CONSTRUCTION, so any
#                    divergence is a correctness bug, not a speedup — and it
#                    would show up as suspiciously high acceptance.
#   2. PERPLEXITY    on a frozen wikitext-2 slice. Tolerance: +0.02 absolute.
#   3. KL-DIVERGENCE against trunk's own stored logits. The strictest signal
#                    available; catches degradation that perplexity averages
#                    away. Tolerance on mean and p99 from BASELINE.md.
#   4. test-backend-ops  correctness floor on the Metal backend.
#
# Exit 0 = all pass. Exit 4 = a check failed (the failing check is named).

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GENERATE=0; ONLY=""; PORT=18081
OV_CTK=""; OV_CTV=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --generate) GENERATE=1; shift ;;
        --only) ONLY="$2"; shift 2 ;;
        --ctk)  OV_CTK="$2"; shift 2 ;;
        --ctv)  OV_CTV="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo "quality.sh: unknown arg $1" >&2; exit 1 ;;
    esac
done

SKIP_COOLDOWN=1 . "$HERE/env.sh" >/dev/null 2>&1 || :
: "${QM_MODEL:?env.sh did not export the frozen configuration}"

CTK="${OV_CTK:-$QM_CTK}"; CTV="${OV_CTV:-$QM_CTV}"
QDIR="$QM_WORK/quality"
REFDIR="$QDIR/reference"
CORPUS="$QM_RUNS/corpus/wikitext-2-raw/wiki.test.raw"
KLDBASE="$QM_RUNS/kld/trunk-base.dat"
mkdir -p "$REFDIR" "$QM_RUNS/kld" "$QDIR"

# Frozen quality-check parameters.
Q_NPREDICT=160
Q_PPL_CTX=512
Q_PPL_CHUNKS=40
Q_PPL_TOL=0.02
Q_KLD_MEAN_TOL=0.010
Q_KLD_P99_TOL=0.050

FAILED=0
_pass() { printf '  PASS  %s\n' "$*"; }
_fail() { printf '  FAIL  %s\n' "$*"; FAILED=1; }

want() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }

# ============================================================ 1. TOKEN-EXACT
SRV_PID=""
start_server() {
    "$QM_BIN/llama-server" -m "$QM_MODEL" -ngl "$QM_NGL" -fa "$QM_FA" \
        -ctk "$CTK" -ctv "$CTV" -c "$QM_CTX" -b "$QM_SERVER_BATCH" -ub "$QM_UBATCH" \
        -np 1 --no-webui --no-context-shift --spec-type none \
        -ctxcp 0 -cram 0 \
        --host 127.0.0.1 --port "$PORT" >"/tmp/qual-server-$PORT.log" 2>&1 &
    SRV_PID=$!
    for _ in $(seq 1 180); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        kill -0 "$SRV_PID" 2>/dev/null || { echo "server died:"; tail -20 "/tmp/qual-server-$PORT.log"; return 1; }
        sleep 2
    done
    return 1
}
stop_server() { [[ -n "$SRV_PID" ]] && { kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null; }; SRV_PID=""; }
trap 'stop_server' EXIT

if want tokens; then
    echo "[1/4] token-exact greedy outputs (ctk=$CTK ctv=$CTV)"
    if start_server; then
        for f in "$QM_WORK"/prompts/*.txt; do
            base="$(basename "$f" .txt)"
            [[ "$base" == "longctx-doc" ]] && continue
            got="$(python3 - "$f" "$PORT" "$Q_NPREDICT" <<'PY'
import json, sys, urllib.request
path, port, npred = sys.argv[1], sys.argv[2], int(sys.argv[3])
body = json.dumps({"prompt": open(path).read(), "n_predict": npred,
                   "temperature": 0.0, "top_k": 1, "seed": 1234,
                   "cache_prompt": False, "return_tokens": True}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", body,
                             {"Content-Type": "application/json"})
r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
print(json.dumps(r.get("tokens") or []))
PY
)"
            ref="$REFDIR/$base.tokens.json"
            if [[ "$GENERATE" == "1" ]]; then
                echo "$got" >"$ref"
                echo "  wrote $(basename "$ref") ($(python3 -c "import json,sys;print(len(json.loads(sys.stdin.read())))" <<<"$got") tokens)"
            elif [[ ! -f "$ref" ]]; then
                _fail "token-exact: no reference for $base (run --generate on trunk)"
            elif [[ "$(cat "$ref")" == "$got" ]]; then
                _pass "token-exact $base"
            else
                n="$(python3 - "$ref" <<PY
import json,sys
a=json.load(open(sys.argv[1])); b=json.loads('''$got''')
i=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b)))
print(f"diverged at token {i} of {len(a)} (ref) / {len(b)} (got)")
PY
)"
                _fail "token-exact $base: $n"
            fi
        done
        stop_server
    else
        _fail "token-exact: server would not start"
    fi
fi

# ============================================================ 2. PERPLEXITY
if want ppl; then
    echo "[2/4] perplexity (wikitext-2 test, ctx=$Q_PPL_CTX, chunks=$Q_PPL_CHUNKS)"
    if [[ ! -f "$CORPUS" ]]; then
        _fail "perplexity: corpus missing at $CORPUS"
    else
        PPL_OUT="$("$QM_BIN/llama-perplexity" -m "$QM_MODEL" -ngl "$QM_NGL" -fa "$QM_FA" \
            -ctk "$CTK" -ctv "$CTV" -f "$CORPUS" -c "$Q_PPL_CTX" --chunks "$Q_PPL_CHUNKS" 2>&1)"
        PPL="$(echo "$PPL_OUT" | grep -oE "Final estimate: PPL = [0-9.]+" | grep -oE "[0-9.]+$")"
        if [[ -z "$PPL" ]]; then
            _fail "perplexity: could not parse result"
            echo "$PPL_OUT" | tail -5
        elif [[ "$GENERATE" == "1" ]]; then
            echo "$PPL" >"$QDIR/ppl-baseline.txt"
            echo "  wrote ppl baseline = $PPL"
        elif [[ ! -f "$QDIR/ppl-baseline.txt" ]]; then
            _fail "perplexity: no baseline (run --generate on trunk)"
        else
            BASE="$(cat "$QDIR/ppl-baseline.txt")"
            if awk -v p="$PPL" -v b="$BASE" -v t="$Q_PPL_TOL" 'BEGIN{exit !(p-b<=t)}'; then
                _pass "perplexity $PPL (baseline $BASE, tol +$Q_PPL_TOL)"
            else
                _fail "perplexity $PPL exceeds baseline $BASE by more than $Q_PPL_TOL"
            fi
        fi
    fi
fi

# ============================================================ 3. KL-DIVERGENCE
if want kld; then
    echo "[3/4] KL-divergence vs trunk logits"
    if [[ "$GENERATE" == "1" ]]; then
        "$QM_BIN/llama-perplexity" -m "$QM_MODEL" -ngl "$QM_NGL" -fa "$QM_FA" \
            -ctk "$QM_CTK" -ctv "$QM_CTV" -f "$CORPUS" -c "$Q_PPL_CTX" \
            --chunks "$Q_PPL_CHUNKS" --kl-divergence-base "$KLDBASE" >/dev/null 2>&1
        if [[ -s "$KLDBASE" ]]; then
            echo "  wrote $KLDBASE ($(du -h "$KLDBASE" | cut -f1))"
        else
            _fail "KLD: base logit file was not written"
        fi
    elif [[ ! -s "$KLDBASE" ]]; then
        _fail "KLD: no base logit file (run --generate on trunk)"
    else
        KOUT="$("$QM_BIN/llama-perplexity" -m "$QM_MODEL" -ngl "$QM_NGL" -fa "$QM_FA" \
            -ctk "$CTK" -ctv "$CTV" -c "$Q_PPL_CTX" \
            --kl-divergence-base "$KLDBASE" --kl-divergence 2>&1)"
        KMEAN="$(echo "$KOUT" | grep -iE "Mean KLD" | head -1 | grep -oE "[0-9.]+e?-?[0-9]*" | head -1)"
        K99="$(echo "$KOUT"  | grep -iE "99.0%? KLD|99th" | head -1 | grep -oE "[0-9.]+e?-?[0-9]*" | tail -1)"
        if [[ -z "$KMEAN" ]]; then
            _fail "KLD: could not parse output"
            echo "$KOUT" | tail -8
        else
            ok=1
            awk -v v="$KMEAN" -v t="$Q_KLD_MEAN_TOL" 'BEGIN{exit !(v<=t)}' || ok=0
            [[ -n "$K99" ]] && { awk -v v="$K99" -v t="$Q_KLD_P99_TOL" 'BEGIN{exit !(v<=t)}' || ok=0; }
            if [[ "$ok" == "1" ]]; then
                _pass "KLD mean=$KMEAN p99=${K99:-NA} (tol $Q_KLD_MEAN_TOL / $Q_KLD_P99_TOL)"
            else
                _fail "KLD mean=$KMEAN p99=${K99:-NA} exceeds tolerance $Q_KLD_MEAN_TOL / $Q_KLD_P99_TOL"
            fi
        fi
    fi
fi

# ============================================================ 4. BACKEND OPS
if want ops; then
    echo "[4/4] test-backend-ops (Metal)"
    OPS_OUT="$("$QM_BIN/test-backend-ops" test -b MTL0 2>&1 | tail -30)"
    if echo "$OPS_OUT" | grep -qE "^[0-9]+/[0-9]+ backends passed"; then
        LINE="$(echo "$OPS_OUT" | grep -E "backends passed" | tail -1)"
        if echo "$OPS_OUT" | grep -q "FAIL"; then
            _fail "test-backend-ops: $LINE"
        else
            _pass "test-backend-ops: $LINE"
        fi
    else
        _fail "test-backend-ops: no pass line found"
        echo "$OPS_OUT" | tail -8
    fi
fi

echo
if [[ "$FAILED" == "0" ]]; then
    echo "quality: ALL CHECKS PASS (ctk=$CTK ctv=$CTV)"
    exit 0
else
    echo "quality: FAILED (ctk=$CTK ctv=$CTV) — see the FAIL line(s) above"
    exit 4
fi
