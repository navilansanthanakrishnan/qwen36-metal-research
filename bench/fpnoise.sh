#!/usr/bin/env bash
# fpnoise.sh — derive the floating-point noise scale of this backend. NEW
# tooling; alters nothing the frozen scripts measure.
#
#   bench/fpnoise.sh [--ntok 48] [--port 18090]
#
# WHY. GOAL.md requires that before the first Class B verdict we know how large
# |delta logit| gets between two runs that differ only by benign numerical
# reordering. Without that number, "it's just floating point" is a hypothesis
# and not an explanation, and a real bug can hide behind it. With it, a token
# divergence whose logit gap is much wider than the noise scale is a bug
# whatever class the candidate was declared.
#
# METHOD. Two arms of the SAME binary on the SAME weights:
#
#   arm A   default
#   arm B   GGML_METAL_FUSION_DISABLE=1 GGML_METAL_GRAPH_OPTIMIZE_DISABLE=1
#
# Both compute the same mathematical function. Fusion changes accumulation
# order (a fused RMS_NORM+MUL accumulates differently from the two ops run
# separately) and graph reordering changes the order independent nodes execute,
# so the last bits move while the algorithm does not. That is precisely the
# class of difference that makes speculative decoding non-bit-exact on this
# backend, which is why it is the right yardstick for it.
#
# No rebuild is involved, so build variance cannot contaminate the scale.
#
# OUTPUT. Per-position top-k logprobs for every frozen prompt under both arms,
# matched by position, reduced to:
#   p50 / p99 / p99.9 / max of |delta logprob|
#   the distribution of the top1-top2 gap (the "near-tie" quantity)
#   a recommended near-tie threshold at 10x the p99.9 noise
# Written to runs/fpnoise/<date>.json and summarised to stdout.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_COOLDOWN=1 . "$HERE/env.sh" >/dev/null 2>&1 || :
: "${QM_MODEL:?env.sh did not export the frozen configuration}"

NTOK=48; PORT=18090
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ntok) NTOK="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo "fpnoise.sh: unknown arg $1" >&2; exit 1 ;;
    esac
done

OUT="$QM_RUNS/fpnoise"; mkdir -p "$OUT"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"; kill ${SRV:-0} 2>/dev/null' EXIT

start_arm() {   # $1 = label
    local label="$1"
    if [[ "$label" == "B" ]]; then
        export GGML_METAL_FUSION_DISABLE=1
        export GGML_METAL_GRAPH_OPTIMIZE_DISABLE=1
    else
        unset GGML_METAL_FUSION_DISABLE GGML_METAL_GRAPH_OPTIMIZE_DISABLE
    fi
    "$QM_BIN/llama-server" -m "$QM_MODEL" -ngl "$QM_NGL" -fa "$QM_FA" \
        -ctk "$QM_CTK" -ctv "$QM_CTV" -c "$QM_CTX" \
        -b "$QM_SERVER_BATCH" -ub "$QM_UBATCH" \
        -np 1 --no-webui --no-context-shift --spec-type none -ctxcp 0 -cram 0 \
        --host 127.0.0.1 --port "$PORT" >"$TMP/srv-$label.log" 2>&1 &
    SRV=$!
    for _ in $(seq 1 200); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        kill -0 "$SRV" 2>/dev/null || { echo "arm $label: server died"; tail -15 "$TMP/srv-$label.log"; return 1; }
        sleep 2
    done
    return 1
}
stop_arm() { kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; sleep 3; }

collect() {   # $1 = label
    local label="$1"
    : > "$TMP/$label.jsonl"
    for f in "$QM_WORK"/prompts/*.txt; do
        local base; base="$(basename "$f" .txt)"
        [[ "$base" == "longctx-doc" ]] && continue
        python3 - "$f" "$base" "$PORT" "$NTOK" >> "$TMP/$label.jsonl" <<'PY'
import json, sys, urllib.request
path, name, port, ntok = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
body = json.dumps({"prompt": open(path).read(), "n_predict": ntok,
                   "temperature": 0.0, "top_k": 1, "seed": 1234,
                   "cache_prompt": False, "n_probs": 20,
                   "post_sampling_probs": False}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", body,
                             {"Content-Type": "application/json"})
try:
    r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
except Exception as e:
    print(json.dumps({"name": name, "error": str(e)})); sys.exit(0)
probs = r.get("completion_probabilities") or []
out = []
for pos in probs:
    top = pos.get("top_logprobs") or pos.get("probs") or []
    out.append([[t.get("id", t.get("tok_str")), t.get("logprob")] for t in top])
print(json.dumps({"name": name, "pos": out}))
PY
    done
}

echo "fpnoise: arm A (default)"
start_arm A || exit 1; collect A; stop_arm
echo "fpnoise: arm B (fusion + graph-optimize disabled)"
start_arm B || exit 1; collect B; stop_arm

python3 - "$TMP/A.jsonl" "$TMP/B.jsonl" "$OUT/$(date -u +%Y%m%dT%H%M%SZ).json" <<'PY'
import json, sys, statistics as st
A={}; B={}
for path,d in ((sys.argv[1],A),(sys.argv[2],B)):
    for line in open(path):
        r=json.loads(line)
        if "error" in r: print(f"  {r['name']}: ERROR {r['error']}"); continue
        d[r["name"]]=r["pos"]

deltas=[]; gaps=[]; npos=0; ndiverge=0; diverge=[]
for name in sorted(set(A)&set(B)):
    for i,(pa,pb) in enumerate(zip(A[name],B[name])):
        if not pa or not pb: continue
        npos+=1
        da={k:v for k,v in pa if v is not None}
        db={k:v for k,v in pb if v is not None}
        for k in set(da)&set(db):
            deltas.append(abs(da[k]-db[k]))
        sa=sorted((v for v in da.values()), reverse=True)
        if len(sa)>=2: gaps.append(sa[0]-sa[1])
        ta = max(da, key=da.get); tb = max(db, key=db.get)
        if ta!=tb:
            ndiverge+=1
            g = sa[0]-sa[1] if len(sa)>=2 else float('nan')
            diverge.append({"prompt":name,"pos":i,"gap":g})

def q(xs,p):
    if not xs: return float('nan')
    xs=sorted(xs); import math
    return xs[min(len(xs)-1, int(math.ceil(p/100*len(xs))-1))]

res={
 "positions_compared": npos,
 "logprob_pairs": len(deltas),
 "abs_delta_logprob": {"p50":q(deltas,50),"p90":q(deltas,90),"p99":q(deltas,99),
                       "p99.9":q(deltas,99.9),"max":max(deltas) if deltas else None},
 "top1_top2_gap": {"p1":q(gaps,1),"p5":q(gaps,5),"p50":q(gaps,50)},
 "argmax_divergences": ndiverge,
 "divergence_examples": diverge[:10],
}
res["recommended_near_tie_threshold"] = 10.0*res["abs_delta_logprob"]["p99.9"] if deltas else None
json.dump(res, open(sys.argv[3],"w"), indent=2)
print(json.dumps(res, indent=2))
PY
echo "fpnoise: written to $OUT"
