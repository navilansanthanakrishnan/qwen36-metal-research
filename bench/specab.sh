#!/usr/bin/env bash
# specab.sh — interleaved A/B for changes that only act under speculation.
# NEW tooling. Alters nothing the frozen scripts measure; it borrows ab.sh's
# statistics verbatim so verdicts are comparable.
#
#   bench/specab.sh --depth 2 --pairs 6 \
#                   --enva "" --envb "GGML_METAL_NO_KQUANT_EXT2=1" \
#                   --la new --lb upstream
#
# WHY THIS EXISTS. bench/ab.sh wraps llama-bench, which decodes one token per
# forward pass — ne11 = 1. A change gated on src1 width (multi-column mat-vec,
# verify-width kernel selection, draft depth policy) is INVISIBLE to it: both
# arms run identical code and the A/B faithfully reports no difference. That is
# the "gate that passes with the feature on and off" failure mode, arrived at
# from the other direction.
#
# Speculative decode is only reachable through llama-server, and the env var is
# read once per process, so the arms are separate server lifetimes rather than
# separate binaries. Interleaved ABBA over pairs, exactly as ab.sh does, because
# a model load between arms is precisely where drift would otherwise be
# attributed to whichever arm ran second.
#
# ALWAYS reports acceptance and mean accepted tokens per forward pass beside
# tok/s. A speculative tok/s without acceptance next to it is unreadable: it
# cannot be told apart from a drafting change.
#
# Exit: 0 = NO DIFFERENCE, 3 = DIFFERENT, 1 = error.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEPTH=2; PAIRS=6; ENVA=""; ENVB=""; LA="A"; LB="B"; PORT=18095; NPREDICT=192
SPEC_TYPE="draft-mtp"; SPEC_TYPE_A=""; SPEC_TYPE_B=""; DEPTH_A=""; DEPTH_B=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --depth)   DEPTH="$2"; shift 2 ;;
        --pairs)   PAIRS="$2"; shift 2 ;;
        --enva)    ENVA="$2"; shift 2 ;;
        --envb)    ENVB="$2"; shift 2 ;;
        --la)      LA="$2"; shift 2 ;;
        --lb)      LB="$2"; shift 2 ;;
        --port)    PORT="$2"; shift 2 ;;
        --npredict) NPREDICT="$2"; shift 2 ;;
        --spec-type) SPEC_TYPE="$2"; SPEC_TYPE_A="$2"; SPEC_TYPE_B="$2"; shift 2 ;;
        # Per-arm drafter/depth, so a drafting change can be A/B'd under the same
        # pairing as an env change. Without these, comparing two spec-types means
        # comparing two separate runs, which is exactly what the interleaving is
        # here to avoid.
        --spec-type-a) SPEC_TYPE_A="$2"; shift 2 ;;
        --spec-type-b) SPEC_TYPE_B="$2"; shift 2 ;;
        --depth-a)     DEPTH_A="$2"; shift 2 ;;
        --depth-b)     DEPTH_B="$2"; shift 2 ;;
        --bina)    BINA="$2"; shift 2 ;;
        --binb)    BINB="$2"; shift 2 ;;
        *) echo "specab.sh: unknown arg $1" >&2; exit 1 ;;
    esac
done

. "$HERE/env.sh" || exit 1
: "${QM_MODEL:?env.sh did not export the frozen configuration}"
BINA="${BINA:-$QM_BIN}"; BINB="${BINB:-$QM_BIN}"
QM_MDE_PCT="${QM_MDE_PCT:-3.0}"

TMP="$(mktemp -d)"
trap 'pkill -f "llama-server.*--port $PORT" 2>/dev/null; rm -rf "$TMP"' EXIT
: >"$TMP/a"; : >"$TMP/b"; : >"$TMP/rows.jsonl"

run_arm() {   # $1=label $2=envstring $3=outfile $4=bindir $5=pairidx $6=arm(a|b)
    local label="$1" envs="$2" out="$3" bin="$4" pidx="$5" arm="${6:-a}"
    local st="$SPEC_TYPE" dp="$DEPTH"
    if [[ "$arm" == "b" ]]; then
        [[ -n "$SPEC_TYPE_B" ]] && st="$SPEC_TYPE_B"
        [[ -n "$DEPTH_B"     ]] && dp="$DEPTH_B"
    else
        [[ -n "$SPEC_TYPE_A" ]] && st="$SPEC_TYPE_A"
        [[ -n "$DEPTH_A"     ]] && dp="$DEPTH_A"
    fi
    local specargs="--spec-type $st"
    [[ "$st" != "none" ]] && specargs="$specargs --spec-draft-n-max $dp --spec-draft-n-min 0"

    ( unset GGML_METAL_NO_KQUANT_EXT2 GGML_METAL_NO_Q4K_MULTICOL
      [[ -n "$envs" ]] && export $envs
      exec "$bin/llama-server" -m "$QM_MODEL" -ngl "$QM_NGL" -fa "$QM_FA" \
        -ctk "$QM_CTK" -ctv "$QM_CTV" -c "$QM_CTX" \
        -b "$QM_SERVER_BATCH" -ub "$QM_UBATCH" \
        -np 1 --no-webui --no-context-shift -ctxcp 0 -cram 0 \
        $specargs --host 127.0.0.1 --port "$PORT" ) >"$TMP/srv.log" 2>&1 &
    local srv=$!

    local ok=0
    for _ in $(seq 1 200); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }
        kill -0 "$srv" 2>/dev/null || break
        sleep 2
    done
    if [[ $ok -ne 1 ]]; then
        echo "    $label: server failed to start"; tail -8 "$TMP/srv.log"
        echo "" >>"$out"; kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null; return 1
    fi

    local mem_before; mem_before="$("$HERE/mem.sh")"
    : >"$TMP/req.jsonl"
    for f in "$QM_WORK"/prompts/*.txt; do
        local base; base="$(basename "$f" .txt)"
        [[ "$base" == "longctx-doc" ]] && continue
        python3 - "$f" "${base%%-*}" "$base" "$PORT" "$NPREDICT" >>"$TMP/req.jsonl" <<'PY'
import json, sys, urllib.request
path, cat, name, port, npred = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
body = json.dumps({"prompt": open(path).read(), "n_predict": npred,
                   "temperature": 0.0, "top_k": 1, "seed": 1234,
                   "cache_prompt": False}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", body,
                             {"Content-Type": "application/json"})
try:
    r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
except Exception as e:
    print(json.dumps({"name": name, "cat": cat, "error": str(e)})); sys.exit(0)
print(json.dumps({"name": name, "cat": cat, "timings": r.get("timings", {})}))
PY
    done
    local mem_after; mem_after="$("$HERE/mem.sh")"
    local sb sa dswap
    sb="$(echo "$mem_before" | tr ' ' '\n' | awk -F= '/swapins_total/{print $2}')"
    sa="$(echo "$mem_after"  | tr ' ' '\n' | awk -F= '/swapins_total/{print $2}')"
    dswap=$(( sa - sb ))

    kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null; sleep 3

    python3 - "$label" "$pidx" "$dswap" "$TMP/req.jsonl" "$out" "$TMP/rows.jsonl" <<'PY'
import json, sys, collections
label, pidx, dswap, path, out, rows = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
rs=[json.loads(l) for l in open(path) if l.strip()]
good=[r for r in rs if "timings" in r]
if not good:
    open(out,"a").write("\n"); print(f"    {label}: all requests failed"); sys.exit(0)
tps=[]; dn=da=pn=0; percat=collections.defaultdict(lambda:[0,0,0,[]])
for r in good:
    t=r["timings"]
    v=t.get("predicted_per_second") or 0
    tps.append(v)
    dn+=int(t.get("draft_n") or 0); da+=int(t.get("draft_n_accepted") or 0)
    pn+=int(t.get("predicted_n") or 0)
    c=percat[r["cat"]]
    c[0]+=int(t.get("draft_n") or 0); c[1]+=int(t.get("draft_n_accepted") or 0)
    c[2]+=int(t.get("predicted_n") or 0); c[3].append(v)
mean_tps=sum(tps)/len(tps)
fwd=pn-da
acc=100.0*da/dn if dn else 0.0
mafp=pn/fwd if fwd else 1.0
# The harness measures tok/s per request and averages; that is the quantity the
# A/B compares. Swap delta is recorded per arm and gated exactly as decode.sh.
status = "ok" if dswap <= 25000 else "DISCARDED_SWAP"
open(out,"a").write((f"{mean_tps}\n") if status=="ok" else "\n")
open(rows,"a").write(json.dumps({"label":label,"pair":pidx,"tps":mean_tps,"accept_pct":acc,
    "mean_acc_per_fwd":mafp,"fwd_passes":fwd,"draft_n":dn,"swapin_delta":dswap,"status":status,
    "percat":{k:{"accept_pct":(100.0*v[1]/v[0] if v[0] else 0.0),
                 "mean_acc_per_fwd":(v[2]/(v[2]-v[1]) if (v[2]-v[1]) else 1.0),
                 "tps":sum(v[3])/len(v[3])} for k,v in percat.items()}})+"\n")
print(f"    {label:<10} tok/s={mean_tps:7.3f}  accept={acc:6.2f}%  mean_acc/fwd={mafp:5.3f}  fwd={fwd:5d}  swap={dswap}  {status}")
PY
}

echo "specab: depth=$DEPTH spec-type=$SPEC_TYPE pairs=$PAIRS ctx=$QM_CTX"
echo "specab: arm A spec-type=${SPEC_TYPE_A:-$SPEC_TYPE} depth=${DEPTH_A:-$DEPTH} | arm B spec-type=${SPEC_TYPE_B:-$SPEC_TYPE} depth=${DEPTH_B:-$DEPTH}"
echo "specab: $LA env=[${ENVA:-none}] bin=$BINA"
echo "specab: $LB env=[${ENVB:-none}] bin=$BINB"
echo "specab: $("$HERE/thermal.sh")"

for ((i=1; i<=PAIRS; i++)); do
    echo "  pair $i:"
    if (( i % 2 == 1 )); then
        run_arm "$LA" "$ENVA" "$TMP/a" "$BINA" "$i" a
        run_arm "$LB" "$ENVB" "$TMP/b" "$BINB" "$i" b
    else
        run_arm "$LB" "$ENVB" "$TMP/b" "$BINB" "$i" b
        run_arm "$LA" "$ENVA" "$TMP/a" "$BINA" "$i" a
    fi
done

OUTDIR="$QM_RUNS/specab"; mkdir -p "$OUTDIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
cp "$TMP/rows.jsonl" "$OUTDIR/$STAMP-d$DEPTH.jsonl" 2>/dev/null || true

python3 - "$TMP/a" "$TMP/b" "$LA" "$LB" "$QM_MDE_PCT" "$TMP/rows.jsonl" <<'PY'
import sys, statistics as st, itertools, math, json, collections
fa, fb, la, lb, mde, rows = sys.argv[1:7]
mde=float(mde)
A=[l.strip() for l in open(fa)]; B=[l.strip() for l in open(fb)]
pairs=[(float(a),float(b)) for a,b in zip(A,B) if a and b]

R=[json.loads(l) for l in open(rows)]
def agg(lbl):
    rs=[r for r in R if r["label"]==lbl and r["status"]=="ok"]
    if not rs: return None
    return (st.mean([r["accept_pct"] for r in rs]),
            st.mean([r["mean_acc_per_fwd"] for r in rs]),
            st.mean([r["fwd_passes"] for r in rs]))
for lbl in (la,lb):
    g=agg(lbl)
    if g: print(f"specab: {lbl:<10} accept {g[0]:6.2f}%   mean accepted/forward {g[1]:5.3f}   fwd passes {g[2]:7.0f}")

# per-category acceptance, because copy-heavy categories behave nothing like prose
cats=collections.defaultdict(lambda: collections.defaultdict(list))
for r in R:
    if r["status"]!="ok": continue
    for c,v in r.get("percat",{}).items():
        cats[c][r["label"]].append(v)
if cats:
    print(f"\nspecab: acceptance by category")
    print(f"  {'category':<10} " + " ".join(f"{l:>22}" for l in (la,lb)))
    for c in sorted(cats):
        cells=[]
        for l in (la,lb):
            vs=cats[c].get(l,[])
            cells.append(f"{st.mean([x['accept_pct'] for x in vs]):6.2f}% / {st.mean([x['mean_acc_per_fwd'] for x in vs]):5.3f}" if vs else "        n/a")
        print(f"  {c:<10} " + " ".join(f"{x:>22}" for x in cells))

if len(pairs)<3:
    print(f"\nspecab: VERDICT = INCONCLUSIVE ({len(pairs)} usable pairs, need >= 3)"); sys.exit(1)
a=[p[0] for p in pairs]; b=[p[1] for p in pairs]
d=[100.0*(bb-aa)/aa for aa,bb in pairs]
n=len(d); mean=st.mean(d); sd=st.stdev(d) if n>1 else 0.0; sem=sd/math.sqrt(n)
tcrit={2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,10:2.228}.get(n-1,1.96)
lo,hi=mean-tcrit*sem, mean+tcrit*sem
cnt=sum(1 for s in itertools.product((1,-1),repeat=n)
        if abs(st.mean([x*y for x,y in zip(s,d)]))>=abs(mean)-1e-12)
p=cnt/2**n
ci=(lo>0) or (hi<0); ex=abs(mean)>=mde; diff=ci and ex
print(f"\nspecab: pairs used        {n}")
print(f"specab: {la:<12} mean {st.mean(a):8.4f} tok/s   ({', '.join(f'{x:.3f}' for x in a)})")
print(f"specab: {lb:<12} mean {st.mean(b):8.4f} tok/s   ({', '.join(f'{x:.3f}' for x in b)})")
print(f"specab: paired delta      {mean:+.3f}%  sd {sd:.3f}  95% CI [{lo:+.3f}%, {hi:+.3f}%]")
print(f"specab: permutation p     {p:.4f}  (exact, 2^{n})")
print(f"specab: CI excludes zero  {ci}")
print(f"specab: |delta| >= MDE    {ex}  (MDE {mde}%)")
print()
if diff:
    print(f"specab: VERDICT = DIFFERENT — {lb} is {'FASTER' if mean>0 else 'SLOWER'} than {la} by {abs(mean):.2f}%"); sys.exit(3)
print("specab: VERDICT = NO DIFFERENCE"); sys.exit(0)
PY
exit $?
