#!/usr/bin/env bash
# spec.sh — speculative / MTP decoding benchmark. FROZEN.
#
#   bench/spec.sh [--depth N[,N,...]] [--prompts DIR] [--npredict N] [--no-env]
#
# llama-bench takes no speculative flags, so this drives llama-server directly
# and reads the per-request `timings` block out of the HTTP response.
#
# Reports, per draft depth and per prompt category:
#   tok/s, draft acceptance rate, mean accepted tokens per forward pass,
#   and total target forward passes.
#
# A tok/s number without acceptance next to it is unreadable, because
# acceptance IS the multiplier on the bandwidth ceiling:
#     ceiling_spec ~= ceiling_1tok x mean_accepted_per_forward_pass
#
# ---------------------------------------------------------------------------
# DO NOT PASS -md / --spec-draft-model FOR MTP.
#
# common/speculative.cpp:2298-2318 has two branches. If a draft path is set,
# it calls llama_model_load_from_file() and loads a SECOND FULL COPY of the
# weights — 16 GiB again, which cannot fit in this machine's 18.2 GiB working
# set and will fail or thrash. With no draft path, the `else if (spec_mtp)`
# branch calls llama_init_from_model(model_tgt, ...) and builds the MTP draft
# context against the already-loaded target model, so the 16 GiB of WEIGHTS are
# shared. That is the only viable path here and it is the one this script uses.
#
# Note carefully: the weights are shared but `is_mem_shared` is FALSE for this
# architecture. src/llama-context.cpp:142 resets cparams.ctx_other = nullptr and
# restores it only for GEMMA4_ASSISTANT, EAGLE3 and DFLASH; qwen35 is in none of
# those lists, so common/speculative.cpp:1335 evaluates false. What is NOT shared
# is the KV / recurrent memory, which is what that flag gates -- so the draft
# context pays a full catch-up decode plus a host round trip every cycle. Do not
# build a cost model that assumes MTP is free because "memory is shared".
# ---------------------------------------------------------------------------
#
# METRIC DERIVATION. The server's JSON exposes only `draft_n` and
# `draft_n_accepted` (tools/server/server-task.cpp:259-262). Each target
# forward pass verifies a draft and always emits at least the one sampled
# token, so
#     predicted_n = n_forward_passes + draft_n_accepted
# hence
#     n_forward_passes = predicted_n - draft_n_accepted
#     mean_accepted_per_forward_pass = predicted_n / n_forward_passes
# This agrees with the server's own internal mean_acc_len =
# 1 + n_draft_accepted / n_draft_verif_steps (server-context.cpp:651).

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEPTHS="1,2,3,4"; PROMPTDIR=""; NPREDICT=192; RUN_ENV=1; PORT=18080
while [[ $# -gt 0 ]]; do
    case "$1" in
        --depth)    DEPTHS="$2"; shift 2 ;;
        --prompts)  PROMPTDIR="$2"; shift 2 ;;
        --npredict) NPREDICT="$2"; shift 2 ;;
        --port)     PORT="$2"; shift 2 ;;
        --no-env)   RUN_ENV=0; shift ;;
        *) echo "spec.sh: unknown arg $1" >&2; exit 1 ;;
    esac
done

if [[ "$RUN_ENV" == "1" ]]; then
    . "$HERE/env.sh" || exit 1
else
    SKIP_COOLDOWN=1 . "$HERE/env.sh" >/dev/null 2>&1 || :
fi
: "${QM_MODEL:?env.sh did not export the frozen configuration}"

PROMPTDIR="${PROMPTDIR:-$QM_WORK/prompts}"
OUTDIR="$QM_RUNS/spec"; mkdir -p "$OUTDIR"
DAY="$(date -u +%Y%m%d)"
LOGDIR="$(mktemp -d)"
trap 'kill %1 2>/dev/null; pkill -f "llama-server.*--port $PORT" 2>/dev/null; rm -rf "$LOGDIR"' EXIT

start_server() {   # $1 = spec args (may be empty)
    # shellcheck disable=SC2086
    "$QM_BIN/llama-server" \
        -m "$QM_MODEL" \
        -ngl "$QM_NGL" -fa "$QM_FA" \
        -ctk "$QM_CTK" -ctv "$QM_CTV" \
        -c "$QM_CTX" -b "$QM_SERVER_BATCH" -ub "$QM_UBATCH" \
        -np 1 --no-webui --no-context-shift \
        -ctxcp 0 -cram 0 \
        --host 127.0.0.1 --port "$PORT" \
        $1 >"$LOGDIR/server.log" 2>&1 &
    SRV_PID=$!
    for _ in $(seq 1 180); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then return 0; fi
        if ! kill -0 "$SRV_PID" 2>/dev/null; then
            echo "spec: server died on startup:" >&2; tail -25 "$LOGDIR/server.log" >&2; return 1
        fi
        sleep 2
    done
    echo "spec: server did not become healthy in 360s" >&2; return 1
}

stop_server() {
    kill "$SRV_PID" 2>/dev/null
    wait "$SRV_PID" 2>/dev/null
    sleep 2
}

run_suite() {   # $1 = label, $2 = depth-or-"off"
    local label="$1" depth="$2"
    : >"$LOGDIR/results.jsonl"
    for f in "$PROMPTDIR"/*.txt; do
        local base cat
        base="$(basename "$f" .txt)"
        [[ "$base" == "longctx-doc" ]] && continue
        cat="${base%%-*}"
        python3 - "$f" "$cat" "$base" "$PORT" "$NPREDICT" >>"$LOGDIR/results.jsonl" <<'PY'
import json, sys, urllib.request
path, cat, name, port, npred = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
body = json.dumps({
    "prompt": open(path).read(),
    "n_predict": npred,
    "temperature": 0.0,
    "seed": 1234,
    "cache_prompt": False,
}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", body,
                             {"Content-Type": "application/json"})
try:
    r = json.loads(urllib.request.urlopen(req, timeout=1200).read())
except Exception as e:
    print(json.dumps({"name": name, "cat": cat, "error": str(e)})); sys.exit(0)
t = r.get("timings", {})
print(json.dumps({"name": name, "cat": cat, "timings": t}))
PY
    done

    python3 - "$label" "$depth" "$LOGDIR/results.jsonl" <<'PY'
import json, sys, collections
label, depth, path = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [json.loads(l) for l in open(path) if l.strip()]
by = collections.defaultdict(list)
tot = collections.Counter()
for r in rows:
    if "error" in r:
        print(f"  {r['name']:14s} ERROR {r['error']}"); continue
    by[r["cat"]].append(r)
print(f"\n=== {label} (draft depth {depth}) ===")
print(f"  {'category':<10} {'tok/s':>8} {'accept%':>8} {'mean_acc':>9} {'fwd_pass':>9} {'draft_n':>8}")
for cat in sorted(by):
    tps=[]; dn=0; da=0; pn=0
    for r in by[cat]:
        t=r["timings"]
        tps.append(t.get("predicted_per_second") or 0)
        dn += int(t.get("draft_n") or 0)
        da += int(t.get("draft_n_accepted") or 0)
        pn += int(t.get("predicted_n") or 0)
    fwd = pn - da
    acc = 100.0*da/dn if dn else 0.0
    mean_acc = pn/fwd if fwd else 1.0
    print(f"  {cat:<10} {sum(tps)/len(tps):8.3f} {acc:8.2f} {mean_acc:9.3f} {fwd:9d} {dn:8d}")
    tot["dn"]+=dn; tot["da"]+=da; tot["pn"]+=pn; tot["n"]+=len(tps); tot["tps"]+=sum(tps)
fwd = tot["pn"]-tot["da"]
acc = 100.0*tot["da"]/tot["dn"] if tot["dn"] else 0.0
mean_acc = tot["pn"]/fwd if fwd else 1.0
print(f"  {'ALL':<10} {tot['tps']/max(tot['n'],1):8.3f} {acc:8.2f} {mean_acc:9.3f} {fwd:9d} {tot['dn']:8d}")
if tot["dn"] == 0 and depth != "off":
    print("  !! draft_n == 0 — NO DRAFT ACTIVITY. This is NOT a speculative run.")
PY
}

echo "spec: model=$QM_MODEL ctx=$QM_CTX npredict=$NPREDICT"
MEM_BEFORE="$("$HERE/mem.sh")"; echo "spec: $MEM_BEFORE"
echo "spec: $("$HERE/thermal.sh")"

# --- speculation OFF (the reference this is measured against) ---------------
if start_server "--spec-type none"; then
    run_suite "speculation OFF" "off" | tee -a "$OUTDIR/$DAY.txt"
    stop_server
fi

# --- MTP at each requested depth --------------------------------------------
IFS=',' read -ra DARR <<<"$DEPTHS"
for d in "${DARR[@]}"; do
    if start_server "--spec-type draft-mtp --spec-draft-n-max $d --spec-draft-n-min 0"; then
        grep -qiE "MTP draft context|draft-mtp" "$LOGDIR/server.log" \
            || echo "spec: WARNING — server log shows no MTP draft context; check --spec-type spelling" >&2
        run_suite "MTP" "$d" | tee -a "$OUTDIR/$DAY.txt"
        stop_server
    fi
done

echo "spec: $("$HERE/mem.sh")"
echo "spec: results appended to $OUTDIR/$DAY.txt"
