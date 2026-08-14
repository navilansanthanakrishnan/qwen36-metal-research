# RESUME

Working state for picking this up in a fresh context window. Scratch — overwrite
freely. `LEDGER.md` is the evidence and is append-only; this file is not.

## STATUS 2026-08-14 — ~29.4 tok/s steady-state, target 30, NOT reached

Was 22.8 steady-state at LEDGER 071, so **+29% this session**. The frozen tool's
6-pair means across today's runs: **29.43, 29.47, 29.86, 29.99**. Steady-state
excluding each run's cold first pair is **~29.4**.

**Do not quote a single run as the level.** Pair 1 of every specab reads 1-3
tok/s high (LEDGER 064's cold-cache artifact). Today that artifact produced
headline readings of 30.41, 29.99 and 31.3 that all evaporated on repeat — see
LEDGER 099, which is the correction of my own claim.

### Best configuration

    GGML_METAL_SGMV_NMIN=3
    LLAMA_MTP_VOCAB_N=98304                      # 105: the optimum, NOT 131072
    LLAMA_MTP_VOCAB_FILE=runs/frspec/rank-mixed.txt
    LLAMA_ARG_SPEC_MTP_MAX=4  LLAMA_ARG_SPEC_EXT_N=3  LLAMA_ARG_SPEC_N_RS_SEQ=3
    --spec-draft-n-max 7 --spec-draft-n-min 0 --spec-type draft-mtp
    ctx 4096, b/ub 512, fa on, KV f16

### What this session established

1. **Shrey's unread stack is worth +10.4% on its own (096).** `frspec-test`
   carries B-nr1 and E-gdnfusion, listed in RESUME as "still unread" since 056
   and never merged, while four kernel mechanisms were invented and rejected
   from scratch (067, 069, 070, 084). Merged: same config, acc/fwd identical,
   cycle 138.2 -> 125.1 ms.
2. **FR-Spec was closed against an N 4x too small (095/097/101/105).** 061
   declared it closed having tested only N <= 32768 of a 248320 vocabulary. Its
   own evidence contradicted itself — 100% coverage AND a 26% acceptance loss —
   and the resolution is that the coverage was measured on the ranking corpus,
   not the benchmark. **Verified KEEP: +6.93%, p=0.0312, VERDICT DIFFERENT.**
   Bisecting: 98304 is the optimum (acc/fwd identical, +1.6% over 131072);
   73728 starts losing acceptance. Curve: 32768 -> 21.3, 65536 -> 28.5,
   73728 -> acc/fwd 3.710, **98304 -> acc/fwd 3.744**, 131072 -> 3.744.

### THE OPEN ITEM — worth ~4%, untested on the final stack

**`bench/specab.sh` never passes `--spec-draft-p-min`**, so every specab number
in LEDGER 101-106 was measured at **p_min = 0**, the llama.cpp default. Separate
6-prompt sweeps (086) measured **p_min 0.4 as ~4% better than 0**. Set it via
`LLAMA_ARG_SPEC_DRAFT_P_MIN` in the arm env — the flag has an env binding, so it
works inside specab without editing the frozen script.

That run was launched and interrupted before its verdict. Its partial arms read
**31.2-32.0 tok/s on both sides**, which is the highest this rig has produced —
but two arms of one run is not a level, and the swap counters were 483-7484
pages, so it needs a clean repeat before it means anything.

If p_min is worth what 086 measured, the level lands at ~30.5 and the target is
met; if it is worth nothing here, ~29.4 stands.

## Superseded status line — ~27 ± 0.5 tok/s

Level was 25.93 at session start. The ±0.5 is real: the six-category mean has
~2.4% run-to-run spread (089), so **paired A/B deltas are quotable and absolute
levels are not**. LEDGER 063 and 089 are both corrections of exactly that error.

Machine: AC, `sudo sysctl iogpu.wired_limit_mb=20480` applied (re-apply after any
reboot before quoting absolute numbers). Sudo is available; the old "Needs
Navilan" section is obsolete and has been deleted.

### Best configuration

    GGML_METAL_SGMV_NMIN=3            # default on branch tree-sim
    --spec-draft-p-min 0.4            # llama.cpp default is 0
    --spec-draft-n-max 7 --spec-draft-n-min 0 --spec-type draft-mtp
    LLAMA_ARG_SPEC_MTP_MAX=4  LLAMA_ARG_SPEC_EXT_N=3  LLAMA_ARG_SPEC_N_RS_SEQ=3
    ctx 4096, b/ub 512, fa on, KV f16

+5.8% over baseline, taken as a 2×2 so the gate and `p_min` could not be the same
effect counted twice: gate +1.5% at both `p_min` settings, `p_min` +4.2% at both
gate settings, no interaction (088).

## Git

`llama.cpp` main checkout is on **`sgmv-q4k`** — the real integration branch. The
branch literally named `trunk` is upstream `0b1bad14f`, i.e. the frozen baseline.
Do not confuse them. Session work is on **`tree-sim`** (off `narrow-tensor`).

Worktree triage complete — all ten closed, every branch pushed to `origin`:

| branch | state | disposition |
|---|---|---|
| `spec-enabled`, `mtp-fit`, `mmap-skip-host-holes`, `s4-q4k-scale-div`, `test-coverage` | 0 commits not in `sgmv-q4k` | landed, worktree closed |
| `narrow-n`, `splitk`, `s0-kquant-multicol` | measured, REJECTED (030/034/062) | closed, branch kept as evidence |
| `d2h-shared-memcpy` | no commits of its own | closed |
| `c1-kquant-ext-gate` | answered by **083/087** | closed; can be deleted |

## What is kept, and what it still needs

- **sgmv width gate at `ne11 >= 3`** (087). T_ver(3) 129.5 → 107.4 ms. Below the
  gate K-quants fall back to a kernel that re-streams all 16.52 GB *per column*.
  The gate belongs at exactly 3 — NMIN=2 measures W=2 *worse* (106.4 vs 92.4).
  **Quality: token-exact 14/14 under speculation** (090). Wants a critic.
- **`narrow-tensor`** (076). Removes a 3× cliff at ne11 ≥ 9 (330 → 180-215 ms).
  Correct, and **honestly inert**: zero narrow pipelines compile during a real
  speculative run, because production width is 8. An enabler only.

**`bench/quality.sh` is BLIND to both.** Its greedy pass is ne11=1 and its
perplexity slice runs at ubatch 512, while these fire at ne11=3 and ne11≥9. It
would pass with the feature on *and* off — the failure mode GOAL names. The
correct check is a token-exactness diff **under speculation**, at the widths that
actually occur. That is what 090 is.

## The cost model — recompute from this

    cycle = T_ver(W) + D × t_draft        tok/s = acc_per_fwd / cycle

Measured per prompt from server timings, not inherited:

    T_ver(4..8) = 110 ms   FLAT — the kernel computes 8 columns at every width
    t_draft     = 6.7 ms   per MTP decode (NOT the 11.5 an older RESUME claimed)
    D           = 4-5 decodes  (1 catch-up + n_max draft steps)
    cycle       = 138.8 - 145.2 ms, spread under 5% across ALL categories
    acc/fwd     = 2.37 (longctx) .. 5.33 (json), 3.70 aggregate

Because the cycle is near-constant, **`tok/s ≈ acc/fwd × 7.1`** and the entire
per-category spread is acceptance.

## THE EXIT CONDITION, AS ONE NUMBER (092)

**40 tok/s requires a drafter with sustained per-step acceptance ≥ 0.84.**

Solving the identity with the measured `T_ver(8) = 110 ms`: per-step **0.929** at
MTP's cost, **0.866** at 2.5 ms/step, **0.839 even with a free drafter**. MTP
measures 0.79 → 0.62 → 0.40. The floor on the requirement is set by `T_ver`,
which 084 showed is itself a floor — **so no amount of cheapening the draft
reaches 40.**

Closed by measurement, do not retry: trees (080), depth (041), FR-Spec on three
rankings (061), the tensor path at verify widths (078), column-exact scalar
verify (084), `mul_mv_ext` tuning across 18 points (083), greedy draft proposals
(089), independent draft models at 0.6B/1.7B/4B (092).

## THE PLAN — EAGLE-class draft head

The only remaining route. Kill criteria are part of it: **if held-out sustained
acceptance stalls below ~0.80 after a reasonable budget, 40 is not reachable on
this machine and the honest answer is the ~33 ceiling of 085. Record it, stop.**

**Step 0 — DONE.** `.venv-train` has torch 2.13.0 with MPS + numpy. The blocker
was that torch ships no wheels for Python 3.14, the only installed interpreter;
`brew install python@3.12` resolved it. The venv is 653 MB and **gitignored** —
nothing large enters the source tree.

**Step 1 — training data.** For each corpus position *t*: the target's hidden
state `h_t` and the token at *t+1*. `h_t` is already exposed — it is exactly what
the MTP head consumes — so no new forward path is needed, only a dump tool:

    #include "../src/llama-ext.h"   // as common/speculative.cpp:12 does
    llama_set_embeddings_nextn(ctx, /*value=*/true, /*masked=*/false);
    const float * h = llama_get_embeddings_nextn_ith(ctx, i);   // n_embd = 5120

Both are `LLAMA_API` but declared in that **staging header, not
`include/llama.h`**, so the tool must live inside the llama.cpp tree.
Scaffolding, verified against the tree:

    tools/eagle-dump/CMakeLists.txt   # copy tools/cli's executable stanza:
        set(TARGET llama-eagle-dump)
        add_executable(${TARGET} dump.cpp)
        target_link_libraries(${TARGET} PRIVATE llama-common ${CMAKE_THREAD_LIBS_INIT})
        target_compile_features(${TARGET} PRIVATE cxx_std_17)
    tools/CMakeLists.txt              # add_subdirectory(eagle-dump), near line 17-23

Per ubatch: `common_batch_add` with **logits=true on every position** (or the
hidden state will not exist there), `llama_decode`, then pair each
`llama_get_embeddings_nextn_ith(ctx, i)` with the token at i+1. Write **f16** —
1M tokens is then ~10 GB rather than 20, and the head trains in half precision
anyway. Output to `~/projects/assets/runs/qwen36-metal/eagle/`, never the source
tree. At the measured 306 tok/s prefill, 1M tokens is ~55 min.

Corpus: `runs/corpus/wikitext-2-raw` exists, but 061 is the warning — a
wikitext-only ranking missed code and JSON entirely. Use a mixed corpus. Note
this trains on the *next token* rather than a proposal trim, so 061's specific
failure mode does not transfer directly.

**Step 2 — train.** ~1 decoder layer at n_embd 5120, ~0.4B params, reusing the
target's embedding and LM head (do not train those). **Validate on held-out
per-step acceptance against the real target, not on loss** — 092's threshold is
0.84 sustained and loss does not tell you that.

**Step 3 — integrate.** GGUF conversion plus an arch entry in llama.cpp's graph
code. `--spec-type draft-mtp` already supports chained heads (`chain_heads`,
`n_mtp_layers`), so the drafting loop may need little change.

## Do not re-derive

- **`ps -o etime` is `[[dd-]hh:]mm:ss`, so "03:29" is 3m29s, not 3h29m.** This
  cost a killed `specab` run: I read a 3-minute-old process as 3.5 hours old,
  concluded 6 pairs would take 40 hours, and killed the correct measurement.
  Cross-check elapsed time against `date` and the log file's creation time
  before concluding anything is slow.
- **Long `sleep`s in a backgrounded agent shell do not reliably sleep.** Poll
  with `until <condition>; do sleep 60; done` instead; a bare `sleep 3000`
  returned in seconds here and made a running job look stalled.
- **Model load is ~2 s once the page cache is warm**, not minutes — so a slow
  arm is never the load. If an arm looks slow, the cause is elsewhere.

- **`bench/env.sh` must be run with `bash`, never sourced from an agent shell.**
  That shell is zsh, where `BASH_SOURCE` is unset, so `dirname ""` resolves to the
  parent and every `QM_*` path comes out wrong. The frozen scripts are fine.
- `bench/spec.sh --depth 1` **aborts** with `LLAMA_ARG_SPEC_EXT_N=3` — the union
  extension overflows a buffer sized by `n_max=1`. Harness interaction at low
  depth only; depth 7 is unaffected.
- **Width 16 is a trap.** Even with `narrow-tensor`, `T_ver(16) = 179 ms`, so a
  16-node draft would need acc/fwd 8.2 to reach 40.
- `simdgroup_float8x8` is **6.1 TFLOP/s**, not 17.6 (074). Any reasoning that
  inherits the old figure is ranking against a ceiling that does not exist.
- Register pressure killed 067 and 070 in the sgmv kernel. Any new variant that
  holds more live state per lane will hit it too.
- Frozen perplexity baseline **PPL = 5.9079 ± 0.14173** at `-c 512 --chunks 40`.
- `llama-cli` one-shot is `-st`. Never pass `-md` for MTP. `llama-server` needs
  `-b` capped at the ubatch plus `-ctxcp 0 -cram 0`.
- Discard pair 1 of any `specab` run (cold-cache outlier); treat swap > ~2000
  pages as measuring memory pressure rather than the change under test.
- llama.cpp's own `AGENTS.md` says fully autonomous agents must **not** open PRs
  there; private forks are exempt. Work in this fork freely, but **prepare**
  upstream patches rather than submitting them. `narrow-tensor` is genuinely
  upstream-worthy — the ne11≥9 cliff is not specific to this model or chip.
