# RESUME

Working state for picking this up in a fresh context window. Overwrite freely —
unlike LEDGER.md, this file is scratch.

## STATUS 2026-08-13 (session 2): ~27 +/- 0.5 tok/s, target 40 — NOT reached

Level was ~25.9 at session start and is ~27 now; see CURRENT LEVEL below for the
configuration and LEDGER 089 for why the honest figure carries +/-0.5 (the
six-category mean has ~2.4% run-to-run spread, so 088's +5.8% is a paired 2x2
delta and must not be quoted as a level).

Machine: AC, `iogpu.wired_limit_mb=20480` already applied, throttle 0.1%, lock
free. Re-apply the sysctl after any reboot before quoting absolute numbers.

**Sudo is available this session** (Navilan confirmed; the machine is a spare and
is fully in scope). The "needs Navilan" section of the old RESUME is obsolete.

## Worktree triage — DONE, all ten closed

Every branch is pushed to `origin`; nothing was lost. Worktrees removed because
`git worktree list` had ten entries and the tree was unreadable.

| branch | state | disposition |
|---|---|---|
| `spec-enabled`, `mtp-fit`, `mmap-skip-host-holes`, `s4-q4k-scale-div`, `test-coverage` | 0 commits not in `sgmv-q4k` | **landed**, worktree closed |
| `narrow-n`, `splitk`, `s0-kquant-multicol` | measured, REJECTED (LEDGER 030/034/062) | **closed**, branch kept as evidence |
| `d2h-shared-memcpy` | no commits of its own, empty | **closed** |
| `c1-kquant-ext-gate` | LEDGER 004, still **PENDING** — never measured | branch kept; see W=3 anomaly below |

`llama.cpp` main checkout sits on **`sgmv-q4k`**, which is the real integration
branch. The branch literally named `trunk` is just upstream `0b1bad14f`, i.e.
the frozen baseline — do not confuse them. New work: `narrow-tensor`.

## What this session established (LEDGER 073-079)

**The previous session's headline conclusion was wrong.** LEDGER 072 declared 40
acceptance-bound and put tree drafting behind a new wide-verify kernel. It held
`T_ver(8)=110` fixed, which is the exact failure mode GOAL warns about.

1. **`simdgroup_float8x8` is not the matrix units (074).** `probe/mmapeak.m`
   measures it at **6.1 TFLOP/s**, the same as scalar fp32 fma (5.6), and
   `simdgroup_half8x8` at 6.2 — no fp16 gain. LEDGER 035 *assumed* 17.6 and
   every roofline since inherited it. This retroactively explains why 055, 067,
   069 and 070 all failed on the sgmv kernel: it was already at 67% of a ceiling
   nobody had measured. **Do not attempt a fifth tweak there.**
2. **The tensor API is a different unit and does reach 16.4 TFLOP/s on Q4_K
   (075).** 3.33 ms/verify-column instead of 8.96.
3. **T_ver has a 3x cliff at W>=9 (073)**: 115 ms at W=8, **328-377 flat** from
   9 to 64, because K-quants fall off `mul_mv_ext` into a 128-wide `mul_mm`
   tile. Present with sgmv disabled too, so it is upstream, not ours.
4. **`narrow-tensor` fixes that (076)**: 180-215 ms over ne11 9-32, i.e. 1.8x,
   gated to ne11 >= 9 so it can never touch the production width. Committed and
   pushed.
5. **The tensor path is closed at width 8 (078).** Its 154 ms fixed cost is the
   dequantize-and-store work, ~8.3 ALU ops per weight — not barriers (a 4x
   barrier cut moved it 5%). Even halved it stays above sgmv's 110.
6. **sgmv is FLAT 4->8 (079).** It always computes 8 columns and only masks the
   store. **So 8 tree candidates cost exactly what today's 4-chain costs.**

## The corrected cost model — recompute from this

    cycle = T_ver(W) + D * t_draft        tok/s = acc_per_fwd / cycle

Measured today, from server timings (not inherited numbers):

    T_ver(8)  = 110 ms   flat for W in 4..8   (61 ms weight stream + ~49 ms MMA)
    t_draft   = 6.7 ms   per MTP decode        (NOT the 11.5 the old RESUME said)
    D         = 4-5 decodes = 27-34 ms of drafting
    cycle     = 136-144 ms measured per prompt
    acc/fwd   = 3.05 (prose-01) .. 4.80 (codeedit-01), 3.738 aggregate

**SUPERSEDED BY 084/085 — read those before using the floors above.** LEDGER 077
took `T_ver(8)`'s floor to be the 61 ms weight stream and concluded 40 was cost
slack, not acceptance-bound. That was wrong: it assumed the arithmetic could be
made to overlap the stream, and 084 measured that the only efficient K-quant
arithmetic on this chip runs at 5.07 TFLOP/s and cannot express fewer than 8
columns. **The real floor is `T_ver(4..8) ~ 110 ms`, not 61**, and sgmv is
already ~99% of the 8-column ideal.

Correct floors: `T_ver(8)` **110 ms**, `t_draft` 4.4 ms. Best achievable cycle
~112 ms, so 40 needs acc/fwd **4.50** against today's 3.70 — i.e. it IS
acceptance-bound after all, which is where 072 came out for the wrong reason.
See the bound in 085 below.

## CURRENT LEVEL: 27.43 mean (LEDGER 088), was 25.93

Best configuration measured this session, on branch `tree-sim`:

    GGML_METAL_SGMV_NMIN=3   (now the default on that branch)
    --spec-draft-p-min 0.4   (llama.cpp default is 0)
    plus the frozen config: --spec-draft-n-max 7, MTP_MAX=4, EXT_N=3, RS_SEQ=3

+5.8% over baseline, verified as a 2x2 so the two changes could not be the same
effect counted twice — the gate is +1.5% at both p_min settings and p_min is
+4.2% at both gate settings, no interaction. Concentrated in prose-02 (+25.1%),
prose-01 (+7.2%), longctx (+7.0%).

**Both still need a fresh-context critic before they are KEEPs outright.**
`p_min` cannot change output (the target verifies every draft token) but the
gate is Class B — it changes which kernel instantiation runs. `test-backend-ops`
MUL_MAT q4_K is 3/3.

**`bench/quality.sh` is BLIND to both of these — do not run it and call them
verified.** Greedy generation is ne11=1 and the frozen perplexity slice runs at
ubatch 512; the sgmv gate fires only at ne11=3 and narrow-tensor only at
ne11>=9, so neither kernel is reachable from that gate and it would pass with
the feature on and off. This is exactly the failure mode GOAL names. The correct
Class B check is a **token-exactness diff under speculation**, where the widths
actually occur: same server config, `GGML_METAL_SGMV_NMIN` 4 vs 3, 14 frozen
prompts at temperature 0, diff the completions, and trace any divergence to a
near-tie below the NOISE.md threshold.

**That check RAN and PASSED: token-exact on 14/14 prompts** (LEDGER 090), ~2240
generated tokens with no argmax flip — stronger than Class B requires. So 087
has its quality evidence and needs only an independent critic. `narrow-tensor`
still has none, but it is inert at production widths (zero narrow pipelines
compile during a real speculative run), so there is nothing to verify until a
configuration actually reaches ne11>=9.

## SPECIFIED, NOT BUILT — fuse the catch-up decode with the first draft step

Worth ~4.9% and nobody has looked at it. Per cycle the MTP path runs **1 catch-up
decode + D draft decodes**, i.e. 5 full MTP forwards at 6.7 ms for D=4. The
catch-up lives in `common_speculative_impl_draft_mtp::process()` and exists
because `is_mem_shared` is false for qwen35, so the MTP block's own recurrent/KV
state has to be advanced over the tokens the target just accepted.

It is a separate forward from the first draft step only by construction. The
catch-up batch covers positions up to `n_past-1`; `draft()` then decodes
`dp.id_last` at `n_past` as its first step. **Append `id_last` to the catch-up
batch and the logits at its position ARE the first draft token**, so step 1 of
the draft loop disappears — 6.7 ms out of a ~137 ms cycle, Class A (scheduling
only, token-exact), no kernel work.

Check before building: confirm `id_last` is not already present in `batch_in`
(if it is, the two decodes overlap by one position and the saving is different),
and that the embedding shifting in `process()` — which memcpys `h_tgt` forward by
one row — still lines up when the batch grows by one.

Do NOT confuse this with sharing memory via `ctx_other`. The MTP block has its
own state; adding qwen35 to that list (llama-context.cpp:142) is not obviously
correct and is a different, larger change.

## READ NEXT — the bound, and the only live lead (LEDGER 084/085/086)

**082's plan was falsified by 084 — do not build a column-exact scalar kernel.**
`probe/scmv.m` is written, correct at every shape, and 3.4x off sgmv on
arithmetic efficiency (1.5 vs 5.07 TFLOP/s), because one MMA retires 512 MACs
where a scalar fma retires 32 per simdgroup. `T_ver(4..8) ~ 110 ms is a floor`
and sgmv is already ~99% of the 8-column ideal.

**The bound (085):** with drafting at its 4.4 ms floor and perfect MMA/stream
overlap, the mean over the six frozen categories is **32.8 tok/s** — json 47.4,
codeedit 42.6, prose 24, longctx 21. 40 aggregate needs acc/fwd **4.50** against
today's 3.70, and trees (080), depth (041), FR-Spec (061), the tensor path (078)
and column-exact verify (084) are all closed by measurement. **40 is reachable
per-category on structured content, not in aggregate**, unless a better drafter
appears. The invariant permits any drafter — the target verifies every token —
so this is a draft-quality problem, not a kernel one.

**The live lead: the draft early-stop threshold is mistuned (086).** `p_min`
defaults to 0 and no one has ever swept it. It is a pure policy knob, output
identical, no code change. Measured means over six categories: **0 -> 25.80,
0.4 -> 26.43, 0.75 -> 26.26**, and the per-category envelope is **27.4**.
longctx-01 goes **16.71 -> 21.46 (+28%)** at 0.75 while json prefers 0.4. So one
global constant is wrong for every category and an **adaptive controller** is
worth ~6%. The right rule is `continue while p > acc * t_draft / cycle`
(~0.18 today), but the draft's reported confidence is not calibrated to actual
acceptance — so drive it off the running per-position acceptance the server
already tracks (`n_accepted_per_pos`, exposed by `-lv 5`).

Second-order but real: early stopping makes short drafts common, and a draft of
2 verifies at **width 3, which costs 134.9 ms — worse than width 8's 111**,
because the sgmv gate starts at ne11>=4 and below it K-quants fall back to a
kernel that re-streams all 16.52 GB per column. `GGML_METAL_SGMV_NMIN=3` on
branch `tree-sim` tests lowering the gate; built, not yet measured.

## SUPERSEDED — a scalar 4-column mat-vec

**Tree drafting is dead (080) and wider verify is dead (079/082). The direction
is a NARROWER, cheaper verify.** Read LEDGER 080-083 before doing anything.

`sgmv` computes a fixed 8-wide simdgroup fragment at every width and masks the
store, so W=4 costs the same ~109 ms as W=8. Against my cost model
`max(61, 8.96*W + 23) * 1.17`, sgmv at W=8 measures 108.6-111.3 against an ideal
110.8 — **it is already ~99% optimal for eight columns.** The waste is running
eight columns to verify four.

An ideal kernel doing only the columns it needs is **flat at 71.4 ms out to
W=4**, because the arithmetic does not exceed the 61 ms weight stream until
W ~ 4.2. That 39 ms is 28% of the cycle. Projected mean over the six frozen
categories: **25.9 -> 35.5 tok/s**, or **38.8** with drafting at its floor
(codeedit 47.0, json 47.2, longctx 29.4 from 16.7).

Neither existing kernel can do it:
- `kernel_mul_mv_q4_K_f32_impl` takes the column as `tgpig.y` and **re-streams
  all 16.52 GB per column** — 94.0 ms at W=2, 134.9 at W=3.
- `mul_mv_ext` reads them once but is 1.73x off, and an 18-point sweep of
  `nxpsg` x `nsg` x `r1ptg` cannot fix it (083): best 122.7 ms, which is the
  existing default. Structural, not a constant.

**Build:** sgmv's structure — register-resident, weight stream read once, zero
threadgroup traffic — with the 8x8 simdgroup MMA replaced by **scalar fma into
4 accumulators**. Arithmetic drops 71.6 -> 35.8 ms and falls back under the
61 ms stream, so the kernel becomes bandwidth-bound like `mul_mv` at width 1.
Prototype in `bench/sgmv.metal` first; that loop iterates in seconds where a
llama.cpp rebuild plus A/B is over half an hour.

Watch the register budget: the width-1 kernel holds `yl[16]+yh[16]`, so four
columns of that is 128 floats and will spill. Restructure to load weights once
and stream 4 activations per k rather than holding whole column tiles — 067 and
070 both died of register pressure in this kernel and this is the same trap.

Then re-derive the best draft depth against the new curve. With verify flat to
W=4 the optimum moves shallow (D=3-4), which is the opposite of today's D=7.

## SUPERSEDED — tree drafting at width 8

This is the one large piece the effort has identified repeatedly and never
built, and 079 makes it much cheaper than 072 thought: **the 8 verify slots are
already paid for and today's config wastes them** on a 4-step MTP chain plus a
3-token union extension.

The hard part is NOT the attention mask (16 layers, standard tree mask). It is
the **48 gated-delta-net layers**: a linear recurrence cannot be masked, so each
tree branch needs its own state. The recurrence is `S_node = f(S_parent, x_node)`,
so the fused GDN kernel needs to walk parent indices in BFS order instead of
assuming sequential positions. Per-layer scratch for 8 live states is ~25 MiB,
which fits. **No published work does tree speculative decoding on a hybrid SSM
model** — this is the novel piece.

Ordered:
1. Tree-aware verify: parent-index scan in the fused GDN + tree attention mask.
2. Tree construction in `common/speculative.cpp` (8 nodes, depth 3-4, expand by
   cumulative probability as EAGLE-2 does).
3. Cut D from 4-5 to 3 and push `t_draft` 6.7 -> ~5.

## Do not re-derive

- **Width 16 is a trap.** `T_ver(16)=179` even with narrow-tensor, so a 16-node
  tree needs acc/fwd 8.2 to reach 40. Width 8 is the target, not more.
- **`bench/env.sh` must be run with `bash`, never sourced from this tool shell.**
  The shell is zsh, `BASH_SOURCE` is unset there, so `dirname ""` resolves to the
  parent and every QM_* path comes out wrong. The frozen scripts are fine.
- `bench/spec.sh --depth 1` **aborts** with `LLAMA_ARG_SPEC_EXT_N=3`: the union
  extension overflows a draft buffer sized by `n_max=1`. Harness interaction at
  low depth only; depth 7 is unaffected. Not the production path.
- **W=3 costs 134.9 ms, worse than W=4's 113.3** (073). Unexplained, and it is
  what `c1-kquant-ext-gate` (LEDGER 004) was opened for and never measured.
- `llama.cpp`'s own `AGENTS.md` says fully autonomous agents must **not** open
  PRs there; private forks are explicitly exempt. So work in this fork freely,
  but **prepare** upstream patches rather than submitting them — that call is
  Navilan's. `narrow-tensor` is genuinely upstream-worthy (the W>=9 cliff is not
  specific to this model or chip).
- Frozen perplexity baseline **PPL = 5.9079 +/- 0.14173** at `-c 512 --chunks 40`.
- `llama-cli` one-shot is `-st`. Never pass `-md` for MTP. `llama-server` needs
  `-b` capped at the ubatch plus `-ctxcp 0 -cram 0`.
- Discard pair 1 of any `specab` run (cold-cache outlier); treat swap > ~2000
  pages as measuring memory pressure rather than the change.

## Config that produces the current number

`--spec-type draft-mtp --spec-draft-n-max 7`, `LLAMA_ARG_SPEC_MTP_MAX=4`,
`LLAMA_ARG_SPEC_EXT_N=3`, `LLAMA_ARG_SPEC_N_RS_SEQ=3`, ctx 4096.
