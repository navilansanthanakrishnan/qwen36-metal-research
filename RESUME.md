# RESUME

Working state for picking this up in a fresh context window. Overwrite freely —
unlike LEDGER.md, this file is scratch.

## STATUS 2026-08-13 (session 2): ~22.8 tok/s, target 40

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

Floors: `T_ver(8)` 61 ms, `t_draft` 4.8 ms (994 MiB head + MTP block at
270.8 GB/s). At the floors with D=4 the cycle is 80 ms, which is **46.7 tok/s at
today's acceptance** — so 40 is NOT acceptance-bound. It is cost slack:
`T_ver(8)` is 1.82x its floor, `t_draft` 1.4x.

**What 40 needs, concretely:** cycle 137 -> ~125 ms (D=3 at the draft floor) and
acc/fwd -> **~5.0**. Both terms move, as GOAL requires.

## NEXT — a scalar 4-column mat-vec. NOT trees, NOT wider verify.

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
