# LEADS — ranked dossier

Every entry carries: the claim, the evidence, the predicted mechanism, a rough
predicted magnitude, and the cheapest experiment that would falsify it. An
unranked list of ideas is not a dossier, so the ranking is the point.

**Nothing here has been tried.** This is Phase H output: what to do next and in
what order, with the reasoning attached so the next phase can disagree with it
on evidence rather than on vibes.

---

## The arithmetic that ranks everything

Two numbers decide whether any given idea is worth a day.

**1. Plain decode is at the memory wall.** `ceiling_1tok = 16.40 tok/s` from
16.52 GB of GPU-resident non-MTP weights against 270.8 GB/s attained. Measured
decode is 14.4–14.6 tok/s, i.e. **~88% of the weights-only ceiling**.

**2. Counting the SSM state traffic, it is closer to 94%.** This model's
recurrent state is not free. Per decode token, on top of the weights:

| term | MB/token | source |
|---|---:|---|
| `build_rs` GET_ROWS pulls the 144 MiB state out of the cache | 302 | `src/llama-graph.cpp:3367` |
| the GDN kernel reads that copy and writes an output tail | 302 | `src/models/delta-net-base.cpp:402,415` |
| `ggml_cpy` writes the tail back into `ssm_states_all` | 302 | `src/models/delta-net-base.cpp:555-558` |
| KV cache at `-d 0` (n_kv padded to 256, 16 layers) | 17 | `MODEL.md` |
| **total with weights** | **≈17,440** | |

17.44 GB ÷ 68.5 ms = **254.7 GB/s = 94% of attained bandwidth.**

> **Status of this second figure: derived from code reading, not measured.** It
> is the most load-bearing unverified claim in this file, and it is *also* the
> most attractive target, because unlike the weights those ~906 MB/token are
> deletable. Falsify or confirm it first — see L8. If it holds, the honest
> statement is that decode has ~6% of headroom, all of it bytes, and **any
> optimization that saves only FLOPs or dispatches is worth ≲1% on decode.**

**Consequence.** The decode-side leads below are all small and bounded. The
large numbers in this file are all on the speculative path, because acceptance
multiplies the wall instead of fighting it:

```
ceiling_spec ≈ 16.40 × mean_accepted_tokens_per_forward_pass
```

And one more measured fact makes speculation unusually attractive here: prefill
runs at ~300 tok/s = 3.23 ms per token of marginal compute, against a 68.5 ms
decode step. **Verifying ~20 tokens costs about the same wall time as verifying
one.** The verify side is nearly free; only the *draft* side costs anything.

---

## Ranked

EV = magnitude × probability ÷ cost. Decode Δ is what the project is actually
graded on.

| # | lead | decode Δ | cost | conf | track |
|---|---|---|---|---|---|
| **L1** | K-quant small-batch matvec is a **non-monotonic staircase** in `ne11`; draft depths 1, 2, 5, 6, 7 cost 2–3 full weight streams per verify step, depths 3, 4 and 8+ cost one | **2–3× between worst and best depth** | **zero — pick the depth** | high (verified) | 2/6 |
| **L2** | ngram / prompt-lookup drafting for codeedit + json, `n_max≈16` | **+200–800% on copy-heavy output** | flags | high | 7 |
| **L3** | Benchmark and gate at explicit `--temp 0`; the GGUF's `general.sampling.*` silently override | acceptance ×1.4–2.0 | flag | high | 7 |
| **L4** | `need_n_rs_seq()` ignores ngram types → checkpoint-restore + full replay on every partial acceptance | ×~1.8 on the ngram path | flags, or 3 lines | high | 7 |
| **L5** | Draft-depth sweep against the 0.073 break-even stopping rule | ±20% on the MTP path | one sweep | high | 7 |
| **L6** | Shrink the MTP draft LM head — it re-reads the 1.04 GB `output` tensor per draft token | break-even 7.3%→2.2%/pos; D_opt 3→7 | ~40 LoC | medium | 7 |
| **L7** | Flash-attention has no GQA head grouping (`nhptg=1`); 6 Q heads each stream the same KV | +0.5% @d=0, **+7% @4k, +25% @16k** | 2–4 d | high | 2 |
| **L8** | GDN→CPY recurrent-state writeback is fused on CUDA, a separate 302 MB/token copy on Metal | +1.5–2.0% | 1–2 d | high | 2 |
| **L9** | Q4_K/Q6_K `mul_mv` `nr0` sweep (`N_R0_Q6_K` first) | +3–6% best case, most points under MDE | 1 rebuild | medium | 5 |
| **L10** | Measure what the Metal tensor API is currently worth (it is already ON) | 0% decode by construction; prefill unknown | **1 env var** | high | 4 |
| **L11** | `{MUL_MAT, MUL_MAT, GLU}` gate+up fusion — CUDA-only | +0.4–0.8% | 3–5 d | medium | 2 |
| **L12** | `UNARY(SILU/SIGMOID/SOFTPLUS)+MUL` fusion — 112 dispatches/token, CUDA has it | +0.3–0.5% | 0.5–1 d | medium | 2 |
| **L13** | `SSM_CONV+SILU` fusion — CUDA and Vulkan both have it | +0.2–0.4% | 0.5 d | medium | 2 |
| **L14** | `-ot token_embd.weight=Metal0` to kill the per-draft-step CPU round trip | up to −30% of MTP draft overhead | flag | low-med | 7 |
| **L15** | `ROPE(+VIEW)+SET_ROWS` fusion; only 16 of 32 ROPE nodes are fusable here | +0.1% | 1–2 d | low | 2 |
| **L16** | Raise `iogpu.wired_limit_mb` to buy context back | 0% speed; unblocks long-context work | sudo | high | A |
| **—** | **CUDA-graph / Indirect Command Buffer equivalent** | **≈0% — do not chase** | — | high | 2 |
| **—** | **Tree drafting** | **infeasible: +149.62 MiB per branch** | — | high | 7 |
| **—** | **Rejection sampling instead of greedy-match** | **0 at temp 0, which is the gate** | — | high | 7 |

---

# L1 — the single highest-value finding

**CLAIM.** The number of times the verify step streams the full 16.5 GB weight
set is a **non-monotonic staircase** in the verify batch width `ne11 = 1 + n_draft`,
driven by two independent things: a type gate that excludes K-quants below
`ne11 = 4`, and the `r1ptg` tile table inside `mul_mv_ext`. The practical
consequence is that **`--spec-draft-n-max` 3 or 4 costs one weight stream while
2 costs three**, at zero code cost.

> **This entry was rewritten after verification.** The first draft of this
> dossier claimed the gate made MTP unable to win at all. That was too strong:
> depths 3 and 4 already land on the good rungs. The gate is fatal only for
> depths 1–2, and the tile table is what spoils 5–7. The corrected version is
> *more* actionable, because exploiting it needs no patch.

**The staircase** (every line verified directly against the pinned tree):

| `--spec-draft-n-max` | `ne11` | path | `r1ptg` | full weight streams |
|---|---|---|---|---|
| 1 | 2 | `mul_mv`, K-quants excluded | — | 2 |
| **2** | 3 | `mul_mv`, K-quants excluded | — | **3 — worst** |
| **3** | 4 | `mul_mv_ext` | 4 | **1 — best** |
| **4** | 5 | `mul_mv_ext` | 5 | **1 — best** |
| 5 | 6 | `mul_mv_ext` | 3 | 2 |
| 6 | 7 | `mul_mv_ext` | 4 | 2 |
| 7 | 8 | `mul_mv_ext` | 4 | 2 |
| **8+** | ≥9 | `mul_mm` | — | **1** |

**EVIDENCE** (verified directly, not just reported):

`ggml/src/ggml-metal/ggml-metal-ops.cpp:2335-2361` —
```c
op->src[0]->type == GGML_TYPE_Q8_0 || ... ) && (ne11 >= 2 && ne11 <= 8)
) || (
 ( op->src[0]->type == GGML_TYPE_Q4_K ||
   op->src[0]->type == GGML_TYPE_Q5_K ||
   op->src[0]->type == GGML_TYPE_Q6_K || ... ) && (ne11 >= 4 && ne11 <= 8)
```
This model is 99.4% Q4_K/Q5_K/Q6_K by bytes.

- The `r1ptg` table **already handles 2 and 3**: `ggml-metal-ops.cpp:2388-2391`
  `case 2: r1ptg = 2; case 3: case 6: r1ptg = 3;`. Nothing but the type gate
  excludes them.
- `ne00 % 128 == 0` (line 2335) holds for every large tensor here (5120, 6144,
  10240, 12288, 17408, 1024).
- Fallback: `ggml-metal-device.cpp:813` `int nr1 = 1;` — never reassigned.
  Dispatch `ggml-metal-ops.cpp:2531` uses `(ne11 + nr1 - 1)/nr1` as `grid.y`, so
  each y-slice is an independent full sweep of a weight matrix far too large to
  be cache-resident.
- The `mul_mv_ext` kernel does the right thing: `ggml-metal.metal:4074-4089`
  dequantizes one chunk into `float4x4` registers and dots it against `r1ptg`
  activation vectors — weights read once for all tokens.
- CUDA has no such exclusion: `ggml-cuda/mmvq.cuh:3` `MMVQ_MAX_BATCH_SIZE 8`,
  weight fetch hoisted out of the `ncols_dst` loop (`mmvq.cu:594-611`),
  instantiated for 1..8 (`mmvq.cu:911-1000`).
- Vulkan likewise: `ggml-vulkan.cpp:389` `mul_mat_vec_max_cols = 8`, K-quant
  weights hoisted in `vulkan-shaders/mul_mat_vec_q4_k.comp:17-48`.

**MECHANISM.** Speculation only pays if verifying *n* tokens costs ≈1 weight
sweep instead of *n*. On Metal today it costs *n*.

**MAGNITUDE.** Plain decode (`ne11`=1): **0%, unaffected.** MTP decode: verify
at `ne11`=2 goes from ~2 weight sweeps to ~1.05. At realistic acceptance and
depth 1–2, end-to-end goes from ≈1.0× (MTP is currently a wash or a loss) to
**1.3–1.8×**. This is the difference between the speculative half of the project
working and not working.

**FALSIFY (cheapest).**
`test-backend-ops perf -o MUL_MAT -b MTL0` filtered to Q4_K `[5120,17408]` at
`ne11` = 1, 2, 4. If per-token time at `ne11`=2 is already close to the `ne11`=1
time, the SLC is absorbing the second sweep and the claim is wrong. Otherwise
change `ne11 >= 4` → `ne11 >= 2` at `ggml-metal-ops.cpp:2360`, rebuild, and A/B
`bench/spec.sh` at depth 1 and 2.

**COST.** One line to test. If `mul_mv_ext` underperforms at `r1ptg`=2 (its
`dequantize_q4_K` reloads scales 16× redundantly, `ggml-metal.metal:739-742`),
the real fix is `nr1 = 2` support in the plain K-quant `mul_mv` kernels, 1–2 days.

**Do this first.** It is one line, it is on the only unbounded lever in the
project, and every other speculative measurement is misleading until it is
resolved.

---

# Track 7 — Acceptance-rate levers

Acceptance is the only unbounded lever, so this track is ranked above all the
kernel work despite the kernel work being cheaper.

## The roofline

```
D          = draft depth
a(D)       = mean accepted draft tokens per verify pass
c_draft    = 4.9 ms (MTP)  |  ~0 ms (ngram)
T_step(D)  = max(68.5, 3.23·(D+1)) + c_draft·D   [+68.5 if the replay path fires]
tok/s      = 1000·(1 + a(D)) / T_step(D)
```

`T_verify(N) ≈ max(68.5, 3.23·N)` — **verify is flat out to N≈21.** (Caveat:
3.23 ms/token comes from `pp512`; per-token cost at N=2…16 is higher because
dispatch overhead is not amortized, so the true free zone is smaller. Falsify
with a `-pg` sweep.)

## L3 — the GGUF's sampler settings silently override your defaults

**CLAIM.** Unless `--temp` is passed explicitly, you are measuring at
**temp 1.0, top_k 20, top_p 0.95** — the values baked into this GGUF
(`general.sampling.*`, see MODEL.md) — not at greedy.

**EVIDENCE.** `common/common.cpp:1159-1216` `common_init_sampler_from_model`
reads `LLAMA_MODEL_META_KEY_SAMPLING_{TOP_K,TEMP,TOP_P}` into `sparams`, each
guarded by `if (config & user_config) return;`, where the bit is set **only**
when the corresponding CLI flag is parsed (`common/arg.cpp:1952` temp, `:1960`
top_k, `:1968` top_p). Called from `common_init_from_params`, which the server
uses at `tools/server/server-context.cpp:1199`.

**MECHANISM.** Acceptance = P(target's draw == draft's argmax). At temp 0 that
is P(argmax==argmax), the maximum achievable. At temp 1 + nucleus it is
`p_tgt(argmax_draft)` renormalized — typically 0.3–0.5 on prose, 0.6–0.85 on code.

**MAGNITUDE.** Mean accepted length at temp 1 is roughly **0.5–0.7×** the temp-0
value. At D=3 that is the difference between ~1.7× and ~1.2× end-to-end.

**FALSIFY.** Same prompt and seed, `--temp 0` vs no flag; compare
`draft_n_accepted / draft_n`. Two 60-second runs.

**Status in this rig:** `bench/spec.sh` sends `"temperature": 0.0` in the request
body, which overrides per-request. Safe — but any future harness that relies on
server defaults will silently measure the wrong thing, and **every acceptance
number must be labelled with its temperature.**

## L2 — ngram / prompt-lookup drafting should beat MTP outright on copy-heavy work

**CLAIM.** For codeedit and json — regenerating text that mostly already exists
in the prompt — an ngram method drafts 48–64 tokens at ~zero cost, and verify is
nearly free out to ~21. This dwarfs anything a learned single MTP head can do.

**EVIDENCE.** The ngram impls draft from `dp.prompt` = the entire context
including pasted code (`tools/server/server-context.cpp:3020`).
`common_ngram_mod::draft_one` (`common/speculative.cpp:1853-1907`) matches a
24-token suffix and emits up to 64 tokens; defaults `n_match=24, n_min=48,
n_max=64` (`common/common.h:333-338`). Draft cost is a CPU hash lookup into a
~4 MiB table (`common/speculative.cpp:1807`). `ngram-mod` self-heals: `accept()`
(`:1927-1952`) resets the table after 5 consecutive rounds below 25% acceptance.

**MAGNITUDE**, using `T_verify(N) = max(68.5, 3.23N)`:

| ngram `n_max` | mean accepted | step | tok/s | vs 14.6 |
|---|---|---|---|---|
| 10 | 5 | 68.5 ms | 87.6 | 6.0× |
| 16 | 8 | 68.5 ms | 131 | 9.0× |
| 64 | 8 | 210 ms | 42.9 | 2.9× |
| 64 | 20 | 210 ms | 100 | 6.9× |

`n_max ≈ 16` is the sweet spot — beyond it verify stops being free. These are
upper bounds assuming the acceptance column is achievable; treat the *shape*
(sweet spot at ~16, not 64) as the reliable part.

**FALSIFY.** `--spec-type ngram-mod --spec-ngram-mod-n-max 16` on `codeedit-*`
and `json-*` vs speculation off. If mean accepted length on codeedit is below
~2, the premise that the output is mostly copied is wrong for these prompts.

**COST.** Flags only. **This is the second experiment to run, after L1.**

## L4 — the ngram + recurrent-state landmine

**CLAIM.** `need_n_rs_seq()` counts only *draft-model* types, so an ngram draft
deeper than `n_rs_seq` forces a checkpoint restore and a **full replay** of the
accepted prefix — an extra ~68.5 ms forward pass on every partial acceptance.

**EVIDENCE.** `common/common.h:386-392` returns `draft.n_max` only if MTP /
EAGLE3 / DFLASH / DSPARK is present, else `0u`. Then
`common/common.cpp:1500-1531` `common_context_can_seq_rm` returns
`..._TYPE_FULL` when `llama_n_rs_seq(ctx) == 0`, and
`tools/server/server-context.cpp:3881-3913` takes the
`ckpt.load_tgt(...)` + `spec_is_replay = true` path.

Two ways to hit it: `--spec-type ngram-mod` alone (`n_rs_seq = 0` → replay every
time); or `ngram-mod,draft-mtp --spec-draft-n-max 3` where ngram drafts 48–64 so
`n_rollback ≫ 3`.

**Fix without code:** set `--spec-draft-n-max K` and `--spec-ngram-mod-n-max K`
to the same K ≤ ~10. **Fix with code (~3–5 lines, high EV):** derive
`need_n_rs_seq()` from `common_speculative_n_max()`, or add an explicit
`--spec-n-rs-seq` flag decoupling rollback depth from MTP draft depth.

**MAGNITUDE.** Removing the replay roughly doubles the ngram path's throughput.

**FALSIFY.** Start with `LLAMA_TRACE=1` and grep for
`accepted %2zu/%2zu draft tokens (restore checkpoint)`
(`server-context.cpp:3889`). One request settles it.

## L5 — draft depth, and the stopping rule

**CLAIM.** With `nextn_predict_layers = 1` the head is fed **autoregressively on
its own output**, and there is no depth compensation anywhere in the code.

**EVIDENCE.** `common/speculative.cpp:1335-1336`:
```cpp
is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt;
chain_heads   = n_mtp_layers > 1 && !is_mem_shared;
```
`src/llama-context.cpp:142-161` sets `cparams.ctx_other` only for
GEMMA4_ASSISTANT / EAGLE3 / DFLASH — `qwen35` is none, so `is_mem_shared=false`;
`n_mtp_layers = 1` so `chain_heads = false`. The draft loop
(`common/speculative.cpp:1513-1662`) re-enqueues the head's own hidden row:
```cpp
1583: const float * h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_last[seq_id]);
1634: common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
1635: std::memcpy(batch.embd + ..., h_row, row_bytes);
```
But `h_row` is `shared_head_norm(mtp_block_out)` (`src/models/qwen35.cpp:625-632`)
while the head was trained on the *trunk's* `output_norm(h_L)`
(`src/models/qwen35.cpp:211-213`) — different tensors, different norms. At draft
position ≥2 the head sees a distribution it was never trained on, so acceptance
decays super-linearly. The `llama_set_nextn_layer_offset` rebasing machinery
(`:1554-1562`) is dead code here.

**Break-even.** Each MTP draft token costs ~4.9 ms because it re-reads the full
LM head — see L6. So

```
speedup(D) = (1 + a(D)) · 68.5 / (68.5 + 5·D)
break-even: cumulative acceptance at position D must exceed 0.073·D
```

| D | required mean accepted `a` |
|---|---|
| 1 | 0.073 |
| 3 (default) | 0.219 |
| 5 | 0.365 |
| 8 | 0.584 |

**Stopping rule: add a draft position while its acceptance probability exceeds
0.073.** That number is read directly off the `#acc rate/pos` trace.

**Hard memory constraint on depth.** `common/common.h:386-392` →
`cparams.n_rs_seq = draft.n_max` → `src/llama-memory-recurrent.cpp:99`
`n_rows = mem_size * (1 + n_rs_seq)`. Each snapshot is the full **149.62 MiB**
recurrent state:

| `--spec-draft-n-max` | RS buffer | Δ |
|---|---:|---:|
| 0 | 149.6 MiB | — |
| 3 (default) | 598.5 MiB | +448.9 |
| 5 | 897.7 MiB | +748.1 |
| 7 | 1197.0 MiB | +1047.3 |

At the frozen ctx of 4096 there is room for D up to ~5; dropping context buys
depth, which is a genuine trade this project can make.

**FALSIFY.** One server run per D ∈ {1,2,3,5}, read `#acc rate/pos`. If
`p_{D-1} < 0.073`, D is past optimum. `bench/spec.sh --depth 1,2,3` does this.

## L6 — the MTP draft step re-reads the 1.04 GB LM head

**CLAIM.** 78% of the MTP draft cost is the output head, because this GGUF has
no `shared_head_head` tensor and the draft graph falls back to `model.output`.

**EVIDENCE.** `src/models/qwen35.cpp:637`:
```cpp
ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
```
MODEL.md lists the only nextn tensors as `eh_proj`, `enorm`, `hnorm`,
`shared_head_norm` — **no `shared_head_head`, no `nextn.embed_tokens`.**

| tensor group | bytes |
|---|---:|
| MTP block 64 body | 233.76 MB |
| `blk.64.nextn.*` | 55.77 MB |
| `output` head (Q6_K) | **1042.96 MB** |
| total | **1332.5 MB** |

1332.5 MB ÷ 270.8 GB/s = **4.92 ms** per draft token ≈ 7.2% of a decode step.

**MECHANISM.** Verification is exact (L-note below), so the draft is *allowed*
to be approximate — any lossy shortcut costs acceptance and nothing else.
Replacing the full head with a shortlist (e.g. top 32k tokens → 134 MB) drops
the draft step to ~1.5 ms.

**MAGNITUDE.** Break-even per position falls 7.3% → 2.2%, moving D_opt from ~3
to ~7–8. At a=1.5, D=5: `2.5·68.5/93.5 = 1.83×` → `2.5·68.5/76 = 2.25×`.

**FALSIFY.** No code needed: run D=1 vs D=3 and check whether tok/s tracks
`(1+a)/(1+0.073D)`. If it does, the cost model holds and so does the payoff.

**COST.** ~40 lines in the MTP graph plus a shortlist tensor.

## Verification is exact — and the two knobs that look lossy are not

`common/sampling.cpp:678-706` is a greedy-match accept: the target samples at
each position in order, and the first mismatch is kept (it is the target's own
draw). Every emitted token is a genuine draw from the target, so the output
distribution is exactly the target's and at temp 0 the sequence is the greedy
sequence. **Speculative decoding here is lossless by construction**, which is
why `bench/quality.sh`'s token-exact check is the right gate.

- `--spec-draft-p-min` (default 0.0) only truncates the *draft*
  (`common/speculative.cpp:1597`), never the verify path. But it is a **footgun**:
  the `p` it compares is renormalized over the top-10 (`:1311-1313` plus the
  `dist` sampler appended at `common/sampling.cpp:396`), so it is systematically
  inflated, and too high a value clears the draft entirely and silently reverts
  to plain decode.
- **`--spec-draft-p-split` is dead code.** It is read at exactly one place in the
  whole tree, `examples/speculative/speculative.cpp:67` — never by
  `common/speculative.cpp` or `tools/server/`. **Do not put it in the harness.**
- `--spec-draft-backend-sampling` (default on) offloads only the top-k reduction
  (`src/llama-sampler.cpp:1478-1502` never writes `probs` or selects a token), so
  it cannot change acceptance. If you ever see an acceptance delta from toggling
  it, that is a bug, not a knob.

One caveat to write into the quality gate: bit-exactness is not *guaranteed*,
because logits for a position are computed in a batch of D+1 rows rather than 1,
and Metal reduction order varies with batch shape, so a near-tie argmax can flip.
That is float nondeterminism, not lossiness.

## Combining spec types — fixed priority, and the CLI order is ignored

`--spec-type a,b` does **not** merge drafts; it builds a fallback chain in a
hardcoded priority order (`common/speculative.cpp:2366-2390`), with **all ngram
methods outranking all learned draft heads**. The merge loop (`:2547-2593`)
stops at the first non-empty draft. So `draft-mtp,ngram-mod` and
`ngram-mod,draft-mtp` are the same run. They are genuinely complementary — ngram
covers verbatim-copy regions, MTP covers novel text — so stacking is the right
default. Residual cost: `common_speculative_process` calls every impl's
`process()` each step (`:2517-2519`), so MTP's KV catch-up (~1.1 ms) is paid even
on steps where ngram supplied the draft.

Dispositions for the rest: `draft-simple` needs a small model with a matching
248,320-token vocab (none exists); `draft-eagle3` / `draft-dflash` /
`draft-dspark` need purpose-trained sidecars (none shipped);
`ngram-map-k` / `-k4v` carry per-value acceptance feedback
(`common/ngram-map.h:44-58`) and are a plausible upgrade over `ngram-mod` for
repeated-structure output such as JSON — worth a sweep.

**Build the harness on `llama-server`, not `llama-speculative-simple`**:
`examples/speculative-simple/speculative-simple.cpp:68-69` omits the server's
RS-bounded guard, so a rollback deeper than `n_rs_seq` silently fails on this
architecture.

## Tree drafting — do not pursue on this machine

Multi-branch verify is genuinely supported by `llama_batch` and demonstrated in
`examples/speculative/speculative.cpp:509-560` via multi-`seq_id` entries. But
each branch costs another MTP head evaluation (~5 ms) **and another 149.62 MiB
recurrent snapshot**. On 24 GiB the memory alone kills it. The verify-side
headroom is better spent on longer *linear* drafts from a cheap source (L2).

## How to instrument acceptance

Per request, from the response JSON — this is what `bench/spec.sh` uses:

```
token_accept_rate            = timings.draft_n_accepted / timings.draft_n
mean_tokens_per_forward_pass = predicted_n / (predicted_n - draft_n_accepted)
```

`draft_n` and `draft_n_accepted` are emitted only when `draft_n > 0`
(`tools/server/server-task.cpp:259-262`). The second quantity is the factor that
multiplies the 16.40 tok/s ceiling; it is exact unless the replay path fires
(L4), so grep for `restore checkpoint` to confirm it did not.

Aggregate and **per-draft-position** rates come from `--metrics`:
```
llamacpp:spec_decode_num_draft_tokens_total
llamacpp:spec_decode_num_accepted_tokens_total
llamacpp:spec_decode_num_drafts_total
llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="i"}
```
(`tools/server/server-context.cpp:4456-4517`). `acc_rate_at_position_i =
per_pos{i} / num_drafts_total` is the input to the 0.073 stopping rule. These
counters are cumulative since server start and are not reset by `reset_bucket()`,
so start a fresh server per configuration or snapshot and subtract.

Useful log lines: `draft acceptance = ...` (default verbosity,
`server-context.cpp:663-665`), `acc per pos = (...)` (needs `-lv 4`),
`accepted N/M draft tokens (restore checkpoint)` (needs `LLAMA_TRACE=1`), and
the startup line `size = ... %2u rs_seq` (`src/llama-memory-recurrent.cpp:121-124`)
which confirms how much recurrent state the draft depth actually allocated.

---

# Track 2 — Metal vs CUDA/Vulkan gap

Estimated Metal dispatch count per decode token: **≈1760** (48 SSM layers × ~29,
16 attention layers × ~23, +3), i.e. ~39 µs/dispatch at 68.5 ms. That number is
what prices every fusion entry below.

## L7 — flash attention has no GQA head grouping

**CLAIM.** With `n_head=24 / n_head_kv=4`, Metal launches one threadgroup column
per **Q head**, so each of the 6 Q heads sharing a KV head independently streams
that head's full K and V — 6× the minimum KV traffic. CUDA packs 8 Q heads per
tile, Vulkan packs exactly `gqa_ratio`=6. Metal hard-codes `nhptg = 1`.

**EVIDENCE.**
- `ggml-metal-ops.cpp:3158` `const int nhptg = 1; // heads per threadgroup` — a
  named constant that is never anything else; used at `:3301`, `:3314` with
  `ne02 = n_head = 24`. In-kernel the Q head is a bare threadgroup coord
  (`ggml-metal.metal:7321`) and K/V are re-derived per head (`:7354-7358`).
- The concept exists elsewhere in the same backend:
  `ggml-metal-impl.h:118` `OP_LIGHTNING_INDEXER_NHPTG 8`.
- CUDA: `ggml-cuda/fattn.cu:91-104` `gqa_ratio > 4 → ncols2 = 8`; tile count
  `ntiles_z_gqa = ceil(gqa_ratio/ncols2)` = 1 instead of 6
  (`fattn-common.cuh:1086-1089`); mask read once per tile.
- Vulkan: `ggml-vulkan.cpp:10786-10794` sets `gqa_ratio = qk_ratio` when
  `qk_ratio <= max_gqa` — ratio 6 fits exactly with zero padding waste, strictly
  better than CUDA's `ncols2=8`.
- **This is a specialization gap, not a fallback.** Metal does support head_dim
  256 (whitelist `ggml-metal-device.m:1286`, kernels `ggml-metal.metal:7155`
  non-vec and `:7816` vec) and FA is confirmed on in the measured runs
  (`"flash_attn": 1` in `runs/decode/*.jsonl`).

**MAGNITUDE.** Strongly depth-dependent, and **the frozen benchmark hides it**:

| depth | excess KV read | decode Δ |
|---|---|---|
| `-d 0` (n_kv padded to 256) | 80 MiB | +0.5% |
| `-d 4096` | 1.34 GB | **+7%** |
| `-d 16384` | ~5 GiB | **+25–30%** |

Prefill ≈0 (FA is ~4% of prefill work). Caveat that cannot be settled
statically: the 6 threadgroups for one KV head may run concurrently and hit in
SLC, in which case DRAM amplification is less than 6×.

**FALSIFY — one command, no code change:**
`llama-bench -p 0 -n 64 -d 0,1024,4096,16384`. Fit decode time vs depth. Slope
≈64 KiB/token/depth → no amplification, claim dead. Slope ≈384 KiB → confirmed.
**This experiment also tells you whether L7 matters at your target context**, and
it is worth running early because it re-ranks the whole file if long context is
the real workload.

**COST.** 2–4 days. `nhptg` is already a named constant so the dispatch side is
trivial; the work is threading a head-group index through
`kernel_flash_attn_ext_vec` (`ggml-metal.metal:7276-7295`) so one threadgroup
accumulates `nhptg` separate `(m,s,o)` accumulators over one K/V stream, plus the
matching change in `..._vec_reduce` (`:7881`).

## L8 — GDN → CPY state writeback is fused on CUDA, a separate copy on Metal

**CLAIM.** The gated-delta-net op writes the new recurrent state into its output
tail and the graph then copies it into `ssm_states_all`. CUDA detects this and
has the kernel write the cache directly. Metal runs the copy: 302 MB of
avoidable read+write per decode token across 48 layers.

**EVIDENCE.**
- Graph: `src/models/delta-net-base.cpp:402` `ggml_gated_delta_net(..., K=1)`,
  `:415-419` `new_state` is a view into the output tail, `:555-558`
  `ggml_build_forward_expand(gf, ggml_cpy(ctx0, new_state, ggml_view_2d(ctx0, ssm_states_all, ...)))`.
- Per-layer state = 128×128×48 f32 = 3 MiB; ×48 = 144 MiB (matches MODEL.md's
  "S/state 144.00 MiB"). Read + write = 302 MB.
- CUDA elides it: `ggml-cuda.cu:2729-2731` — *"match gated_delta_net + the
  strided cpy that scatters its state snapshots into the cache … so the kernel
  can write them and skip the cpy"*; matcher `ggml_cuda_try_gdn_cache_fusion`
  (`:2732`), dispatch `:3273-3286`. Landed as `5a460dea9` (#23940), whose commit
  message names the MTP case explicitly: *"With MTP draft length 3, target decode
  uses K=4, so that becomes 4 extra ggml_cuda_cpy calls."*
- Metal: `ggml-metal-ops.cpp:1813-1885` `ggml_metal_op_gated_delta_net` ends
  `return 1;` — no lookahead, no fusion.

**MAGNITUDE.** 302 MB of 17.44 GB = **+1.5–2.0%** decode (14.6 → ~14.85), plus
48 dispatches. Prefill +~1%. Small, but it is one of only two places where
decode *bytes* can actually be deleted.

**FALSIFY.** `GGML_METAL_GRAPH_DEBUG=1` and confirm 48 `CPY` nodes of 3 MiB per
decode step; then patch the graph to skip the CPY and check that generation
breaks — if the model still produces correct text the state is already aliased
and the claim is wrong.

**COST.** 1–2 days, mirroring `ggml_cuda_try_gdn_cache_fusion`.

**Adjacent and the same size, but not a backend gap:** `build_rs`
(`src/llama-graph.cpp:3367`) issues a GET_ROWS that copies the whole 3 MiB state
out of the cache before the GDN reads it — another 302 MB/token that *no* backend
elides. Fixing that is a ggml-graph change worth the same ~1.7%. Together these
two are the entire ~3.5% of removable decode bytes.

## L11 — `{MUL_MAT, MUL_MAT, GLU}` gate+up fusion (CUDA-only)

**EVIDENCE.** The graph shape is confirmed for this model:
`src/llama-graph.cpp:1707` up, `:1729` gate, `:1775` `ggml_swiglu_split` →
`{MUL_MAT, MUL_MAT, GLU}`; `src/models/qwen35.cpp:477-482` uses
`LLM_FFN_SILU, LLM_FFN_PAR` for all 64 layers. CUDA matcher
`ggml-cuda.cu:3029-3055`, kernel takes a second weight pointer
(`mmvq.cu:538`, dual accumulate reusing the same activation block at `:604-611`,
SwiGLU epilogue `:659-687`). **Vulkan does not have this** — its fusion table
(`ggml-vulkan.cpp:17078-17108`) has only MUL_MAT+ADD variants. Metal has **no
mat-mat fusion at all**: `ggml-metal-ops.cpp:2536` `return 1;` unconditionally.

**MAGNITUDE.** Decode: bytes saved 17.8 MB of 17.44 GB = 0.10%; the value is the
128 removed dispatches → **+0.4–0.8%**. Prefill: intermediates are 35.6 MB each,
saving ~9.1 GB per ubatch → **+1.5–2%**. Compounds with L1: at `ne11`=2–3 the
fused form also halves FFN activation traffic.

**COST.** 3–5 days — Metal needs the whole fusion-args plumbing CUDA has.

## L12 — `UNARY(SILU|SIGMOID|SOFTPLUS) + MUL` fusion

**EVIDENCE.** CUDA: `ggml-cuda.cu:3979-3983` (commit `e34f04215`, #21665).
This model emits the pair **112 times per decode token**:
`src/models/qwen35.cpp:374-377` SOFTPLUS+MUL ×48; `:253-255` SILU+MUL ×48;
`:327-330` SIGMOID+MUL ×16. Metal routes each half to its own dispatch
(`ggml-metal-ops.cpp:298-310`), and its only unary-ish fusion is a 5-op snake
chain this model never emits. **The tensors are 48, 6144 and 6144 elements — all
three are pure launch overhead with essentially zero real work.**

**MAGNITUDE.** −112 dispatches/token (6.4% of node count) → **+0.3–0.5%** decode.

**FALSIFY.** Metal capture and read the wall time of the 112 UNARY dispatches. If
they sum to <0.2 ms of 68.5 ms, the per-dispatch cost assumption is wrong and
**L11, L12, L13 and L15 should all be marked down together.**

**COST.** 0.5–1 day for all three, since Metal's bin kernels already take a
fusion count (`ggml-metal-ops.cpp:3449-3547`).

## L13 — `SSM_CONV + SILU` fusion (CUDA *and* Vulkan have it)

**EVIDENCE.** Adjacency confirmed: `src/models/qwen35.cpp:395` `ggml_ssm_conv`,
`:398` `ggml_silu` with no node between. CUDA `ggml-cuda.cu:3974-3977`
(commit `098705a29`, #22478); Vulkan `ggml-vulkan.cpp:16531-16585`
(commit `3fbadb06d`, #22653). Metal `ggml-metal-ops.cpp:1601-1690` has a nice
batched-prefill variant but no epilogue fusion.

**MAGNITUDE.** Decode **+0.2–0.4%** (value is the 48 dispatches; bytes are 0.02%).
Prefill +0.4%.

**COST.** 0.5 day — copy CUDA's matcher, add an `FC_` function constant for the
SILU epilogue, return 2.

## L15 — `ROPE(+VIEW)+SET_ROWS` fusion

CUDA `ggml-cuda.cu:3364-3372` and `:3949-3957`; Vulkan `ggml-vulkan.cpp:17149-17156`.
Metal has neither (`ggml-metal-ops.cpp:392-396`, `:370-374` both `return 1`).
**Important negative:** the 5-op `{RMS_NORM, MUL, ROPE, VIEW, SET_ROWS}` form does
**not** match this model — `src/models/qwen35.cpp:279`/`:290` (Q/K norms) are
separated from `:303`/`:309` (the `ggml_rope_multi` calls) by the K and V
projections. Only `{ROPE, VIEW, SET_ROWS}` for K matches, ×16 layers. **Do not
port this expecting the CUDA PR's headline number.**
Decode +0.1%, prefill +0.2%. 1–2 days.

## Explicitly checked and found NOT to be gaps

Recorded so nobody re-derives them:

- **CUDA graphs / Indirect Command Buffers — do not chase.** Metal re-encodes
  ~1760 dispatches per token on the CPU, each with two `snprintf`s, an `NSLock`,
  and two `std::string`-keyed map lookups (`ggml-metal-device.m:350-368`,
  `ggml-metal-device.cpp:54-59`). It reads badly and costs ≈0: encoding is split
  across the main thread and one worker (`ggml-metal-context.m:445`, `:550`) and
  overlaps GPU execution, so ~1.8 ms of CPU hides inside 68.5 ms of GPU. Metal
  *already* has the barrier elision Vulkan has —
  `ggml-metal-common.cpp:209-373` `ggml_metal_graph_optimize_reorder` plus a
  `MTLDispatchTypeConcurrent` encoder with targeted
  `memoryBarrierWithScope` (`ggml-metal-ops.cpp:143-147`). CUDA's multi-stream
  equivalent is opt-in and off by default. **This is parity, not a gap.**
  30-second check if you doubt it: set `n_cb=2` and observe no change.
- **`RMS_NORM + MUL (+ADD)` and multi-`ADD` chains are already fused on Metal**,
  at parity with CUDA and Vulkan (`ggml-metal-ops.cpp:3742-3796`, `:3455-3496`).
  Vulkan's extra `add_rms_fusion` is decode-only and worth 2.6 MB/token here
  (0.015%) — not worth porting.
- **Metal FA does support head_dim 256.** There is no silent fallback to
  `soft_max + 2 mul_mat`.
- **K-quant mat-vec does not over-fetch the quant payload.** Each block's `qs` is
  read exactly once, fully coalesced (`ggml-metal.metal:8435-8437`). The only
  redundancy is scale bytes, which Vulkan fixes with a shared-memory `sccache`
  that Metal cannot use as written (K-quant pipelines are built with `smem == 0`).
  Since decode is at ~94% of *DRAM* ceiling and this is L1/L2 traffic, ~0.

---

## Track 4 — Metal tensor API

### T4.1 — The tensor path is LIVE on this machine right now. There is nothing to enable.

**CLAIM.** On this M5 Pro, `has_tensor == true` and `GGML_METAL_HAS_TENSOR` **is**
defined in the shader the process actually compiles. The "disabled by default"
gate is a *pre-M5* gate and it does not fire here. No env var or build flag needs
flipping to turn it on; the only flag that matters is the one that turns it off.

**EVIDENCE.**
- `ggml-metal-device.m:743` — `dev->props.has_tensor = [dev->mtl_device supportsFamily:MTLGPUFamilyMetal4_GGML];` with `MTLGPUFamilyMetal4_GGML = 5002` at `:27`. Init log prints `MTLGPUFamilyMetal4 (5002)` → true.
- `:753-760` — the actual gate:
  ```
  if (getenv("GGML_METAL_TENSOR_ENABLE") == NULL &&
      ![[dev->mtl_device name] containsString:@"M5"] && ... ) { has_tensor = false; }
  ```
  `[mtl_device name]` is `"Apple M5 Pro"`, so `containsString:@"M5"` is true → the
  disable branch is skipped. `GGML_METAL_TENSOR_ENABLE` is a no-op on this chip.
- `:763-830`, `:842-859` — the two runtime compile probes; the log shows both, and
  `:940` prints `has tensor = true`.
- `ggml-metal-device.m:221-228` — `if (props->has_tensor) [prep setObject:@"1" forKey:@"GGML_METAL_HAS_TENSOR"]`, applied to `newLibraryWithSource:options:`.
- `GGML_METAL_EMBED_LIBRARY=ON` means the source path with the macro dictionary is
  the one taken. (An AOT `default.metallib` build passes no `-DGGML_METAL_HAS_TENSOR`,
  so the tensor path would be dead code even on M5 — not this build.)

**MECHANISM.** Three independent conditions all pass and the macro is injected.
**MAGNITUDE.** Zero change available — it is already on. What is unmeasured is what it is currently *worth*.
**FALSIFY.** `GGML_METAL_TENSOR_DISABLE=1`, no rebuild. Expect prefill to move, decode not to.
**COST.** One env var, via a shim dir passed as `ab.sh --b`. Cheapest real experiment in the project.

### T4.2 — The TODO, verbatim

`ggml-metal-device.m:748-752`:
```
// note: disable the tensor API by default for old chips because with the current implementation it is not useful
// - M2 Ultra:   ~5% slower
// - M4, M4 Max: no significant difference
//
// TODO: try to update the tensor API kernels to at least match the simdgroup performance
```
`GGML_METAL_HAS_TENSOR` appears in exactly two places, both mat-**mat**:
`ggml-metal.metal:10042-10163` (`kernel_mul_mm`, legacy `#else` at `:10167-10381`)
and `:10467,10522,10543,10681,10719` (`kernel_mul_mm_id`). **There is no tensor-API
mat-vec kernel at all.**

| | legacy | tensor |
|---|---|---|
| A×B tile | 64×32 | **64×128** |
| B staging | threadgroup memory | none — wraps device memory |
| accumulator | 16 regs/thread | 64 regs/thread |
| threadgroup mem | 6144–8192 B | 4096 B |

### T4.3 — None of it is on the decode hot path

**CLAIM.** For this model at batch 1 the tensor API contributes **0.00%** of decode time.
**EVIDENCE.**
- `ggml-metal-ops.cpp:2331` `ne11_mm_min = 8`; `:2440` selects `mul_mm` only when `ne11 > 8`. Decode has `ne11 = 1` → always `get_pipeline_mul_mv` (`:2486`).
- The `mul_mv_ext` branch (`:2335-2363`) needs `ne11 >= 4`.
- **The 48 GDN layers do no `ggml_mul_mat` at decode at all**: `src/models/delta-net-base.cpp:434-439` routes `n_seq_tokens == 1` to `build_delta_net_fused` (`cparams.fused_gdn_ar = true`, `src/llama-context.cpp:232`), emitting a single `ggml_gated_delta_net` op (`:402`). Only `build_delta_net_chunking` has mat-muls, and it is prefill-only.
- The 16 attention layers use `flash_attn_ext`.
- `MUL_MAT_ID` never fires — the model is dense.
- At prefill (`ub=512`) every weight matmul goes through tensor `kernel_mul_mm`.
- Bounds check clean: `bc_out = (ne0 % 64 || ne1 % 128)`; all `ne0` divisible by 64, `ne1=512` by 128.

**MAGNITUDE.** Decode 0% (structural). Prefill: unmeasured, plausibly large.
**FALSIFY.** `GGML_METAL_TENSOR_DISABLE=1` must leave tg128 inside the noise band while moving pp512.

### T4.4 — What tuning is left, and what breaks

`ggml-metal-impl.h:8-15` (`// TODO: become function constants`):
`SZ_SIMDGROUP 16, N_MM_NK 2, N_MM_BLOCK_X 4, N_MM_BLOCK_Y 2, N_MM_SIMD_GROUP_X 2, N_MM_SIMD_GROUP_Y 2`.
- `NRB=128`/`NRA=64` are at the register wall (64 f32/thread for the accumulator). Not candidates.
- **`N_MM_NK` 2→4 is free**: smem 4096→8192 B against a 32 KiB limit; K descriptor 64, still a multiple of 16; K ∈ {5120,6144,17408} divides exactly. Halves barrier pairs.
- **Re-split at constant tile**: `N_MM_SIMD_GROUP_X 2→4` with `N_MM_BLOCK_X 4→2` keeps the tile but halves accumulator registers to 32/thread. Fails loudly at pipeline compile if `matmul2d` rejects it.

**MAGNITUDE.** Decode 0% by construction. Prefill ±3% (`N_MM_NK`), ±10% (re-split).
**COST.** One rebuild + one A/B. Rank last — this is a prefill project and prefill is not the target.

### T4.5 — Dead hook: `device_id` is computed and never read

`ggml-metal-device.h:226-247` defines `GGML_METAL_DEVICE_M5_PRO`; `:269` declares
`device_id`; the only occurrence in the backend is the write at
`ggml-metal-device.m:881`. Nothing reads it. Combined with `ggml-metal-impl.h:22`
(`// TODO: for optimal performance, become function of the device and work size`)
this is the upstream-sanctioned landing site for Track 5's results. ~20 lines.

---

## Track 5 — Q4_K/Q6_K mul_mv template parameters

> The signature in the brief, `kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K, N_SG_Q4_K, N_SIMDWIDTH>`,
> is the **old** upstream signature. At `0b1bad14f` it is `template<int nr0, typename args_t>`
> — one numeric parameter. `N_SG` became a **runtime function constant** in `da4495332`.
> This makes the two axes cost very different amounts to sweep.

### T5.1 — Where the parameters live

| symbol | definition | value | binding |
|---|---|---|---|
| `N_R0_Q4_K` | `ggml-metal-impl.h:54` | **2** | compile-time template arg (`metal:8495`) |
| `N_SG_Q4_K` | `ggml-metal-impl.h:55` | **2** | host-only → function constant `FC_MUL_MV+0` → `metal:3541` `FC_mul_mv_nsg` |
| `N_R0_Q5_K` | `:57` | **1** | template arg (`metal:8626`) |
| `N_SG_Q5_K` | `:58` | **2** | host-only (`device.cpp:897`) |
| `N_R0_Q6_K` | `:60` | **2** | template arg (`metal:8734`) |
| `N_SG_Q6_K` | `:61` | **2** | host-only (`device.cpp:902`) |
| `N_SIMDWIDTH` | `ggml-metal.metal:28` | **32** | `#define`, not tunable |

`N_R0_Q5_K = 1` because of `b54124110` — "metal : fix q5_k mul_mv register spill (#20399)",
a one-line revert of 2→1. **That commit is the empirical ceiling marker for this exercise.**

`N_SIMDWIDTH` has no legal alternative: it appears zero times inside the K-quant impls,
which hardcode the lane split (`tiisg/8`, `tiisg%8` etc.) and use `simd_sum`; dispatch
hardcodes 32 threads in x (`ops.cpp:2531`).

### T5.2 — Legal parameter space, derived from the kernel body

**The 32 KiB threadgroup-memory limit is not a constraint on these kernels — they use zero threadgroup memory.**
- `device.cpp:812` `smem = 0`; the Q4_K case (`:891-894`) never touches it; `ops.cpp:2523` sets 0. Reduction is `simd_sum` (`metal:8478`), intra-simdgroup only. (Contrast F32/F16/BF16/Q8_0, which reduce through `helper_mv_reduce_and_write` and are capped at `NSG ≤ 32` by a 32-slot array. **K-quants have no such cap.**)
- `metal:8404` `first_row = (r0*NSG + sgitg)*nr0` — each simdgroup owns its own rows, so any `NSG ≥ 1` is correct.
- **No assert on threadgroup size.** `ops.cpp:2531` dispatches `(32, nsg, 1)`; exceeding `maxTotalThreadsPerThreadgroup` is a validation failure at dispatch, not a fallback.
- **Unguarded input over-read.** Output is guarded (`metal:8477`), input is not (`:8409`); the last threadgroup can read up to `nr0*NSG - 1` rows past the tensor. **Rule: pick `nr0*NSG` dividing `ne01` for every matrix.** GCD of {17408,5120,10240,12288,1024,6144,248320} = **512** → any power of two `nr0*NSG ≤ 512` is safe. Exclude `nr0 ∈ {3,5,6,7}`.
- K dimension: `nb = ne00/256` ∈ {20,68,24}, all divisible by the 4-way lane stride → balanced.
- **Register pressure is the binding constraint.** Q4_K live state ≈ `52 + 9·nr0` regs (`yl[16]+yh[16]`, `sumf[nr0]`, `sumy`, `sc16[4]`, ~14 for pointers, `acc1[4]+acc2[4]` per unrolled row). 128 regs before spilling. **nr0=4 is the boundary; nr0=8 will spill.**

Legal grid for Q4_K (correctness + no over-read + no tail): `nr0 ∈ {1,2,4}` (8 = spill risk) × `NSG ∈ {1,2,4,8,16}`, all satisfying `nr0*NSG | 512`.

### T5.3 — Q6_K and Q5_K

**Q6_K** (`metal:8630-8722`), 26.5% of bytes, carries `attn_qkv` on all 48 SSM layers, `ffn_down`, and `output.weight`:
- `smem = 0`, same structure, same over-read.
- **Half the vector register state of Q4_K**: `float yl[16]` only (`metal:8665`), no `yh`. ≈ `35 + 5·nr0` regs. **Most headroom of the three.**
- K split is only **2-way** per simdgroup (`metal:8668`), so each lane runs a long sequential chain — more independent row streams is exactly what it wants.
- `output.weight` has `ne01 = 248320 = 2^9 × 485`, so `nr0*NSG ≤ 512` is the binding limit.

**Q5_K** (`metal:8499-8614`), 6.1% of bytes, is **`ssm_out` [6144,5120] on all 48 SSM layers and nothing else**:
- Heaviest register footprint: `yl[16]+yh[16]`, `acc1[4]+acc2[4]`, `sumy`, `sc16[4]`, `hm1..hm4`, six pointers.
- **Raising `N_R0_Q5_K` re-creates a bug upstream explicitly fixed (`b54124110`). Do not.**
- `N_SG_Q5_K` costs zero registers and zero smem (sole use at `metal:8516`) — the only free knob on this path.

### T5.4 — Dispatch geometry, and an honest negative result

`ggml-metal-ops.cpp:2531`: grid is 1-D, `ceil(ne01 / (nr0·NSG))` threadgroups of `32·NSG` threads.

| tensor | type | K | rows | rows/TG | TGs | tail |
|---|---|---|---|---|---|---|
| ffn_gate, ffn_up | Q4_K | 5120 | 17408 | 4 | 4352 | none |
| ffn_down | Q4_K/Q6_K | 17408 | 5120 | 4 | 1280 | none |
| attn_gate (SSM) | Q4_K | 5120 | 6144 | 4 | 1536 | none |
| attn_q | Q4_K | 5120 | 12288 | 4 | 3072 | none |
| attn_k/v | Q4_K | 5120 | 1024 | 4 | 256 | none |
| attn_output | Q4_K | 6144 | 5120 | 4 | 1280 | none |
| attn_qkv (SSM) | Q6_K | 5120 | 10240 | 4 | 2560 | none |
| output.weight | Q6_K | 5120 | 248320 | 4 | 62080 | none |
| ssm_out (SSM) | Q5_K | 6144 | 5120 | **2** | 2560 | none |

**CLAIM (negative). There is no bad shape interaction to exploit.** Every `ne01`
divides by the current `nr0·NSG`; every `nb` divides by the K stride. Tail
elimination — normally the easiest win in this kind of tuning — is worth **exactly
zero** here.

- **`NSG` is nearly a no-op.** Total simdgroups launched = `ne01/nr0`, independent of NSG. It only regroups them into fewer, fatter threadgroups. Predicted ±1%.
- **`nr0` is first-order.** It gives each thread `nr0` independent load streams at stride `nb01` over the same registers — the classic latency-hiding lever for a matvec.

**MAGNITUDE and the wall.** Total DRAM weight traffic is **invariant** under `(nr0, NSG)`.
Decode runs at 85–89% of attained bandwidth, so the absolute ceiling on all Track 5
work is `1/0.87 − 1 ≈ +15%`, requiring 100% of attained bandwidth and zero non-matmul
time. **Realistic best case +3% to +6% decode.** Amdahl weights: Δ% kernel improvement
yields **0.69·Δ** (Q4_K), **0.27·Δ** (Q6_K), **0.06·Δ** (Q5_K) end-to-end.
**Prefill effect: exactly 0%** — at `ub=512` the path is `mul_mm`, which never reads these constants.

**Sobering.** Most individual points will land below the 3.0% MDE and be reported as
NO DIFFERENCE. Plan for the combination, not the singletons.

### T5.5 — Ranked, executable sweep plan

**Step 0 — make the sweep cheap.** Either (A) edit `ggml-metal-impl.h` and rebuild per
point (~2–3 min each; one edit keeps host and device consistent because
`CMakeLists.txt:46` splices the header into the embedded shader), or **(B, recommended)**
two small patches make the whole grid env-driven with one rebuild total:
1. NSG (host-only, ~6 lines): wrap `device.cpp:892/897/902` with `getenv`. The pipeline
   cache key already includes `nsg` (`device.cpp:963`), so values coexist safely.
2. nr0 (~10 lines): copy the existing dispatch pattern at `metal:4345-4350`
   (`kernel_mul_mv_t_t_disp` already does `switch (args.nr0)`), add a
   `kernel_mul_mv_q4_K_f32_disp`; `args.nr0` is already populated (`ops.cpp:2512`).

**Step 1 — free register screen before spending a bench slot.** Load once with `-v` and
`grep 'th_max' | grep kernel_mul_mv_q[456]_K_f32`. `th_max` is a direct proxy for
register pressure (`device.m:428-431`); if it drops vs baseline, kill that point
unbenched. It also gives the hard NSG ceiling (`NSG ≤ th_max/32`).

**Step 2 — ranked points**, each `bench/ab.sh --pairs 5` against trunk:

| # | change | rationale |
|---|---|---|
| 1 | `N_R0_Q6_K 2→4` | Best value/risk: 26.5% of bytes, lightest register footprint (no `yh`), K loop only 2-way split so most latency-starved. |
| 2 | `N_R0_Q4_K 2→4` | Largest Amdahl weight (×0.69) but same `yl+yh` pattern that spilled Q5_K. **Gate on the Step-1 `th_max` check.** |
| 3 | `N_SG_Q5_K 2→4, →8` | Only free knob for the crippled path (`ssm_out`, 48 layers, nr0=1). Small weight (×0.06); run mainly as evidence on the threadgroup-slot hypothesis. |
| 4 | `N_SG_Q4_K 2→4, →8` | Free, largest weight, cleanest test of "is occupancy threadgroup-slot-limited?". A real move here reframes the track. |
| 5 | `N_SG_Q6_K 2→4` | Same hypothesis on the 26.5% path; cuts `output.weight`'s 62080 threadgroups to 31040. |
| 6 | **combine the winners** | The singletons will mostly sit under the MDE; the combination is where the track clears the bar or is honestly declared dead. Run even if every singleton said NO DIFFERENCE. |
| 7 | `N_R0_Q6_K 4→8` (only if #1 won and `th_max` held) | Q6_K's register model says 8 may fit where Q4_K's cannot. |
| 8 | `N_R0_Q5_K 1→2` | **Only after a register diet.** As-is this re-creates the exact spill `b54124110` fixed. Do not run blind. |

**Do not try:** `nr0 ∈ {3,5,6,7}` (tails + over-read); `nr0=8` for Q4_K/Q5_K (spill);
`32·NSG > th_max` (dispatch failure, no assert to catch it); any change to `N_SIMDWIDTH`.

**FALSIFY the whole track cheaply.** Before points 1–8 run
`test-backend-ops perf -o MUL_MAT -b Metal -p 'type_a=q4_K.*n=1'`. It isolates the
kernel with far lower variance than end-to-end tg128, so it resolves a 2% kernel change
that `ab.sh` would call noise. **If the isolated Q4_K/Q6_K matvec already runs at >95%
of the 270.8 GB/s probe, the entire track is dead** and effort should move to the
non-matmul decode time — the GDN kernels and per-node encoder overhead that occupy the
remaining 11–15%.

**COST.** Step 0(B)+1: one rebuild, ~10 min, no bench slot. Points 1–6: six `ab.sh` runs
at `--pairs 5`, ~3 h wall. Still, T4.1's `GGML_METAL_TENSOR_DISABLE=1` A/B is cheaper
(zero rebuilds) and should run first.

---

# Track 3 — The unified-memory embedding copy

**Verdict up front: this is a dead end as a speed lever, and that is a useful
result.** The copy PR #22673 introduced is real, is still present at the pinned
commit, and is completely ungated on unified memory — and fixing it perfectly
buys **≤0.5% of decode and ~0.005% of prefill**. The value of the investigation
is the four larger things found sitting next to it.

## What #22673 did, and what actually got fixed

[PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673), merged 2026-05-16
(`255582687`), added MTP as a second `llama_context` over the same model plus the
`embeddings_nextn` plumbing. Its body says the quiet part out loud: *"Prompt
processing (PP) speed typically takes a negative hit when MTP is enabled mainly
due to Device-To-Host (D2H) embedding transfers."* Its TODO **"Avoid D2H + H2D
pre-norm embedding transfers somehow?" is still unchecked.**

But the regression as shipped was not mainly the 20 KiB embedding row. It was
that the target context reported `need_embd() == true`, so the server flagged
every prompt token as an output and the **LM head ran over all 512 tokens of
every ubatch** — 1.30 TFLOP/ubatch and ~496 MiB/ubatch of logits coming back.
That part **was fixed**, by [#23198](https://github.com/ggml-org/llama.cpp/pull/23198)
(`3e12fbdea`, 2026-05-17), which split `need_embd` from `need_embd_pre_norm`.
[#23433](https://github.com/ggml-org/llama.cpp/pull/23433) then made draft
catch-up decodes skip the LM head, and
[#23287](https://github.com/ggml-org/llama.cpp/pull/23287) kept draft sampling
on-GPU. Nothing has touched the `h_nextn` D2H itself, and the upstream direction
is tensor sharing between contexts
([#26636](https://github.com/ggml-org/llama.cpp/pull/26636), open).

## Still present, still ungated

Call sites: `src/llama-context.cpp:1954` (decode) and `:1551` (encode); sizing at
`:2059-2065` allocates `n_embd_out * n_batch`, not `* n_outputs`. The tensor is
forced live by `ggml_set_output(t_h_nextn)` at `src/llama-graph.cpp:1344-1347`.
The host round trip runs through `common/speculative.cpp:1443-1444` and `:1583`
and back in via `src/models/qwen35.cpp:514-515`.

There is **no** unified-memory gate anywhere on this path, and one could not work
as written: **Metal reports `is_host == false` for all three of its buffer
types** (`ggml/src/ggml-metal/ggml-metal.cpp:275-279`, `:351-355`, `:427-431`)
even though `use_shared_buffers = true` here.

## T3-4 — the real defect: an asymmetry inside ggml-metal

The *synchronous* extraction path has the unified-memory fast path; the *async*
one does not.

```
ggml-metal-device.m:1902-1906   // ggml_metal_buffer_get_tensor
    if (buf->is_shared) { memcpy(data, (const char *) tensor->data + offset, size); return; }
```
versus `ggml-metal-context.m:351-391` (`ggml_metal_get_tensor_async`), which does
`newBufferWithBytesNoCopy` + a fresh command buffer + a blit + commit.
`ggml_backend_tensor_get_async` takes the async path because Metal's
`iface.get_tensor_async` is non-NULL. So on unified memory every logits /
embedding / `h_nextn` extraction allocates an `MTLBuffer` and submits an extra
command buffer where a `memcpy` would do. (The **H2D leg is already free** —
`ggml-metal-device.m:1849` has the check for `set`.)

**MAGNITUDE.** Bytes are nil: 20,480 B per row → 77 µs against a 1707 ms ubatch
(0.0045% of prefill); 140 KiB per MTP cycle (0.0015% of decode). What remains is
fixed per-submit overhead — 4 extra command-buffer commits and 4 VM mappings per
cycle at 30–100 µs each → **0.2–0.5% of decode with MTP on.** That is the honest
ceiling on "fix the D2H copy".

**FALSIFY.** Add `if (ggml_metal_buffer_is_shared(buf)) { memcpy(...); return; }`
at the top of `ggml_metal_get_tensor_async` (the helper already exists at
`ggml-metal-device.m:1816`) and A/B `bench/spec.sh --depth 3`. If it does not
clear the MDE, close this thread permanently. **COST** ~4 lines, 1 h.

## T3-2 — CORRECTION: `is_mem_shared` is FALSE for qwen35

**This corrects an error in this project's own earlier notes.** `bench/spec.sh`
originally said the MTP path yields `is_mem_shared = true`. It does not.
`common/speculative.cpp:2298` sets `cparams.ctx_other = ctx_tgt`, but the context
constructor **resets it to `nullptr`** at `src/llama-context.cpp:142` and
restores it only for `LLM_ARCH_GEMMA4_ASSISTANT` (`:145-152`) and
`LLM_ARCH_EAGLE3` / `LLM_ARCH_DFLASH` (`:154-161`). **`LLM_ARCH_QWEN35` is in
neither list** (verified directly). So `common/speculative.cpp:1335` is false and
the catch-up block at `:1430-1499` is *not* skipped.

The weights *are* shared — `llama_init_from_model(model_tgt, …)` at `:2323` — so
the memory-feasibility conclusion stands and `-md` must still be omitted. What is
**not** shared is the KV/recurrent memory, which is exactly what the flag gates.
Any cost model assuming "MTP is free because memory is shared" understates each
cycle by a full extra MTP forward pass plus the host round trip. The comment in
`bench/spec.sh` has been corrected.

## T3-1 — the MTP draft step re-reads the 1.043 GB output head

Independent confirmation of Track 7's L6, with the cycle arithmetic:
`src/models/qwen35.cpp:637-640` falls back to `model.output` because this GGUF
has no `nextn.shared_head_head`. Cycle bytes at `n-max 3` = 16.517 (target
verify) + 0.290 (catch-up) + 3 × 1.333 (draft: 0.290 MTP block + 1.043 head) =
**20.82 GB, of which the repeated head is 3.13 GB = 15.0%.**
**MAGNITUDE: decode +10–15% with MTP on**, zero without.
**FALSIFY:** check whether `bench/spec.sh --depth 1,2,3` tok/s scales as
`(1+n_accepted)/(16.5 + 0.29 + n·1.33)` rather than `/(16.5 + n·0.29)`.
Mitigation is awkward — a draft-only lower-precision head costs ~0.35 GB of
headroom — so treat this primarily as the cost model that bounds MTP.

## T3-3 — a last-layer optimization silently disabled for ALL qwen35 users

`#23198` gated the standard last-layer `ggml_get_rows` early-exit on
`cparams.embeddings_nextn_masked`, which defaults to **false**
(`src/llama-context.cpp:119`). So plain, non-MTP prefill of Qwen3.5/3.6 runs the
final layer's post-attention norm + FFN + residuals and `output_norm` over *every*
token instead of just the output rows. See `src/models/qwen35.cpp:178-181` and
`src/models/qwen35moe.cpp:201`, versus the unconditional idiom everywhere else
(`src/models/qwen3.cpp:114`, `src/models/llama.cpp:174`).

One layer's FFN is 3 × 5120 × 17408 = 267.4 M params = 534.8 MFLOP/row; for a
512-token ubatch with one output row that is **273 GFLOP of pure waste** against
~27.98 TFLOP. **Prefill ≈ +1.0%**, decode 0 (the branch is inert at
`n_tokens == 1`). Two-line fix in two files; **worth sending upstream regardless
of whether it clears our MDE**, since it affects every user of this architecture.

## T3-5 / T3-6 / T3-7 — smaller

- **T3-5.** `t_h_nextn` is flagged a graph output even when MTP is off
  (`src/models/qwen35.cpp:212-213` assigns unconditionally), and graph outputs are
  never freed (`ggml/src/ggml-alloc.c:692-695`). Pins 10 MiB at ub=512 (40 MiB at
  2048) inside the compute buffer. **0% speed**, but it is reclaimable headroom on
  a machine where headroom is the binding constraint. 1 line.
- **T3-6.** `token_embd` is host-resident and Metal's `offload_op` accepts only
  `MUL_MAT`/`MUL_MAT_ID` (`ggml/src/ggml-metal/ggml-metal.cpp:758-765`), so
  `GET_ROWS` can never move to the GPU and every graph — including each ~5 ms MTP
  draft step — opens with a CPU→GPU split. **Decode +0.5–2% with MTP on.**
  Test with `-ot token_embd\.weight=MTL0`, but that moves 682 MiB into the working
  set — check `mem.sh` first.
- **T3-7.** Latent shape hazard: in the MTP graph `t_h_nextn` is captured *before*
  `get_rows` (`src/models/qwen35.cpp:631-634`) while the masked extraction copies
  `n_outputs` rows from offset 0 (`src/llama-context.cpp:1946-1954`). Correct only
  because MTP draft batches happen to have `n_tokens == n_outputs`. Correctness
  hazard, 0% speed. Worth an upstream note.

---

# Track 6 — Prior art

> **Coverage caveat, stated plainly.** The exhaustive `gh pr list --state closed`
> sweep of 90 days of ggml-metal PRs did **not** complete. T6-4 and T6-6 are
> rejected/contested items found while chasing other threads, not the systematic
> sweep the brief asked for. **That sweep should be re-run before the next phase
> concludes anything about what upstream has already refused.**

## T6-5 + T6-6 — the tensor path is ON by default here, and there is a correctness question attached

**T6-5.** ggml-metal already routes the prefill GEMM (`mul_mm`, including
`q4_K_f32` and `q6_K_f32`) through `mpp::tensor_ops::matmul2d` on M5-class
hardware, gated by a **device-name whitelist** containing "M5"/"M6"/"A19"/"A20"
(`ggml-metal-device.m:743-761`). This machine matches. The enabling commit's own
note says the tensor kernels were *"~5% slower"* on M2 Ultra and *"no significant
difference"* on M4 — **nobody has published an M5 number.** Prefill is
compute-bound here, so this is the right lever and the cheapest experiment:
`GGML_METAL_TENSOR_DISABLE=1`, one env var, no rebuild.

**T6-6 — the attached warning.** An independent fork measured Apple's M5 (`g17`)
neural-accelerator path producing **numerically wrong results** in MLX and raised
its gate from `gen >= 17` to `gen >= 18`
(`~/projects/forks/mlx-prism/mlx/backend/metal/device.cpp:798-806`, dated
2026-07-05: *"fp16 GEMM max|err| ~4 vs fp32 for every shape routed to
steel_gemm_fused_nax; quantized qmm_t ~400 abs err; non-nax fallback bit-matches
stock mlx"*). Two different codebases drive the same `MetalPerformancePrimitives`
matmul on the same silicon, so if the fault is in the driver rather than MLX's
kernel, ggml's tensor `mul_mm` is exposed too — **and it is on by default on this
exact machine.**

**Status on this rig: substantially reassuring, and it is already measured.**
`test-backend-ops test` ran **14022/14022 passed, zero failures**, on the default
build — i.e. **with the tensor path enabled** — and that harness compares every op
against the CPU backend. That is direct evidence that ggml's M5 tensor path is
numerically sound here, at least across the shapes `test-backend-ops` covers.
The remaining check is a perplexity comparison with `GGML_METAL_TENSOR_DISABLE=1`,
which `bench/quality.sh` can do unchanged. Until that is run, treat this as
*probably fine, cheaply confirmable* rather than settled.

## T6-2 — MLX enables the equivalent kernel from M ≥ 2, and documents the trap

Apple's own framework uses the same kernel design as `mul_mv_ext` but routes
affine-quantized weights into it at `M >= 2` on gen-15+ hardware
(`mlx/backend/metal/quantized.cpp:1759`, `use_qmv_wide()` at `:537-540`), and
picks the matvec→matmul crossover from `(arch_gen, arch_size, K, N)` — returning
**33 / 25 / 13** by shape (`:84-120`) — rather than llama.cpp's flat
`ne11_mm_min = 8`. The kernel shape matches ggml's exactly: 2 simdgroups, 8
k-lanes, tile cap 5.

**The trap, and probably why the gate is at 4.** Because ggml excludes K-quants
below `ne11 = 4`, the `nxpsg = 16` branch at `ggml-metal-ops.cpp:2374-2376`
(`ne00 % 256 == 0 && ne11 < 3`) has **never been exercised for K-quants**. Naively
lowering the gate to 2 would route Q4_K straight into `nxpsg = 16`, which MLX
documents as the wrong setting for affine quants. The correct change is
**gate → 2 AND restrict the `nxpsg = 16` branch to non-K-quant types.** Host-side
only: the `_r1_2` and `_r1_3` pipelines already exist
(`ggml-metal.metal:4231-4232`) and `nxpsg` is a function constant (`:3542`), so no
new kernel source. ~5 lines. **This is how you make every draft depth good
instead of only 3, 4 and 8+** — i.e. it generalizes L1 from "pick the right rung"
to "remove the staircase".

## T6-3 / T6-4 — flash-attention vec kernel, and the PR that was rejected

`ggml_metal_op_flash_attn_ext_use_vec` selects the vec kernel whenever
`ne01 < 20` (`ggml-metal-ops.cpp:2796`), and that kernel's `NQPSG = 1`
(`ggml-metal-impl.h:112`, versus 8 for the non-vec path at `:109`), so a 4-token
verify reads the KV cache 4× instead of once. Because this model charges KV for
only 16 of 64 layers, this is second-order here — **~0% at ≤2k context, ~10–18%
at 16k** with MTP on — where on a dense 27B it would be first-order.

**Someone already built this fix and it was rejected.**
[PR #23114](https://github.com/ggml-org/llama.cpp/pull/23114) (closed unmerged
2026-05-23) implemented Q=2 queries per threadgroup, measuring 1.2–1.7× on the FA
kernel and up to **1.10× end-to-end on Qwen3.6-27B MTP**. ggerganov's reason is
the most useful thing in this track: *"I'm worried that such kind of special
casing would quickly become unmaintainable… it would be better to design a
generic solution that would work for all head sizes and provide some tooling to
measure the optimal settings (NE, Q, etc.) on a given hardware. These 'per-device'
configs would then become part of the ggml Metal backend."*

Two hard facts from the author's own PoC, worth knowing before anyone retries it:
moving accumulators to threadgroup memory to relieve register pressure made it
**worse** (144.85 µs → 166.95 µs → 504.96 µs as more state moved to SMEM;
occupancy fell 23.7% → 18.4%), and **Metal `function_constant` values cannot be
used as array dimensions**, so `Q` and `NE` must stay template parameters, i.e.
separate kernel instantiations (~82 → ~105 for the proposed scope).

**Consequence for planning:** a local Q=2 patch is a fork-only change that will
not be accepted upstream. Budget it as a fork experiment (1 day) or not at all.

## T6-7 / T6-8 — ik_llama.cpp and llamafile

**ik_llama.cpp** *does* ship Metal kernels for its whole IQ*_K family (I counted
21–24 kernel-name occurrences each for `iq4_k`, `iq4_ks`, `iq4_kss`, `iq5_k`,
`iq5_ks`, `iq6_k`, `iq2_k`, `iq3_k`, `iq2_kt`…, in a 10,424-line
`ggml/src/ggml-metal.metal`), contrary to the common assumption that it is
CPU/CUDA-only. The `_R4`/`_R8` row-interleaved variants are CPU-only and do not
cross to Metal. But **every one of its wins here is a re-quantization win** —
IQ4_KS at ~4.25 bpw against Q4_K_M's ~4.7 bpw would cut bytes/token, which is the
only currency decode responds to — and **this model is frozen**. So: **+8–12% if
the freeze were ever lifted, 0% as scoped.** Recorded so it is not re-discovered.

**llamafile / tinyBLAS**: CPU GEMM (AVX2/AVX512/NEON) for prompt processing. No
Metal kernels. All 65 layers are GPU-resident here, so the CPU path is idle.
**0% decode, 0% prefill.** Stated plainly so nobody spends a day on it.

## T6-9 — roofline framing (external numbers NOT independently verified)

Agent-supplied and flagged as unverified: M5 ridge points ~6.5 FLOP/byte (naive
float4) to ~31 (optimized scalar); Neural Accelerator feed requirement 93.44 GB/s
per GPU core. Also cited: Chordiya, *Lossless but Not Free: An Empirical Anatomy
of Speculative Decoding on Consumer Hardware*, arXiv:2607.17283 — 1.61× at K=6 on
Apple silicon, but **3 of 5 configurations decelerated**, attributed partly to
quantized Metal backends verifying serially. Treat the paper as corroborating
mechanism, not as a number.

The useful framing: MTP *should* be worth **2–3×** here
(`mean_accepted × 16.4 tok/s`). If measured MTP throughput falls far short of
that, the gap is a specific bug — L1's staircase, T3-1's repeated head, T6-3's KV
re-reads, or the 48 GDN layers, whose scan is a serial per-token loop with two
`simd_sum` reductions inside (`ggml-metal.metal:2638-2696`) and is therefore a
*latency* limit rather than a bandwidth one.

---

## Recommended order for the next phase

| # | action | cost | why |
|---|---|---|---|
| 1 | `GGML_METAL_TENSOR_DISABLE=1` A/B on pp512, plus `bench/quality.sh` under both | 1 h | T6-5 + T6-6. One env var. A prefill lever *and* a correctness gate on every future measurement. `test-backend-ops` already passed with it on, so this is confirmation, not a blocker. |
| 2 | `llama-bench -p 1,2,3,4,5,6,7,8,9,12,16 -n 0` | 30 min | L1. Maps the whole staircase in one run; confirms or kills the largest decode lever. |
| 3 | Set `--spec-draft-n-max` to 3 or 4 — never 1, 2, 5, 6, 7 | **0** | L1, if step 2 confirms. Free. |
| 4 | `bench/spec.sh --depth 1,2,3,4` with acceptance per category | 1 h | Establishes the real MTP baseline and the 0.073 stopping rule input. |
| 5 | `--spec-type ngram-mod --spec-ngram-mod-n-max 16` on codeedit + json | 1 h | L2, the largest number in this file. |
| 6 | `GGML_SCHED_DEBUG` to count `ctx_dft` decodes and graph splits per cycle | 30 min | Settles T3-2 and T3-6 empirically. |
| 7 | Lower the K-quant gate to 2 **with** the `nxpsg` restriction | half day | T6-2. Generalizes L1 from "pick the rung" to "remove the staircase", backed by MLX's own constants. |
| 8 | The two-file `get_rows` / `set_output` fix in qwen35 + qwen35moe | 1 h | T3-3 + T3-5. ~1% prefill, plus reclaimed headroom. Send upstream regardless. |
