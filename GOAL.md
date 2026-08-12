# GOAL

> No GOAL.md was supplied to copy in, so this one was authored during setup.
> The numbers block is filled from measurements taken on this machine this week.
> See SETUP-LOG.md, "ASSUMPTION: GOAL.md was authored, not copied".

## The goal

**Maximum decode speed for `Qwen3.6-27B-Q4_K_M.gguf` on this M5 Pro, at
unchanged output quality.**

"Unchanged" is not a judgement call. It is whatever `bench/quality.sh` says:
token-exact greedy output against the frozen trunk references, perplexity within
+0.02, KL-divergence within tolerance, and `test-backend-ops` clean. A change
that is faster and fails any of those is not a win and does not enter LEDGER.md
as one.

## The numbers

Measured, not estimated. Every one of these is reproducible from this tree.

```
MACHINE
  chip                        Apple M5 Pro (binned): 16 GPU cores, 15 CPU (5 Super + 10 Perf)
  unified memory              24 GiB (25,769,803,776 B)
  vendor peak bandwidth       307 GB/s        (Apple, quoted for the full M5 Pro part)
  attained bandwidth          270.8 GB/s      (bench/bwprobe, median of 20)
  attained / peak             88.2%           (gate 60-90%: PASS)
  GPU clock reference         6258 GFLOP/s    (bench/gpuinfo --clock, cold idle)

MODEL
  file                        Qwen3.6-27B-Q4_K_M.gguf
  sha256                      a7cbd3ecc0e3f9b333edee61ae66bc87ed713c5d49587a8355814722ed329e0f
  size on disk                17,106,773,120 B (15.932 GiB)
  architecture                qwen35, 65 blocks = 64 main + 1 MTP
                              48 SSM layers + 16 full-attention layers
  MTP tensors                 PRESENT (blk.64.nextn.eh_proj / enorm / hnorm / shared_head_norm)
  weights read per decode step 16,516,723,261 B (GPU-resident, excluding MTP and
                              host-resident token_embd)

CEILING
  ceiling_1tok                16.40 tok/s     = 270.8e9 / 16,516,723,261
  ceiling_spec                16.40 x mean_accepted_tokens_per_forward_pass

MEMORY BUDGET (frozen ctx = 4096)
  GPU working set available   18186 MiB       (iogpu.wired_limit_mb = 0, driver default)
  model, GPU-resident         16027.69 MiB
  KV cache                      256.00 MiB    (64 KiB/token x 4096, 16 attn layers only)
  SSM recurrent state           149.62 MiB    (constant in context length)
  compute buffer                152.13 MiB
  total                       16585.44 MiB    -> 1600.81 MiB headroom
  largest context that RUNS      4096         (see below -- not the same as what the
                                               static fitter accepts)

BASELINE (trunk 0b1bad14f, speculation off)  -- see BASELINE.md
  decode  tg128               14.567 +/- 0.404 tok/s   (n=10; settled machine 14.829 +/- 0.189)
  prefill pp512               306.75 +/- 6.09 tok/s
  fraction of ceiling_1tok    88.8%          <-- decode is at the bandwidth wall
  test-backend-ops            14022/14022 passed, 0 failures

MEASUREMENT SENSITIVITY  -- see NOISE.md
  noise floor                 7.21% raw spread over 10 runs; 3.14% paired
  measured resolution         0.64%   (8 pairs, quiet machine, null test)
  minimum detectable effect   3.0%    (frozen KEEP threshold, QM_MDE_PCT)
  required cooldown           60 s
  null test                   PASS -- NO DIFFERENCE between identical binaries
  positive control            DETECTED -- slower by 5.88%, p=0.0078, every pair agreed
```

## What follows from those numbers

**Plain decode is nearly finished.** It sits at 88.8% of a hard bandwidth
wall. The remaining headroom on the ordinary decode path is single-digit
percent, and it is bounded — no amount of kernel work can take 16.4 tok/s past
16.4 tok/s. Anyone proposing to spend a day tuning `mul_mv` for decode should be
shown this line first.

**Prefill is a different story.** 306.75 tok/s is compute-bound, not
bandwidth-bound, and is where fusion, graph optimization, and kernel
specialization can actually pay.

**Acceptance rate is the only unbounded lever.** Everything else in this project
is a fight for single-digit percentages against a wall. Speculative decoding
multiplies the wall itself. A move from 1.5 to 2.5 mean accepted tokens per
forward pass is worth more than every kernel optimization in the dossier
combined.

## Rules

1. No number enters LEDGER.md without a `bench/ab.sh` verdict against trunk.
2. Nothing below the minimum detectable effect in NOISE.md is a result.
3. `bench/quality.sh` must pass. Speculative decoding is lossless by
   construction; if it isn't reproducing the reference token-for-token, that is
   a bug being mistaken for a speedup.
4. Numbers from other machines are not evidence. There is prior work on this
   model on a larger Mac; its techniques are leads, its numbers are not
   baselines.
