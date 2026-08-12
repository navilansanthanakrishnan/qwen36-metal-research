# Qwen3.6-27B decode on an Apple M5 Pro

The research half of an effort to make one model decode faster on one machine:
**Qwen3.6-27B-Q4_K_M** on a binned **M5 Pro** — 16 GPU cores, 24 GiB unified
memory. Context stays at 4096 and the quantization is never changed.

Decode went from **14.567 tok/s to 23.76 tok/s (1.63×)**. The target was 35 and
has not been reached; [LEDGER.md](LEDGER.md) records why, entry by entry,
including the failures.

The kernels and llama.cpp changes live in the implementation repo:
**[llama.cpp-qwen3.6-m5](https://github.com/navilansanthanakrishnan/llama.cpp-qwen3.6-m5)**
(branch `sgmv-q4k`). This repo is the method, the measurement harness, and the
evidence.

## The result

| | before | after |
|---|---|---|
| decode, MTP speculative | 14.567 tok/s | **23.76 tok/s** |
| width-8 verification | 215.2 ms | **110.4 ms** |
| Q4_K mat-vec, m=4096 k=14336, n=8 | 510 µs | **235 µs** |

Decode at one token per forward pass is capped at **16.40 tok/s** — 16.5 GB of
weights over 270.8 GB/s of measured bandwidth. Everything above that line comes
from accepting more than one token per pass, which is why every speculative
figure is reported with its acceptance rate beside it.

## What is here

```
LEDGER.md      every change tried, with its measured effect and verdict.
               The primary artifact. Rejections are kept, with the reason.
GOAL.md        the target and the rules the work runs under
HARDWARE.md    measured machine truth: bandwidth, FLOP/s, thermal reference
MODEL.md       tensor census by quant type, KV and recurrent-state sizes
BASELINE.md    the frozen baseline and how it was taken
NOISE.md       the null test, the positive control, and the 3.0% MDE
LEADS.md       ranked hypotheses, most now closed by measurement
RESUME.md      working state for picking the effort back up
progress.html  a dashboard of the above

bench/         the measurement harness (see below)
prompts/       14 frozen prompts across 5 categories, with a manifest
quality/       the quality oracle: greedy references, perplexity, KLD
```

`models/` and `runs/` are symlinks to weights and run output kept outside the
source tree; they are not in git. Create them yourself (see Replicating).

## The harness

The numbers in `LEDGER.md` are only worth reading because of how they were
taken. The rules, and the scripts that enforce them:

- **`bench/env.sh`** — a gate that refuses to benchmark on battery, in Low Power
  Mode, while Spotlight is indexing, above a load average of 3.0, or with any
  process over 50% CPU. Apple Silicon drops GPU clocks on battery, so a number
  taken there is not comparable to one that is not.
- **`bench/lock.sh`** — an exclusive measurement lock. There is one GPU and a
  model that occupies 16 of 24 GiB. Two benchmarks at once do not produce
  obviously-bad numbers; they produce plausible ones.
- **`bench/ab.sh`, `bench/specab.sh`** — interleaved ABBA A/B with an exact
  sign-flip permutation test. Both arms are the **same binary**, selected by an
  environment variable, so a rebuild cannot be mistaken for an effect. Six pairs
  is the minimum that can reach p < 0.05 (p = 2/2ⁿ).
- **`bench/decode.sh`** — discards any run over 25000 swap pages or 5% relative
  standard deviation, and checks prefill health so a regression cannot hide.
- **`bench/quality.sh` / `quality-run.sh`** — greedy token-exactness, perplexity,
  KL-divergence, and `test-backend-ops`. Run through `quality-run.sh`:
  `llama-perplexity` defaults to a batch that allocates 2 GB of logits and OOMs.
- **`bench/spec.sh`** — speculative decoding, always reporting acceptance and
  mean accepted tokens per forward pass. A speculative tok/s without acceptance
  beside it cannot be told apart from a drafting change.
- **`bench/sgmv.m` / `sgmv.metal`** — a standalone Metal probe for the kernel
  work. Iterating here takes seconds; a llama.cpp rebuild plus an A/B takes half
  an hour. Most of the kernel design was found in this file.
- **`bench/bwprobe/`, `bench/gpuinfo/`, `bench/sgmap.m`** — the machine-truth
  probes: attained bandwidth, GPU clock and throttle, and the empirical recovery
  of the simdgroup matrix lane→element layout.

The minimum detectable effect is **3.0%**, established by a null test (two
identical arms, which must report no difference) and a positive control (a
deliberate 20 ms/token slowdown, which must be detected — it was, at p=0.0078).

## Replicating

You need a GGUF of Qwen3.6-27B-Q4_K_M and an Apple Silicon machine. Nothing here
assumes an M5 Pro except the recorded constants, which is the point: any
constant tuned for a different core count has to be re-derived, not inherited.

```bash
git clone https://github.com/navilansanthanakrishnan/llama.cpp-qwen3.6-m5 llama.cpp
cd llama.cpp && git checkout sgmv-q4k
cmake -B build -DGGML_METAL=ON && cmake --build build -j
cd ..

mkdir -p models runs                 # or symlink them somewhere with space
cp /path/to/Qwen3.6-27B-Q4_K_M.gguf models/

bash bench/env.sh                    # must pass before any timing
bash bench/decode.sh --label baseline --tag t1
```

To reproduce the headline comparison — both arms, one binary:

```bash
bash bench/lock.sh with env LLAMA_ARG_SPEC_N_RS_SEQ=3 \
  bash bench/specab.sh --depth 7 --pairs 6 \
    --enva "LLAMA_ARG_SPEC_EXT_N=3 LLAMA_ARG_SPEC_MTP_MAX=4" \
    --envb "LLAMA_ARG_SPEC_EXT_N=3 LLAMA_ARG_SPEC_MTP_MAX=4 GGML_METAL_SGMV_DISABLE=1" \
    --la new --lb upstream
```

The kernel probe needs no model and runs in seconds:

```bash
clang -fobjc-arc -O2 -framework Foundation -framework Metal bench/sgmv.m -o bench/sgmv
./bench/sgmv                # M=40960 K=5120 N=8
./bench/sgmv 4096 14336     # a narrower, taller shape
```

## Reading the ledger

Entries are numbered and append-only. A verdict of **KEEP** means an A/B cleared
the 3.0% MDE with p ≤ 0.05; **REJECT** means it was measured and lost;
**NO DIFFERENCE** means the test could not distinguish it from noise. Rejections
are the majority and are kept deliberately — several of this effort's best ideas
are in there, refuted.

A few entries worth reading on their own:

- **036, 037** — `simdgroup_matrix::thread_elements()` is writable, and the
  empirical recovery of its implementation-defined lane→element layout. This is
  what made the kernel possible.
- **038** — the kernel's progression from 68 to 170 GB/s, and the measured
  loads-only ceiling of 213 GB/s that bounds it.
- **041, 052** — the speculative cost model, solved from measurements, and what
  it says 35 tok/s actually requires.
- **054** — a correction: width-scaling is matmul, not the gated-delta-net
  recurrence, which I had attributed it to.
- **057, 058** — a frequency-ranked draft vocabulary that works mechanically
  (the cycle drops 160 → 112 ms) and loses anyway, on acceptance.

## Prior work

The multi-column K-quant mat-vec direction, the `dequantize_q4_K` scale bug, and
the `test-backend-ops` coverage gap come from
[shreyvishen/llama.cpp-qwen3.6-metal](https://github.com/shreyvishen/llama.cpp-qwen3.6-metal),
an M4 Max effort on the same model. His tile constants do not transfer to 16 GPU
cores and were re-derived or rejected here; the scale fix and the coverage
transfer unchanged and are merged.
