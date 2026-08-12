# BASELINE — trunk, unmodified, measured on this machine

**MTP tensors are PRESENT.** `blk.64.nextn.{eh_proj,enorm,hnorm,shared_head_norm}`
survived this conversion, `qwen35.nextn_predict_layers = 1`, and the pinned
llama.cpp implements `--spec-type draft-mtp`. The speculative half of this
project is live. See MODEL.md.

_(Numbers below are filled from the runs recorded in
`runs/decode/*.jsonl` and `runs/spec/*.txt`. Every entry carries its full
command, the commit SHA, mean ± sd as the tool reported it, the thermal line,
wall clock, and whether the machine had been idle.)_

---

## What the numbers mean

**Plain decode is at the memory-bandwidth wall, and tuning the decode matmul is
very nearly finished before anyone spends a day on it.**

Measured decode is **14.567 tok/s** against a hard ceiling of
**16.40 tok/s** — the ceiling being 16.52 GB of GPU-resident non-MTP weights
divided by 270.8 GB/s of attained bandwidth. That is
**88.8% of `ceiling_1tok`**. Counting the ~906 MB/token of SSM
recurrent-state traffic that this hybrid architecture also moves (LEADS.md, "the
arithmetic that ranks everything"), the figure rises to roughly 94% of attained
bandwidth. Either way the conclusion is the same and it is blunt: **there is
single-digit headroom on the ordinary decode path, it is bounded, and no kernel
can take 16.4 tok/s past 16.4 tok/s.** The remaining decode-side wins are all
*byte deletions* — the recurrent-state copies — not better arithmetic.

Prefill is a different regime: **306.75 tok/s** is compute-bound, not
bandwidth-bound, and is where fusion, graph optimization, and the tensor-API
path can actually pay.

**The only unbounded lever is draft acceptance**, because it multiplies the wall
rather than fighting it. And the most important single finding of setup is that
the cost of a verify step is a **non-monotonic staircase** in the draft depth:
Metal's small-batch mat-vec excludes K-quants below `ne11 = 4`, and the `r1ptg`
tile table inside it spoils several widths above that. Depths 1, 2, 5, 6 and 7
cost 2–3 full 16.5 GB weight streams per verify step; depths **3, 4 and 8+ cost
one**. Since this model is 99.4% K-quant by bytes, choosing the depth is worth
2–3× and costs nothing. See LEADS.md L1, where every rung is verified against the
pinned source. **Any speculative number taken at depth 1, 2 or 5–7 is measuring
that staircase, not MTP's potential.**

---

## Configuration (frozen — see `bench/env.sh`)

| | |
|---|---|
| commit | `0b1bad14ff204627636aeb1de22ddcd5acb859d4` (b10380) |
| model | `Qwen3.6-27B-Q4_K_M.gguf`, sha256 `a7cbd3ec…29e0f` |
| `-ngl` | 99 (all 65 blocks on Metal) |
| `-fa` | on |
| `-ctk` / `-ctv` | f16 / f16 |
| `-b` / `-ub` | 2048 / 512 (llama-bench); 512 / 512 (llama-server, see below) |
| `-p` / `-n` | 512 / 128 |
| `-r` | 3 |
| context | 4096 |

The llama-server scripts cap `-b` at the ubatch size because the server
allocates batch-sized buffers that llama-bench does not; at `-b 2048` it OOMs
during warmup. This does not affect the frozen llama-bench decode numbers.

## Primary loop

```
bench/decode.sh
  llama-bench -m models/Qwen3.6-27B-Q4_K_M.gguf -ngl 99 -fa on \
              -ctk f16 -ctv f16 -b 2048 -ub 512 -p 512 -n 128 -r 3 -o json
```

Wall clock ~37 s per invocation, which is what makes a claim killable in under a
minute.

### Decode and prefill, speculation off

Ten gated runs on unmodified trunk, 21:27:45Z–21:59Z, machine idle beforehand.
Full record in `runs/decode/20260811.jsonl` (tags `nf-1`…`nf-10`).

```
tg128   13.998  14.080  14.641  13.945  14.882  14.686  14.701  14.777  14.994  14.967
        mean 14.567 tok/s   sd 0.404   (rsd 2.78%)

pp512   mean 306.75 tok/s   sd 6.09    (rsd 1.99%)
```

Wall clock 37–41 s per run. Thermal line on every run, e.g.
`thermal gpu_gflops=6254.9 ref_gflops=6258.0 throttle_pct=0.0 power=AC`.
Swap-in delta recorded per run; all ten passed the validity gates.

**Under settled conditions the baseline is higher and much tighter**: the 16
clean-arm runs of the null test, taken after the machine had quiesced, give
**14.829 ± 0.189 tok/s** (rsd 1.27%). The 14.567 figure is the frozen baseline
because it is the 10-run protocol measurement, but the spread between the two is
the machine-settling drift characterised in NOISE.md, not a disagreement.

| quantity | value |
|---|---|
| decode `tg128`, speculation off | **14.567 ± 0.404 tok/s** |
| decode, settled machine | 14.829 ± 0.189 tok/s |
| prefill `pp512`, cold KV per rep | **306.75 ± 6.09 tok/s** |

## `test-backend-ops`

```
llama.cpp/build/bin/test-backend-ops test
  Backend 1/2: MTL0   14022/14022 tests passed   OK
  Backend 2/2: CPU    skipped
  2/2 backends passed
```

Zero failures. This is the correctness floor and `bench/quality.sh` re-checks it.

## Speculative / MTP

PLACEHOLDER_SPEC_PENDING

## Attained fraction of the ceiling

```
attained_bandwidth   = 270.8e9 B/s          (bench/bwprobe, median of 20)
weight_bytes         = 16,516,723,261 B     (GPU-resident, excluding MTP and
                                             host-resident token_embd)
ceiling_1tok         = 270.8e9 / 16.5167e9  = 16.40 tok/s
measured decode      = 14.567 tok/s
fraction of ceiling  = 88.8%
```

Two other bases, for anyone who wants to argue about which is honest:

| basis | bytes | ceiling | fraction |
|---|---:|---:|---:|
| whole GGUF file | 17,106,773,120 | 15.83 tok/s | 92.0% |
| GPU-resident incl. MTP | 16,806,251,069 | 16.11 tok/s | 90.4% |
| **GPU-resident excl. MTP** | **16,516,723,261** | **16.40 tok/s** | **88.8%** |

The third is the right one for speculation-off decode: `token_embd` is
host-resident and gathered one row at a time, and the MTP block is not touched
when speculation is off.
