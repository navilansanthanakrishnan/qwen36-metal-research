# LEDGER

One row per experiment. Append only — never edit or delete a row, including
the ones that failed. A ledger you can rewrite is not evidence.

Rules:

- **Every row cites a `bench/ab.sh` verdict**, not a single run. A number
  without a paired, interleaved comparison against trunk is not a result.
- `KEEP` requires the A/B verdict to be `DIFFERENT` in the faster direction
  **and** `bench/quality.sh` to pass. Speed at degraded quality is not a win
  and does not get a KEEP.
- The minimum detectable effect is in NOISE.md. Anything below it is `NOISE`,
  regardless of how good the mean looks.
- Record `REVERTED` rows too. The fact that an idea was tried and failed is
  worth as much as the fact that one worked, and it stops the same idea being
  re-tried in three months.

| # | date | lead | change | commit/patch | A/B verdict | Δ decode | Δ prefill | accept | quality | decision | notes |
|---|------|------|--------|--------------|-------------|----------|-----------|--------|---------|----------|-------|
| 001 | 2026-08-12 | infra | `bench/lock.sh` — exclusive measurement lock, held across a whole A/B | `ebf53a0` | n/a | — | — | — | n/a | **INFRA** | Not a candidate. env.sh's load/CPU gates miss a second llama-bench that has not ramped yet; a lock released between arms lets another run land inside yours and produces a believable number instead of an obviously broken one. |
| 002 | 2026-08-12 | Shrey test block | `test-backend-ops`: K-quant mat-vec row gates, partial tiles, large k | `5497474d6` on `test-coverage` | n/a | — | — | — | pending | **PENDING** | Verified the gap first-hand: the only K-quant mul_mat coverage in the suite is ne01=16, n=1..9, k=256. Any backend switching kernels on row count is untested. Upstream-worthy, backend-independent. |
| 003 | 2026-08-12 | S4 | `dequantize_q4_K`: divide the second-half scale in float, not half | `s4-q4k-scale-div` | n/a (quality) | 0 expected | 0 expected | — | pending | **PENDING** | Independently measured on this model, not inherited: over 14,919,680 Q4_K super-blocks the median `d` is 6.187e-05, so `d/16` = 3.87e-06 is subnormal in half for **99.9998%** of blocks. Median relative error 3.66e-03 (~8 mantissa bits), p99 9.6e-03, max 1.0 — some blocks flush to zero. Expect PPL to *fall*. Upstream-worthy. |
| 004 | 2026-08-12 | L1 / S0-cheap | K-quants use `mul_mv_ext` at ne11 2–3 (gate 4→2) + nxpsg=16 restricted to non-K-quants | `c1-kquant-ext-gate` | pending | pred: large at depth 1–2, **0 at depth 0** | 0 | — | pending | **PENDING** | Class B. Mechanism: below the gate the weight matrix is streamed once *per column*. Predict 2→1 streams at ne11=2, 3→1 at ne11=3. Env toggle `GGML_METAL_NO_KQUANT_EXT2=1` for one-binary A/B. If plain decode moves, the result is suspect. |

