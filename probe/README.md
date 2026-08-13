# probe/ — standalone Metal probes

New tooling added alongside `bench/`, which is frozen. Nothing here alters what
any frozen script measures; these exist because a llama.cpp rebuild plus an A/B
takes half an hour and these answer a question in seconds.

Each one was written to kill a specific belief, and each one did.

## `mmapeak.m` — peak arithmetic throughput, zero memory traffic

```bash
clang -fobjc-arc -O2 -framework Foundation -framework Metal probe/mmapeak.m -o probe/mmapeak
./probe/mmapeak
```

Measures `simdgroup_float8x8`, `simdgroup_half8x8` and a scalar `fma` control
with every operand register-resident, a true dependency through the accumulator
so nothing can be hoisted, and an iteration sweep to confirm time is linear in
work (a constant offset would mean the loop was folded).

**Result (LEDGER 074):** MMA **6.07–6.15 TFLOP/s**, half **6.04–6.18**, scalar
control **5.6–5.7**. The simdgroup matrix intrinsics run on the ordinary ALUs.

This matters because LEDGER 035 asserted "the matrix units run at 17.6 TFLOP/s"
— that figure was `mul_mm`'s n=512 rate, which comes from the **tensor API**, a
different unit on a different code path — and built the register-resident kernel
on it. Every roofline in the effort inherited the error. It explains, after the
fact, why 055, 067, 069 and 070 all failed: the kernel was already at 67% of a
ceiling nobody had measured.

## `scmv.m` — column-exact scalar multi-column Q4_K mat-vec

```bash
clang -fobjc-arc -O2 -framework Foundation -framework Metal probe/scmv.m -o probe/scmv
./probe/scmv                # M=5120  K=5120 — too small, launch-overhead bound
./probe/scmv 40960 5120     # the shape to compare against, 112.5 MB of weights
```

Sweeps N (1,2,3,4,6,8) × NROW (2,4,8), checking every point against a CPU
reference. The design: lane *t* owns one `uint` of `qs`, so `g = t/8`, both
activation reads are aligned `float4` and the weight read is one aligned `uint`.
Weights are streamed once, nothing goes through threadgroup memory.

**Result (LEDGER 084):** correct everywhere (max rel err 2.8e-06 … 2.2e-05) and
**3.4× off** the simdgroup kernel on arithmetic efficiency — 1.5 TFLOP/s against
5.07. Cost grows ~0.16 ms per column instead of staying flat.

The reason is instruction issue, not FLOPs: one `simdgroup_multiply_accumulate`
retires 512 MACs where a scalar fma retires 32 per simdgroup, so the same work
needs ~16× the instructions and Q4_K's scale decode is amortised over 8 values
per lane instead of 8192. This refuted a plan proposed two ledger entries
earlier in the same session.

## Why they are here and not in `bench/`

`bench/**` is frozen: editing what an existing script measures voids the run.
Adding new tooling beside it is expected. Neither probe loads the model, so
neither needs `bench/lock.sh` for correctness — but both were run under it
anyway, because a probe competing with a benchmark corrupts the benchmark, not
the probe.
