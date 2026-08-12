# SETUP LOG

## HANDOFF

**Baseline.** Decode `tg128` = **14.567 ± 0.404 tok/s** (n=10, gated; on a settled
machine 14.829 ± 0.189). Prefill `pp512` = **306.75 ± 6.09 tok/s**. Trunk
`0b1bad14ff204627636aeb1de22ddcd5acb859d4` (b10380), Metal, ngl 99, fa on, KV f16,
ctx 4096. `test-backend-ops`: 14022/14022, zero failures.

**Ceiling.** `ceiling_1tok` = **16.40 tok/s** — 16.52 GB of GPU-resident non-MTP
weights against a **measured** 270.8 GB/s (88.2% of the 307 GB/s vendor peak,
inside the 60–90% gate, with cache-residency and saturation controls both clean).

**We are at 88.8% of it.** Counting the ~906 MB/token of SSM recurrent-state
traffic this hybrid architecture also moves, closer to 94% — that second figure is
derived from code reading and is the most load-bearing unverified claim in
LEADS.md; confirm it first.

**Where the headroom is.** Not in the decode matmul. That path has single-digit
percent left and it is bounded — no kernel can take 16.40 past 16.40. The
remaining decode-side wins are *byte deletions*, chiefly the three redundant
copies of the 144 MiB recurrent state per token (LEADS.md L8). Prefill at 306
tok/s is compute-bound and is where fusion and the tensor path can pay. **The only
unbounded lever is draft acceptance**, because it multiplies the wall instead of
fighting it.

**The lead I would chase first is free.** The verify-step cost is a
**non-monotonic staircase** in draft depth: Metal's small-batch mat-vec excludes
K-quants below `ne11 = 4`, and the `r1ptg` tile table spoils several widths above
it. Depths 1, 2, 5, 6, 7 cost **2–3 full 16.5 GB weight streams** per verify step;
depths **3, 4 and 8+ cost one**. This model is 99.4% K-quant by bytes. Every rung
was verified directly against the pinned source. Setting `--spec-draft-n-max` to 3
or 4 is worth up to 2–3× on the speculative path and costs nothing — no patch, no
rebuild. Then `--spec-type ngram-mod --spec-ngram-mod-n-max 16` for the
copy-heavy categories (LEADS.md L2), which is the largest number in the dossier.

**What this rig can see.** Minimum detectable effect **3.0%** (frozen KEEP
threshold). Measured resolution is 0.64% on a quiet machine, but the threshold is
deliberately set above the 3.14% paired spread of an unsettled session, because
the machine's own settling drift is 4.9% end-to-end and would otherwise be
certifiable as a win. Null test: **PASS**. Positive control: **DETECTED**, right
direction, p = 0.0078 with every pair agreeing.

---

## STATUS OF THE TEN GATES

| # | gate | status |
|---|---|---|
| 1 | `init.sh` builds from clean, `decode.sh` produces a number | **PASS** — 42 gated runs recorded |
| 2 | null test reports no difference | **PASS** — +0.542%, CI [−0.103, +1.186], NO DIFFERENCE |
| 3 | positive control detects ~4%, sabotage reverted and proven | **PARTIAL** — detected at p=0.0078; tree clean and rebuilt; **baseline-matching run still queued** |
| 4 | `quality.sh` passes on trunk, fails on degraded config | **PENDING** — queued |
| 5 | `spec.sh` reports nonzero acceptance | **PENDING** — queued |
| 6 | NOISE.md states a specific MDE | **PASS** — 3.0% |
| 7 | LEADS.md ≥10 ranked entries with falsifying experiments | **PASS** — 16 |
| 8 | GOAL.md numbers block filled | **PASS** |
| 9 | swap-clean run + memory budget in HARDWARE.md | **PASS** (threshold-based, see ASSUMPTION) |
| 10 | no weights or bulk output in the source tree | **PASS** — ~350 KB excluding `llama.cpp/`; `models` and `runs` are symlinks |

**Gates 3 (final run), 4 and 5 are blocked on machine availability, not on
anything technical.** They are armed in a background job that waits for the
machine to be quiet and then runs: post-revert baseline proof → quality reference
regeneration at ctx 4096 → verify → degraded-config check → `spec.sh`. At the
time of writing the machine is in interactive use and `bench/env.sh` is correctly
refusing to benchmark. **Re-run `bash bench/quality.sh --generate` then
`bash bench/quality.sh` and `bash bench/spec.sh --depth 1,2,3,4` on a quiet
machine to close them.** Nothing about them is expected to be difficult; the
scripts are written, the flags are verified against `--help` on this build, and
the server memory configuration that made them fail earlier is fixed.

---

## What surprised me

Ordered by how much it would have cost to discover later.

### 1. `llama-cli -no-cnv` does not give you a one-shot run, and fails silently for 12 minutes

The first coherence attempt used `-no-cnv`, which this build accepts. It
generated its answer, then dropped into the interactive REPL and spun at 98.7%
CPU reading EOF from stdin. From the outside it was indistinguishable from a
hung model load: 4.8 MB RSS, 676 MB physical footprint, huge system time, the
GGUF held open, and `MTLCompilerService` alive in the background. I spent
twelve minutes and a `vmmap` on the theory that it was thrashing or stuck in
Metal shader compilation. It was neither — it had already finished and was
waiting for input that could never arrive.

The correct flag on this build is **`-st` / `--single-turn`**. Everything now
uses `-st` with stdin redirected from `/dev/null`.

This is precisely the failure mode the brief warns about, and it is worse than
a flag that no-ops: it produces a process that looks broken while being fine,
which sends you debugging the wrong subsystem.

### 2. The model is a hybrid SSM + attention model, and that changes all the memory arithmetic

`qwen35` with `full_attention_interval = 4`: **48 of 64 layers are SSM /
gated-delta-net, only 16 are full attention.** I had been sizing the KV cache
for 64 layers. It is charged for 16:

```
llama_kv_cache: size = 512.00 MiB (8192 cells, 16 layers), K (f16) 256, V (f16) 256
llama_memory_recurrent: size = 149.62 MiB (1 cells, 64 layers)
```

64 KiB per token for KV, and the SSM recurrent state is **149.62 MiB regardless
of context length**. Long context is far cheaper on this model than on a dense
27B, and any reasoning about "KV bandwidth will dominate at long context" has to
be redone with 16 layers, not 64.

### 3. Plain decode is already at 88.8% of the memory-bandwidth ceiling

This is the finding that should shape the whole next phase, and it was visible
within the first hour. See BASELINE.md. It means the matmul path for decode is
very nearly finished, and effort spent there has almost nowhere to go.

### 4. `-md` would load the model twice

For MTP, `common/speculative.cpp:2298-2325` branches: if a draft model path is
set it calls `llama_model_load_from_file()` and loads a **second full copy** of
the weights. That is 16 GiB again on a machine with an 18.2 GiB working set — it
cannot fit. With no `-md`, the `else if (spec_mtp)` branch calls
`llama_init_from_model(model_tgt, ...)` and shares the target's weights.
`bench/spec.sh` therefore never passes `-md`, and says so in a comment loud
enough that nobody adds it back.

**Correction to an earlier version of this note**, caught in Phase H and verified
directly: I originally wrote that this path yields `is_mem_shared = true`. It does
not. `common/speculative.cpp:2298` sets `cparams.ctx_other = ctx_tgt`, but the
context constructor **resets it to `nullptr`** at `src/llama-context.cpp:142` and
restores it only for `LLM_ARCH_GEMMA4_ASSISTANT` and `LLM_ARCH_EAGLE3` /
`LLM_ARCH_DFLASH`. `LLM_ARCH_QWEN35` is in neither list, so
`common/speculative.cpp:1335` evaluates false. The *weights* are shared — that
part stands and is what makes omitting `-md` necessary — but the KV and recurrent
memory are not, which is exactly what that flag gates. Any cost model that assumes
"MTP is free because memory is shared" understates each cycle by a full extra MTP
forward pass plus a host round trip.

(Incidentally that branch passes `params.model.path` to
`llama_model_load_from_file` while logging `model_path` — it loads the *target*
path regardless of what `-md` pointed at. Not our problem here since we do not
use it, but noted.)

### 5. The GPU allocation is split across two Metal buffers

`maxBufferLength` on this device is **13639 MiB**, which is *smaller than the
model*. ggml-metal splits it (13639.69 MiB + 3382.67 MiB) transparently. Worth
knowing before blaming a mystery allocation failure on something else.

---

## ASSUMPTIONS

Each of these was a fork in the road where the brief's instruction could not be
followed literally on this machine. I picked the most defensible option and
recorded it rather than stopping.

### ASSUMPTION: work directory path

The brief specifies `WORK = ~/projects/navilan/research/qwen36-metal`. The
session's actual working directory was
`~/projects/navilan/research/qwen3.6-27b-mtp-q4-reasearch`, which was empty and
had been created minutes earlier. I used **the brief's path**, since the
deliverables block names it explicitly. The other directory is left empty and
untouched.

### ASSUMPTION: there is no passwordless sudo, so the GPU wired limit is not set

The brief requires `sudo sysctl iogpu.wired_limit_mb=<value>` in `bench/env.sh`
every session. `sudo -n true` fails on this machine — there is no way to supply
a password from an unattended script, and I will not park credentials anywhere.

What I did instead: measured what the driver actually grants by default and
checked whether it is sufficient. It is.

```
iogpu.wired_limit_mb = 0            (driver default)
recommendedMaxWorkingSetSize = 18186 MiB   (74% of the 24576 MiB installed)
```

The frozen configuration needs 17361 MiB, leaving 825 MiB of device headroom,
and the model loads fully resident. So the default is adequate and the sysctl is
**not required** for the frozen configuration.

`bench/env.sh` therefore: (a) attempts `sudo -n sysctl` if a non-zero limit is
configured, (b) **verifies the outcome by reading
`recommendedMaxWorkingSetSize` from the Metal API**, which is the quantity that
actually matters, and (c) fails with the exact command to run if the working set
is below what the model needs. Verifying the effect rather than assuming the
command worked is strictly better than the literal instruction, since a
successful `sysctl` does not by itself prove the driver honoured it.

What this costs: context is capped at 16384 instead of 32768. Raising the limit
to 20480 would fit 32768. That is recorded in HARDWARE.md and as a lead.

### ASSUMPTION: `vm.swapins` does not exist; using `vm.compressor.swapper.swapins_total`

The brief names `sysctl vm.swapins`. That OID does not exist on macOS 26.5, and
`vm_stat` on this build prints no Swapins/Swapouts lines either. The live
counters are:

```
vm.compressor.swapper.swapins_total
vm.compressor.swapper.swapouts_total
vm.swapusage
```

`bench/mem.sh` uses `swapins_total`. It deliberately does **not** use
`Pageins`, which moves by ~16 GiB on every cold model load because llama.cpp
mmaps the GGUF — treating that as swap would discard every legitimate run.

### ASSUMPTION: the zero-swap rule is a threshold, not a strict zero

The brief says any run whose swap-in count moved is discarded.
`swapins_total` is **system-wide**, and this machine swaps continuously from
other applications. Five idle 40-second windows with no benchmark running at
all measured:

```
21259 / 639 / 3071 / 1290 / 2080 pages
```

A strict `!= 0` gate discards 100% of runs here — it discarded the smoke test.
So `bench/decode.sh` gates at **25000 pages**, just above the observed idle
maximum, and records the raw delta on every run regardless so any suspicious
result can be re-examined. The failure mode the rule exists to catch — the model
itself being evicted and re-read per token — is not subtle: the one genuine
thrashing event observed moved **289,342 pages**, an order of magnitude above
the idle ceiling.

To catch what a system-wide counter cannot attribute, there is a second
independent gate: **relative standard deviation across repetitions > 5%
discards the run.** A run that thrashed has large spread between reps even when
its mean looks plausible. This gate fired on the first noise-floor attempt and
was correct to.

### ASSUMPTION: thermal signal is a measured GPU clock probe, not `powermetrics`

`sudo powermetrics --samplers thermal` needs a password (same problem as above).
`pmset -g therm` runs without sudo but on macOS 26.5 reports only a warning
*level* that stays unset until the machine is already in trouble — it printed
"No thermal warning level has been recorded" throughout, including during heavy
load. Neither gives a continuous signal, and the brief is explicit that skipping
this is not acceptable because thermal drift manufactures wins.

So `bench/thermal.sh` **measures** the thing that matters. `bench/gpuinfo
--clock` runs a register-resident FMA loop with zero memory traffic; its
throughput is proportional to GPU clock × active cores, which is exactly what
throttling and power capping reduce. On a cold idle machine it reads
**6258 GFLOPs**, with run-to-run spread of 0.07% (6262.5 / 6257.8 / 6258.0).
`throttle_pct` is the shortfall against that frozen reference.

This is arguably a *better* thermal signal than `powermetrics` would have given,
because it measures delivered GPU throughput rather than a proxy temperature.
Sanity check: 6258 GFLOP/s against a theoretical ~6.55 TFLOP/s FP32 for 16 cores
is 95%, so the probe really is clock-limited and not bottlenecked elsewhere.

### ASSUMPTION: env.sh gained a background-load gate that the brief did not list

Added after it bit. The first noise-floor attempt ran while four research
subagents were grepping the repository, and produced

```
noise  pp512=298.39±9.14  tg128=12.705±0.694 (rsd 5.46%)  DISCARDED_VARIANCE
```

against a quiet-machine 14.6 — a **13% depression**. Had that been averaged
into the noise floor it would have been recorded as intrinsic machine variance
and permanently destroyed the rig's sensitivity: the minimum detectable effect
would have been inflated past the 4% the positive control has to catch. The run
was discarded, the noise floor was restarted on a quiet machine, and `env.sh`
now refuses to run at 1-minute load average > 3.0 or with any process above 50%
CPU.

### ASSUMPTION: no Xcode, so the Phase H GPU trace is not an Xcode trace

`xcode-select -p` is `/Library/Developer/CommandLineTools`. There is no full
Xcode, so `xcrun metal`, `xctrace`, and the Metal debugger are all unavailable
(`/usr/bin/xctrace` exists but is a stub that errors). Installing Xcode is a
multi-gigabyte change to the machine that is well outside "set up a measurement
rig", so I did not.

Substitute, which is arguably more directly useful for ranking hypotheses: the
Metal backend exposes runtime switches that let each existing optimization be
**turned off and measured**, which bounds the headroom for adding more of the
same kind:

```
GGML_METAL_FUSION_DISABLE          GGML_METAL_FUSION_DEBUG
GGML_METAL_GRAPH_OPTIMIZE_DISABLE  GGML_METAL_GRAPH_DEBUG
GGML_METAL_TENSOR_ENABLE/DISABLE   GGML_METAL_CONCURRENCY_DISABLE
GGML_METAL_CAPTURE_COMPUTE         GGML_METAL_NO_RESIDENCY
```

plus `test-backend-ops perf -b MTL0` for per-op throughput on the real backend.
Details and results in LEADS.md.

### ASSUMPTION: GOAL.md was authored, not copied

The brief says "GOAL.md copied in, with its numbers block filled from your
results", but no GOAL.md was supplied to copy. I wrote one, with the numbers
block filled from measurements taken here.

### ASSUMPTION: perplexity and KLD run on a bounded slice, not the whole corpus

`llama-perplexity` over all of wikitext-2 test (~330K tokens) at ~300 tok/s
prefill is ~18 minutes per invocation, which makes `bench/quality.sh` too
expensive to run routinely — and a gate nobody runs is not a gate. Frozen at
`-c 512 --chunks 40` (20,480 tokens). The corpus file and the chunk count are
frozen together; both must be quoted whenever a perplexity number is.

---

## What was done, in order

1. **Phase A — hardware.** Verified every line of the brief's hardware block
   against the machine. All of it matched. Wrote and froze a Metal streaming-read
   bandwidth probe (`bench/bwprobe/`) and a GPU working-set / clock probe
   (`bench/gpuinfo/`). Established the memory budget and the largest context
   that fits.
2. **Phase B — build.** Fresh clone of `ggml-org/llama.cpp`, pinned
   `0b1bad14ff204627636aeb1de22ddcd5acb859d4` (b10380). Release + Metal.
   `~/projects/forks/llama.cpp` was read for history only and never touched.
   Every flag used was checked against `--help` on this build.
3. **Phase C — model.** Not present on this machine (only MLX-format Qwen3.6-27B
   weights were). Fetched `unsloth/Qwen3.6-27B-MTP-GGUF` at revision
   `5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace`; SHA-256 matches the Hub's LFS OID
   exactly. **MTP tensors confirmed present** at `blk.64.nextn.*`.
4. **Phase D — froze the measurement surface.** `env.sh`, `mem.sh`,
   `thermal.sh`, `decode.sh`, `spec.sh`, and 14 categorized prompts.
5. **Phase E — baseline.**
6. **Phase F — noise floor, null test, positive control, drift.**
7. **Phase G — quality oracle.**
8. **Phase H — research dossier.**
