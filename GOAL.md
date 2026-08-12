# GOAL — maximum decode speed for Qwen3.6-27B-Q4_K_M on this M5 Pro, at unchanged quality

Make llama.cpp generate tokens faster on this exact model and this exact machine,
by any method that doesn't cost output quality. Kernels, graph structure,
speculation policy, build configuration, the machine itself. Run many
experiments. Do not stop.

Setup is done and the rig is calibrated. You are not starting from zero: there is
a measured baseline, a measured ceiling, a proven noise floor, a 16-entry ranked
dossier, and a fork by someone who already did a version of this work on
different silicon. Read `SETUP-LOG.md` before anything else — the handoff at the
top is worth more than this file's first three sections.

## The numbers

Measured on this machine, reproducible from this tree. Not estimated, not
inherited.

```
MACHINE
  chip                        Apple M5 Pro: 16 GPU cores, 15 CPU (5 Super + 10 Perf)
  unified memory              24 GiB
  attained bandwidth          270.8 GB/s   (bench/bwprobe, 88.2% of 307 GB/s peak)
  GPU clock reference         6258 GFLOP/s (bench/gpuinfo --clock, cold idle)
  GPU working set             18186 MiB    (driver default; wired limit unset, no sudo)
  maxBufferLength             13639 MiB    — smaller than the model; ggml splits it

MODEL
  file                        Qwen3.6-27B-Q4_K_M.gguf  (unsloth/Qwen3.6-27B-MTP-GGUF)
  sha256                      a7cbd3ecc0e3f9b333edee61ae66bc87ed713c5d49587a8355814722ed329e0f
  architecture                qwen35 — 48 SSM layers + 16 full-attention layers + 1 MTP block
  MTP tensors                 PRESENT, single head at blk.64.nextn.*
  weights read per decode step 16.52 GB

CEILING
  ceiling_1tok                16.40 tok/s
  ceiling_spec                16.40 × mean accepted tokens per forward pass

BASELINE  (clean upstream trunk 0b1bad14f, speculation off)
  decode  tg128               14.567 ± 0.404 tok/s   (settled machine 14.829 ± 0.189)
  prefill pp512               306.75 ± 6.09 tok/s
  fraction of ceiling         88.8%
  test-backend-ops            14022/14022

SENSITIVITY
  minimum detectable effect   3.0%   ← the KEEP threshold
  measured resolution         0.64% on a quiet machine
  required cooldown           60 s
  null test PASS · positive control DETECTED (5.88%, p=0.0078)

FROZEN CONFIG
  context 4096 · ngl 99 · fa on · KV f16 · b/ub 512
```

## What follows from those numbers

**The decode matmul is closed.** 88.8% of a hard wall, possibly ~94% once SSM
recurrent-state traffic is counted. No kernel takes 16.40 past 16.40. If you find
yourself tuning `mul_mv` to make plain decode faster, you have misread this file.
The pre-registered pivot has already fired.

Three fronts remain, in order of headroom:

1. **Acceptance rate.** The only unbounded lever. It multiplies the wall instead
   of fighting it. Going from ~1.5 to ~2.5 mean accepted tokens per forward pass
   is worth more than every kernel in the dossier combined.
2. **Byte deletion on the decode path.** Not faster reads — *fewer* reads. Chiefly
   the three redundant copies of the 144 MiB recurrent state per token
   (`LEADS.md` L8). This is the only remaining way to move `ceiling_1tok` itself.
3. **Verify-step and prefill compute.** 306 tok/s prefill is compute-bound, so
   fusion, tiling, and kernel specialisation pay here — and the verify pass of
   speculative decoding runs on the same kernels, which is why prefill work feeds
   decode.

## Where you start: Shrey's fork

`https://github.com/shreyvish5678/llama.cpp-qwen3.6-metal`, already wired in as
remote `shrey`, branch `shrey-shipped`. He did this work for the same model on a
**32-core M4 Max, 410 GB/s, 36 GB** — roughly twice this GPU. His fork notes
(`QWEN36-METAL.md` on that branch) are unusually honest and you should read them
in full before touching anything.

**Your first job is to port everything of his that is relevant to this chip, and
re-derive everything that isn't.** Not "evaluate whether to" — port it. He has
already done months of work you would otherwise repeat, and the parts that
transfer are free. Go change by change through `git log 3653e6d6..shrey/shipped`
and `git diff` each one; read `QWEN36-METAL.md` on that branch in full first.

**His stack is not trunk, though. It is your first six candidates, pre-written.**
Trunk stays at clean upstream `0b1bad14f`, because that's what the baseline, the
noise floor, and the quality references were measured against. Porting is free;
*claiming* is not. Each change enters the ledger as a candidate, on its own
branch, in dependency order, and earns its KEEP on this machine or doesn't. A
change of his that loses here is a finding about the difference between 32 cores
and 16, and it is worth writing down as carefully as a win.

Note his base is `3653e6d6`, older than our trunk. Rebase his commits onto
`trunk`; don't move trunk back to meet him.

| | change | his measurement | env toggle |
|---|---|---|---|
| S0 | multi-column K-quant mat-vec (`_r1_2/3/4` for q4_K/q5_K/q6_K) — everything below builds on it | 1.431× speculative decode | — |
| S1 | sixteen lanes per super-block, Q4_K multi-column mat-vec | decode +6.7% | `GGML_METAL_NO_Q4_L16=1` |
| S2 | wider row tile for Q6_K multi-column mat-vec at verify width ≥3 | decode +4.1%, bit-exact | `GGML_METAL_NO_Q6_NR0=1` |
| S3 | 32×32 accumulator tile for quantised `mul_mm` | prefill +3.9%, bit-exact | — |
| S4 | `dequantize_q4_K` half-precision scale division | quality fix, PPL 6.6534 → 6.6475 | — |
| S5 | draft depth 2 → 3 | +6.01% | policy, not code |

Plus two riders on S0 — a gated-delta-net state-write fusion that fixes upstream
PR #25788 firing 0/192 times, and an FR-Spec draft-vocabulary trim he measured at
+2.8% at ~1.6σ and explicitly did **not** confirm. Treat the trim as an unproven
lead, not a shipped change.

**Where his thresholds will be wrong here, and it is not subtle.** He says it
himself: *"the thresholds are tuned to this chip."* `ne01 >= 4096` is really a
threadgroups-per-core quantity — 16 per core on a **32-core** part. This machine
has **16 cores**, so that same row count yields 32 per core. Every gate of that
shape must be **re-derived from core count**, not inherited. The same applies to
`N_R0_*_R1` register-pressure constants and to the `ne11 < 64` regression in S3.
A candidate that ports a threshold verbatim and wins is luckier than it is right;
say so in the ledger.

**Two of his findings are worth more than his kernels.**

- **`test-backend-ops` passes on things it cannot see.** His change S2 passed
  1143/1143 with its gate both on *and* off — both arms ran identical code, so the
  test could not have failed. Verify a kernel-selection change by **counting
  gated pipelines compiled**, never by a green test. And `strings` on the dylib
  does not work for this: with `GGML_METAL_EMBED_LIBRARY` the kernel names are
  built by the Metal preprocessor and exist nowhere in the binary. His added test
  cases — `ne01 ∈ {4088, 4095, 4096, 4100, 4104}` × width {2,3,4}, `mul_mm`
  partial tiles, large-stride shapes — are independent of every Metal change and
  are the single most portable thing in the fork. Take them early.
- **`test-backend-ops` never calls `graph_optimize`.** A graph-level fusion can
  pass every unit test and fire zero times in production. That is exactly what
  upstream's gated-delta-net fusion does. Any fusion you write gets a firing-count
  check, not a pass.

**One pre-existing bug to know about, not to trip over.** The K-quant mat-vec
kernels guard the store but not the load, so the last threadgroup reads up to
`nr0*nsg - 1` rows past `ne01` — 3 rows on upstream, 7 with his changes. Latent
here because every affected tensor has `ne01` divisible by 8, and invisible to
`test-backend-ops` because results stay correct. If you change `nr0`, you widen
it. Fixing it is upstream-worthy on its own.

**Also upstreamable, independent of us:** `--spec-draft-n-max` is unguarded for
single-head MTP models — the clamp sits inside `if (chain_heads)` and
`chain_heads` requires `n_mtp_layers > 1`. This model has exactly one head, so
any depth is accepted without complaint. That matches what setup found
independently about `is_mem_shared` being false for `LLM_ARCH_QWEN35`.

## The bar

Three things you cannot argue with. None of them is your opinion about a diff.

**1. The frozen baseline** in `BASELINE.md`, measured on trunk by a rig that
passed a null test and a positive control. It does not move. Re-measure it under
the frozen protocol and write a new dated entry if you must; never quietly adopt
a better-looking number. It is re-measured after every merge to trunk.

**2. Physics.** `ceiling_1tok` = 16.40 tok/s for anything reading all weights per
token. We are at 88.8%. The only ways past it are accepting more tokens per
forward pass, or reading fewer bytes.

**3. Quality**, as `bench/quality.sh` defines it: greedy output against the frozen
references, perplexity within +0.02 on the frozen slice (`-c 512 --chunks 40`,
quoted together with every perplexity number), KL-divergence within tolerance,
`test-backend-ops` clean on touched ops. There is no exchange rate between tok/s
and quality here.

### Bit-exactness: the rule setup wrote is wrong, and here is the correction

Setup's GOAL.md said speculative decoding is lossless by construction, so any
divergence is a bug. **That is true of the algorithm and false of this backend**,
and following it literally would burn days chasing a non-bug.

Shrey measured it directly: on his fork, speculative output matches unspeculated
greedy on **9 of 10** frozen prompts, at *every* depth, with a different prompt
diverging at depth 2 than at depth 3. The cause is structural — the verify pass
selects its mat-vec kernel by batch width, and those instantiations partition the
`simd_sum` reduction differently from the width-1 path. The last bits differ,
and wherever two tokens are near-tied the argmax flips. The first divergence he
traced sat mid-JSON-schema with both continuations valid.

So classify every candidate **before you build it**, declare the class in the
ledger, and let the critic verify the class against the diff:

- **Class A — arithmetic untouched.** Scheduling, dispatch, caching, residency,
  memory layout that preserves values, pure policy changes. Must be
  **token-exact** against the references. Divergence is a real bug.
- **Class B — arithmetic changed.** Different reduction partitioning, tile
  geometry, accumulation order, fusion, vectorization — and **anything that
  changes which kernel instantiation runs**, which includes most speculative-depth
  changes on this backend. Judged by **KL-divergence and perplexity**, with token
  divergence tolerated only where the divergence point is traced to a near-tie.

**Before the first Class B candidate, derive the floating-point noise scale.**
Build trunk twice with benign numerical differences, diff logits across the frozen
prompts, take p99.9 of `|Δlogit|`, record a near-tie threshold at a small multiple
of it in `NOISE.md`. Until that exists, "it's just floating point" is a hypothesis,
not an explanation — show the logit gap. A divergence with a wide gap is a bug
whatever class it was declared.

A change declared Class B whose diff doesn't touch arithmetic is gate shopping and
is treated as a failed candidate.

## What each result means — decided in advance

- **Above `ceiling_1tok` with speculation off** → the benchmark is broken. There
  is no kernel that beats bandwidth. Find the bug before logging anything.
- **Above `ceiling_1tok` with speculation on and nonzero accepts** → working as
  designed. Keep pushing.
- **A decode win claimed from matmul tuning with speculation off** → suspect it.
  We are at 88.8%; the room isn't there. Look for a measurement fault first.
- **A win below 3.0%** → not a win. It's noise wearing a hypothesis.
- **Acceptance above ~90% at depth ≥3** → check it against the verifier before
  celebrating. Shrey saw 0.925 legitimately, so this is a look-twice, not an
  automatic bug — but an inflated acceptance is exactly how a verification bug
  presents.
- **A Class B token divergence with a wide logit gap** → a bug, not floating point.
- **A run whose swap-in delta exceeds 25000 pages, or whose rsd across reps
  exceeds 5%** → discarded, not averaged. Both gates already fired correctly
  during setup and both were right to.
- **A gate that passes with the feature switched on and off** → the test is blind.
  Count gated pipelines instead.

## You are autonomous

Never ask for clearance. Never stop to check in. Never wait. **There is no step in
this project that requires my approval** — not creating branches, not pushing to
our fork, not opening an upstream PR, not changing the machine, not deciding what
to try next. If something is blocked or ambiguous, write your reasoning to the
ledger, pick the next hypothesis, and keep moving. I read the ledger and stop the
run myself when I want to.

The single hard blocker is `sudo`, which cannot be automated — see *The machine is
in scope*. Route around it; never stall on it.

Decide your own decomposition — acceptance policy, draft/verify structure,
recurrent-state traffic, mat-vec instantiation, graph fusion, dispatch overhead,
build and shader flags, machine configuration. Fan out builders. Don't ask me
which.

Delegate for genuinely independent tracks. Don't delegate what you can finish in
a handful of tool calls, and never spawn a subagent to double-check your own
reasoning — that's the critic, and it only works because it's independent.

**Parallelism is for thinking and building; measurement is serialized.** One GPU,
18186 MiB of working set, a model that takes 16 GiB of it. Two benchmark
processes at once means both numbers are garbage. `bench/env.sh` already refuses
above load 3.0 or with any process over 50% CPU — that catches the common case,
but add `bench/lock.sh` (atomic lockfile, PID, start time, stale timeout) and
hold it across an entire A/B, never per invocation. A lock released between arms
lets someone else's run land inside yours and produces a plausible number instead
of an obviously broken one.

**Abandon rule.** After three consecutive rejected candidates on one hypothesis,
or four hours on it, write what you learned and take the largest remaining gap.
"Do not stop" applies to the run, not to a dead line of attack.

## Builders don't grade themselves

Every candidate goes to a **critic subagent with fresh context** that has never
seen the builder's reasoning. The critic gets the diff, the declared class, the
frozen bench scripts, and `BASELINE.md`. Nothing else — no narrative, no "here's
why this should be faster." The explanation is exactly what must not reach the
grader.

The critic rebuilds from the branch, takes the lock, runs `bench/ab.sh` itself,
runs `bench/quality.sh` itself, and rules.

**Only critic numbers enter the ledger.** A builder reporting its own tok/s is
producing fiction.

- **KEEP** — beats baseline by more than 3.0%, reproduced independently at a
  different time and thermal state, quality clean. Merges to trunk one at a time,
  then trunk is re-benchmarked and a new dated baseline entry written. Two
  verified wins can remove the same stall and fail to add; you will not notice
  unless you look.
- **REJECT** — didn't clear the bar, or quality failed. Builder takes the largest
  remaining gap.
- **NOISE** — inside the floor, or builder and critic disagree in magnitude.
  Re-run once at a different time. If they still disagree in *direction*, escalate
  once to the heavy protocol in `NOISE.md`, rule, and stop arguing.

**Where an env toggle exists, A/B on one binary.** `GGML_METAL_NO_Q4_L16=1` and
`GGML_METAL_NO_Q6_NR0=1` switch arms without a rebuild, which removes build
variance from the comparison entirely and halves the wall clock. Build the same
toggle into your own gated changes — an env-gated change is cheaper to verify,
cheaper to bisect, and cheaper to withdraw.

To be clear about what these are: a **measurement device**, not an escape hatch
for me. Nothing in this project waits on a human flipping a switch.

## Measurement protocol

`bench/env.sh`, `mem.sh`, `thermal.sh`, `decode.sh`, `spec.sh`, `ab.sh`,
`quality.sh`, `bwprobe/`, `gpuinfo/`, and `prompts/` are **frozen**. Editing them
voids the run. Adding *new* tooling alongside them is fine and expected;
altering what an existing script measures is not.

- **Replication, not reps.** Independent reproduction — different agent, different
  time, different thermal state — is the evidence. One run agreeing with itself
  is not.
- **A/B interleaved**, never all-A then all-B. Settling drift on this machine is
  4.9% end to end, which is why the 3.0% threshold sits above the 3.14% paired
  spread of an unsettled session rather than at the 0.64% the rig can resolve when
  quiet.
- **Thermal is measured, not inferred.** `throttle_pct` against the frozen
  6258 GFLOP/s reference. Log it with every result.
- **`bench/mem.sh` before and after**, gate at 25000 pages, and record the raw
  delta regardless.
- **Always report acceptance rate and mean accepted tokens per forward pass next
  to any speculative tok/s**, per prompt category. Aggregate acceptance hides that
  copy-heavy categories behave nothing like prose, and that difference is itself a
  lead (`LEADS.md` L2).
- Re-run `bench/env.sh` at the start of every session.

### Policy numbers expire

Shrey's sharpest observation, and it applies to us immediately:

> A configuration decision is a measurement against a particular kernel stack. It
> expires when the stack changes.

He measured depth 3 as a loss, wrote it down as closed, and was correct at the
time — then shipped two kernels gated on verify width, which made depth 3 worth
+6.01%. He had re-derived kernel numbers constantly and never re-derived a policy
number.

`LEADS.md` L1 — the non-monotonic verify-cost staircase in draft depth — is
exactly such a policy number. **Re-derive every depth, threshold, and gate after
any change to the kernels they select between**, and mark policy entries in the
ledger as expiring rather than settled.

Build the instrument that makes this cheap: per-draft-step thresholds, with a
step's threshold set above 1.0 to disable it, so one server expresses several
depths inside a single model load seconds apart instead of two servers twelve
minutes apart. Shrey used exactly this and did not ship it; on a machine where a
model load costs what it costs here, it may be the highest-leverage tooling in the
project.

## Git and fork discipline

The fork exists and is wired up. Use it. Nothing here needs my approval.

```
origin    https://github.com/navilansanthanakrishnan/llama.cpp-qwen3.6-m5   ← ours, push freely
upstream  https://github.com/ggml-org/llama.cpp                            ← read only
shrey     https://github.com/shreyvish5678/llama.cpp-qwen3.6-metal         ← read only
```

Branches already present: `trunk` (clean upstream `0b1bad14f`, pushed to origin),
`shrey-shipped` (his stack), `master` (upstream tracking).

- **Every hypothesis gets its own branch off `trunk`, in its own `git worktree`,**
  named for the hypothesis. Commit early and often with real messages: what
  changed, the predicted mechanism, the gated variant if any. **Push every branch
  to `origin` as you go** — a crash or a compaction must never cost work.
- **Critics build from the branch**, never from a builder's working tree. A
  working tree carries uncommitted files and a warm cache; rebuilding from the
  branch is what makes the diff the only variable.
- **Only KEEPs merge to `trunk`**, one at a time, and `trunk` is pushed after each
  merge with the new baseline recorded in the commit message.
- **The research tree** — commit after every ledger entry, every baseline
  re-measurement, every finding. `LEADS.md`, `LEDGER.md`, `NOISE.md`,
  `BASELINE.md` are the project's memory; an uncommitted memory is one crash away
  from gone.
- **Never touch `~/projects/forks/llama.cpp`.** It tracks upstream; read its
  history, write nothing.
- Nothing large enters the source tree. Weights in
  `~/projects/assets/models/qwen36-metal`, bulk output in
  `~/projects/assets/runs/qwen36-metal`, both symlinked in.

**Upstreaming is yours to do.** Two things are upstream-worthy on Shrey's own
account and independent of this project: S4's `dequantize_q4_K` half-precision
division fix, which makes every Q4_K model on Metal slightly more accurate, and
the `test-backend-ops` coverage block. The unguarded `--spec-draft-n-max` clamp
for single-head MTP models is a third. When you have one that is genuinely
upstream-worthy — measured here, tested, minimal, and not tuned to this chip —
prepare it as a clean branch against `upstream/master` and open the PR from our
fork. Log it in the ledger. Don't upstream anything whose thresholds are
M5-specific; that's what our fork is for.

Identity is `Navilan Santhanakrishnan
<143132458+NavilanSanthanakrishnan@users.noreply.github.com>`. Never commit
secrets or raw transcripts.

## The machine is in scope

The machine is a legitimate optimization surface and some of the largest wins may
be there. Power settings, background services, indexing, scheduler and thermal
conditions are all fair game, all reversible, and all yours to change without
asking.

**The one thing you cannot do is `sudo`.** There is no passwordless sudo on this
machine and you will not park credentials anywhere to get it. The GPU wired limit
is therefore unset (`iogpu.wired_limit_mb = 0`), leaving 18186 MiB of working set
and capping context at 4096; raising it to 20480 would fit 32768.

**Do not block on this.** Try `sudo -n` once, and when it fails, write the exact
command into `RESUME.md` under a heading I'll see, and carry on at context 4096.
Everything in this project is measured at 4096 and the frozen configuration
assumes it. A larger context is an opportunity, not a prerequisite, and a run that
stalls waiting for a password has failed at the only thing it was asked to do.

**Machine changes are not candidates.** A configuration change invalidates the
frozen baseline, so it is never a "win" — it is a new configuration requiring
re-baselining before any candidate is measured against it. Record the change, the
exact command that reverses it, and the new baseline. Never let a machine change
and a code change land in the same comparison.

## Where to look

`LEADS.md` — 16 ranked entries with predicted mechanisms, magnitudes, and
falsifying experiments. Work it top-down, re-rank as evidence arrives.

`RESUME.md` names the first three: L1 (the depth staircase, free, no patch), L2
(ngram drafting for copy-heavy categories, the largest number in the dossier),
L10 (`GGML_METAL_TENSOR_DISABLE=1`, one env var, doubles as a correctness check).

Standing reminders:

- **There is no GPU trace on this machine** — no Xcode, and installing it wasn't
  in scope. Setup's substitute is better for ranking anyway: the Metal backend's
  runtime switches (`GGML_METAL_FUSION_DISABLE`, `GRAPH_OPTIMIZE_DISABLE`,
  `TENSOR_ENABLE/DISABLE`, `CONCURRENCY_DISABLE`, `NO_RESIDENCY`,
  `CAPTURE_COMPUTE`, `GRAPH_DEBUG`) let you turn each existing optimization *off*
  and measure it, which bounds the headroom for adding more of the same kind. Plus
  `test-backend-ops perf -b MTL0` for per-op throughput on the real backend.
- **This is M5, and Shrey's fork is M4.** He states flatly that *"there is no
  matrix unit on M4"* — `matmul2d` lowers onto the ordinary shader path, and his
  prefill sits at ~77% of the scalar roof. M5's GPU cores are a different
  generation. **Establish early whether this part exposes matrix/neural
  acceleration that Metal's matmul path or the tensor API can reach.** If it does,
  his entire prefill analysis has a different ceiling here and L10 is much bigger
  than it looks. If it doesn't, you've closed a large question for one experiment.
- **The most load-bearing unverified claim in the dossier** is the ~94%-of-ceiling
  figure that counts SSM recurrent-state traffic. It gates how much decode headroom
  exists at all. Confirm it with `GGML_METAL_GRAPH_DEBUG=1` node counts before
  trusting L8's magnitude.
- Diffing Metal against CUDA and Vulkan remains the highest-yield source of ports.

Use `ggml_can_fuse()` for fusions — hand-rolled pattern detection shipped a
correctness bug in this project by not checking for other consumers of the
intermediate. And check that a fusion actually *fires*: upstream's gated-delta-net
fusion fires 0 times out of 192 here because `graph_optimize_reorder()` hoists a
node between the pair.

## The lead list is a starting rank, not a fence

`LEADS.md` is where the known yield is, not the boundary of what's allowed. When
it's exhausted, or when nothing on it works, go find new hypotheses. Read the
literature. Port from other runtimes. Invent something nobody has published. The
freedom is in where hypotheses come from; the bar they clear does not move.

**Reserve one experiment in four for off-list ideas.** A greedy ranking never
leaves its own list, and the list was written before you knew anything.

A paper is worth reading only if it ends in a `LEADS.md` entry — mechanism,
whether that mechanism survives *this* chip, quantization and memory-bound regime,
predicted magnitude, cheapest falsifying experiment. Published speedups routinely
evaporate against a different memory hierarchy. Translating the claim into a
prediction about this machine is the work; reading the paper is not. Time-box it:
a literature pass that yields no ranked leads yielded nothing.

Expect quality-affecting methods to die at the gate — pruning, sparsity, cache
eviction, early exit, heavier quantization. Test one as a separately labeled
configuration if you believe it's near-lossless; it never mixes into the headline
number. Speculative-decoding variants are the opposite case — n-gram and
prompt-lookup drafting, tree attention, adaptive depth by category, FR-Spec-style
draft-vocabulary trimming. They act on acceptance rate, which is the only
unbounded lever, and that is where external ideas are most likely to pay.

Search externally only where the repo doesn't answer, scoped to Metal and Apple
GPU writeups, inference and speculative-decoding papers, llama.cpp forks, and
CUDA/Vulkan implementations worth porting.

## Files

Yours: `ggml/src/ggml-metal/**`, `common/speculative.*`, `src/llama-graph.cpp`,
`src/llama-model.cpp`, `src/models/qwen35.cpp`, `tools/server/server.cpp`,
`ggml/src/ggml.c`, `tests/test-backend-ops.cpp`, build and shader configuration.
Go wherever the hypothesis leads.

Frozen: `bench/**`, `prompts/**`, `quality/**`, `BASELINE.md`, `NOISE.md`,
`HARDWARE.md`, the GGUF, and any conversion tooling. The frozen context length and
the frozen perplexity slice are part of the measurement surface, not tunables.

## Things that are not speedups

- **A quality knob turned down.** Not the quantization, not KV cache type, not
  context length, not the sampler, not a shorter `-n`. A near-lossless setting can
  be argued as a separately labeled configuration with its own full quality run;
  it never joins the headline number.
- A prefix-cache hit measured as prefill.
- A result labeled MTP with zero draft activity.
- A win only one agent could reproduce.
- A Class A divergence, or a Class B divergence you can't trace to a near-tie.
- A change declared Class B whose diff doesn't touch arithmetic.
- Perplexity up more than 0.02 on the frozen slice, KLD out of tolerance, or
  `test-backend-ops` failing on a touched op.
- A gain that only appears when the machine is hot, or only when it's cold.
- A gain bought with memory the model needs, or with a machine configuration
  change smuggled into a code comparison.
- **A number from Shrey's machine.** His techniques are leads; his measurements
  were taken on twice this GPU and never enter this ledger as evidence. Every one
  of his changes is re-measured here or it doesn't count.
- A threshold inherited from a 32-core part without being re-derived for 16.

## Ledger

`LEDGER.md` — every experiment including failures. Hypothesis, declared class,
**predicted mechanism and predicted magnitude before running**, `git diff --stat`,
critic's numbers ± sd, acceptance rate and tokens per forward pass, quality
result, `throttle_pct`, swap delta, verdict, and the commit.

Predicting magnitude first is what makes this a learning loop rather than a log.
When the outcome wasn't predicted, write what was wrong with your model of the
hardware. That correction is worth more than the experiment.

Mark policy entries — depths, thresholds, gates — as **expiring**, with the kernel
stack they were measured against.

Live `progress.html` I can open from my phone: decode tok/s vs baseline vs
ceiling, speculation on and off, acceptance and tokens/forward by category,
fraction of ceiling, experiments run, in flight, landed.

Roughly 5 in 30 experiments landing is normal. A wave where eight of ten died is a
successful wave. Do not manufacture wins. If the honest answer is "the decode
matmul is finished and everything left is acceptance rate," that is already
written above — go work on acceptance rate.

## Before any optimization: close the three open gates

Setup is complete except for three gates blocked on machine availability, not on
anything technical. **Do these first.** Optimizing on an unproven rig is how a
month of results turns out to be fiction.

```bash
cd ~/projects/navilan/research/qwen36-metal

# gate 3 — prove the reverted tree is back at baseline (expect ~14.6–14.9)
for i in 1 2 3; do bash bench/decode.sh --label reverted --tag "post-revert-$i"; done

# gate 4 — quality oracle: regenerate at ctx 4096, verify, prove it FAILS degraded
bash bench/quality.sh --generate
bash bench/quality.sh
bash bench/quality.sh --ctk q4_0 --ctv q4_0    # must FAIL, naming the check

# gate 5 — MTP acceptance across the L1 staircase
bash bench/spec.sh --depth 1,2,3,4
```

Then fill `PLACEHOLDER_SPEC_PENDING` in `BASELINE.md`, flip gates 3–5 in
`SETUP-LOG.md`, and re-run `bash bench/phase-f.sh null` — the prefill-health gate
was added to `decode.sh` *after* the null test ran, so confirm the procedure
didn't become credulous. Commit and push all of it.

Then, in order:

1. **Derive the floating-point noise scale and near-tie threshold.** S0–S3 are
   Class B and cannot be adjudicated without it.
2. **Add `bench/lock.sh`** and route every measurement through it.
3. **Take Shrey's `test-backend-ops` coverage block first.** It is independent of
   every Metal change, it is the piece he rates most portable, and it is what
   makes the rest of the porting safe — without it, a kernel-selection change can
   pass 14022/14022 while doing nothing.
4. **Port S0, then S1, S2, S3, S4**, each on its own branch, each re-deriving any
   threshold that encodes a core count, each through the critic.
5. **Re-derive the depth policy (S5 and `LEADS.md` L1) last**, after the kernels
   land — because a policy number is a measurement against a kernel stack and his
   own depth-3 result only appeared once the width-gated kernels were in.

Then the dossier, top-down, one experiment in four off-list.

## Known-good facts — do not re-derive

- `llama-cli` one-shot is **`-st`**, not `-no-cnv`. `-no-cnv` is accepted, finishes,
  then spins in the REPL at 98.7% CPU looking exactly like a hung model load.
- **Never pass `-md`** for MTP — it loads a second full copy of the weights, 16 GiB
  on an 18.2 GiB working set. Without it the MTP branch shares the target's
  weights. Note the weights are shared but KV and recurrent memory are **not**:
  `ctx_other` is reset to `nullptr` for every arch except `GEMMA4_ASSISTANT`,
  `EAGLE3` and `DFLASH`, so `is_mem_shared` is false here and each cycle costs a
  full extra MTP forward pass plus a host round trip.
- `llama-server` needs `-b` capped at the ubatch (512), plus `-ctxcp 0 -cram 0`,
  or it allocates 32 context checkpoints at 149.6 MiB and an 8 GiB prompt cache
  and dies.
- Frozen context is **4096**. 16384 passes the static fitter and OOMs in warmup;
  8192 loads and decodes at 5.69 tok/s.
- KV is charged for **16 layers, not 64** — 64 KiB/token. The SSM recurrent state
  is 149.62 MiB regardless of context. Long context is far cheaper on this
  architecture than on a dense 27B.
- `vm.swapins` doesn't exist on macOS 26.5; the live counter is
  `vm.compressor.swapper.swapins_total`. Don't use `Pageins` — it moves ~16 GiB on
  every cold load because the GGUF is mmapped.
- The `.venv/` is only needed for `gguf_dump.py`.

## Continuity

Your context window will be compacted; do not stop early over token budget. Save
state to `LEDGER.md` and `RESUME.md` before it refreshes, and commit. Never
artificially stop because the window is filling.

On a fresh window: `pwd`; `bash bench/env.sh`; read `RESUME.md`, the tail of
`LEDGER.md`, `BASELINE.md`, `NOISE.md`, `SETUP-LOG.md`'s handoff, and `git log`;
check open worktrees. **Resume in-flight work before starting anything new** — most
of the good ideas are already written down, and the failure mode of a fresh window
is starting a fifth thing while four sit half-finished.

Keep looping. There is no final round.
