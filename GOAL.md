# GOAL — 40 tok/s decode on Qwen3.6-27B-Q4_K_M, this M5 Pro, unchanged quality

The target is **40 tok/s decode**. It does not move, and the run does not stop
until the number is real.

**This file holds what does not change.** Current numbers live in `RESUME.md` and
`LEDGER.md`; read those for state. Anything written here that turns out to be a
point-in-time measurement is a bug in this file — delete it rather than update it.

## The one invariant

**The model's output must stay indistinguishable from trunk's.** Same weights,
same quantization, same context length, same sampling semantics, the same
distribution over tokens for any prompt.

No smaller context, no lighter quant, no lossy KV compression, no shortened
generation, no warm-cache measurement, no benchmark-specific special-casing —
nothing that is fast because it is quietly doing less of the work. A change that
makes the model cheaper by making it different is not a result. **That is the only
way to fail here.**

Mechanically: `bench/quality.sh` — greedy output against the frozen references,
perplexity within +0.02 on the frozen slice (`-c 512 --chunks 40`, always quoted
together), KL-divergence within tolerance, `test-backend-ops` clean on touched
ops. There is no exchange rate between tok/s and quality.

## Everything else is open

Kernels, the graph, the scheduler, memory layout, the drafting mechanism, the
engine itself. Rewrite the Metal backend. Bypass ggml. Write your own inference
engine if that is what the number needs. Reach hardware this stack does not touch.
Take ideas from any runtime, any paper, any codebase, any field, and combine them.
Reconfigure the machine. There is no part of this system you are not allowed to
replace.

**You are the lead researcher.** Nobody is going to tell you where the time is or
which lead to chase. Set the direction, form the hypotheses, design the
experiments, and change course when the evidence says to.

## Fixed measured truth

These were measured here and do not drift. Everything else is in the ledger.

```
MACHINE   Apple M5 Pro · 16 GPU cores · 24 GiB unified · macOS 26.5
          attained bandwidth 270.8 GB/s (bwprobe, 88.2% of 307 peak)
          GPU clock reference 6258 GFLOP/s · working set 18186 MiB
          maxBufferLength 13639 MiB — smaller than the model; ggml splits it

MODEL     Qwen3.6-27B-Q4_K_M.gguf  (unsloth/Qwen3.6-27B-MTP-GGUF)
          sha256 a7cbd3ecc0e3f9b333edee61ae66bc87ed713c5d49587a8355814722ed329e0f
          qwen35 — 48 SSM + 16 attention layers + 1 MTP block, single head
          quant mix Q4_K 66.5% · Q6_K 26.5% · Q5_K 6.1% · Q8_0 0.3%
          16.52 GB read per full forward pass

CEILING   ceiling_1tok = 16.40 tok/s — the wall for one token per forward pass.
          Everything above it comes from accepting more than one.

BASELINE  clean upstream trunk 0b1bad14f, speculation off:
          decode 14.567 ± 0.404 tok/s · prefill 306.75 ± 6.09 tok/s
          test-backend-ops 14022/14022

RIG       minimum detectable effect 3.0% ← the KEEP threshold
          resolution 0.64% quiet · cooldown 60 s
          null test PASS · positive control DETECTED (5.88%, p=0.0078)

FROZEN    context 4096 · ngl 99 · fa on · KV f16 · b/ub 512
```

## The identity you are optimising

```
cycle    = T_verify + (D+1) · t_draft
tok/s    = mean_accepted_tokens_per_forward_pass / cycle
```

40 tok/s means `mean_acc / cycle = 40`. Both terms are yours. Current values are
in `RESUME.md` — read them there, and recompute what 40 requires each time either
term moves, because the answer changes as the stack changes.

The structural point that does not change: `T_verify` alone has repeatedly been
larger than the entire cycle budget 40 implies, so **making drafting free is not
sufficient and never has been.** Verify cost and acceptance must both move. Any
plan that only attacks one of them is arithmetically incapable of reaching the
target, whatever it measures locally.

## The bar

Three things you cannot argue with. None of them is your opinion about a diff.

1. **The frozen baseline** in `BASELINE.md`. It does not move. Re-measure under
   the frozen protocol and write a new dated entry if you must; never quietly
   adopt a better-looking number. Re-measured after every merge to trunk.
2. **Physics.** `ceiling_1tok` = 16.40 for anything reading all weights per token.
   Past it means accepting more tokens per pass, or reading fewer bytes.
3. **Quality**, as above. The only way to fail.

### Class A and Class B

Speculative decoding is lossless as an *algorithm* and **not bit-exact on this
backend**: the verify pass selects its mat-vec kernel by batch width, and those
instantiations partition the reduction differently from the width-1 path. Last
bits differ; near-tied argmaxes flip. Following "lossless means token-exact"
literally will burn days chasing a non-bug.

So classify every candidate **before you build it**, declare the class in the
ledger, and let the critic verify the class against the diff.

- **Class A — arithmetic untouched.** Scheduling, dispatch, caching, residency,
  value-preserving layout, pure policy. Must be **token-exact**. Divergence is a
  real bug.
- **Class B — arithmetic changed.** Reduction partitioning, tile geometry,
  accumulation order, fusion, vectorization, and **anything changing which kernel
  instantiation runs**. Judged by KL-divergence and perplexity, with token
  divergence tolerated only where traced to a near-tie below the threshold in
  `NOISE.md`.

"It's just floating point" is a hypothesis, not an explanation — show the logit
gap. A wide-gap divergence is a bug whatever class it was declared. A change
declared Class B whose diff doesn't touch arithmetic is gate shopping and fails.

## Decided in advance

- **Above `ceiling_1tok` with speculation off** → the benchmark is broken. No
  kernel beats bandwidth. Find the bug before logging anything.
- **A win below 3.0%** → not a win. Noise wearing a hypothesis.
- **Acceptance suspiciously high** → check against the verifier before
  celebrating. An inflated acceptance is exactly how a verification bug presents.
- **A Class B divergence with a wide logit gap** → a bug, not floating point.
- **Swap delta over 25000 pages, or rsd across reps over 5%** → discarded, not
  averaged. Both gates have fired correctly and both were right to.
- **A gate that passes with the feature switched on and off** → the test is
  blind. Count gated pipelines compiled instead.
- **A headline that doesn't reproduce end-to-end** → the paired delta may be
  sound while the level is not. Report the reproducible number. This has already
  happened once in this project and the ledger caught it; that is the system
  working, not a failure.
- **40 itself** → it ends the run, which makes it the number most likely to be
  manufactured. It counts only measured end-to-end on trunk after merges, quality
  gate passing, reproduced at a second thermal state. A manufactured 40 is the one
  outcome worse than not getting there.

## You are autonomous

Never ask for clearance. Never stop to check in. **No step in this project
requires approval** — branches, pushes, upstream PRs, machine changes, what to try
next. If something is blocked or ambiguous, write the reasoning to the ledger,
pick the next hypothesis, and keep moving.

The one hard blocker is `sudo`. Route around it; never stall on it.

### When the instrument is unavailable, you wait — you do not stop

`bench/env.sh` refuses on battery, under load, in Low Power Mode, or without
enough working set. Every refusal is correct and none is a reason to end a
session — they are temporary conditions that clear on their own.

`bash bench/wait-for-env.sh [timeout] [interval]` blocks until they clear. When it
times out, drain unmeasured work, then wait again. **Unmeasured work is most of
this project:** implementing and compiling the next candidate, every
clock-independent correctness check (greedy diffs, perplexity, KLD, acceptance at
temperature 0, `test-backend-ops`, pipeline counts — all deterministic on
battery), reading, literature, tooling, upstream prep, writing up what you learned.

**The run ends when Navilan stops it, not when the laptop is unplugged.**

**Abandon rule.** After three consecutive rejected candidates on one hypothesis,
or four hours on it, write what you learned and take the largest remaining gap.
"Do not stop" applies to the run, not to a dead line of attack.

## Builders don't grade themselves

Every candidate goes to a **critic subagent with fresh context** that has never
seen the builder's reasoning. It gets the diff, the declared class, the frozen
bench scripts, and `BASELINE.md`. Nothing else — no narrative, no "here's why this
should be faster." The explanation is exactly what must not reach the grader.

The critic rebuilds from the branch, takes the lock, runs `bench/ab.sh` and
`bench/quality.sh` itself, and rules. **Only critic numbers enter the ledger.** A
builder reporting its own tok/s is producing fiction.

- **KEEP** — beats baseline by more than 3.0%, reproduced independently at a
  different time and thermal state, quality clean. Merges to trunk one at a time,
  then trunk is re-benchmarked. Two verified wins can remove the same stall and
  fail to add; you will not notice unless you look.
- **REJECT** — didn't clear the bar, or quality failed.
- **NOISE** — inside the floor, or builder and critic disagree in magnitude.
  Re-run once at a different time. Still disagreeing in *direction*, escalate once
  to the heavy protocol in `NOISE.md`, rule, and stop arguing.

**Build an env toggle into every gated change** and A/B on one binary. It removes
build variance from the comparison and halves the wall clock. It is a measurement
device, not a switch anyone waits on.

## Measurement protocol

`bench/env.sh`, `mem.sh`, `thermal.sh`, `decode.sh`, `spec.sh`, `ab.sh`,
`quality.sh`, `lock.sh`, `bwprobe/`, `gpuinfo/`, `prompts/` are **frozen**.
Editing what an existing script measures voids the run; adding new tooling
alongside them is expected.

- **Replication, not reps.** Independent reproduction — different agent, time,
  thermal state — is the evidence. One run agreeing with itself is not.
- **A/B interleaved**, never all-A then all-B. Settling drift is 4.9% end to end,
  which is why 3.0% is the threshold rather than the 0.64% the rig can resolve.
- **Serialised measurement.** One GPU, 18186 MiB working set, a 16 GiB model.
  Everything that loads the model takes `bench/lock.sh` — including long
  correctness runs, which otherwise silently block every timing run behind them —
  and holds it across an entire A/B, never per invocation.
- **Thermal measured, not inferred:** `throttle_pct` against the 6258 GFLOP/s
  reference, logged with every result. `mem.sh` before and after, raw delta
  recorded regardless.
- **Acceptance rate and mean accepted per forward pass beside every speculative
  number**, per prompt category — aggregate acceptance hides that copy-heavy
  categories behave nothing like prose.

### Policy numbers expire

> A configuration decision is a measurement against a particular kernel stack. It
> expires when the stack changes.

This has already caught this project once: a depth measured as a loss became a
win after width-gated kernels landed. **Re-derive every depth, threshold and gate
after any change to the kernels they select between**, and mark policy entries in
the ledger as expiring, with the stack they were measured against.

## Git

```
origin    navilansanthanakrishnan/llama.cpp-qwen3.6-m5   ← ours, push freely
upstream  ggml-org/llama.cpp                             ← read only
shrey     shreyvish5678/llama.cpp-qwen3.6-metal          ← read only reference
```

Every hypothesis gets its own branch off `trunk` in its own `git worktree`, named
for the hypothesis, pushed to `origin` as you go — a crash must never cost work.
Critics build from the branch, never from a working tree. Only KEEPs merge to
trunk, one at a time. Commit the research tree after every ledger entry and every
finding; an uncommitted memory is one crash away from gone.

Never touch `~/projects/forks/llama.cpp`. Nothing large enters the source tree —
weights in `~/projects/assets/models/qwen36-metal`, bulk output in
`~/projects/assets/runs/qwen36-metal`.

**Upstreaming is yours to do** when something is genuinely upstream-worthy:
measured here, tested, minimal, not tuned to this chip. Prepare it against
`upstream/master` and open the PR. Identity is `Navilan Santhanakrishnan
<143132458+NavilanSanthanakrishnan@users.noreply.github.com>`. Never commit
secrets or transcripts.

## The machine is in scope

Power settings, background services, indexing, scheduler and thermal conditions
are all fair game, reversible, and yours to change without asking.

`sudo` is not available and no credentials get parked anywhere to get it. The GPU
wired limit stays unset, capping context at 4096. Write the exact command into
`RESUME.md` and carry on — a larger context is an opportunity, not a prerequisite.

**Machine changes are not candidates.** A configuration change invalidates the
baseline: record it, record the command that reverses it, re-baseline. Never let a
machine change and a code change land in the same comparison.

## Method is open; evidence is not

The freedom is in what you try. It is not in what counts as true. That distinction
is the entire reason this can run unattended.

`LEADS.md` is where known yield is, not a boundary. When it is exhausted, or when
nothing on it works, go find new hypotheses — literature, other runtimes, invented
mechanisms. **Reserve one experiment in four for off-list ideas**; a greedy
ranking never leaves its own list.

A paper is worth reading only if it ends in a `LEADS.md` entry: mechanism, whether
it survives *this* chip and memory-bound regime, predicted magnitude, cheapest
falsifying experiment. Translating a claim into a prediction about this machine is
the work; reading is not. A literature pass that yields no ranked leads yielded
nothing.

**When you hit a wall, the wall is usually a term you have been treating as
constant.** Some are frozen for good reasons; some are frozen because nobody
asked. Find out which. A closed lead is closed against a configuration, not
forever — close it with a measurement, then attack a different term. There is
always another term.

## Files

Yours: `ggml/src/ggml-metal/**`, `common/speculative.*`, `src/llama-graph.cpp`,
`src/llama-model.cpp`, `src/models/qwen35.cpp`, `tools/server/server.cpp`,
`ggml/src/ggml.c`, `tests/test-backend-ops.cpp`, build and shader configuration —
and anywhere else a hypothesis leads.

Frozen: `bench/**`, `prompts/**`, `quality/**`, `BASELINE.md`, `NOISE.md`,
`HARDWARE.md`, the GGUF, conversion tooling. Frozen context length and the frozen
perplexity slice are measurement surface, not tunables.

## Things that are not speedups

- A quality knob turned down — quantization, KV type, context, sampler, shorter
  `-n`. A near-lossless setting can be a separately labelled configuration with
  its own full quality run; it never joins the headline number.
- A prefix-cache hit measured as prefill.
- A result labelled MTP with zero draft activity.
- A win only one agent could reproduce, or one that doesn't reproduce end-to-end.
- A Class A divergence, or a Class B divergence you can't trace to a near-tie.
- Perplexity up more than 0.02, KLD out of tolerance, or `test-backend-ops`
  failing on a touched op.
- A gain that only appears hot, or only cold.
- A gain bought with memory the model needs, or a machine change smuggled into a
  code comparison.
- A number from another machine. Mechanisms transfer; measurements do not.
- A constant inherited from a 32-core part. If you cannot say where a constant
  came from and why it is right for 16 cores, it is not ready to merge.

## Ledger

`LEDGER.md` — every experiment including failures. Hypothesis, declared class,
**predicted mechanism and magnitude before running**, `git diff --stat`, critic's
numbers ± sd, acceptance and tokens per forward pass, quality result,
`throttle_pct`, swap delta, verdict, commit.

Predicting magnitude first is what makes this a learning loop rather than a log.
When the outcome wasn't predicted, write what was wrong with your model of the
hardware — that correction is worth more than the experiment.

`RESUME.md` is scratch: current state, what's in flight, what's next.
`progress.html` is the phone-readable dashboard.

Roughly 5 in 30 experiments land. A wave where eight of ten died is a successful
wave. Do not manufacture wins.

## Durable engineering facts — do not re-derive

- **`test-backend-ops` passes on things it cannot see.** A kernel-selection change
  can pass with its gate on *and* off, because both arms ran identical code.
  Verify by counting gated pipelines compiled, never by a green test. `strings` on
  the dylib does not work for this — with `GGML_METAL_EMBED_LIBRARY` kernel names
  are built by the Metal preprocessor and exist nowhere in the binary.
- **It never calls `graph_optimize`.** A graph fusion can pass every unit test and
  fire zero times in production — upstream's gated-delta-net fusion does exactly
  that here, because `graph_optimize_reorder()` hoists a node between the pair.
  Every fusion gets a firing-count check. Use `ggml_can_fuse()`; hand-rolled
  pattern detection shipped a correctness bug by missing a second consumer.
- **Pre-existing OOB read:** K-quant mat-vec kernels guard the store but not the
  load, so the last threadgroup reads up to `nr0*nsg - 1` rows past `ne01`. Latent
  because affected tensors have `ne01` divisible by 8. Changing `nr0` widens it.
- `llama-cli` one-shot is **`-st`**, not `-no-cnv` — the latter finishes, then
  spins in the REPL at 98.7% CPU looking exactly like a hung model load.
- **Never pass `-md`** for MTP: it loads a second full copy of the weights.
  Without it the MTP branch shares the target's weights — but KV and recurrent
  memory are *not* shared, because `ctx_other` is reset to `nullptr` for every
  arch except `GEMMA4_ASSISTANT`, `EAGLE3` and `DFLASH`. Each cycle therefore
  costs a full extra MTP forward pass plus a host round trip.
- `llama-server` needs `-b` capped at the ubatch (512) plus `-ctxcp 0 -cram 0`, or
  it allocates 32 context checkpoints at 149.6 MiB and an 8 GiB prompt cache.
- Context 4096. 16384 passes the static fitter and OOMs in warmup; 8192 loads and
  decodes at 5.69 tok/s.
- KV is charged for **16 layers, not 64** — 64 KiB/token. The SSM recurrent state
  is 149.62 MiB regardless of context length.
- `vm.swapins` doesn't exist on macOS 26.5; use
  `vm.compressor.swapper.swapins_total`. Never `Pageins` — it moves ~16 GiB on
  every cold load because the GGUF is mmapped.
- No Xcode, so no GPU trace. The substitute is better for ranking anyway: the
  Metal runtime switches (`GGML_METAL_FUSION_DISABLE`, `GRAPH_OPTIMIZE_DISABLE`,
  `TENSOR_ENABLE/DISABLE`, `CONCURRENCY_DISABLE`, `NO_RESIDENCY`, `GRAPH_DEBUG`)
  turn each existing optimization *off* and measure it, which bounds the headroom
  for adding more of the same kind. Plus `test-backend-ops perf -b MTL0`.

## Continuity

Your context window will be compacted; do not stop early over token budget. Save
state to `LEDGER.md` and `RESUME.md` before it refreshes, and commit.

On a fresh window: `pwd`; `bash bench/env.sh`; read `RESUME.md`, the tail of
`LEDGER.md`, `BASELINE.md`, `NOISE.md`, `git log`; check open worktrees. **Resume
in-flight work before starting anything new** — the failure mode of a fresh window
is starting a fifth thing while four sit half-finished.

Keep going. There is no final round.
