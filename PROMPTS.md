# Session prompts

Paste one of these into a fresh session. The specification is `GOAL.md`; these
just set disposition and the first move.

---

## Kickoff — the first session

```
Read GOAL.md first, then the handoff at the top of SETUP-LOG.md, then LEADS.md,
BASELINE.md, NOISE.md and RESUME.md. GOAL.md is the specification — follow it.

The goal: maximum decode tok/s for Qwen3.6-27B-Q4_K_M on this M5 Pro at unchanged
quality. Baseline is 14.567 tok/s; the bandwidth ceiling for one token per forward
pass is 16.40 and we are already at 88.8% of it. Plain decode is closed. Acceptance
rate is the only unbounded lever, and byte deletion is the only way to move the
ceiling itself.

Work autonomously. Never ask for clearance, never stop to check in, never wait.
Nothing in this project needs my approval — branching, pushing to our fork,
opening upstream PRs, changing the machine, deciding what to try next. If you are
blocked or something is ambiguous, write your reasoning to LEDGER.md, pick the
next hypothesis, and keep moving. I read the ledger and stop the run myself when
I want to. `sudo` is the one thing you cannot do: route around it, never stall on
it.

First, close the three open gates in SETUP-LOG.md — do not optimize anything on an
unproven rig. Then derive the floating-point noise scale, add the measurement
lock, and write the test coverage. Then implement Shrey's six mechanisms yourself,
derived for 16 cores; his repo is reference only and never becomes a branch. Then
work LEADS.md top-down.

Builders don't grade themselves. Every candidate goes to a fresh-context critic
that gets the diff and nothing else — no narrative. Only critic numbers enter the
ledger. Predict the mechanism and the magnitude before you run, and when the
outcome surprises you, write down what was wrong with your model of the hardware.

Read papers. Diff against CUDA and Vulkan. Invent mechanisms that exist in no
paper and no backend. One experiment in four goes to something off the list.
Treat apparent impossibility as a hypothesis to test rather than a conclusion —
the bandwidth ceiling is real, but everything above it is speculation and
everything below it is engineering. Ambition belongs in what you attempt; it never
reaches the ledger as a number that wasn't measured.

Commit and push constantly. Keep LEDGER.md, RESUME.md and progress.html current.

Keep going. There is no final round.
```

---

## Push to 30 — the standing continuation prompt

```
Target: 30 tok/s decode on Qwen3.6-27B-Q4_K_M, at unchanged quality. Baseline is
14.567. That is a 2.06x speedup and it is the whole job now.

Know what 30 requires before you start. ceiling_1tok is 16.40 tok/s — the wall for
one token per forward pass — so 30 tok/s means an effective multiplier of 1.83
after verify cost is paid, not before. Shrey realised 1.68x on twice this GPU
(18.6 -> 31.3) from 2.94 accepted tokens per forward pass, so the token multiplier
is only ~57% realised once verification is charged for. 30 tok/s therefore needs
one of: higher acceptance than he got, a cheaper verify pass than he got, or bytes
deleted from the per-token stream so the wall itself moves. Most likely all three.
Say in the ledger which one a candidate is buying, every time.

Start with what is already on the table. Do not go looking for new ideas while
finished work sits unverified:

1. Regenerate runs/kld/trunk-base.dat — it is 80K and looks truncated from a
   killed process. No KLD verdict is trustworthy until it is rebuilt.
2. Close the three open gates in SETUP-LOG.md.
3. Measure the pending ledger rows: 002 (test coverage), 003 (S4 scale division),
   004 (K-quant mul_mv_ext gate at ne11 2-3, your own finding and the most
   interesting thing in the ledger).
4. Finish s0-kquant-multicol — it is committed WIP and does not compile.
5. Then LEADS.md top-down: L1 depth staircase, L2 ngram drafting on copy-heavy
   categories, L10, L6 (the MTP draft step re-reading the 1.04 GB LM head), L8.

Then run the real program, and treat it as research rather than a task list:

- Find the actual bottleneck. Not the assumed one. Instrument it — per-op
  throughput, gated pipeline counts, graph node counts, the GGML_METAL_*_DISABLE
  switches turned off one at a time to price each existing optimisation. The
  ~94%-of-ceiling figure that counts SSM recurrent-state traffic is still
  unverified and it gates how much decode headroom exists at all. Settle it.
- Understand the machine, not just the code. This is a 16-core M5, not a 32-core
  M4. Establish whether this part exposes matrix or neural acceleration Metal can
  reach — if it does, every prefill and verify-width conclusion inherited from an
  M4 has a different ceiling here.
- Read the literature. Speculative decoding is the unbounded lever, so that is
  where papers pay: tree and multi-branch drafting, adaptive depth by context,
  draft-vocabulary trimming, n-gram and prompt-lookup hybrids, verification
  batching. Every paper exits as a ranked LEADS.md entry with a mechanism, a
  predicted magnitude for this chip, and a falsifying experiment — or it produced
  nothing.
- Try things nobody has published. You have a calibrated rig, a quality oracle
  and an honest ledger; that is exactly the apparatus that makes a wild idea cheap
  to test and safe to be wrong about. One experiment in four goes off-list.
  Impossibility is a hypothesis to test, not a conclusion to accept — but it gets
  tested, never asserted.

Non-negotiable, because these fail silently: builders never grade themselves —
fresh-context critic, gets the diff and no narrative, and only critic numbers
enter the ledger. Predict mechanism and magnitude before running. Report
acceptance and tokens-per-forward next to every speculative number. No constant
inherited from a 32-core part without being re-derived for 16.

Never stop. bash bench/wait-for-env.sh blocks until the machine is measurable
instead of failing; when it times out, drain the unmeasured queue — implement the
next candidate, run the clock-independent correctness checks, read, diff against
shrey/shipped, prepare upstream branches, re-rank leads — then wait again. The run
ends when I stop it.

And do not narrate intent. If you write "next I will X", do X in the same turn.
An announcement is not an action, a plan to continue is not continuing, and a
session that ends on "I'll keep going" has stopped. End your turns having done
the thing, not having described it.

Commit and push constantly. Keep LEDGER.md, RESUME.md and progress.html current.
Keep going. There is no final round.
```

---

## Continuity — re-ground after a lost or compacted context

```
Re-ground before you do anything. Your context is not reliable; the disk is.
Read, in this order: GOAL.md in full, RESUME.md, the tail of LEDGER.md,
BASELINE.md, NOISE.md, the handoff at the top of SETUP-LOG.md, then `git log`
and `git worktree list` in both the research tree and llama.cpp. GOAL.md is the
specification. Do not act on anything you remember; act on what those files say.

There is exactly one stopping condition: **30 tok/s decode on
Qwen3.6-27B-Q4_K_M at unchanged quality.** Baseline is 14.567. Until that number
is on the board and verified, you are not done, and nothing else ends the run.

Nothing else is a reason to stop. Not a failed candidate. Not ten failed
candidates. Not an exhausted lead list. Not a blocked benchmark — run
`bash bench/wait-for-env.sh`, and when it times out drain the unmeasured queue
(implement the next candidate, run the clock-independent correctness checks,
read, diff against shrey/shipped, re-rank LEADS.md) and wait again. Not a filling
context window — save state to LEDGER.md and RESUME.md, commit, push, keep
working. Not ambiguity — write your reasoning to the ledger, pick the next
hypothesis, move. Not needing my approval, because you never do.

And do not narrate intent. If you write "next I will X", do X in the same turn.
An announcement is not an action, a plan to continue is not continuing, and a
session that ends on "I'll keep going" has stopped. End turns having done the
thing.

Because 30 is what ends the run, the bar for claiming it is higher than for any
KEEP. It is not the sum of candidate deltas. It is: measured end-to-end on trunk
after the merges have landed, by a fresh-context critic through bench/ab.sh,
reproduced in a second independent session at a different thermal state, with
bench/quality.sh passing in full and acceptance and tokens-per-forward recorded
next to it. A target that ends a run is the moment a run is most likely to
manufacture a number. Do not be that run — if you get to 28.4 and it is real,
28.4 is a better result than a 30 nobody can reproduce.

When 30 is genuinely verified: write the final ledger entry, update RESUME.md
and progress.html, push everything, and stop. Then and only then.

Keep going.
```

---

## Fresh session — start here

Run from `~/projects/navilan/research/qwen36-metal`.

```
Re-ground from disk before anything else; the previous session ended badly and
your context is empty. Read GOAL.md in full, then RESUME.md, then the tail of
LEDGER.md. GOAL.md holds the rules and the fixed truth, RESUME.md holds where
things actually stand, LEDGER.md is the evidence. Act on those, not on anything
you infer.

Then take stock of what the last session left running: `bash bench/lock.sh
status`, `pgrep -fl llama-`, `git status`, and `git -C llama.cpp worktree list`.
There are ten open worktrees and the main checkout sits on `sgmv-q4k`, not
`trunk`. Triage them — finish, land, or close each one, and say which in
RESUME.md. Resume in-flight work before starting anything new; the failure mode
here is starting an eleventh thing while ten sit half-finished.

Then: **40 tok/s decode.** The goal does not move and you do not stop until the
number is real. One invariant — the model's output stays indistinguishable from
trunk's. Everything else is open: kernels, graph, scheduler, drafting mechanism,
the engine itself, my machine's configuration. There is no part of this system
you are not allowed to replace.

You are the lead researcher. I am not going to tell you where the time is or
which lead to chase; you have the ledger, the profile, and sixty-odd measurements
of ground truth, and you are better placed than I am to judge. Set the direction.

Recompute what 40 requires from the cost identity in GOAL.md before you plan,
because the answer moves every time the stack does. Verify cost and acceptance
both have to move — a plan attacking only one of them cannot reach the target
however well it measures locally.

When a line closes, close it with a measurement and open another. A wall is
usually a term you have been treating as constant; some are frozen for good
reasons and some because nobody asked, and telling those apart is the research.
Go wide for ideas — papers, other runtimes, other fields — and combine them; most
of what is left will come from two known ideas meeting rather than one new one
arriving. Invent what nobody has published. You have a rig that falsifies an idea
in an hour and a quality oracle that stops a wrong one from being believed, which
is the argument for trying things that would be reckless without it.

The discipline is unchanged and it is what makes this safe to run unattended:
predict before you measure, a fresh-context critic measures every candidate and
only its numbers count, failures go in the ledger. 40 ends the run, so it is the
number most likely to be manufactured — end-to-end on trunk, quality clean,
reproduced. A manufactured 40 is the one outcome worse than not getting there.

When the machine is busy, `bash bench/wait-for-env.sh` and then drain unmeasured
work. Never stop. Commit and push constantly.
```

---

## /goal — 40 (current)

```
Read GOAL.md, then RESUME.md and the tail of LEDGER.md. GOAL.md holds the rules
and the fixed truth; RESUME.md holds where things actually stand. Act on those,
not on memory.

40 tok/s decode. The goal does not move and you do not stop until the number is
real. One invariant: the model's output stays indistinguishable from trunk's.
Everything else — kernels, graph, scheduler, drafting mechanism, the engine
itself, my machine's configuration — is open. There is no part of this system you
are not allowed to replace.

You are the lead researcher. I am not going to tell you where the time is or
which lead to chase; you have the ledger, the profile, and sixty-odd measurements
of ground truth, and you are better placed than I am to judge. Set the direction.

Recompute what 40 requires from the identity in GOAL.md before you plan, because
the answer moves every time the stack does. Verify cost and acceptance both have
to move — a plan that attacks only one of them cannot reach the target however
well it measures locally.

When a line closes, close it with a measurement and open another. A wall is
usually a term you have been treating as constant; some are frozen for good
reasons and some because nobody asked, and telling those apart is the research.
Go wide for ideas — papers, other runtimes, other fields — and combine them; most
of what is left will come from two known ideas meeting rather than one new one
arriving. Invent what nobody has published. You have a rig that falsifies an idea
in an hour and a quality oracle that stops a wrong one from being believed, which
is the argument for trying things that would be reckless without it.

The discipline is unchanged and it is what makes this safe to run unattended:
predict before you measure, a fresh-context critic measures every candidate and
only its numbers count, failures go in the ledger. 40 ends the run, so it is the
number most likely to be manufactured — end-to-end on trunk, quality clean,
reproduced. A manufactured 40 is the one outcome worse than not getting there.

Keep going.
```

---

## /goal — 40, no matter what

```
40 tok/s decode. The goal does not move, and you do not stop until the number is
real.

**One invariant, and it is the only one: the model's output must stay
indistinguishable from trunk's.** Same weights, same quantization, same context
length, same sampling semantics, the same distribution over tokens for any
prompt. No smaller context, no lighter quant, no lossy KV compression, no
shortened generation, no warm-cache measurement, no benchmark-specific
special-casing — nothing that is fast because it is quietly doing less of the
work. A change that makes the model cheaper by making it different is not a
result. That is the only way to fail here.

**Everything else is open.** Kernels, the graph, the scheduler, memory layout,
the drafting mechanism, the engine itself. Rewrite the Metal backend. Bypass ggml.
Write your own inference engine if that is what the number needs. Reach hardware
this stack does not currently touch. Take ideas from any runtime, any paper, any
codebase, any field, and combine them. Reconfigure my Mac. There is no part of
this system you are not allowed to replace.

**You are the lead researcher.** I am not going to tell you where the time is or
which lead to chase. You have the profile, the ledger, and sixty-odd measurements
of ground truth, and you are better placed than I am to judge what is worth
trying. Set the direction, form the hypotheses, design the experiments, and change
course when the evidence says to.

40 may be unreachable by any route currently in the ledger. That is a statement
about what has been tried, not about what is possible — this project has already
reversed one such conclusion and gained 6% doing it. When a line closes, close it
with a measurement and open another. There is always another term in the equation.

The discipline is unchanged, and it is what makes "no matter what" ambitious
rather than reckless: predict before you measure, a fresh-context critic measures
every candidate and only its numbers count, failures go in the ledger. 40 is what
ends the run, which makes it the number most likely to be manufactured — it counts
end-to-end on trunk after merges, quality gate passing, reproduced at a second
thermal state. A manufactured 40 is the one outcome worse than not getting there.

Keep going.
```

---

## /goal — decode to 40

```
Target is 40 tok/s decode. Everything in GOAL.md still stands.

Know what 40 costs before you plan it. At today's 3.70 accepted tokens per
forward pass it needs a 92 ms cycle, and verify alone is 110 ms — so 40 is
unreachable by making drafting free. It is the same wall ledger 052 found at 35,
one step further out. Two terms can move it:

- **The 71.1 ms width-1 verify floor.** Width 5 to 8 now costs 3.2 ms, so the
  matmuls are finished. What remains is the gated-delta-net recurrence across 48
  layers, attention and norms — untouched, and now the largest single item in the
  cycle.
- **Acceptance.** At the current ~137 ms cycle, 40 tok/s means 5.48 accepted
  tokens per forward pass instead of 3.70. LEDGER 092 sharpens this: the binding
  form is sustained per-step acceptance >= 0.84, and T_ver is itself a floor.

Neither alone is obviously enough. Together they are. That is the shape of the
problem; the route is yours to find.

**When you hit a wall, the wall is usually a term you have been treating as
constant.** List what you have been holding fixed and work out which of them is
actually load-bearing: a single draft head, MTP as the only drafting mechanism,
one draft per step, linear rather than tree drafting, greedy verification, the
draft coming from this model at all, batch 1, the layer split, what the recurrence
recomputes every step. Some are frozen for good reasons. Some are frozen because
nobody asked. Find out which is which — that is the research, and it is where the
next order of magnitude lives rather than in another kernel.

Go wide for ideas, and be specific about where you look. Papers on speculative and
parallel decoding, tree attention, draft distillation, state-space inference.
Other runtimes — MLX, vLLM, SGLang, TensorRT-LLM, tinygrad, ik_llama.cpp — for
things nobody has ported to Metal. Apple's own frameworks for hardware ggml does
not reach. Then combine them: most of the real gains here will come from two known
ideas meeting for the first time, not from one new idea arriving. Identify the
studies worth running, and run them.

Invent. You have a rig that can falsify an idea in an hour and a quality oracle
that stops a wrong one from being believed. That apparatus is exactly what makes
an unreasonable idea cheap to test — which is the argument for trying things that
would be reckless without it, not the argument for caution.

Do not stop at a wall, reframe it. A closed lead is closed against a
configuration, not forever; this project has already reopened one and gained 6%.
If a line is genuinely exhausted, write the measurement that closed it in the
ledger and go attack a different term in the equation. There is always another
term.

Everything still enters the ledger the same way — prediction before measurement,
critic numbers only, failures logged. 40 is what ends the run, which makes it the
number most likely to be manufactured: end-to-end on trunk after merges, quality
clean, reproduced.
```

---

## Decode to 30 — lead-researcher continuation

```
Read GOAL.md, then RESUME.md, the tail of LEDGER.md, LEADS.md, BASELINE.md and
NOISE.md. Act on what those files say, not on what you remember.

Decode only this run. You are at ~27 tok/s. **You stop at 40 and nothing else
ends this run.**

The cycle is ~137 ms at 3.70 accepted tokens per forward pass — verify 110,
drafting 27. 40 tok/s needs 92 ms. So ~45 ms comes out, or acceptance rises,
or both. Everything inside that cycle is yours: acceptance rate, draft depth and
stopping policy, the 9.2 ms draft step, the 1 GB output-head read at width 1, the
host round trip qwen35 pays for not being on the `ctx_other` sharing list, and the
width-8 verify matmul that is two thirds of the cycle.

**You are the lead researcher here, not a task runner.** Set the direction.
Decide what is worth knowing and then find out: profile before you theorise, read
the literature on speculative decoding and drafting policy, port what other
backends already prove, and invent what nobody has published. Ledger 052 says 35
was unreachable *at that acceptance* — a statement about one configuration, not a
law. This project has already reversed exactly that kind of conclusion once, when
depth 3 went from a closed loss to a +6% win after the kernels changed. Treat
impossibility as a hypothesis and test it.

**The freedom is in what you try. It is not in what counts as true.** That
distinction is the whole reason you can be left running unattended:

- Predict the mechanism and the magnitude in LEDGER.md *before* you measure.
- A fresh-context critic measures every candidate: it gets the diff and no
  narrative, and only its numbers enter the ledger. Subagent depth is capped at 1
  here, so delegated work cannot spawn its own critic — you spawn it.
- Log the failures. Roughly five in thirty land; a wave where eight of ten died
  is a successful wave. A negative result written honestly is worth more than a
  positive one nobody can reproduce.
- 30 ends the run, which makes it the number most at risk of being manufactured.
  It counts only when measured end-to-end on trunk after merges, quality gate
  passing, reproduced at a second thermal state. A real 28.4 beats a 30 nobody
  can reproduce.

One GPU: build in parallel, measure one at a time through `bench/lock.sh`, and
anything that loads the model takes the lock. When the machine is busy, run
`bash bench/wait-for-env.sh`; when it times out, drain unmeasured work and wait
again. Never stop.

Do not narrate intent. If you write "next I will X", do X in the same turn. A
session that ends on "I'll keep going" has stopped.

Commit and push constantly. Keep LEDGER.md, RESUME.md and progress.html current.
Keep going.
```

---

## Two-track push to 30

```
Read GOAL.md, then RESUME.md, the tail of LEDGER.md, LEADS.md, BASELINE.md and
NOISE.md. Act on what those files say, not on what you remember.

You are at ~27 tok/s. **You stop at 40 tok/s decode and nothing else ends this
run.** Verified means measured end-to-end on trunk after merges, by a
fresh-context critic, quality gate passing, reproduced at a second thermal state.

The arithmetic you are working against: cycle ~137 ms — verify 110, drafting
27 — at 3.70 accepted tokens per forward pass. 40 tok/s needs a 92 ms cycle.
So roughly 33 ms comes out, or acceptance goes up. Both are open.

Run two persistent tracks as subagents:

- **decode** — acceptance rate, the 9.2 ms draft step, the 1 GB output-head read
  at width 1, the host round trip qwen35 pays for not being on the `ctx_other`
  sharing list, draft depth and stopping policy.
- **prefill** — the verify pass is prefill-shaped, so every millisecond off wide
  mat-mul is a millisecond off the 110. Tiles, fusion, the matrix-unit path,
  graph and dispatch overhead.

You orchestrate. Subagent depth is capped at 1 on this machine, so the track
agents cannot spawn their own critics — **you** spawn a fresh-context critic per
candidate, hand it the diff and nothing else, and only its numbers reach the
ledger.

One GPU. The tracks build in parallel and measure one at a time through
`bench/lock.sh`; anything that loads the model takes the lock. When the machine
is busy run `bash bench/wait-for-env.sh`, and when it times out drain unmeasured
work — implement the next candidate, run the clock-independent correctness
checks, read, diff against `shrey/shipped` — then wait again. Never stop.

Read the literature and then test it: speculative decoding variants, drafting
policy, wide-batch GEMM on Apple silicon. Every paper exits as a ranked LEADS.md
entry with a mechanism, a predicted magnitude for this chip, and a falsifying
experiment — then gets run.

Take initiative and do not accept impossibility. Ledger 052 says 35 was
unreachable *at that acceptance* — a statement about one configuration, not a
law. This project has already overturned exactly that kind of conclusion once:
depth 3 was measured as a loss and closed, and became a +6% win after the
width-gated kernels landed. You have a calibrated rig, a quality oracle and 61
entries of ground truth, which is the apparatus that makes an unreasonable idea
cheap to test and safe to be wrong about. Test impossibility; never assert it.

Do not narrate intent. If you write "next I will X", do X in the same turn. A
session that ends on "I'll keep going" has stopped.

Commit and push constantly. Keep LEDGER.md, RESUME.md and progress.html current.
Keep going.
```

---

## Resume — every session after

```
Continue autonomous work per GOAL.md, RESUME.md, LEDGER.md and LEADS.md.
Target: decode tok/s on Qwen3.6-27B-Q4_K_M at unchanged quality.

First: pwd, run bash bench/env.sh, read RESUME.md, the tail of LEDGER.md,
BASELINE.md and NOISE.md, check git log and open worktrees. Resume in-flight work
before starting anything new — most of the good ideas are already written down.

Still binding: only critic numbers enter the ledger; bench/, prompts/, quality/,
BASELINE.md, NOISE.md and the frozen context are frozen; declare the change class
before you build; report acceptance and tokens-per-forward next to every
speculative number; any run over 25000 swap pages or 5% rsd is discarded; hold the
measurement lock across a whole A/B. No constant inherited from a 32-core part
without being re-derived for 16.

Work autonomously — nothing here needs my approval. One experiment in four goes
off-list: papers, other runtimes, or something nobody has published.

Commit and push. Keep LEDGER.md, RESUME.md and progress.html current. Keep going.
```
