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
