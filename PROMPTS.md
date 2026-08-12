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
