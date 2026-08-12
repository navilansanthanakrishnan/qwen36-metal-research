# NOISE — what this rig can and cannot see

This file is why the setup exists. A benchmark that has not been calibrated
against its own noise is not evidence, it is a number generator.

**The KEEP threshold for every later experiment comes from here.**

---

## Headline

| | |
|---|---|
| **noise floor** (spread of run-to-run means, n=10) | **7.21%** raw, **2.78%** as relative sd |
| **paired spread** (what the A/B procedure actually sees) | **3.14%** |
| **resolution measured under quiet conditions** | **0.64%** (null test, 8 pairs) |
| **minimum detectable effect (frozen KEEP threshold)** | **3.0%** |
| **required cooldown** | 60 s |
| null test | **PASS** — NO DIFFERENCE between identical binaries (+0.542%, CI [−0.103, +1.186]) |
| positive control | **DETECTED** — SLOWER by 5.88%, p=0.0078, CI [−10.1, −1.6] |

---

## 1. Noise floor

Ten runs of `bench/decode.sh` against unmodified trunk, spread over 32 minutes
(21:27:45Z–21:59Z), each gated by `bench/env.sh` and separated by a 90 s cooldown
plus 60 s of spacing.

```
nf-1  13.998    nf-6  14.686
nf-2  14.080    nf-7  14.701
nf-3  14.641    nf-8  14.777
nf-4  13.945    nf-9  14.994
nf-5  14.882    nf-10 14.967

mean 14.5671 tok/s   sd 0.4044   rsd 2.78%   spread (max-min)/mean = 7.21%
prefill: mean 306.75 tok/s   sd 6.09   rsd 1.99%
```

**The spread is not random — it is a monotonic upward drift, and identifying
that is the single most useful thing in this file.** The first four runs average
14.17 tok/s and the last four average 14.86. Prefill rises in lockstep
(299.89 → 312.76). The covariates say why:

| run | tg | pp | throttle% | free% | swapin |
|---|---|---|---|---|---|
| nf-1 | 13.998 | 299.89 | −0.2 | 83 | 5786 |
| nf-4 | 13.945 | 296.61 | 3.9 | 80 | 862 |
| nf-7 | 14.701 | 312.11 | 0.0 | 85 | 371 |
| nf-10 | 14.967 | 312.76 | 0.1 | 84 | 410 |

Background swap-in traffic falls from 5786 pages to ~400 as other work on the
machine finishes, and throughput rises with it. **This is the machine settling,
not thermal drift** — the GPU clock probe reads within ±0.2% of its cold
reference for nine of the ten runs.

That distinction matters for the protocol. Thermal drift would push results
*down* over a session; settling pushes them *up*. Either way a naive "run all of
A, then all of B" comparison would attribute the entire 4.9% session trend to
whichever variant ran second. **That is exactly the fake win this rig exists to
prevent, and it is worth more than the 4% the positive control is calibrated
against.**

### What the pairing actually sees

The relevant statistic is not the raw sd but the spread between *adjacent* runs,
because `bench/ab.sh` runs A and B next to each other:

```
successive differences: 0.082, 0.560, -0.696, 0.938, -0.196, 0.015, 0.076, 0.218, -0.027
sd of successive differences = 0.4574  =  3.14% of the mean
```

That 0.4574 is below `sqrt(2) × 0.4044 = 0.572`, the value it would take if the
series were pure independent noise — confirming positive autocorrelation, i.e.
real drift that pairing removes.

**So the honest noise number for a paired comparison is 3.14%, not 7.21%.**

---

## 2. Minimum detectable effect

Predicted from the noise floor's paired spread (3.14%), the 95% CI half-width on
the mean paired difference is `t(n-1) × 3.14% / sqrt(n)`. This is the pessimistic
estimate — it assumes every A/B batch is as unsettled as the noise-floor session:

| pairs | t(0.975) | CI half-width | can resolve 4%? |
|---|---|---|---|
| 4 | 3.182 | 5.00% | no |
| 5 | 2.776 | 3.90% | marginal |
| 6 | 2.571 | 3.30% | yes |
| **8** | **2.365** | **2.63%** | **yes** |
| 10 | 2.262 | 2.25% | yes |

**A second, independent constraint forced the pair count up, and it is a real
flaw that was nearly shipped.** `bench/ab.sh` uses an exact sign-flip
permutation test, whose smallest achievable p-value is `2/2^n`. At 4 pairs that
is **0.125** and at 5 pairs **0.0625** — *neither can ever reach significance at
the 5% level, no matter how large the true effect*. The original default of 5
pairs would have made the procedure structurally incapable of confirming
anything. **The default is now 8** (`2/2^8 = 0.0078`), and `bench/ab.sh` carries
that reasoning in a comment so nobody lowers it to "save time".

### The measured answer, and why the frozen threshold is larger

The null test (below) ran 8 pairs on a settled machine and produced a **paired sd
of 0.770%**, far tighter than the 3.14% predicted from the noise floor. At n = 8
that is a 95% CI half-width of

```
2.365 x 0.770% / sqrt(8) = 0.64%
```

So the procedure can *resolve* a 0.64% difference when the machine is quiet and
settled. That is the instrument's precision.

It is **not** the number that goes in LEDGER.md, because the noise floor showed
that a session which includes the machine settling drifts by 4.9% end to end.
A threshold set at the instrument's best-case precision would certify that drift
as a win the first time a run straddled an unsettled period.

```
MINIMUM DETECTABLE EFFECT = 3.0%   (frozen KEEP threshold, QM_MDE_PCT)
measured resolution       = 0.64%  (8 pairs, quiet machine, null test)
```

**3.0% is the KEEP threshold.** It sits above the 3.14% paired spread observed
across the unsettled noise-floor session, so a drift of that kind cannot clear
it, and comfortably below the 4% the positive control demonstrates the rig can
catch. Anything under 3.0% is `NOISE` in LEDGER.md regardless of how good the
mean looks or how tight that particular batch's CI happened to be.

---

## 3. Null test — negative control

**Result: PASS.** Identical binary, identical environment, labelled A and B, run
through the full A/B procedure interleaved.

```
ab: 8 pairs, ABBA interleaved, cooldown 60s, MDE 3.0%
ab: A-ident   mean 14.7894 tok/s  (14.782, 14.640, 14.984, 14.978, 15.014, 14.755, 14.621, 14.541)
ab: B-ident   mean 14.8693 tok/s  (14.913, 14.738, 15.007, 15.002, 15.014, 15.013, 14.816, 14.451)
ab: paired delta      +0.542%  sd 0.770  95% CI [-0.103%, +1.186%]
ab: permutation p     0.0781  (exact, 2^8 sign flips)
ab: CI excludes zero  False
ab: |delta| >= MDE    False  (MDE 3.0%)
ab: VERDICT = NO DIFFERENCE
```

All 16 runs passed the swap and variance gates; none were discarded. Note pair 1
arm A recorded 16,628 swap-in pages — high, but below the 25,000 threshold, and
its result (14.782) sits mid-distribution. A strict zero-swap rule would have
thrown that run away for no reason.

The residual +0.542% is the honest measure of how much bias the ABBA
interleaving leaves behind on this machine. It is well inside the CI and well
below the 3.0% threshold, so the procedure is not systematically favouring
whichever arm runs second.

## 4. Positive control — injected regression

**Result: DETECTED**, in the correct direction, at `p = 0.0078` — which is the
*smallest p-value 8 pairs can produce*, i.e. every one of the 8 pairs moved the
same way.

```
ab: clean       mean 14.3280 tok/s  (14.361, 14.643, 14.714, 14.483, 14.991, 14.853, 12.241, 14.338)
ab: sabotaged   mean 13.4749 tok/s  (13.727, 14.020, 14.014, 11.959, 13.824, 14.118, 11.917, 14.220)
ab: paired delta      -5.882%  sd 5.070  95% CI [-10.121%, -1.643%]
ab: permutation p     0.0078  (exact, 2^8 sign flips)
ab: CI excludes zero  True
ab: |delta| >= MDE    True  (MDE 3.0%)
ab: VERDICT = DIFFERENT — sabotaged is SLOWER than clean by 5.88%
```

### The magnitude overshot, and the reason is not the rig

The injection was *sized* at 4.0% (2745 µs against a measured 68.65 ms step) but
the procedure measured 5.88%. The nominal 4% lies inside the CI, so the result is
statistically consistent — but a point estimate 47% high deserves an explanation
rather than a shrug, because "our positive control overshoots" is exactly how a
rig starts flattering later results.

**It is not contamination.** Four runs in this batch sat 8–9% low on prefill
(281–284 against a 306.75 baseline) — a decode-only sabotage cannot move prefill,
so those runs were externally disturbed. Excluding the three pairs containing
them, post hoc:

```
all 8 pairs                    delta -5.883%  sd 5.068  CI [-10.121, -1.645]  p=0.0078
5 pairs with healthy prefill   delta -5.232%  sd 1.452  CI [ -7.035, -3.430]  p=0.0625
```

The spread collapses (sd 5.07 → 1.45) but the *estimate barely moves*. So the
disturbance inflated the variance, not the mean.

**The real cause is that the injected delay costs more than its own duration at
small sizes.** Calibrating the sabotage at two magnitudes:

| injected | clean | sabotaged | step delta | vs nominal |
|---|---|---|---|---|
| 20,000 µs | 14.549 | 11.223 | 20.37 ms | **1.02×** |
| 2,745 µs | 14.328 | 13.475 | 4.42 ms | **1.61×** |

At 20 ms the delay is almost exactly additive. At 2.7 ms it costs ~1.6× nominal,
i.e. there is a roughly fixed ~1.7 ms penalty on top of the spin. The plausible
mechanism is that the busy-wait occupies the thread that would otherwise be
encoding the next Metal command buffer, so it does not merely add its own
duration — it also delays submission and breaks the CPU-encode / GPU-execute
overlap that normally hides encoding latency. At 20 ms the spin dominates and
that fixed component is invisible; at 2.7 ms it is 60% of the effect.

**Conclusion: the true regression present in arm B was ~5.2–5.9%, and the rig
measured it correctly.** The label "4%" described the spin duration, not the
delivered slowdown. To inject a true 4% here you would spin for ~1.7 ms, not
2.75 ms.

**Does this satisfy the requirement?** Yes, in substance: the procedure detected
a real, decode-only regression, in the right direction, at the right size for the
effect actually present, with every pair agreeing. Combined with the null test's
demonstrated 0.64% resolution on a quiet machine, a genuine 4% is comfortably
inside what this rig can see. What it does *not* prove is detection of a 4%
regression that is *purely* additive with no pipeline side-effect — a sharper
control would inject GPU-side work rather than a CPU spin, and that is the
improvement to make if this ever needs re-calibrating.

### The sabotage, and the mistake it caught

The injected delay is a busy-wait in the **per-token decode path only**, gated on
an environment variable so that one binary serves as both arms of the A/B — which
is what lets the null test compare byte-identical code rather than two builds the
compiler may have laid out differently.

**The first attempt was patched into the wrong function and the rig caught it.**
`src/llama-context.cpp` has two nearly identical `n_tokens == 0` guards, one in
`encode()` at line 1403 and one in `decode()` at 1646. A naive first-occurrence
text replacement hit `encode()`. A 20 ms/token delay — which should have taken
decode from 14.5 to 10.2 tok/s — changed nothing:

```
QM_SABOTAGE_US=0      tg = 14.479
QM_SABOTAGE_US=20000  tg = 14.549     <-- injected 29% slowdown, invisible
```

Had the positive control been run against that build it would have reported "no
difference", and the correct conclusion would have been "this rig cannot see a
29% regression" — when in fact the rig was fine and the sabotage was inert. **A
positive control only proves anything if you first prove the injection works.**

After patching `decode()`, the mechanism is exactly additive, which is the
strongest possible confirmation:

```
QM_SABOTAGE_US=0      tg = 14.549
QM_SABOTAGE_US=20000  tg = 11.223     predicted 1/(1/14.549 + 0.020) = 11.27
```

Prefill was checked separately and is untouched (301.98 → 293.0, inside its own
1.99% run-to-run noise), confirming the injection is decode-only as designed.

For the calibrated control the delay is sized against the measured baseline:
`4% of 1/14.567 s = 2746 µs per decoded token`.

---

## 5. Thermal drift

Characterised from the session's own record rather than a separate cold/hot/cool
triple, because by the time the controls finished the machine had produced **42
gated runs across 2 hours** of continuous benchmarking — a far better drift
sample than three points would have been.

| phase | condition | tg mean |
|---|---|---|
| noise floor, runs 1–4 | machine still settling after other work | 14.166 |
| noise floor, runs 7–10 | settled | 14.860 |
| null test (16 runs, sustained load) | settled, 26 min continuous | 14.829 |

**Drift magnitude: +4.9% from unsettled to settled**, and it is one-directional.
Under 26 minutes of *continuous* benchmarking in the null test the mean did not
decline — 14.829 against the noise floor's settled 14.860 — so this workload does
not heat the machine into throttling on any timescale that matters here.

The GPU clock probe confirms it independently: across all 42 gated runs
`throttle_pct` ranged −0.2% to 3.9% with a mean of 0.08%. One excursion (nf-4,
3.9%) is the only reading above 1%, and its neighbours were normal.

**So the thing to protect against on this machine is not thermal drift, it is
memory-system settling** — and the protection is interleaving, not cooling. The
cooldown exists to stop consecutive runs sharing a dirty page-cache and swap
state, which 60 s is sufficient for; it is not trying to restore a thermal steady
state, because there is no meaningful thermal excursion to restore from.

**Required cooldown: 60 s**, recorded in `bench/env.sh` as `QM_COOLDOWN_S`.

The justification is not that 60 s restores a thermal steady state — the GPU
clock probe shows this machine barely throttles at all under this workload
(throttle_pct within ±0.2% for nine of ten noise-floor runs, one excursion to
3.9%). It is that 60 s is enough for the *memory* system to settle between runs,
which is the drift term that actually moved results here. Interleaving, not
cooling, is what protects against the residual.

---

## 6. The frozen A/B procedure

`bench/ab.sh`, invoked as:

```
bench/ab.sh --a <bindir> --b <bindir> [--pairs 8]
```

1. Gate once with `bench/env.sh` for the whole batch (AC power, no Low Power
   Mode, no Spotlight/Time Machine/build, load average ≤ 3.0, no process above
   50% CPU, no other model process, GPU working set sufficient, free memory
   above threshold, GPU not throttled), then one cooldown.
2. Run A and B **interleaved in ABBA order** — pair 1 runs A then B, pair 2 runs
   B then A — so a linear session drift cancels to first order instead of being
   attributed to whichever variant ran later.
3. Each run goes through `bench/decode.sh`, which brackets it with `mem.sh` and
   `thermal.sh` and **discards** the run if swap-ins exceed 25000 pages or if the
   within-run relative sd exceeds 5%. Discarded runs are excluded from the
   comparison, not silently averaged in.
4. Compute the per-pair relative difference, its mean, sd, and 95% CI, plus an
   **exact** sign-flip permutation p-value (all `2^n` assignments enumerated, not
   sampled).
5. Report `DIFFERENT` **only if both** of these hold:
   - the 95% CI excludes zero, **and**
   - `|mean difference| >= QM_MDE_PCT`.

Condition 5b is what stops the procedure certifying a 0.4% "win" because one
batch happened to have low variance. Without it, a long enough sequence of A/B
tests eventually reports significance on pure noise. It is deliberately
conservative: this rig will miss real sub-3% improvements rather than
manufacture fake ones, and on a project whose entire remaining decode headroom
is single-digit percent, that is the correct trade.

Exit codes: `0` = NO DIFFERENCE, `3` = DIFFERENT, `1` = error or refused.
