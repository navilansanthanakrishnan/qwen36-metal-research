# RESUME

Working state for picking this up in a fresh context window. Overwrite freely —
unlike LEDGER.md, this file is scratch.

## STATUS 2026-08-12 (later): 22.48 tok/s at depth 4, target 35 — NOT reached

The register-resident Q4_K simdgroup kernel is **built, correct, integrated and
kept**: llama.cpp branch `sgmv-q4k`, +12.05% under the frozen speculative A/B
(p=0.0312, CI [+9.87,+14.23], acceptance unchanged). See LEDGER 038-041.

Run it as: `LLAMA_ARG_SPEC_N_RS_SEQ=3` (swept, LEDGER 042), MTP depth 4, ctx 4096.
`GGML_METAL_SGMV_DISABLE=1` selects the old path for A/B on one binary.

**The cost model is now solved, and it says what has to happen next.**

    cycle = T_ver + (D+1)*t_draft      t_draft = 8.9 ms, T_ver = 122.4 ms

35 tok/s at depth 4 needs a 94.1 ms cycle. Drafting is only 44.5 ms of the
current 166.9, so **making drafting free is not enough on its own** -- T_ver
alone already exceeds the budget. Two things must both happen:

  1. T_ver 122 -> ~90 ms. The standalone probe shows this kernel runs at 170
     GB/s against a 213 GB/s loads-only ceiling for its access pattern, so
     roughly 25% is still on the table inside the kernel itself.
  2. mean_acc/fwd 3.29 -> ~4.7. Depth does not do this (041): each extra step
     buys ~0.12 accepted tokens for 8.9 ms.

**DONE (LEDGER 045): Q6_K extended, +24.3% end to end.** Superseded note follows.
**Quality (LEDGER 046): token-exact on all 14 prompts, perplexity 5.9079 vs
5.9079, backend-ops pass. KLD did NOT complete** — it dies at chunk 30 with
`failed reading log-probs`, which is the stored base file running short, not a
divergence (chunks 1-29 show KL 0.0008 at 100.000% same-token). **First job
next session: regenerate the KLD base at the matching chunk count and re-run
`bench/quality-run.sh`.** Then tree drafting. Old note: extend the kernel to Q6_K (LEDGER 044). The kernel
covers Q4_K, which is 66.5% of the weight bytes; Q6_K is another 26.5% and
carries `ffn_down` and the fused `attn_qkv`, still on the scalar path. That is
why a 1.25-2.17x per-shape win became 1.17x on the model. Everything carries
over -- the lane layout, the reduction permutation, the fma fold -- only the
unpack differs: Q6_K keeps 4 low bits in `ql`, 2 high bits in `qh`, and an
int8 scale per 16 values, so it needs a shift-and-combine before
`unpack_unorm4x8_to_float` instead of one mask. Work in
`ggml/src/ggml-metal/ggml-metal.metal` next to `kernel_mul_mv_sgq4k_f32`, and
prototype in `bench/sgmv.metal` first -- that loop iterates in seconds where a
llama.cpp rebuild plus A/B is over half an hour.

**After that, tree drafting rather than depth.** Verify now
costs the same at width 8 as at width 5 (233-235 us flat, per-shape), so a
tree of 8 candidates costs what a chain of 5 costs today. That raises
mean_acc/fwd without adding draft steps, which is exactly the term the model
says is missing. It needs tree attention masks in speculative.cpp.

## Where things stand (earlier)

**35 tok/s = 2.13x ceiling_1tok, so it is reachable only through speculation.**
Depth 3 needs mean accepted 1.96 of 3 (65%); depth 8 lands back on a one-stream
rung via mul_mm and needs only 35%. ngram at zero draft cost reaches 36.4 tok/s
at a=1.5. The target is a well-posed engineering problem, not a wall.

**Speculation currently does not run: everything OOMs.** The chain of causes is
now fully traced and the fix is identified — see LEDGER 009-017. The single
remaining blocker is that `token_embd` (682.03 MiB) is wired into the Metal
buffer as a hole in the middle of a contiguous min→max mapping span, even though
it is host-assigned. Splitting that mapping is the unlock.

## Old status


Setup is complete except for three gates that are blocked on **machine
availability**, not on anything technical. Nothing has been optimized. The
handoff at the top of SETUP-LOG.md has the baseline, the ceiling, and the first
lead to chase.

## Blocked, and how to unblock

`bench/env.sh` correctly refuses to benchmark while the machine is in interactive
use. At the end of the setup session a game was running and Spotlight was
re-indexing after the rebuild. On a quiet machine, run these three in order:

```bash
cd ~/projects/navilan/research/qwen36-metal

# gate 3 — prove the reverted tree is back at baseline (expect ~14.6-14.9 tok/s)
for i in 1 2 3; do bash bench/decode.sh --label reverted --tag "post-revert-$i"; done

# gate 4 — quality oracle. --generate first (references were invalidated when the
# frozen context changed from 16384 to 4096), then verify, then prove it FAILS
# on a deliberately degraded KV cache.
bash bench/quality.sh --generate
bash bench/quality.sh                      # must PASS
bash bench/quality.sh --ctk q4_0 --ctv q4_0  # must FAIL, naming the check

# gate 5 — MTP acceptance. Depths chosen to map the LEADS.md L1 staircase.
bash bench/spec.sh --depth 1,2,3,4
```

Then fill `PLACEHOLDER_SPEC_PENDING` in BASELINE.md with the acceptance table and
flip gates 3–5 in the SETUP-LOG status table.

## Known-good facts that took time to establish (do not re-derive)

- **`bench/quality.sh` must be run through `bench/quality-run.sh`** (or with
  `LLAMA_ARG_BATCH=512` exported). llama-perplexity defaults to `-b 2048`, which
  allocates 2.03 GB of logits (2048 x 248320 x 4) on top of the 16 GiB model and
  OOMs. Perplexity is batch-independent, so this changes what completes, not what
  is measured. quality.sh itself is frozen and untouched.
- **Frozen perplexity baseline: PPL = 5.9079 +/- 0.14173** at
  `-c 512 --chunks 40` on `runs/corpus/wikitext-2-raw/wiki.test.raw`. Always
  quote the chunk count with the number.

- Frozen context is **4096**, not what the fitter says. 16384 passes llama.cpp's
  fitter and OOMs in `llama-server` during warmup; 8192 loads but decodes at
  5.69 tok/s. See HARDWARE.md.
- `llama-server` needs `-b` capped at the ubatch (512), plus `-ctxcp 0 -cram 0`.
  Without the latter two it allocates up to 32 context checkpoints at 149.6 MiB
  each plus an 8 GiB prompt cache, and dies.
- `llama-cli` one-shot is `-st`, **not** `-no-cnv`.
- Never pass `-md` for MTP — it loads a second full copy of the weights.
- The venv at `.venv/` (29 MB, numpy + pyyaml) is only needed for `gguf_dump.py`.

## Next action

LEADS.md is ranked. L1 is free and needs no patch: set `--spec-draft-n-max` to 3
or 4 and confirm the staircase with
`llama-bench -p 1,2,3,4,5,6,7,8,9,12,16 -n 0`. Then L2 (ngram drafting for the
copy-heavy categories), then L10 (`GGML_METAL_TENSOR_DISABLE=1`, one env var, and
it doubles as the correctness check described in Track 6's T6-6).

## Open questions

- The ~94%-of-bandwidth figure that counts SSM recurrent-state traffic is derived
  from code reading, not measured. It is the most load-bearing unverified claim
  in LEADS.md and it gates how much decode headroom actually exists. Confirm with
  `GGML_METAL_GRAPH_DEBUG=1` node counts before trusting L8's magnitude.
- The prefill-health gate in `decode.sh` was added **after** the null and positive
  controls ran. That is conservative for the positive control but lenient for the
  null test — re-run `bash bench/phase-f.sh null` under the new gate to confirm
  the procedure did not become credulous.
- The exhaustive `gh pr list --state closed` sweep of 90 days of ggml-metal PRs
  did not complete. What upstream has already *rejected* is only partially known.

## Needs Navilan (do not block on it)

### AC POWER — blocks all timing measurement

At 2026-08-12 07:53Z the machine was **on battery** (98%, discharging) and
`bench/env.sh` correctly refused every timing run: Apple Silicon drops GPU clocks
on battery, so tok/s taken there is not comparable to the frozen baseline.

**Plug the laptop in.** That is the whole fix. Nothing else is blocked.

Correctness work is clock-independent and proceeds regardless — greedy outputs,
perplexity, KL-divergence, acceptance rate at temperature 0, `test-backend-ops`,
and gated-pipeline counts are all deterministic. Only tok/s needs AC.


The GPU wired limit is unset, capping context at 4096. Raising it fits 32768.
There is no passwordless sudo, so this is the one thing the run cannot do:

```bash
sudo sysctl iogpu.wired_limit_mb=20480
```

It resets on reboot. Everything in this project is measured at ctx 4096 and the
frozen configuration assumes it — a larger context is an opportunity, not a
prerequisite. Carry on without it.
