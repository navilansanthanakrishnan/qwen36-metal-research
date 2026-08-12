# RESUME

Working state for picking this up in a fresh context window. Overwrite freely —
unlike LEDGER.md, this file is scratch.

## STATUS 2026-08-12: 20.670 tok/s mean (best 25.620), target 35 — NOT reached

Measured across all 14 frozen prompts, MTP depth 4, `--spec-n-rs-seq 1`, ctx 4096,
quant and quality untouched. Baseline was 14.567.

**The remaining step is LEDGER 035** (032/033 named the wrong mechanism; 034
refuted the tile-shape thesis). 35 tok/s is provably out of reach for scalar
arithmetic: it needs 6.94 ms/verify-column and scalar FP32 peak is 9.36. The one
kernel that can clear it does not exist in any llama.cpp backend — dequantize
K-quant weights into simdgroup 8x8 fragments held in REGISTERS (not threadgroup
memory) and drive simdgroup_multiply_accumulate against an 8-wide B fragment.
mul_mv/ext are register-resident but scalar; mul_mm uses matrix ops but stages A
through threadgroup memory. Old note:: a narrow-N tile for
`mul_mm` (4M×1N simdgroup arrangement, NR0=128/NR1=16). Verify at width 5 runs at
3.88 TFLOP/s where the same kernel reaches 17.6 at n=512, because it computes a
32-wide N tile for 5 columns. Predicted T_ver(5) 136.6 → ~95 ms → ~35 tok/s.
Six coupled sites must change together; do not attempt without budget to validate.

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
