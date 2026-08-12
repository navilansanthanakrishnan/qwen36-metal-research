# RESUME

Working state for picking this up in a fresh context window. Overwrite freely —
unlike LEDGER.md, this file is scratch.

## Where things stand

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

The GPU wired limit is unset, capping context at 4096. Raising it fits 32768.
There is no passwordless sudo, so this is the one thing the run cannot do:

```bash
sudo sysctl iogpu.wired_limit_mb=20480
```

It resets on reboot. Everything in this project is measured at ctx 4096 and the
frozen configuration assumes it — a larger context is an opportunity, not a
prerequisite. Carry on without it.
