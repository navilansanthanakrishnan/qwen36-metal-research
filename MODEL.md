# MODEL — Qwen3.6-27B Q4_K_M (MTP)

The experimental subject. **Frozen.** Never re-quantized, never converted, never
moved out of `~/projects/assets/models/qwen36-metal/`.

## Identity

| | |
|---|---|
| file | `~/projects/assets/models/qwen36-metal/Qwen3.6-27B-Q4_K_M.gguf` |
| symlinked as | `$WORK/models/Qwen3.6-27B-Q4_K_M.gguf` |
| repo | `unsloth/Qwen3.6-27B-MTP-GGUF` |
| revision | `5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace` |
| size | 17,106,773,120 B (15.926 GiB) |
| **SHA-256** | `a7cbd3ecc0e3f9b333edee61ae66bc87ed713c5d49587a8355814722ed329e0f` |

The SHA-256 matches the repo's LFS OID for that revision exactly, so the file is
byte-identical to what the Hub serves — the download is verified, not merely
completed.

It was **not** present on this machine at setup; the only local Qwen3.6-27B
weights were MLX-format (`mlx-community/Qwen3.6-27B-{4bit,8bit,OptiQ-4bit}`),
which are not GGUF and not usable by llama.cpp. It was fetched fresh.

## Architecture

Declared `general.architecture = qwen35` (Qwen3.6 uses the Qwen3.5 arch family).

| key | value |
|---|---|
| block_count | 65 (64 main + 1 MTP) |
| n_layer (effective) | 64 |
| embedding_length | 5120 |
| feed_forward_length | 17408 |
| attention.head_count | 24 |
| attention.head_count_kv | 4 |
| attention.key_length | 256 |
| attention.value_length | 256 |
| n_embd_k_gqa / n_embd_v_gqa | 1024 / 1024 |
| context_length | 262144 |
| rope.freq_base | 10,000,000 |
| rope.dimension_count | 64 |
| rope.dimension_sections | [11, 11, 10, 0] (mrope) |
| rms_norm_eps | 1e-6 |
| vocab | 248,320 tokens, 247,587 merges, `gpt2` tokenizer, pre `qwen35` |
| BOS / EOS / PAD | 248044 / 248046 / 248055 |
| file_type | 15 (Q4_K_M) |
| params | 27.32 B |
| imatrix | `unsloth_calibration_Qwen3.6-27B.txt`, 496 entries, 76 chunks |

### This is a hybrid SSM + attention model, not a dense transformer

The single most consequential fact about it, and it changes the memory
arithmetic completely:

| key | value |
|---|---|
| full_attention_interval | 4 |
| ssm.conv_kernel | 4 |
| ssm.state_size | 128 |
| ssm.group_count | 16 |
| ssm.time_step_rank | 48 |
| ssm.inner_size | 6144 |

**48 of the 64 main layers are SSM / gated-delta-net layers**; only **16 are full
attention** (every 4th). SSM layers carry `attn_qkv` (fused, Q6_K [5120,10240]),
`attn_gate` (Q4_K [5120,6144]), `ssm_out` (Q5_K [6144,5120]), and the small F32
`ssm_a / ssm_alpha / ssm_beta / ssm_conv1d / ssm_dt / ssm_norm` tensors.
Attention layers carry separate `attn_q / attn_k / attn_v / attn_output`.

Consequences:

- **KV cache is charged for 16 layers, not 64.** llama.cpp confirms:
  `llama_kv_cache: size = 512.00 MiB (8192 cells, 16 layers)` → exactly
  **64 KiB per token**, split K (f16) 32 KiB + V (f16) 32 KiB.
- **The SSM recurrent state is constant in context length**: 149.62 MiB total
  (R/conv 5.62 MiB + S/state 144.00 MiB) for 1 sequence, at *any* context. It
  does not grow.
- So context length is far cheaper here than for a dense 27B, and long-context
  decode does not degrade toward a KV-bandwidth wall as quickly.

## MTP tensors — PRESENT, confirmed

`qwen35.nextn_predict_layers = 1`. The MTP head lives at block 64 and was
**not** stripped by this conversion:

```
blk.64.nextn.eh_proj.weight            Q8_0  [10240, 5120]   55.7 MB
blk.64.nextn.enorm.weight              F32   [5120]
blk.64.nextn.hnorm.weight              F32   [5120]
blk.64.nextn.shared_head_norm.weight   F32   [5120]
```

plus a full transformer block body at `blk.64.*` (attn_q/k/v/output,
ffn_gate/up/down, norms) totalling 233.8 MB. Combined MTP cost: **289.5 MB**.

`nextn_predict_layers = 1` means a **single** draft head, so multi-head chained
drafting (`chain_heads` in `common/speculative.cpp`) is unavailable; drafts
deeper than 1 must come from feeding the one head autoregressively. That is the
central fact governing how acceptance decays with draft depth — see LEADS.md.

The pinned llama.cpp supports this: `LLM_TENSOR_NEXTN_*` in
`src/llama-arch.cpp:511-518`, `COMMON_SPECULATIVE_TYPE_DRAFT_MTP` in
`common/speculative.cpp:35`, selected with `--spec-type draft-mtp`.

## Tensor accounting

Total tensor bytes 17,095,778,304 (15.922 GiB) across 866 tensors.

| group | tensors | bytes | GiB | share |
|---|---:|---:|---:|---:|
| main blocks 0–63 | 848 | 15,048,124,416 | 14.015 | 88.0% |
| output head | 2 | 1,042,964,480 | 0.971 | 6.1% |
| token_embd | 1 | 715,161,600 | 0.666 | 4.2% |
| MTP block 64 body | 11 | 233,760,768 | 0.218 | 1.4% |
| MTP nextn.* | 4 | 55,767,040 | 0.052 | 0.3% |

By quantization type:

| type | tensors | bytes | GiB | share |
|---|---:|---:|---:|---:|
| Q4_K | 294 | 11,370,332,160 | 10.589 | 66.5% |
| Q6_K | 67 | 4,526,592,000 | 4.216 | 26.5% |
| Q5_K | 48 | 1,038,090,240 | 0.967 | 6.1% |
| F32 | 456 | 105,058,304 | 0.098 | 0.6% |
| Q8_0 | 1 | 55,705,600 | 0.052 | 0.3% |

Q4_K is two thirds of the bytes, which is why `kernel_mul_mv_q4_K_f32_impl` is
the decode hot path. Q6_K is a full quarter and carries `ffn_down` and
`attn_v` — it is not a rounding error and any tuning pass must cover it too.

## Runtime footprint (measured, ngl=99, ctx=16384)

| buffer | MiB |
|---|---:|
| MTL0_Mapped model (GPU) | 16027.69 |
| CPU_Mapped model (`token_embd`, stays on host) | 682.03 |
| KV cache (16 layers × 16384 × 64 KiB) | 1024.00 |
| recurrent SSM state (64 layers, ctx-independent) | 149.62 |
| MTL0 compute | 160.13 |
| CPU compute | 28.02 |
| output | 0.95 |
| **device total** | **17361.44** of 18186.25 available |

`token_embd` is deliberately host-resident (llama.cpp reports it *cannot* use
the `CPU_REPACK` buffer type and falls back to `CPU`), so it is gathered a row
at a time and does **not** contribute to per-token weight bandwidth. That is why
the honest decode ceiling is computed from 16.52 GB, not the 17.1 GB file size —
see HARDWARE.md.

The GPU allocation is **split across two Metal buffers** (13639.69 MiB +
3382.67 MiB) because `maxBufferLength` on this device is 13639 MiB, below the
model size. This is handled transparently by ggml-metal but is worth knowing
before blaming a mystery on allocation.

## Coherence

Loads and generates correct, coherent text at temperature 0:

```
llama-cli -m models/Qwen3.6-27B-Q4_K_M.gguf -ngl 99 -c 4096 -n 200 --temp 0 \
          --seed 1234 -st -p "Write a Python function that returns the n-th
          Fibonacci number iteratively."
```

produces a correct chain-of-thought derivation of the iterative Fibonacci
recurrence (the model emits a `[Start thinking]` block by default). Verified
before any measurement was taken.
