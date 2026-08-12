# HARDWARE — what every later number is a ratio against

Every line here was read off this machine, not copied from a spec sheet. The
brief supplied a hardware block to verify; **every item in it matched**, with
one clarification about binning noted below.

## Chip

| | measured | how |
|---|---|---|
| chip | **Apple M5 Pro** | `sysctl -n machdep.cpu.brand_string` |
| **GPU cores** | **16** | `system_profiler SPDisplaysDataType` → "Total Number of Cores: 16"; `ioreg -l \| grep gpu-core-count` → 16 |
| CPU cores | **15** = 5 Super + 10 Performance | `hw.perflevel0.name=Super, physicalcpu=5`; `hw.perflevel1.name=Performance, physicalcpu=10`; `hw.ncpu=15` |
| SoC | T6050 | `uname -a` |
| unified memory | **25,769,803,776 B = 24 GiB** | `sysctl hw.memsize` |
| macOS | **26.5 (25F71)**, Darwin 25.5.0 arm64 | `sw_vers`, `uname -a` |
| disk free | 290–307 GiB | `df -h /` |

### This is a binned M5 Pro — the part matters

Apple's M5 Pro is "up to 20-core GPU", "18-core CPU (6 super + 12 performance)",
"up to 64GB unified memory", "up to 307GB/s memory bandwidth". This machine is
the **binned** variant: 16/20 GPU cores, 15/18 CPU cores, 24 GB.

That is not a discrepancy with the brief — the brief's block already said 16 GPU
cores and 5+10 CPU, and that is what the machine reports. It matters because
**the 307 GB/s figure is quoted for the full part**, and it is the denominator
of the attained-bandwidth ratio below. Apple does not publish a separate
bandwidth figure for the binned configuration. On Apple Silicon the memory
controller is generally not binned alongside GPU cores, and the measurement
below (88.2% of 307 GB/s) is consistent with the full figure applying here — if
the real peak were lower, the attained fraction would be implausibly high.

### Metal

```
GPU name   : MTL0 (Apple M5 Pro)
GPU family : MTLGPUFamilyApple10 (1010), MTLGPUFamilyCommon3 (3003), MTLGPUFamilyMetal4 (5002)
simdgroup reduction   = true      has unified memory = true
simdgroup matrix mul. = true      has bfloat         = true
has tensor            = true      use residency sets = true
                                  use shared buffers = true
maxThreadgroupMemoryLength   = 32 KiB
maxBufferLength              = 13639 MiB      <-- SMALLER THAN THE MODEL
recommendedMaxWorkingSetSize = 18186 MiB (19069.67 MB)
```

`maxBufferLength` being below the model size means ggml-metal splits the weights
across two Metal buffers (observed: 13639.69 MiB + 3382.67 MiB). It is handled
transparently, but it is the kind of thing that looks like a bug when you meet
it for the first time at 2am.

`has tensor = true` and family `Metal4` are directly relevant to LEADS.md — the
tensor-API path is gated on exactly this.

---

## Measured memory bandwidth

Probe: `bench/bwprobe/` — **frozen**. A 2 GiB private-storage buffer (orders of
magnitude larger than any cache), read exactly once per dispatch with fully
coalesced `float4` loads and four independent grid-stride streams for
memory-level parallelism, timed with the command buffer's own GPU timestamps,
median of 20 dispatches after 3 warmups. The accumulator is consumed by a
comparison that is never true, so nothing is dead-code eliminated and nothing is
written back — this is read bandwidth, not read+write.

```
device            Apple M5 Pro
buffer_bytes      2147483648 (2.00 GiB)
threads           1048576  threadgroup 256
read_GBps_median  270.8
read_GBps_best    281.4
read_GBps_worst   260.6
```

### `attained_bandwidth = 270.8 GB/s` — 88.2% of the 307 GB/s vendor peak

The brief's gate is 60–90%: **PASS**, near the top of the band, which is what a
correctly written streaming read on Apple Silicon should look like.

Two controls prove the probe is neither measuring cache nor under-saturating:

**Not cache-resident** — bandwidth is flat as the buffer grows far past any
cache. A cache-resident measurement would be several times faster at 64 MiB:

| buffer | 64 MiB | 256 MiB | 1 GiB | 2 GiB | 4 GiB |
|---|---|---|---|---|---|
| GB/s | 241.3 | 254.7 | 258.0 | 275.0 | 239.7 |

**Saturated** — throughput is flat across a 16× range of thread counts, so the
number is a property of the memory system, not of the dispatch geometry:

| threads | 262144 | 524288 | 1048576 | 2097152 | 4194304 |
|---|---|---|---|---|---|
| GB/s | 272.1 | 271.6 | 273.4 | 269.5 | 271.9 |

### GPU clock reference (the thermal signal)

`bench/gpuinfo --clock`, a register-resident FMA loop with zero memory traffic:
**6258 GFLOP/s** on a cold idle machine, run-to-run spread 0.07%. Against a
theoretical ~6.55 TFLOP/s FP32 for 16 cores that is ~95%, confirming the probe
is clock-limited. `bench/thermal.sh` reports shortfall against this as
`throttle_pct`. See SETUP-LOG.md for why this replaced `powermetrics`.

---

## The decode ceiling

```
attained_bandwidth = 270.8e9 B/s
```

What is actually read from memory on every ordinary decode step is **not** the
whole file:

- `token_embd` (682.03 MiB) is **host-resident** — llama.cpp reports it cannot
  use the `CPU_REPACK` buffer type and keeps it on `CPU`. It is gathered one row
  at a time per token, so it contributes ~20 KiB, not 682 MiB.
- the **MTP block** (`blk.64` body + `nextn.*`, 276.1 MiB) is not touched at all
  when speculation is off.

| basis | bytes | GiB | `ceiling_1tok` |
|---|---:|---:|---:|
| GGUF file on disk | 17,106,773,120 | 15.932 | 15.830 tok/s |
| all tensors | 17,095,778,304 | 15.922 | 15.840 tok/s |
| GPU-resident (`MTL0_Mapped`) | 16,806,251,069 | 15.652 | 16.113 tok/s |
| **GPU-resident minus MTP (the honest one)** | **16,516,723,261** | **15.382** | **16.396 tok/s** |

```
ceiling_1tok = 270.8e9 / 16,516,723,261 = 16.40 tok/s
```

**This is a hard wall.** Nothing that reads all the weights once per token can
exceed it, ever — not a better kernel, not a better scheduler, not a better
compiler. The only way past is to stop reading all the weights once per token.

### Past the wall

Speculative decoding is the escape, and it works by amortizing one weight read
over several accepted tokens:

```
ceiling_spec  ~=  ceiling_1tok  x  mean_accepted_tokens_per_forward_pass
```

Treat this as an order-of-magnitude target, not a computed limit: it ignores the
cost of running the draft head and the shape of the acceptance distribution.
With this model's single MTP head (`nextn_predict_layers = 1`) and a mean
acceptance of, say, 2.0 accepted tokens per forward pass, the target is ~33
tok/s. Acceptance is the only unbounded lever in the project, which is why
LEADS.md ranks it first.

---

## Memory budget — this is a 24 GiB machine holding a 16 GiB model

The dominant constraint on the project, not a footnote. The weights are 66% of
installed RAM.

Device budget is set by `recommendedMaxWorkingSetSize = 18186 MiB`, of which
llama.cpp sees 18035 MiB free. Fixed costs, all measured:

| item | MiB | scales with |
|---|---:|---|
| model, GPU-resident | 16027.69 | — |
| SSM recurrent state (64 layers) | 149.62 | **nothing** — constant in context |
| compute buffer | 152–176 | ubatch, weakly on ctx |
| KV cache | 64 KiB/token | context, 16 attention layers only |

KV arithmetic: 16 attention layers × 4 KV heads × 256 head dim × 2 (K and V) ×
2 bytes (f16) = **64 KiB per token**. Confirmed by llama.cpp:
`llama_kv_cache: size = 512.00 MiB (8192 cells, 16 layers)`.

### Largest context that fits

| ctx | model | KV | SSM | compute | total | headroom of 18186 | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 8192 | 16027.69 | 512.00 | 149.62 | 152.13 | 16841.44 | 1344.81 | fits |
| **4096** | 16027.69 | **256.00** | 149.62 | 152.13 | **16585.44** | **1600.81** | **FROZEN** |
| 16384 | 16027.69 | 1024.00 | 149.62 | 160.13 | 17361.44 | 824.81 | fitter accepts, **does not run** |
| 24576 | 16027.69 | 1536.00 | 149.62 | 168.13 | 17881.44 | 304.81 | too tight |
| 32768 | 16027.69 | 2048.00 | 149.62 | 176.13 | 18401.44 | −215.19 | does not fit |

llama.cpp's own fitter agrees, targeting 1024 MiB of free device memory:

```
ctx 16384: projected 16679 MiB vs 18035 free -> will leave 1356 >= 1024, no changes needed
ctx 24576: projected 17199 MiB vs 18035 free -> cannot meet target, need to reduce by 187 MiB
ctx 32768: projected 17719 MiB vs 18035 free -> cannot meet target, need to reduce by 707 MiB
```

### The fitter's answer is not the largest context that actually runs

**`QM_CTX = 4096` is part of the frozen configuration**, and getting there was the
most surprising correction of the setup.

llama.cpp's static fitter approves 16384 and the arithmetic above says it leaves
824 MiB of device headroom. It does not work. A `llama-server` holding a 16384
context **OOMs during warmup** with
`kIOGPUCommandBufferCallbackErrorOutOfMemory`, before serving a single request.
At 8192 it loads and runs — but decodes at **5.69 tok/s instead of ~14.6**, i.e.
it is memory-starved rather than failing outright, which is the more dangerous
outcome because it looks like a working configuration. At 4096 it loads and
sustains full speed (15.07 tok/s on a single-request check).

Two things are going on. `recommendedMaxWorkingSetSize` is a *static driver
number*; it does not account for what macOS and other applications currently hold,
and this machine runs with roughly 3 GiB wired before llama.cpp starts. And
`llama-server` allocates batch-sized buffers that `llama-bench` does not, so it
additionally needs `-b` capped at the ubatch size (at `-b 2048` it OOMs even at
smaller contexts).

**The lesson for anyone re-deriving this: a context that passes the fitter and
loads is not proven. It has to sustain full decode speed under the memory
pressure the machine actually has.**

### GPU wired limit

```
iogpu.wired_limit_mb = 0   (driver default)  ->  recommendedMaxWorkingSetSize = 18186 MiB
```

18186 MiB is 74% of the 24576 MiB installed, leaving 6390 MiB to macOS. **The
default is sufficient for the frozen configuration** and no sysctl is required.

There is no passwordless sudo on this machine, so `bench/env.sh` cannot set the
limit unattended; it instead *verifies* the working set the driver actually
grants, which is the quantity that matters. See SETUP-LOG.md.

If it can be raised interactively:

```
sudo sysctl iogpu.wired_limit_mb=20480     # resets on reboot
```

20480 MiB would fit ctx 32768 with ~2 GiB headroom. Whether that is wise on a
24 GiB machine is a separate question — it takes macOS's share from 6.4 GiB to
4 GiB, and this machine already swaps continuously from other applications.

### Swap

**A run that swaps is not a measurement.** `bench/mem.sh` records
`vm.compressor.swapper.swapins_total` before and after every run and
`bench/decode.sh` gates on it. The counter is system-wide and this machine has
substantial background swap traffic at idle, so the gate is a measured
threshold rather than a strict zero — the reasoning, the idle measurements, and
the second variance-based gate are in SETUP-LOG.md.
