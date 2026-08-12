# BUILD — the pinned trunk

Everything in this project is a diff against this. Reproduce with `./init.sh`
(add `--clean` to wipe `build/` first).

## Pinned commit

| | |
|---|---|
| repo | `https://github.com/ggml-org/llama.cpp.git` (fresh clone) |
| **commit** | `0b1bad14ff204627636aeb1de22ddcd5acb859d4` |
| build tag | `b10380` |
| date | 2026-08-11 15:15:20 -0500 |
| subject | `chat : fix muse-glimmer detection of tool calls after EOM (#26879)` |
| depth | 10,380 commits |

`~/projects/forks/llama.cpp` is an upstream-tracking fork and was **not** used
or modified. It sits at `1d1d9a9ed` (2026-07-10), about a month behind; it may
be read for history but nothing here touches it.

## Toolchain

| | |
|---|---|
| cmake | 4.4.0 (Homebrew) |
| compiler | Apple clang 21.0.0 (clang-2100.1.1.101), `/usr/bin/cc` / `/usr/bin/c++` |
| target | arm64-apple-darwin25.5.0 |
| macOS SDK | 26.5 |
| xcode-select | `/Library/Developer/CommandLineTools` |

**Command Line Tools only — full Xcode is not installed.** Two consequences
that matter and are not obvious:

1. `xcrun metal` does not exist, so the Metal shader library cannot be compiled
   ahead of time. `GGML_METAL_EMBED_LIBRARY=ON` therefore embeds the *source*
   and every pipeline is compiled at runtime on first use. This is why the very
   first model load is slow and later ones are not.
2. `xctrace` and the Metal debugger are unavailable (`/usr/bin/xctrace` is a
   stub that errors). Phase H's GPU trace had to be obtained another way — see
   SETUP-LOG.md and LEADS.md.

## CMake configuration (frozen)

```
cmake -S llama.cpp -B llama.cpp/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DGGML_BLAS=OFF \
    -DGGML_ACCELERATE=ON \
    -DLLAMA_CURL=ON \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DBUILD_SHARED_LIBS=OFF
cmake --build llama.cpp/build --config Release -j 15
```

Resolved cache values worth recording: `GGML_NATIVE=ON`, `GGML_CPU=ON`,
`GGML_METAL_SHADER_DEBUG=OFF`, `CMAKE_CXX_FLAGS=""` (no local flag tampering).

## Binaries verified present and executable

`llama-bench`, `llama-cli`, `llama-server`, `llama-perplexity`,
`test-backend-ops` — all under `llama.cpp/build/bin/`. `init.sh` fails the build
if any is missing.

## Flags — checked against `--help` on THIS build, not from memory

The brief's warning is justified; two traps were found here.

**Confirmed and used:**

| tool | flags |
|---|---|
| `llama-bench` | `-m -p -n -pg -d -b -ub -ctk -ctv -t -ngl -fa <on\|off\|auto> -sm -mg -nkvo -lm -ot -r --delay --no-warmup -o <csv\|json\|jsonl\|md\|sql> -fitt -fitc` |
| `llama-cli` | `-m -ngl -c -n --temp --seed -st -p -v -lv --log-file --no-warmup` |
| `llama-server` | `--spec-type <none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,ngram-simple,ngram-map-k,ngram-map-k4v,ngram-mod,ngram-cache>`, `--spec-draft-model/-md`, `--spec-draft-n-max`, `--spec-draft-n-min`, `--spec-draft-p-min`, `--spec-draft-p-split`, `--spec-draft-ngl/-ngld`, `--spec-draft-backend-sampling`, `-ctkd/-ctvd` |

**Traps found:**

1. **`-mmp / --mmap` is DEPRECATED in favour of `--load-mode` (`-lm`)** on this
   build. Exactly the brief's scenario: the old spelling still parses. Use
   `-lm <auto|none|mmap|mlock|mmap+mlock|dio>`. Nothing here passes `-mmp`.
2. **`-no-cnv` is accepted by `llama-cli` but does not give you a one-shot
   run.** The first attempt with `-no-cnv` generated its answer and then sat in
   the interactive REPL spinning at 100% CPU on stdin EOF for 12 minutes before
   being killed — it looked exactly like a hung model load, and it was not.
   The correct flag on this build is **`-st` / `--single-turn`**. Everything in
   `quality/` and the coherence checks uses `-st` and redirects stdin from
   `/dev/null`.

`llama-bench` takes **no** speculative flags, which is why `bench/spec.sh` is
built on `llama-server` instead.

## Metal device as this build sees it

```
GPU name   : MTL0 (Apple M5 Pro)
GPU family : MTLGPUFamilyApple10 (1010), MTLGPUFamilyCommon3 (3003), MTLGPUFamilyMetal4 (5002)
simdgroup reduction   = true
simdgroup matrix mul. = true
has unified memory    = true
has bfloat            = true
has tensor            = true
use residency sets    = true
use shared buffers    = true
recommendedMaxWorkingSetSize = 19069.67 MB  (18186 MiB)
```

## Runtime switches exposed by ggml-metal

These make several Phase H hypotheses testable without writing a kernel, by
turning an existing optimization *off* and measuring what it was worth. That
bounds the headroom for adding more of the same:

```
GGML_METAL_TENSOR_ENABLE / GGML_METAL_TENSOR_DISABLE
GGML_METAL_FUSION_DISABLE / GGML_METAL_FUSION_DEBUG
GGML_METAL_GRAPH_OPTIMIZE_DISABLE / GGML_METAL_GRAPH_DEBUG
GGML_METAL_CONCURRENCY_DISABLE
GGML_METAL_CAPTURE_COMPUTE
GGML_METAL_NO_RESIDENCY / GGML_METAL_RESIDENCY_KEEP_ALIVE_S
GGML_METAL_SHARED_BUFFERS_ENABLE / GGML_METAL_SHARED_BUFFERS_DISABLE
GGML_METAL_BF16_DISABLE
GGML_METAL_DEVICES
GGML_OP_OFFLOAD_MIN_BATCH
```

and from ggml/llama core: `GGML_SCHED_DEBUG`, `LLAMA_GRAPH_REUSE_DISABLE`,
`LLAMA_KV_CACHE_DEBUG`, `LLAMA_BATCH_DEBUG`, `LLAMA_TRACE`.

## `test-backend-ops`

The correctness floor. Result recorded in BASELINE.md and re-checked by
`bench/quality.sh`.
