#!/usr/bin/env bash
# quality-run.sh — run the frozen bench/quality.sh with the environment this
# machine requires. NEW tooling; quality.sh itself is untouched.
#
# WHY. llama-perplexity defaults to -b 2048 and quality.sh does not pass -b
# (correctly -- it is measuring quality, not batching). On this machine that
# allocates 2048 x 248320 x 4 = 2.03 GB of logits on top of a 16 GiB model in an
# 18186 MiB working set, and the run dies with
# kIOGPUCommandBufferCallbackErrorOutOfMemory before producing a number.
#
# LLAMA_ARG_BATCH is the documented env form of -b (common/arg.cpp:1606) and a
# CLI flag still overrides it, so this affects only llama-perplexity: the
# llama-server invocations inside quality.sh pass -b explicitly.
#
# Perplexity is mathematically independent of batch size, so this changes what
# the run CAN complete, not what it measures.
export LLAMA_ARG_BATCH="${LLAMA_ARG_BATCH:-512}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/quality.sh" "$@"
