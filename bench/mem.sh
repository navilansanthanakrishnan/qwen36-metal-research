#!/usr/bin/env bash
# mem.sh — one line of memory + swap state. FROZEN.
#
# Every recorded result carries this line before and after the run. Any run
# whose swapins_total moved is discarded, not averaged in.
#
# NOTE ON COUNTERS: the sysctl the setup brief names (`vm.swapins`) does not
# exist on macOS 26.5, and `vm_stat` on this build does not print Swapins /
# Swapouts either. The live counters are under vm.compressor.swapper.*, which
# is where the kernel's compressed-memory swapper accounts real swap traffic.
# swapins_total is the one that matters: it is monotonic and nonzero movement
# means pages came back from the swap file during the run.
#
# `Pageins` is deliberately NOT used as the swap signal. llama.cpp mmaps the
# GGUF, so pageins move by ~16 GiB on every cold model load as a matter of
# course. Treating that as swap would discard every legitimate run.

set -euo pipefail

read -r SWAPIN SWAPOUT <<<"$(sysctl -n vm.compressor.swapper.swapins_total \
                                       vm.compressor.swapper.swapouts_total | tr '\n' ' ')"

SWAPUSED_MB="$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/')"

# vm_stat page counts -> MiB (page size is 16384 on Apple Silicon)
PGSZ="$(vm_stat | head -1 | sed -E 's/.*page size of ([0-9]+) bytes.*/\1/')"
eval "$(vm_stat | awk -v p="$PGSZ" '
  /^Pages free/            {gsub(/\./,"",$3); printf "FREE_MB=%d\n",  $3*p/1048576}
  /^Pages wired down/      {gsub(/\./,"",$4); printf "WIRED_MB=%d\n", $4*p/1048576}
  /^Pages inactive/        {gsub(/\./,"",$3); printf "INACT_MB=%d\n", $3*p/1048576}
  /^Pages occupied by compressor/ {gsub(/\./,"",$5); printf "COMPR_MB=%d\n", $5*p/1048576}
')"

PRESSURE="$(memory_pressure 2>/dev/null | awk -F': ' '/free percentage/{gsub(/%/,"",$2); print $2}')"

printf 'mem free_pct=%s free_MiB=%s wired_MiB=%s inactive_MiB=%s compressor_MiB=%s swap_used_MB=%s swapins_total=%s swapouts_total=%s\n' \
    "${PRESSURE:-NA}" "$FREE_MB" "$WIRED_MB" "$INACT_MB" "$COMPR_MB" \
    "$SWAPUSED_MB" "$SWAPIN" "$SWAPOUT"
