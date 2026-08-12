# LEDGER

One row per experiment. Append only — never edit or delete a row, including
the ones that failed. A ledger you can rewrite is not evidence.

Rules:

- **Every row cites a `bench/ab.sh` verdict**, not a single run. A number
  without a paired, interleaved comparison against trunk is not a result.
- `KEEP` requires the A/B verdict to be `DIFFERENT` in the faster direction
  **and** `bench/quality.sh` to pass. Speed at degraded quality is not a win
  and does not get a KEEP.
- The minimum detectable effect is in NOISE.md. Anything below it is `NOISE`,
  regardless of how good the mean looks.
- Record `REVERTED` rows too. The fact that an idea was tried and failed is
  worth as much as the fact that one worked, and it stops the same idea being
  re-tried in three months.

| # | date | lead | change | commit/patch | A/B verdict | Δ decode | Δ prefill | accept | quality | decision | notes |
|---|------|------|--------|--------------|-------------|----------|-----------|--------|---------|----------|-------|
