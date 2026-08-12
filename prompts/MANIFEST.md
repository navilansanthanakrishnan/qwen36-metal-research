# Frozen prompt set

**IMMUTABLE.** Editing, adding, or removing a prompt invalidates every
acceptance-rate and quality number measured against this set. If a new prompt
is genuinely needed, add it as a *new* category file and leave these alone.

Acceptance rate under speculative decoding is a function of how predictable the
continuation is, and that varies enormously by category — code with rigid syntax
drafts far better than open prose. An aggregate acceptance number across
categories is meaningless, so every result is reported per category.

All prompts are real text. None are synthetic token soup: synthetic input cannot
measure speculative decoding, because the draft model's hit rate depends on the
statistical structure of real language.

| file | category | note |
|---|---|---|
| `codegen-01.txt` | codegen | write a function from a spec |
| `codegen-02.txt` | codegen | write a class with several methods |
| `codegen-03.txt` | codegen | write a shell script from a spec |
| `codeedit-01.txt` | codeedit | fix a bug in supplied code, reprint it |
| `codeedit-02.txt` | codeedit | refactor supplied code, reprint it |
| `codeedit-03.txt` | codeedit | add error handling to supplied code |
| `prose-01.txt` | prose | explanatory technical prose |
| `prose-02.txt` | prose | narrative prose |
| `prose-03.txt` | prose | argumentative / analytical prose |
| `longctx-01.txt` | longctx | retrieval from a long real document |
| `longctx-02.txt` | longctx | multi-fact retrieval from the same document |
| `json-01.txt` | json | structured record extraction |
| `json-02.txt` | json | nested config generation |
| `json-03.txt` | json | tabular data as JSON array |

Categories, and why each is here:

- **codegen** — the workload this project exists to speed up. Highly
  predictable token-to-token; expected to show the *highest* draft acceptance.
- **codeedit** — regenerating mostly-unchanged code. Should show acceptance
  even higher than codegen, because most of the output is copied from context.
  If it doesn't, the drafting path is leaving the largest win on the table.
- **prose** — the low-acceptance floor. Open-ended text has genuine entropy per
  token, so this bounds how much speculation can ever help on average.
- **longctx** — a different regime entirely: KV cache is large, attention cost
  per token is higher, and the memory-bandwidth ceiling arithmetic changes
  because KV traffic joins weight traffic. Also the case where a prefill
  regression is most visible.
- **json** — rigid grammar, very low entropy. Expected acceptance ceiling.

The long-context document is `longctx-doc.txt`, built from the frozen
wikitext-2 test corpus under `runs/corpus/`. It is real encyclopedic prose, not
generated filler, and it is identical across runs.
