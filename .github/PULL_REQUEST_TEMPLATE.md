<!--
KVShareArena submission template.

`kva submit` fills this in for you; if you are opening the pull request by hand,
replace the angle-bracket placeholders and delete the lines that do not apply.

In the public arena repository this file lives at .github/PULL_REQUEST_TEMPLATE.md.
-->

## Submission: <method display name>

| field | value |
| --- | --- |
| method key | `<method_key>` |
| track / subset | `<re or ar>` / `<subset>` |
| receiver model | `<model id, main board is Qwen3-8B>` |
| backend | `<hf-transformers / vllm / ...>` |
| implementation | `<official / port / reference>` |
| samples | `<n, 100 on the frozen queryset>` |
| result file | `submissions/<method>/<file>.json` |
| method card | `submissions/<method>/method_card.md` |

### Self-reported cost axes

| axis | value |
| --- | --- |
| payload fraction recomputed at answer time | `<x%>` |
| KV bytes per sample | `<bytes>` |
| TTFT per sample | `<ms>` |

Cost numbers are self-reported and are shown as such on the leaderboard; the scorer
recomputes quality (PGR and the paired interval) from the per-sample outputs in the
result file, and maintainers spot-check cost claims.

### Method card

<!-- paste the method card, or point at the file added in this pull request -->

### Checklist

- [ ] frozen queryset only, no sample selection after seeing results
- [ ] `build_cache` never saw the question
- [ ] `kva validate` passes locally
- [ ] method card states backend, versions, training data (if any), and how each cost number is measured

---

Maintainer review is required. This release does not enable an `/evaluate` bot or
automatic admission to the leaderboard.
