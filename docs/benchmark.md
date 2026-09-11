# Benchmark settings

## Two kinds of change

**Context change:** the receiver sees the source text in a different prompt.
Separately encoded sources can have different prefixes, token positions, and attention
history from a full prefill of the assembled prompt. Moving cache positions addresses
one part of this mismatch. It does not recreate all previous attention computations.

**Checkpoint change:** the receiver uses different model weights. Even compatible
cache shapes and tokenizers do not make the cached values interchangeable. This setting
can occur during a handoff between a general model and a specialist. Reuse is useful
only if moving and repairing the cache costs less than recomputing it at the required
answer quality.

The research considers both changes. This website release covers context change with
the same Qwen3-8B checkpoint on both sides. Cross-checkpoint results will require a
separate, explicitly labeled export.

## Workloads

- **Retrieved Evidence:** cache each source independently, then answer a question over
  the assembled sources. Qasper, MultiFieldQA-en, and HotpotQA each contain 100 frozen
  questions. The FRAMES column is a separately labeled snapshot variant.
- **Agent Reports:** independent agents write reports without seeing the final question.
  The receiver answers from those reports. The full-prefill reference uses exactly the
  same reports, not the original source documents.

The public generation command currently implements Retrieved Evidence only. Agent
Reports data and measured results are available; its generation runner is not included.

## Quality and costs

The three LongBench subsets use word F1. FRAMES uses its autorater and is not scored by
the package's word-F1 command. PGR uses the no-context and full-prefill scores as its
references. Raw PGR is not clipped for reporting.

| Measurement | Definition |
| --- | --- |
| Compute saved | One minus the fraction of payload layer-tokens recomputed at answer time |
| Memory saved | One minus held KV-cache bytes divided by the dense bf16 reference bytes |
| TTFT saved, cache ready | One minus receiver TTFT divided by same-backend full-prefill TTFT |
| TTFT saved, cache build included | The separately reported accounting that includes cache construction |

Cache construction, training, transfer, and serving infrastructure are not interchangeable
costs. Do not interpret cache-ready TTFT as an end-to-end deployment speedup. A byte
count is not peak GPU memory. Timing comparisons require the same backend and protocol.

## Comparisons

Official implementations, our ports, and reference rows are separated. Reference rows
are not ranked. For measured quality comparisons, paired bootstrap uses sample IDs,
seed 20260730, 10,000 resamples, and a 95% interval. An interval containing zero means
that the comparison did not detect a difference; it does not establish equivalence.

The website combines quality and cost with a user-selected weighted harmonic mean.
Only that preference calculation clips inputs to [0, 1]. Every selected cost must be
available for a row to receive a score. Raw negative values remain visible. The website
therefore need not reproduce rankings from a paper score with different missing-data rules.

## Limits of this release

The result snapshot is a measured-data export, not a live service connected to an HPC.
The scorer can recompute quality from predictions and supplied answers. Public admission
also requires a maintainer to check answers against the frozen question set, model and
prompt identity, complete coverage, implementation provenance, and cost accounting.
The standalone schema check is not a certification of those claims.
