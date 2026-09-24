# KVShareArena

Compare KV-cache reuse methods on shared tasks and quality–cost measurements.

**[Explore the leaderboard](https://xishi404.github.io/KVShareArena/)** ·
[Benchmark settings](docs/benchmark.md) · [Frozen data](data/queryset/README.md) ·
[Add a method](CONTRIBUTING.md)

## Why this benchmark?

A KV cache depends on more than its source text. It also depends on the preceding
context and the model weights. Reusing the same text does not guarantee a usable cache.

- **Across contexts:** cache sources separately, then combine them in a new prompt.
  This occurs in document-based question answering and agent-to-agent report sharing.
- **Across model checkpoints:** reuse a cache after changing the model weights.
  A general model can hand a long conversation to a specialist. Recomputing the full
  conversation adds work. Direct cache reuse can change answer quality.

Different methods repair or compress caches in different ways. KVShareArena compares
their answer quality, payload recomputation, cache size, and time to first token.
It is an evaluation framework, not a cache storage or serving engine.

## What is released now?

| Component | Current coverage |
| --- | --- |
| Interactive leaderboard | Qwen3-8B, same producer and receiver checkpoint; two workloads |
| Frozen question sets | Qasper, MultiFieldQA-en, HotpotQA: 100 questions each |
| Agent reports | 100 HotpotQA questions; frozen reports and original token IDs |
| FRAMES variant | 120 test and 60 calibration IDs, fingerprints, and a reconstruction tool |
| Evaluation package | CPU scoring and validation; two example generation drivers |
| Generation drivers | Retrieved Evidence: position alignment and SnapKV, on the HF backend |

This is an initial public release, not the full research workspace. The website does
not yet include cross-checkpoint results or the new model comparisons. It does not
claim that those experiments are absent from the paper. Full method-specific research
runners are not included in this release. Community submissions use maintainer review;
automatic admission and ranking are not enabled.

## Quickstart

Python 3.10 or newer is required. The scorer needs no GPU or third-party runtime packages.

```bash
git clone https://github.com/xishi404/KVShareArena.git
cd KVShareArena
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
kva data check --data_dir data/queryset
kva score examples/position_alignment_hotpotqa.json --track re --subset hotpotqa
```

The example contains real, previously measured model outputs. It is a legacy quality
record, not a submission file with complete cost measurements. See [examples](examples/README.md).

### Run a method on your GPU

```bash
python -m pip install -e '.[runtime]'
kva methods
kva run --method naive_pos --track re --subset hotpotqa \
  --model Qwen/Qwen3-8B --data_dir data/queryset --out runs/naive_pos.json
kva validate runs/naive_pos.json --data_dir data/queryset
kva score runs/naive_pos.json --data_dir data/queryset
```

For SnapKV, install `python -m pip install -e '.[runtime,kvpress]'` and use
`--method snapkv_r025`. The default removes 25% of cache entries. Generation requires a CUDA GPU with enough memory for the model,
source caches, and activations. Exact outputs can depend on GPU libraries and versions.
These GPU runs were not repeated during public-release packaging.

### Preview the website

```bash
python3 -m http.server 8765 --directory leaderboard
# Visit http://localhost:8765
```

Adjust the quality–cost balance and the compute and memory weights. TTFT is
reported separately and does not enter the preference score. Search and filter methods,
inspect individual results, and export the current comparison as CSV. There is no
server-side inference, sign-in requirement, analytics, or external JavaScript dependency.

### Run checks

```bash
python -m unittest discover -s tests -v
node leaderboard/test_explorer.cjs
python tools/build_site.py --check
```

## Measurements

Quality is reported as **performance-gap recovered (PGR)**:

`PGR = (method score − no-context score) / (full-prefill score − no-context score)`

PGR is not accuracy. It can be negative or exceed 1. Each workload has its own full-prefill
reference. Cost dimensions remain separate: payload computation, KV-cache bytes, and
same-backend time to first token. Missing values stay missing. A preference score is
not a statistical test. See [the measurement definitions](docs/benchmark.md).

## Repository layout

```text
kvsharearena/    Evaluation interface, drivers, scorer, and packaged references
data/queryset/  Frozen inputs, reports, FRAMES fingerprints, and checksums
leaderboard/    Static website and the released result snapshot
examples/       A real quality-scoring example
docs/           Settings, release scope, and deployment instructions
tests/          CPU-only release checks
tools/          Website data build and release audit
```

## License and attribution

Original harness and website code: Apache-2.0. Third-party code and datasets retain
their own licenses. The code license does not relicense source documents or model
weights. See [NOTICE](NOTICE), [third-party licenses](licenses/), and the
[dataset card](data/queryset/README.md).
