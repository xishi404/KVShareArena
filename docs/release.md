# Initial public release

Release prepared: 2026-09-11. The website measurement snapshot was exported on
2026-09-10. Publishing a snapshot does not rerun experiments or add new results.

## Included

- A standalone installable evaluation package and two example generation drivers.
- Frozen inputs, reports, scoring references, and FRAMES reconstruction fingerprints.
- The existing interactive leaderboard with unchanged quality and cost measurements.
- A real historical quality-scoring example, CPU tests, and GitHub Pages deployment.

## Not included

- Private development history, internal planning or review documents, cluster account
  details, scheduler scripts, experiment logs, slides, or model weights.
- Unfinished experiments and preliminary results from ongoing model extensions.
- The full collection of method-specific reproduction runners and per-method outputs.
- A hosted GPU runner, PyPI release, Hugging Face mirror, or automatic submission service.

The public repository starts with fresh history. It is not a fork containing earlier
private commits. The original research repository remains separate.

## Provenance and release changes

The frozen LongBench question files, report token IDs, real scoring example, and all
numeric leaderboard fields are copied without changes. Metadata omits machine-specific
paths. The checksum manifest covers the released data files.

The package no longer searches internal cluster directories. CPU scoring installs
without GPU dependencies; generation dependencies are available through the `runtime`
extra. Website links and documentation describe only the supported public workflow.

The leaderboard JSON is the canonical snapshot for this release. `tools/build_site.py`
only bundles that JSON for browser use. It does not regenerate measurements from raw
experiments. Full reproduction of every paper row is not claimed by this release.

Community result files require manual validation against the frozen answers and
settings. A submission's own gold answers, quality scores, or cost numbers are not
independent evidence. No workflow executes submission code or automatically admits rows.
