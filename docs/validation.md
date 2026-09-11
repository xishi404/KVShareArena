# Release validation

Checks performed locally on 2026-09-11:

- Installed the package in a new virtual environment, with no runtime dependencies.
- Passed six CPU tests: command-line help, frozen IDs and references, checksums, the
  real output example, report and FRAMES coverage, and scoring edge cases.
- Recomputed the 100-question example: mean F1 0.2740, PGR -0.1961; no stored-F1 drift.
- Passed JavaScript tests for score weights, endpoints, missing measurements, negative
  values, dataset coverage, and cost mappings.
- Verified that all 745 numeric fields in the leaderboard equal the source export.
- Verified byte equality for all three QA question files and the frozen report file.
- Opened the website in Chrome at 1440 px and 390 px widths. Both loaded 18 rows in
  the default view and 16 Agent Reports rows. Slider and details-dialog checks passed.
  No JavaScript errors or page-level horizontal overflow were detected.

Not tested here: GPU generation, FRAMES network reconstruction, public pull-request
submission, and GitHub-hosted deployment. Deployment requires creation of the public
repository and selection of GitHub Actions as its Pages source.

These are release checks, not new research measurements.
