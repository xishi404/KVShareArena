# Frozen evaluation inputs

| Files | Contents |
| --- | --- |
| `qasper_n100.jsonl` | 100 Qasper questions from LongBench v1 |
| `multifieldqa_en_n100.jsonl` | 100 MultiFieldQA-en questions from LongBench v1 |
| `hotpotqa_n100.jsonl` | 100 HotpotQA questions from LongBench v1 |
| `agent_reports/` | Frozen question-blind reports for 100 HotpotQA questions |
| `frames_variant/` | 120 test and 60 calibration IDs, fingerprints, reconstruction metadata |
| `manifest.json` | Sampling and chunking settings for the three LongBench subsets |
| `sha256.json` | Checksums for the released input and metadata files |

LongBench inputs were shuffled with seed 42 and the first 100 eligible questions were
kept per subset. A 29,000-token cap used the Qwen3-8B tokenizer; it removed no questions
in these three pools. Natural passage boundaries define multi-hop chunks. Single-document
inputs use 512-token chunks. Frozen files, not a fresh tokenizer run, define this release.

Each question contains `_id`, `context`, `input`, `answers`, and `chunks`.
`n_tok_ctx` counts **context plus question**, not context alone. Do not sum it with the
question length again. Chunk tokenization may differ from tokenization of the full string.

Agent report token IDs are authoritative. Do not re-tokenize report text for decoded-cache
replay. Generation settings and report statistics are in its manifest. Reports are model
outputs, not verified source facts.

FRAMES uses a pinned Wikipedia snapshot. It is a **variant**, not the original FRAMES
context release. Only IDs and fingerprints are included here. To rebuild text locally:

```bash
python -m pip install -e '.[frames]'
kva data rebuild-frames --data_dir data/queryset --split test --out_dir frames_payloads
```

The tool downloads the pinned sources and checks reconstructed text against fingerprints.
It is not a GPU evaluation or the FRAMES answer autorater.

## Attribution and licenses

These are subsets and derived inputs, not newly authored source datasets. Selection,
chunking, and generated reports are the modifications. Original content remains under
its upstream terms; the repository's Apache-2.0 code license does not replace them.

- [LongBench v1](https://github.com/THUDM/LongBench/tree/main/LongBench), Bai et al., ACL 2024:
  source of the three QA exports. Its repository is MIT-licensed; underlying datasets
  retain their own terms. See `licenses/LongBench-MIT.txt` at the repository root.
- [Qasper](https://allenai.org/data/qasper), Dasigi et al., NAACL 2021: CC BY 4.0 for
  the dataset. Underlying scientific papers retain their applicable terms.
- [HotpotQA](https://hotpotqa.github.io/), Yang et al., EMNLP 2018: CC BY-SA 4.0.
  The included HotpotQA subset and derivative report data are distributed under these
  same terms. The included scoring example also derives from this dataset.
- MultiFieldQA-en: the English single-document QA subset released with LongBench v1.
  Refer to the LongBench release for source-document provenance and usage terms.
- [FRAMES](https://huggingface.co/datasets/google/frames-benchmark): official question
  references; [MLCommons](https://github.com/mlcommons) supplies the pinned snapshot
  named in `frames_variant/fingerprints.json`. Wikipedia text retains its applicable
  attribution and share-alike terms. This repository does not bundle that text archive.

Please cite the upstream datasets as well as KVShareArena when using these inputs.
