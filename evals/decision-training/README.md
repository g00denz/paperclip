# Decision Training Corpus

This directory contains reproducible, sanitized datasets derived from
`decision_training_examples`.

Artifacts emitted by `scripts/eval-decision-training-corpus.ts`:

- `corpus-raw.jsonl` — one DecisionTrainingExample per line (source). Never committed with live credentials; treat as sensitive.
- `corpus-all.jsonl` — sanitized corpus records ready for evaluation/training.
- `corpus-train.jsonl` / `corpus-holdout.jsonl` — issue-grouped split.
- `corpus-manifest.json` — provenance, split, distributions, rejections, hashes.
- `baseline-corpus.json` — corpus-readiness baseline (model-free) scored by the runner's eval kernel.

All schema strings are asserted on read, so format drift fails closed.
