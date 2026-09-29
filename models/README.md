# Experiment records

Result files written by individual runs of the original research pipeline
(`ml/train.py`, `ml/train_problem2_lstm.py`). They are kept as a record of how
the work developed.

**These are not the numbers to quote.** They come from intermediate evaluations
on partial data and earlier versions of the pipeline, so they do not match
either of the two metric sets that matter:

- the **paper's** results, measured on the real pilot data — see the README
- the **demo's** results, reproducible from this repository — see
  `artifacts/metrics.json` and `artifacts/model_card.md`

Nothing in the running service reads this directory.
