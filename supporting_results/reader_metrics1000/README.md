# CTI-Bench RCM: detailed metrics for every reader

This package extends the current 1,000-question diagnostics to **all 99 selected reader conditions** (11 models × 9 methods), alongside the eight Table 3 classifiers: **107 conditions in total**. It uses the primary 120,238-reference corpus results and unchanged frozen answers. No new answer-model inference, label changes or condition selection was performed.

- [metrics.json](metrics.json): mean per-question identifier-set P/R/F1 and exact match, class-level macro/micro P/R/F1, class-union sizes and empty/multiple-ID counts.
- [class_results/](class_results/): per-CWE support, predicted counts, true positives, precision, recall and F1; full confusion summaries for every condition.
- [support_bins.csv](support_bins.csv): minority/common class-support breakdowns (at most five versus more than five gold questions), with exact-correct counts and mean-question F1.
- [question_scores.jsonl.gz](question_scores.jsonl.gz): all 107,000 case-level extracted sets and scores.
- [inputs.json](inputs.json): release paths, original uncompressed answer hashes and expected main-table F1 scores; complete saved reader answers are under `answers/`.

## Example results

All values below are percentages. Class-macro F1, class-micro F1 and mean-question F1 are different averaging units.

| Condition | Mean-question F1 | Class macro P/R/F1 | Class micro P/R/F1 |
|---|---:|---|---|
| CVE Label Aggregation | 76.30 | 39.53 / 39.46 / 37.67 | 76.30 / 76.30 / 76.30 |
| Majority vote | 75.70 | 40.00 / 39.25 / 37.80 | 75.70 / 75.70 / 75.70 |
| Gemma 4 31B Naive RAG | 75.83 | 35.99 / 39.30 / 35.88 | 65.40 / 80.70 / 72.25 |
| Gemma 4 31B RAG Fusion | 77.36 | 36.50 / 40.79 / 36.62 | 67.05 / 81.60 / 73.61 |

The table reports class-level metrics. The [paired analysis](../paired_statistics1000/README.md) uses mean per-question F1 as its statistical endpoint.

## Scoring and confusion conventions

The frozen selected scorer extracts case-insensitive `CWE-\d+` matches from the **entire saved response**, uppercases and deduplicates them, and compares the set against the single canonical CTI-Bench gold CWE. Empty sets remain failures. Digit strings are retained; no numeric zero-padding changes, final-line-only selection, answer repair or Unicode-hyphen preprocessing is introduced. This preserves the recorded CTI-Bench metric. CTIConnect's separate declared Unicode-hyphen preprocessing and accepted-alternative scoring are not imported into this benchmark.

Class metrics evaluate each label as a binary prediction across all 1,000 questions. Macro averages over each method's gold/predicted label union, with undefined scores zero. Micro pools TP/FP/FN across labels. Union sizes can differ by condition. Free-text readers can mention multiple CWEs, so their class-micro F1 does not generally equal exact accuracy or mean-question F1. For the eight complete singleton classifiers, micro P/R/F1 equal accuracy.

Each `<condition>_confusion.csv` is a full long-form confusion summary with columns `canonical_gold`, `complete_predicted_id_set`, and `questions`. Every question contributes once, including empty and multi-ID answers; each file sums to 1,000. These are set-valued summaries rather than matrices that force a free-text answer into one selected CWE. No labels are collapsed into Other. Conventional square matrices for the eight singleton classifiers remain available in [class_metrics1000/](../class_metrics1000/README.md).

These diagnostics use the same saved answers as the main comparison. The recorded Phi A-RAG parser configuration is described in [protocol coverage](../../docs/PROTOCOL_COVERAGE.md#variants-and-scoring).

## Rebuild and verify offline

From the repository root, using standard-library Python:

```bash
python supporting_results/reader_metrics1000/compute.py
python supporting_results/reader_metrics1000/verify.py
```

No GPU, weights or API key is required. The package reuses `../class_metrics1000/gold.jsonl` and `predictions.jsonl` for the unchanged benchmark gold and eight classifier ledgers. [checksums.json](checksums.json) binds those inputs and this package. All 99 headline reader F1 scores reproduce; all eight existing classifier class-metric rows match. The verifier checks confusion totals and their per-CWE support/prediction/TP marginals. All 642 class-metric values (107 conditions × six metrics) were independently checked against scikit-learn.

The reader answers originate from release v1.0.0 `rcm_results_and_protocols.zip`; unchanged uncompressed export hashes are recorded in `inputs.json`. The source release retains saved requests, chat templates and model/protocol lineage. These commands rebuild metrics from saved model answers.
