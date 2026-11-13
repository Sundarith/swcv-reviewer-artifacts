# Current 1,000-question class metrics and confusion matrices

These artifacts cover all 1,000 CTI-Bench RCM questions under the primary 433-reference exclusion mask (120,238 references), matching the paper's Table 3.

- [All macro/micro precision, recall and F1](metrics.json), as fractions.
- [CVE Label Aggregation confusion matrix](cosine_sum_confusion.csv).
- [CVE Label Aggregation per-CWE results](cosine_sum_per_cwe.csv).
- [Majority-vote confusion matrix](majority_confusion.csv).
- Each of the eight classifiers has a full `<method>_confusion.csv` and `<method>_per_cwe.csv`.
- Exact frozen [predictions](predictions.jsonl), [gold](gold.jsonl) and [file hashes](checksums.json) are included.

Matrices have gold labels in rows and predictions in columns; no labels are collapsed into Other. Each matrix sums to 1,000, and its diagonal sums to the reported correct count. Per-CWE CSVs include support, predicted count, correct count, precision, recall and F1. Zero-support predicted-only labels are retained.

Macro scores average over each method's union of benchmark and predicted labels, setting undefined scores to zero. Union sizes can differ across methods. Micro scores pool TP/FP/FN across classes. All eight classifiers emit exactly one label per question, so micro precision, recall and F1 equal accuracy. This equality does not apply generally to free-text reader outputs: the paper's reader tables use mean per-question identifier-set F1. This package covers the eight singleton classifiers, not class-level metrics for all reader conditions.

| Classifier | Macro P (%) | Macro R (%) | Macro F1 (%) | Micro P/R/F1 (%) |
|---|---:|---:|---:|---:|
| CVE Label Aggregation | 39.53 | 39.46 | 37.67 | 76.30 |
| Majority vote | 40.00 | 39.25 | 37.80 | 75.70 |
| Nearest neighbor | 30.90 | 30.23 | 28.42 | 68.60 |
| Inverse distance | 39.56 | 39.49 | 37.70 | 76.40 |
| Softmax | 39.52 | 39.49 | 37.65 | 76.40 |
| BM25 | 27.98 | 25.65 | 24.84 | 59.50 |
| Embedding ridge | 44.17 | 40.54 | 39.93 | 74.40 |
| TF-IDF naive Bayes | 10.71 | 4.86 | 4.51 | 37.30 |

Weighting does not improve macro-F1 over majority voting. High micro scores should not be read as strong minority-class performance.

## Rebuild offline

Run `python supporting_results/class_metrics1000/build.py` from the repository root. Standard-library Python suffices; no inference or API request occurs. The builder validates unique official question IDs, full question coverage and matrix row/column totals.

Input bytes are copied unchanged from release v1.0.0 `rcm_results_and_protocols.zip`, paths `aggregation_artifacts/voting_and_classifiers.jsonl` and `aggregation_artifacts/ctibench_gold.jsonl`. Newly generated metrics match the release's `documentation/per_cwe.csv` and all eight macro rows of paper Table 3.

## All reader conditions

[reader_metrics1000](../reader_metrics1000/README.md) extends the detailed diagnostics to all 99 current CTI-Bench reader conditions and reproduces the eight classifiers here. This directory retains the classifiers' conventional square confusion matrices.
