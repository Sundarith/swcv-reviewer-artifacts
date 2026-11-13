# CTIConnect RCM: metrics, paired uncertainty and overlap sensitivity

This package covers all **290 official question IDs**, the paper's **99 selected reader conditions** (11 models × 9 methods), and five label-selection controls. It retains the primary scores and released gold labels. Selection of aggregation k=10 is retrospective on these same questions. Naive RAG uses20 references: the reported comparison is not a matched-k10 experiment.

## Metrics and confusion summaries

[metrics.json](metrics.json) reports official mean-question precision/recall/F1 and accepted-alternative correctness for all104conditions, plus explicitly separate canonical-label macro/micro P/R/F1. [support_bins.csv](support_bins.csv) reports question counts and official scores for minority/common gold classes. [class_results/](class_results/) contains full per-CWE P/R/F1/support and confusion counts for every condition.

| Condition | Official F1 (%) | Canonical macro P/R/F1 (%) | Canonical micro P/R/F1 (%) |
|---|---:|---|---|
| Aggregation k10 | 55.52 | 35.13 / 37.57 / 33.76 | 51.72 / 51.72 / 51.72 |
| Majority k10 | 54.83 | 35.57 / 38.53 / 34.59 | 51.03 / 51.03 / 51.03 |
| GPT-OSS Naive RAG | 60.00 | 34.50 / 38.77 / 34.05 | 54.03 / 55.52 / 54.76 |
| GPT-OSS reranked RAG | 63.68 | 39.53 / 42.41 / 38.88 | 58.31 / 59.31 / 58.80 |

The official scorer credits declared accepted alternatives (present for27questions). Canonical class metrics instead evaluate each extracted label against the released single canonical CWE. Aggregation is correct on150canonical labels and161under the official accepted-alternative rule. Thus **canonical micro-F151.72% is not a replacement for official55.52%**. Both are reported to make the convention explicit; canonical diagnostics can penalize officially valid alternatives.

Class metrics use one binary decision per question/label, the method's canonical-gold/predicted label union, and zero for undefined P/R/F1. Macro averages class scores; micro pools TP/FP/FN. Union sizes differ by method. A reader can mention multiple IDs, so its micro metrics do not generally equal accuracy.

Every `<condition>_confusion.csv` is a full, uncollapsed long-form confusion summary: canonical gold CWE, **complete extracted prediction set**, number of questions. Each question contributes once, so counts sum to290even for multiple-ID or empty answers. These are set-valued confusion summaries, not matrices that falsely force a reader to one label. The five singleton controls also have conventional square `<condition>_matrix.csv` files with gold in rows and prediction in columns. Per-CWE CSVs support class-specific error analysis. [question_scores.jsonl.gz](question_scores.jsonl.gz) retains all extracted sets and official per-question scores.

Uniform Unicode-hyphen normalization is applied before whole-response CWE extraction, matching the selected CTIConnect table protocol. The frozen official scorer is included unchanged. All99reader F1 cells reproduce to numerical precision.

## Paired uncertainty and significance

[statistics.json](statistics.json) includes paired95%percentile bootstrap intervals (200,000resamples), two-sided paired randomization tests (200,000draws), seeds, Monte Carlo errors, individual method intervals and secondary binary exact McNemar tests. Positive differences favor aggregation k10.

| Comparator | F1 difference (points) | Paired95%CI | Randomization p | Holm p |
|---|---:|---|---:|---:|
| Majority k10 | +0.69 | [−1.03,+2.41] | 0.6877 | 0.6877 |
| Strongest Naive RAG: GPT-OSS | −4.48 | [−8.97,0.00] | 0.0613 | 0.1226 |
| Strongest reader/method: GPT-OSS reranked RAG | −8.16 | [−13.62,−2.76] | 0.00374 | 0.01122 |

The strongest reranked reader outperforms aggregation on this reported evaluation. The other two F1 differences are nonsignificant at0.05; that does not prove equivalence. Weighting does not improve canonical macro-F1 over majority voting. The same-k20 aggregation/majority control metrics are also retained, without selecting a new k.

The majority comparison has binary singleton official outcomes: both correct157, aggregation-only4, majority-only2, neither127; exact McNemar p=0.6875. Reader F1 can be fractional, so their binary McNemar tests in the JSON concern **official accepted-alternative correctness**, not mean-question F1. Do not substitute one test's endpoint for the other.

Bootstrap draws resample paired question outcomes; multinomial sampling of identical score pairs is equivalent. Randomization independently swaps method labels within each question, using the absolute mean difference and `(extreme+1)/(draws+1)`. Holm correction covers these three CTIConnect F1 comparisons, not the study's historical exploration or other benchmarks; intervals are pointwise. The analyses condition on frozen outputs, assume independent question units and do not account for vulnerability-family clustering, model-run variability or retrospective comparator/k selection.

## Benchmark-specific overlap sensitivity

The primary119,889-reference corpus inherits the CTI-Bench base exclusions and removes349additional CTIConnect-specific source-ID/exact-description records from the120,238base. The frozen original descriptor records351source/exact records before accounting for overlap with those base exclusions. [exclusions.json](exclusions.json) publishes the full current eligible-row mask. This inherited filtering alone was not a dedicated CTIConnect semantic audit.

The additional audit covers **282distinct source CVEs for all290questions**, using their source descriptions from the pinned released CTIConnect corpus. It screens every eligible reference description with fixed rules: normalized exact matching, word5gramJaccard>0.8 AND normalized unit-cost Levenshtein similarity>0.8, or symmetric description-embedding cosine≥0.95. No additional exact pairs remain. The lexical/semantic union identifies253references (37lexical and229semantic, including13in both). Direct matches to operationally specific source/advisory/commit URLs add a conservative shared-disclosure stress condition totaling291additional references. Shared disclosures may contain distinct flaws; these are sensitivity exclusions, not confirmed leakage verdicts. No transitive family closure or product-only exclusion is applied.

| Fixed-k10 aggregation condition | References | Official correct/290 | Official F1 (%) |
|---|---:|---:|---:|
| Primary reported corpus |119889|161|55.52|
| Additional description exclusions |119636|160|55.17|
| Description + shared-disclosure stress |119598|161|55.52|

All290questions and original labels remain. The two additional conditions change one and two predictions respectively. Masks, enumerated pairs, source inputs/provenance hashes, frozen predictions, top20neighbors and results are in [overlap/](overlap/). Only aggregation is recomputed under additional masks; reader scores remain those of the primary corpus. Do not present these as reader reruns under tighter masks.

The audit reused161identical cached source-description embeddings and encoded121additional source descriptions with the same frozen Qwen encoder on twoGPUs. It made **zero new answer-model requests**. Primary retrieval reproduces every saved top20row and similarity exactly. Thresholds/masks and all sensitivity predictions were frozen before scoring. The analyses use the released benchmark labels and the recorded benchmark-specific corpus filters.

## Reproduce

From the repository root:

```bash
python supporting_results/cticonnect290/verify.py
python supporting_results/cticonnect290/compute.py
```

Verification uses standard-library Python; metrics/statistics rebuilding also needs NumPy. The verifier checks [checksums.json](checksums.json), all confusion totals and every overlap vote/score. The computation reproduces the99reader table cells and all104metrics from complete saved answers. [inputs.json](inputs.json) records unchanged uncompressed answer hashes and release paths. Saved requests/templates remain available in release v1.0.0, alongside the other benchmark's artifacts.

Full overlap regeneration is separate from offline verification. Set `SWCV_ENCODER_PATH` to the recorded model snapshot and `SWCV_ASSET_ROOT` to the matching external frozen assets listed by hash in [overlap/overlap_protocol.json](overlap/overlap_protocol.json), then run `python supporting_results/cticonnect290/overlap/regenerate.py`. It requires the recorded GPU/runtime dependencies, reference description cache, original full-record/query embeddings, documents and reference provenance; these large external assets and weights are not duplicated here. [overlap/environment.json](overlap/environment.json) records the actual runtime. Output receipts and original/public code hashes identify the portable path adaptation.

## Extended neighborhood sensitivity

The [six-setting aggregation sweep](neighborhood_sensitivity/README.md) extends the sensitivity analysis to k=1,5,10,20,30,50. It retains the same primary corpus, official questions and labels. The selected k=10 result remains 55.52%; `selection_candidates` records the original selection history separately from this later sweep. Verify it with `python supporting_results/cticonnect290/neighborhood_sensitivity/verify.py` from the repository root.
