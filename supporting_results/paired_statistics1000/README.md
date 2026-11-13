# Current 1,000-question paired reader comparisons

These analyses compare the frozen primary CVE Label Aggregation predictions (763/1,000; 120,238-reference corpus) with the strongest Naive RAG reader and strongest reader–method condition in the current CTI-Bench main table. Both readers are Gemma 4 31B. Selection is retrospective on these same evaluated questions; it is not a prespecified independent comparison.

The primary endpoint matches the paper: **mean per-question identifier-set F1**. All CWE identifiers in each complete answer are extracted using the frozen scorer's case-insensitive `CWE-\d+` rule and deduplicated. Empty sets score zero. Aggregation emits one CWE, so its per-question F1 is binary; reader F1 can be fractional. No final-line-only parsing, new retry, or answer repair is applied.

Differences below are **aggregation minus reader**, in percentage points.

| Reader condition | Reader F1 (%) | Difference | Paired 95% bootstrap CI | Two-sided paired randomization p | Holm-adjusted p |
|---|---:|---:|---:|---:|---:|
| Strongest main-table Naive RAG | 75.83 | +0.47 | [−1.27, +2.19] | 0.5992 | 0.5992 |
| Strongest main-table reader–method: RAG Fusion | 77.36 | −1.06 | [−2.94, +0.80] | 0.2675 | 0.5349 |

Neither F1 difference is statistically significant at 0.05, before or after correction. This does **not** establish equivalence or noninferiority. The small observed Naive RAG advantage must not be described as an established gain. RAG Fusion has the higher observed F1.

The existing current majority-vote comparison remains separate: +0.60 points, paired bootstrap CI [−0.10,+1.30], exact McNemar p=0.145996. These new reader comparisons do not replace that result.

## Statistical procedure

- All 1,000 official question IDs are paired, with no exclusions for empty or malformed answers.
- Percentile paired nonparametric bootstrap: 200,000 resamples. Multinomial sampling of identical paired score categories is equivalent to resampling question pairs. Individual method intervals and pointwise difference intervals appear in `results.json`.
- Paired randomization: independently swap method labels within each question, 200,000 Monte Carlo draws. The statistic is the absolute mean F1 difference. Report `(extreme + 1)/(draws + 1)`; seeds and Monte Carlo standard errors are saved. Grouped binomial draws implement the same independent sign flips efficiently.
- Holm correction is applied to the two primary F1 tests in this package. Bootstrap intervals are pointwise.
- Calculations pair the frozen predictions by question. Bootstrap resampling and randomization operate on individual question units; each analysis uses the selected saved generation for each condition.

### Secondary binary endpoint

`results.json` also reports exact-match paired counts and exact two-sided McNemar tests. Here reader correctness requires its entire extracted CWE set to equal the singleton gold set. Naive RAG has676/1000 exact matches; RAG Fusion has703/1000. Fractional F1 credit for multi-ID answers explains why these counts differ from the main-table F1 scores. McNemar evaluates binary exact-match correctness; the paired randomization test evaluates mean per-question F1. Secondary p-values are unadjusted.

## Rebuild and provenance

Run from the repository root with Python and NumPy:

```bash
python supporting_results/paired_statistics1000/compute.py
```

The release's complete saved answer files are included as deterministic gzip files. `inputs.json` records their release paths, original uncompressed SHA-256 hashes and main-table scores. Aggregation predictions and gold are reused from `../class_metrics1000/`; no new inference is needed. `question_scores.csv` records every extracted reader set and paired F1 value. `results.json` contains all counts, intervals, tests and seeds; `checksums.json` binds the files and reused inputs. Current results match the frozen release scorer.
