# Retrieval-corpus overlap controls

## CTI-Bench RCM: current 1,000-question evaluation

The reference pool starts with 121,295 mapped CVEs. All 1,000 target CVE identifiers are excluded globally;607occur in this pool. Casefolding and whitespace collapse are applied to target/reference descriptions for exact matching; 17 additional references match normalized target descriptions. These exclusions leave 120,671 records before related-record screening.

Lexical description screening requires both word 5-gram set Jaccard strictly above 0.8 and normalized character edit similarity strictly above 0.8. Edit similarity is 1 minus unit-cost Levenshtein distance divided by the maximum character length. Semantic screening uses description-only Qwen3-Embedding-4B vectors and cosine similarity at least 0.95, without using benchmark correctness to select the threshold. The lexical/semantic union removes 429 unique references (30 lexical, 427 semantic, 28 overlapping).

Source-supported vulnerability relationships add 4 references to that union, for 433 additional exclusions beyond target-ID/exact matching. The final corpus has 120,238 references. Source evidence considers operationally specific advisories, patches, commits/changesets and documented relationships; shared product names or generic vendor pages alone do not establish a relationship. Shared advisories may cover distinct flaws. Grouping uses direct source relationships.

The exact final 433-ID list, original corpus row indices and 120,238 eligible rows are in corpus/final_source_supported_exclusions.json. Its SHA-256 is 14fb27040fa8bdeddc1b92c6aceb7643d22ccb8b92182c1c31b99cdc58ac6f80. The same frozen eligible row order appears in the release's aggregation_artifacts/final_corpus_selection.json. configs/near_duplicate_detection.json records lexical/semantic normalization, model revision and thresholds.

A separate conservative shared-disclosure stress condition removes 574 references beyond target/exact exclusions and leaves 120,097. The sensitivity conditions retain the released benchmark labels. Aggregation accuracy is 76.30% under the primary 433-reference mask, 76.40% under the broader 574-reference mask and 76.50% after target/exact exclusions alone. Every evaluation question remains in the denominator. Naive RAG and aggregation share the primary corpus and the same ordered top 20 CVEs per question; other methods may alter retrieval by design.

## CTIConnect benchmark quality versus our retrieval controls

CTIConnect was designed to evaluate retrieval-augmented CTI analysis. Its RCM questions are generated from authoritative CVE–CWE mappings. In Section 3.1.7, its authors report GPT-4 quality filtering, independent verification by two domain practitioners and senior-annotator adjudication. Sources: https://arxiv.org/html/2510.11974v2#S3.SS1.SSS7 and https://arxiv.org/html/2510.11974v2#S4.SS1 .

CTIConnect’s quality checks concern question/answer construction. Our experiment separately filters its mapped-CVE retrieval corpus to 119,889 records. The 120,238-record corpus above applies to CTI-Bench. CTIConnect k=10 was selected using the evaluated 290 questions.

## Source dates and annotations

The corpus uses official CVE/NVD annotations without per-target disclosure-date cutoffs. Aggregation settings were selected using the evaluated questions. Source versions and acquisition records are listed in [corpus snapshots](CORPUS_SNAPSHOTS.md).

## Broader shared-disclosure sensitivity artifacts

The complete [574-reference exclusion mask](../corpus/shared_disclosure_exclusions.json) preserves the original frozen bytes and includes excluded CVE IDs, original corpus rows and all 120,097 eligible rows. Its [provenance manifest](../corpus/shared_disclosure_manifest.json) records source/public SHA-256 hashes. The associated 1,000 predictions and twenty-neighbor lists are in [baseline_predictions.jsonl.gz](../supporting_results/ablations/point5/baseline_predictions.jsonl.gz), arm `shared_disclosure`: 764 correct (76.40%). This broader stress condition is separate from the primary 433-reference mask; no experiment was rerun for publication.

## CTIConnect-specific source-description and disclosure sensitivity

The [complete290-question audit](../supporting_results/cticonnect290/README.md#benchmark-specific-overlap-sensitivity) publishes its119,889-reference primary mask and dedicated screening of all282source CVEs. Beyond the primary source-ID/exact exclusions, lexical/semantic source-description screening removes253references and yields160/290(55.17%). Adding operationally specific shared-disclosure URL relationships removes291references in total and yields161/290(55.52%), equal to primary161/290. Predictions and masks are frozen, every question retained, and source descriptions/relationship pairs/hash provenance are published. These sensitivity results apply to aggregation k10; the LLM readers were not rerun under additional masks.
