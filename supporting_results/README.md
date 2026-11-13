# Supporting prediction ledgers

This directory supplies the per-question evidence for the Muse prompt sensitivity analysis and Table 4 representation ablations. Files ending in `.gz` contain the exact original bytes, compressed without changing prediction text or neighbor lists. `manifest.json` records the original freeze hashes, uncompressed hashes and download hashes. No experiment was rerun for this release.

Run from a fresh repository clone with Python 3.10 or newer; no additional dependencies, model endpoint or GPU is required:

```bash
python supporting_results/verify.py
```

The verifier checks hashes and 1,000 unique official question identifiers per ledger, reproduces Muse precision/recall/F1/exact match with the recorded scorer, and reconstructs ablation predictions by similarity-weighted voting over every saved neighborhood before checking exact accuracy.

## Muse sensitivity

`muse/` contains all six 1,000-question ledgers, the frozen prepared questions/messages and twenty retrieved references, and the exact prompt variants/settings/results. Only the final user instruction changes in the rewrite condition. The six ledgers retain full answer text, recorded finish reasons and request-failure fields. Model checkpoint, chat-template hash, scorer hash and available run dates are recorded in `protocol_and_results.json`. The prepared input hash matches the historical protocol. The answer ledgers were recovered read-only from the completed Server 2 experiment and match all six previously published freeze hashes.

| Condition | Mean per-question identifier-set F1 (%) |
|---|---:|
| original-1 | 75.55 |
| rewrite-1 | 75.42 |
| original-2 | 76.00 |
| rewrite-2 | 75.47 |
| original-3 | 75.90 |
| rewrite-3 | 76.52 |

These are sensitivity runs, separate from the selected main-table Muse result. The scorer extracts all CWE identifiers from the complete response, not only its final line.

## Representation ablations and controls

| File / arm | Correct / 1,000 | Role |
|---|---:|---|
| `point4/predictions.jsonl.gz`: `full_question` | 763 | Cached full-question/full-record control |
| same file: `description_only` | 714 | Description-only query with the same cached records |
| `point5/predictions.jsonl.gz`: `full_reference` | 759 | Newly embedded full-record control |
| same file: `description_product` | 734 | Newly embedded description/product records |
| `point5/baseline_predictions.jsonl.gz`: `source_supported` | 763 | Frozen cached baseline matching point4's full-question arm |

Paths in this table are relative to `ablations/`. Each prediction arm includes the predicted CWE and twenty retrieved CVEs with similarities and CWE labels. The baseline ledger also preserves historical alternative exclusion controls under their original field names; they are not additional Table 4 arms. The 759 and 763 controls are distinct and must not be substituted for one another.

The release includes original prediction-freeze records, query/description input text, the reference mask and reference metadata, recorded preflight/validation files and full-1,000-question reports. Inputs for both point5 representation arms occupy consecutive halves of `description_inputs.jsonl.gz`. Large embedding arrays and model weights are not included; the release verifies the stored predictions and votes without recomputing embeddings. Broader corpus reconstruction artifacts are linked from the repository root.
