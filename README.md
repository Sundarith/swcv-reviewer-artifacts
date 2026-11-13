# CVE Label Aggregation

Code and evaluation artifacts for **Retrieval-Based CWE Classification through CVE Label Aggregation**. The method retrieves similar CVE records and combines their stored CWE labels using cosine similarity, without an answer-generating language model.

## Overview

The evaluation covers eleven language models across nine closed-book and retrieval-based configurations on two root-cause mapping benchmarks.

| Benchmark | Questions | Reference CVEs | Label aggregation F1 (%) | k |
|---|---:|---:|---:|---:|
| CTI-Bench RCM | 1,000 | 120,238 | 76.30 | 20 |
| CTIConnect RCM | 290 | 119,889 | 55.52 | 10 |

Both benchmarks retain their released questions and labels. CTIConnect scoring credits its accepted alternative CWE labels. The evaluated questions were used to select aggregation neighborhood sizes; [selection settings](configs/evaluation_selection.json) record the configurations and sensitivity sweeps.

## Quickstart

Python 3.10 or newer is sufficient for the saved-result checks below; no GPU, model weights or API key is required.

```bash
git clone https://github.com/Sundarith/swcv-reviewer-artifacts.git
cd swcv-reviewer-artifacts
python -m pip install -e .

# Reconstruct the current CTI-Bench aggregation result: 763/1,000 (76.30%).
swcv-reproduce --run swcv_union_filtered

# Verify the saved language-model scores and class diagnostics on both benchmarks.
python supporting_results/reader_metrics1000/verify.py
python supporting_results/cticonnect290/verify.py
```

The [reviewer walkthrough](docs/REPRODUCING_PAPER.md) maps every paper table and analysis to its artifacts and provides metric/statistical rebuild commands.

## Evaluation and supporting results

| Analysis | Artifacts and instructions |
|---|---|
| CTI-Bench: 99 model conditions and eight classifiers | [Metrics, predictions and confusion summaries](supporting_results/reader_metrics1000/README.md) |
| CTIConnect: 99 model conditions and five controls | [Official scores, canonical class metrics, paired tests and overlap sensitivity](supporting_results/cticonnect290/README.md) |
| Label-selection baselines | [Eight classifier metrics and square confusion matrices](supporting_results/class_metrics1000/README.md) |
| Representation and prompt sensitivity | [Frozen inputs, prediction ledgers and verifiers](supporting_results/README.md) |
| Paired CTI-Bench comparisons | [Bootstrap intervals and significance tests](supporting_results/paired_statistics1000/README.md) |
| Neighborhood size | [CTIConnect six-setting sweep](supporting_results/cticonnect290/neighborhood_sensitivity/README.md) |
| Corpus overlap | [Screening rules, exclusion masks and sensitivity results](docs/LEAKAGE_CONTROLS.md) |
| Computational cost | [Corpus encoding, latency and memory measurements](docs/computational_cost.md) |

Scores use the recorded identifier extraction and benchmark-specific scoring rules. Class macro/micro metrics and mean per-question F1 are labeled separately. Model revisions, prompts, decoding settings, generation records and implementation variants are indexed in [protocol coverage](docs/PROTOCOL_COVERAGE.md).

For a new model run, see [API usage](docs/API_USAGE.md) and [examples](examples/README.md). The [recorded regeneration source](regeneration/README.md) provides additional retrieval adapters and baseline-fitting code with their input and dependency requirements.

## Data and downloads

[Release v1.0.0](https://github.com/Sundarith/swcv-reviewer-artifacts/releases/tag/v1.0.0) contains the larger archives:

- `rcm_results_and_protocols.zip`: all 198 selected model-condition exports, retained request payloads, prompts/templates, ordered retrieval, scoring inputs and verifiers.
- `historical_corpus_inputs.zip`: prepared starting CVE documents and structured mappings for corpus reconstruction.

Check downloads against [release_manifest.json](release_manifest.json). Later supporting analyses are included in the current repository tree. The [walkthrough](docs/REPRODUCING_PAPER.md#5-inspect-or-rebuild-the-archived-corpus) gives extraction and corpus-rebuild commands; [corpus snapshots](docs/CORPUS_SNAPSHOTS.md) document source dates, versions and mappings. Model weights and embedding arrays are external assets.

## Repository layout

```text
src/swcv/          Voting, saved-result reproduction and API client
configs/           Evaluation, model and provider settings
prompts/           Recorded prompt templates
benchmarks/        Question identifiers and scoring labels
artifacts/         Saved aggregation retrieval and predictions
supporting_results/ Model answers, metrics, ablations, tests and overlap analyses
regeneration/      Recorded retrieval and baseline-fitting source
examples/          API configuration and execution examples
docs/              Walkthrough, protocols, corpus records and cost measurements
tests/             Portable code and API transport tests
```

## Source attribution

Benchmark data, adapted code and model assets retain their source-specific terms. See [third-party notices](THIRD_PARTY_NOTICES.md) for CTI-Bench, CTIConnect and CVE/NVD/MITRE attribution. This repository accompanies our experiments; it is separate from the official benchmark releases.
