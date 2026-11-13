# Reviewer walkthrough: reproduce the current paper

This guide maps **Retrieval-Based CWE Classification through CVE Label Aggregation** to its saved artifacts. Offline checks use Python 3.10 or newer; they require no model weights, GPU or API key.

## 1. Obtain the current repository

```bash
git clone https://github.com/Sundarith/swcv-reviewer-artifacts.git
cd swcv-reviewer-artifacts
```

The default branch contains the current supporting results. [Release v1.0.0](https://github.com/Sundarith/swcv-reviewer-artifacts/releases/tag/v1.0.0) preserves the larger request/prompt and corpus-input archives; later supporting analyses also live in the current repository tree. Downloading only the release ZIPs omits these later additions.

## 2. Check the saved paper results

Run from the repository root:

```bash
PYTHONPATH=src python -m unittest discover -s tests
PYTHONPATH=src python -m swcv.reproduce --run swcv_union_filtered
python supporting_results/verify.py
python supporting_results/reader_metrics1000/verify.py
python supporting_results/cticonnect290/verify.py
python supporting_results/cticonnect290/neighborhood_sensitivity/verify.py
```

The primary CTI-Bench aggregation result is **763/1,000 (76.30%), k=20**, from 120,238 references. The primary CTIConnect aggregation result is **161/290 (55.52%), k=10**, from 119,889 references. The first aggregation command reports the `full1000` result; [earlier exclusion conditions](../reproduce/README.md#earlier-exclusion-condition) are retained separately.

The other checks validate saved answers and hashes, the representation ablations and six Muse sensitivity runs, class/confusion diagnostics, overlap-condition predictions and the extended CTIConnect k sweep. These commands check existing results rather than making model requests.

## 3. Locate each paper table and analysis

| Paper item | Saved evidence |
|---|---|
| Table 1: CTI-Bench language-model comparison | [99 conditions and metrics](../supporting_results/reader_metrics1000/README.md), [model/method score index](full1000_comparison.json) |
| Table 2: CTIConnect language-model comparison | [99 conditions, official scores and diagnostics](../supporting_results/cticonnect290/README.md) |
| Table 3: CTI-Bench label-selection baselines | [Eight classifier metrics and predictions](../supporting_results/class_metrics1000/README.md), [baseline fitting source](../regeneration/README.md#stronger-baseline-fitting) |
| Table 4: representation ablations | [Frozen inputs and matched-control predictions](../supporting_results/README.md#representation-ablations-and-controls) |
| Table 5: cross-benchmark class metrics | [CTI-Bench macro/micro metrics](../supporting_results/reader_metrics1000/metrics.json), [CTIConnect canonical-label metrics](../supporting_results/cticonnect290/metrics.json) |
| Section 5.1: prompt/run sensitivity | [Six Muse conditions](../supporting_results/README.md#muse-sensitivity) |
| Section 5.2: paired comparisons | [CTI-Bench reader tests](../supporting_results/paired_statistics1000/README.md), [CTIConnect tests](../supporting_results/cticonnect290/README.md#paired-uncertainty-and-significance); majority results also appear in the release's `aggregation_artifacts/` |
| Section 5.3: neighborhood sensitivity | [Selection settings](../configs/evaluation_selection.json), [extended CTIConnect sweep](../supporting_results/cticonnect290/neighborhood_sensitivity/README.md); CTI-Bench sweep/ranked-label records are in the release's `aggregation_artifacts/` |
| Section 5.5: minority classes and confusion | [CTI-Bench diagnostics](../supporting_results/reader_metrics1000/README.md), [CTIConnect diagnostics](../supporting_results/cticonnect290/README.md#metrics-and-confusion-summaries) |
| Section 5.6: overlap sensitivity | [Primary and broader filtering rules/masks](LEAKAGE_CONTROLS.md), [CTIConnect pairs and predictions](../supporting_results/cticonnect290/overlap/) |
| Supporting computational cost | [Encoding, latency and memory measurements](computational_cost.md) |

Class-level macro/micro metrics differ from mean per-question identifier-set F1. CTIConnect's main scores credit accepted alternatives; its class diagnostics use canonical labels. Use the metric names and averaging rules in the corresponding package.

## 4. Optionally rebuild derived metrics and statistics

Class metrics and confusion summaries can be rebuilt with standard-library Python:

```bash
python supporting_results/class_metrics1000/build.py
python supporting_results/reader_metrics1000/compute.py
python supporting_results/reader_metrics1000/verify.py
```

The following statistical rebuilds additionally require **NumPy**. The recorded CTIConnect statistical environment is Python 3.13.12 with NumPy 2.2.6; see [statistics_environment.json](../supporting_results/cticonnect290/statistics_environment.json). Use a separate environment if installing dependencies, for example:

```bash
python -m venv .venv-statistics
. .venv-statistics/bin/activate
python -m pip install numpy==2.2.6
python supporting_results/paired_statistics1000/compute.py
python supporting_results/cticonnect290/compute.py
python supporting_results/cticonnect290/verify.py
```

These builders rewrite derived metric/statistical files from the saved prediction ledgers. Run them in a disposable clone if you want to keep your checkout untouched. They do not generate new model answers. Seeds, resampling counts, comparison families and input hashes are retained with the statistical results.

## 5. Inspect or rebuild the archived corpus

Download both release ZIPs and check their SHA-256 values against [release_manifest.json](../release_manifest.json). Extract `rcm_results_and_protocols.zip` and `historical_corpus_inputs.zip` into separate directories. From the extracted results directory:

```bash
python verify.py
python aggregation_artifacts/verify_results.py
python verify_corpus.py PATH_TO_EXTRACTED_CORPUS_INPUTS
```

This reconstructs the frozen prepared 120,238-record corpus. [Corpus snapshots](CORPUS_SNAPSHOTS.md) and [overlap controls](LEAKAGE_CONTROLS.md) describe source versions, mappings and benchmark-specific exclusions.

## 6. Run a new model experiment

[API usage](API_USAGE.md) gives the prepared-input schema, endpoint configuration, generation settings and output records for the portable Naive RAG/closed-book client. [Examples](../examples/README.md) show configurable API calls. [Recorded regeneration source](../regeneration/README.md) covers the additional retrieval methods and baseline fitting, with its required external assets and environments. Saved-score verification and new inference are separate workflows.

[Third-party notices](../THIRD_PARTY_NOTICES.md) identify benchmark/scorer provenance and applicable source terms.
