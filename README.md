# CVE Label Aggregation: reproducibility artifacts

Code and saved evaluation artifacts for **Retrieval-Based CWE Classification through CVE Label Aggregation**. This is the public reviewer repository; the development repository and its history are separate. It is not an official CTI-Bench or CTIConnect release.

## Reproduce the paper tables

Python 3.10 or newer is sufficient for offline verification; no GPU, model weights or API key is required. Download the two ZIPs from [release v1.0.0](https://github.com/Sundarith/swcv-reviewer-artifacts/releases/tag/v1.0.0) and verify their SHA-256 values against `release_manifest.json`.

Extract `rcm_results_and_protocols.zip` into a directory, then run:

```bash
cd PATH_TO_EXTRACTED_RESULTS
python verify.py
python aggregation_artifacts/verify_results.py
```

The verifier reproduces all 198reader table cells (11 models x 9 methods on each benchmark), CTIConnect k=10 aggregation, and the CTI-Bench aggregation/ranked-label/baseline results. The saved payloads and templates document retained prompts and decoding settings; this is offline score reproduction, not a fresh model rerun.

| Benchmark | Questions | Selected reader conditions | CVE Label Aggregation |
|---|---:|---:|---:|
| CTI-Bench RCM |1000|99|76.30%, k=20|
| CTIConnect RCM |290|99|55.52%, k=10|

CTIConnect uses official accepted alternatives and the declared Unicode-hyphen normalization. Its k=10 was selected on these same questions and is exploratory. Corrected Phi CTIConnect A-RAG scores 35.92%; the original 5.00% diagnostic is superseded in the selected table. CTI-Bench Phi A-RAG retains its separate parser qualification. IRCoT and bounded-evidence A-RAG are excluded.

## Rebuild the filtered corpus

Extract `historical_corpus_inputs.zip` separately and run from the results directory:

```bash
python verify_corpus.py PATH_TO_EXTRACTED_CORPUS_INPUTS
```

This decompresses the archived starting documents and structured source mappings, checks their hashes, and runs the supplied corpus builder. All 120,238filtered CVE documents reproduce the frozen input content/order and verified output hash. The starting document pool, not a fresh upstream download, is the rebuild boundary. Model weights and embeddings are not distributed.

## Code and protocol

- `src/swcv/`: portable voting, saved-result reproduction and configurable API inference.
- `configs/`, `prompts/`, `examples/`: recorded settings and provider-neutral examples; keys belong in environment variables.
- `docs/condition_index.json`: model asset/checkpoint lineage, exact retained decoding variants, template references and source/export hashes for every selected condition.
- `docs/PROTOCOL_COVERAGE.md` and `protocol_coverage.csv`: historical evidence and missing-date disclosures.
- Release archives: 153,889 retained payloads, hashed chat templates, ordered aggregation retrieval, selected IDs, predictions, scorer/gold, corpus-building inputs and offline verifiers.

Historical original/reused/continuation phase dates are not fully recoverable. Response creation windows exist for 40 conditions and completion receipts for 187; a queue receipt is not an individual generation date. Exact CVE acquisition time/commit is unavailable. Evaluation remains retrospective, with shared upstream annotation sources and no per-target temporal cutoff. Cross-benchmark results do not establish an untouched independent evaluation.

## Portable code checks

```bash
PYTHONPATH=src python -m unittest discover -s tests
PYTHONPATH=src python -m swcv.reproduce --run swcv_exact_baseline
PYTHONPATH=src python -m swcv.reproduce --run swcv_union_filtered
```

The two `swcv_*`conditions above are preserved historical overlap-sensitivity checks (120,671 and 120,242 references), distinct from the current 120,238-reference final corpus in the release. See [third-party notices](THIRD_PARTY_NOTICES.md) for source attribution and license scope.
