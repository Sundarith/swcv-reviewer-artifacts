# Recorded regeneration source

This source snapshot complements saved-score verification. It includes our experiment adapters for query generation/HyDE/Fusion/EtR, reranking, LinearRAG, Local CRAG, agent retrieval/reading, stronger-baseline fitting and paired statistics. See source_manifest.json for original/public source hashes. Machine-specific workspace/asset prefixes are replaced; original algorithms remain intact.

## Materialize the recorded layout

    python regeneration/stage_sources.py --workspace /YOUR/EXPERIMENT_WORKSPACE --asset-root /YOUR/LOCAL_ASSET_ROOT

The staging command creates new source files only and refuses to overwrite existing files. It never loads a model or executes inference. Recorded relative directory layout matters because runners resolve neighboring assets and input files.

## Stronger baseline fitting

Entry point after staging:

    python /YOUR/EXPERIMENT_WORKSPACE/analysis/ctibench1000_stronger_baselines_20261110/run.py

The runner includes nearest/majority/cosine/inverse-distance/softmax voting, reference-only temperature selection, BM25, embedding ridge and supervised TF-IDF multinomial naive Bayes. stronger_baseline_inputs/protocol.json and temperature_selection.json record original input hashes, dependency/version information when retained, and selection outputs. Supply the exact frozen documents, document/query vectors, query ledger, gold-scoring input, neighborhood ledger and exclusion mask at the paths declared by that runner. The public corpus input release supplies prepared documents/mappings; model assets and vector arrays are not distributed. Generating compatible embeddings is a separate GPU stage; never replace them with arbitrary vectors.

## Retrieval methods

The sources preserve the actual experiment adapters rather than substitute implementations. Main entry points are prepare_retrieval.py/query_retrieval.py, generate_queries.py, prepare_reranked_windowed.py, prepare_linear_native.py and prepare_crag.py under the CTI-Bench code directory, plus the CTIConnect full-adapter directory and declared recovery code. Their recorded top-level constants/imports specify required upstream/prepared artifacts and external method packages. Reader payloads, model/template manifests and decoding settings are in release v1.0.0 and docs/condition_index.json. Use the recorded condition-specific settings for these mapped-CVE task adaptations.

## Runtime and input requirements

These runners require the archived inputs, recorded external packages/model checkpoints and experiment-specific environments. Some source modules execute work on import; inspect and provision dependencies before invoking them. CPU stronger baselines require NumPy, scikit-learn, bm25s and threadpoolctl; neural/retrieval stages additionally require the recorded torch/transformers/embedding runtimes and suitable GPUs. Do not upgrade a historical environment silently. Source syntax and staging are verified; no training or model inference was rerun for publication.

Third-party packages retain their licenses; see ../THIRD_PARTY_NOTICES.md. No model weights, credentials, private serving controllers or original Git history are included.
