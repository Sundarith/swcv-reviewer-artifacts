# CTIConnect aggregation neighborhood sensitivity

All 290 questions and the primary 119,889-reference corpus are retained. The extended sweep tests k=1,5,10,20,30,50. This is retrospective analysis, not held-out configuration selection. The primary k=10 result remains unchanged.

| k | Official accepted-alternative F1 (%) |
|---:|---:|
|1|45.86|
|5|51.03|
|10|55.52|
|20|54.48|
|30|53.10|
|50|54.83|

Cached query and document embeddings were ranked on both local GPUs, using FP32 matrix-vector products with TF32 disabled, stable corpus-row ordering and lexical label ties. The first twenty IDs, scores, labels and record contents exactly match the previous retrieval for every question. Each neighbor contributes its full cosine similarity to every stored CWE label. All six complete prediction ledgers were frozen before scoring. No answer-model requests were made. Other retrieval-method results were not changed.

`ranked50.jsonl.gz` supplies the ordered neighbors, similarities and structured labels. `results.json` and `protocol.json` contain scores, input fingerprints and configuration. Run `python verify.py` to rebuild votes and independently reproduce official scores using the parent package's released gold and scorer.
