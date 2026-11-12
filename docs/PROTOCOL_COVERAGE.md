# Protocol coverage for the two RCM tables

This index covers eleven models and nine methods on each benchmark (198 selected conditions). It records original ledger hashes, exported-answer hashes, checkpoint-profile lineage and model asset manifests, saved chat templates, retained exact generation payloads and all observed decoding-setting variants. `saved_requests.jsonl.gz` preserves the exact retained prompts/messages and tool definitions; a request example alone is not the full protocol. Omitted request fields are marked OMITTED, not represented as explicit values. Server chat-template kwargs/context and checkpoint manifests are separately indexed. Remaining output/parser/context failures stay in the denominator.

## Historical dates

The package has server-supplied response creation windows for 40 conditions and recorded completion receipts for 187. The CSV distinguishes a condition receipt from an ancestor queue receipt. A queue completion is not an individual request timestamp, and a response window does not necessarily identify every reused answer's original generation phase. Exact original/replay/continuation boundaries are not fully recoverable from these retained structured receipts. Missing dates remain unavailable; no date was inferred from current filesystem modification time or invented from an experiment directory name. `historical_dates.json` preserves the evidence rather than asserting a full historical timeline.

## Variants and scoring

CTI-Bench uses the original identifier extraction and canonical reference label. CTIConnect retains official accepted alternatives with the declared uniform Unicode-hyphen normalization. The packaged pinned scorer and offline verifier reproduce every selected table cell. CRAG fallback, Fusion parser and full-evidence context continuations are distinct recorded variants. Corrected Phi CTIConnect A-RAG uses the nested-array parser recovery and scores35.92%; its original5.00% is historical. The CTI-Bench Phi A-RAG parser diagnostic remains unresolved and is not replaced by that separate correction. IRCoT and bounded-evidence A-RAG are excluded.

The original ten-reader NaiveRAG manifest is retained for that historical scope; it is not silently extended to the other178conditions. Current condition_index.json and per-condition exports cover the expanded comparison. Prompts and responses are not claims of independently reproduced inference; offline score reproduction is verified, historical regeneration still requires appropriate model assets/dependencies and corpus inputs.
