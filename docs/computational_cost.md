# Computational cost

These measurements were previously summarized in the manuscript's computational-cost section. The manuscript now uses that space for reviewer-requested class-level metrics. No new profiling or inference was performed for this move.

Corpus encoding is a one-time cost. Encoding and vector-array writing for 120,238 CVEs took 31.5 minutes on two RTX 4090 GPUs. Flat-index assembly from cached embeddings took 4.59 seconds on one CPU thread; warm index materialization took 1.90 seconds.

On the recorded 128-question profiling sample, median/95th-percentile raw-question-to-output latency was 0.166/0.252 seconds for CVE Label Aggregation and 6.327/10.041 seconds for Naive RAG with Gemma 4 E4B. The encoder and answer model occupied one RTX 4090 each, with peak sampled device memory of 8.82 and 21.31 GiB. The profiling client, including the encoder, peaked at 7.38 GiB host RSS.

The recorded [measurement results and sample identifiers](matched_online.json) include concurrency, failure/truncation counts, and the saved-neighbor equality diagnostic. Profiling covers the recorded 128-question sample; the main accuracy evaluation covers 1,000 questions. Neighbor-equality counts are recorded in the measurement file. CVE Label Aggregation does not make answer-generation requests.
