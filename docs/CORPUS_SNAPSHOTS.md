# Corpus dates, versions and retained evidence

| Source/component | Retained date or version | Interpretation |
|---|---|---|
| NVD JSON 2.0 annual feeds |2026-10-17, 19:46:01.905952–19:47:13.158247 UTC|Recorded feed retrieval interval; 25 annual feeds spanning 2002–2026.|
| CVE source tree |Upstream delta marker 2026-05-06T00:12:51.523Z|A retained source-tree marker, not a verified local acquisition date or upstream Git commit.|
| Current comparison's CWE data |No standalone CWE definition corpus|Voting labels come from structured NIST/CNA/ADP mappings attached to CVE records; record text may contain source-provided weakness information.|
| Historical CWE catalog lineage |CWE 4.20, 2026-04-30|Provenance retained for earlier mixed-corpus baselines; not a separate definition corpus used by the current mapped-CVE comparison.|

[corpus/snapshot_manifest.json](../corpus/snapshot_manifest.json) preserves the actual NVD receipt timestamps, source URLs, feed-modification metadata and compressed hashes, the deterministic CVE import digest, historical catalog evidence, benchmark checkout commit and known source hashes. The same snapshot record is included in release v1.0.0 under aggregation_artifacts/corpus_snapshots.json. This new repository file exposes existing evidence directly; it does not change any corpus, date, score or release archive.

## Acquisition records

The exact CVE local acquisition timestamp and source commit are null in the manifest. The retained delta marker identifies a source-tree version. NVD feed retrieval occurred on October 17; corpus assembly occurred on October 18. Retained input hashes identify the files used for the verified rebuild.

## Temporal scope

Records were filtered by the documented target-ID, description and source-relationship rules. No per-target disclosure-date or benchmark-creation date cutoff was applied. Source mappings use the recorded NVD/CNA/ADP priority.

## Rebuild inputs

The public historical-input archive begins with retained prepared CVE documents and structured source mappings rather than newly downloaded raw upstream feeds. verify_corpus.py reproduces all 120,238 filtered records with the frozen output hash, encoded text, labels and row order. Only machine-local path prefixes in unused mapping provenance were redacted. Model assets and embeddings are not bundled.
