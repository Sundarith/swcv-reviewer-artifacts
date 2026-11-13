# Release verification — November 12, 2026

- Public repository and release pages returned HTTP 200 without an Authorization header or cookie jar.
- Both release ZIPs were downloaded without credentials; their byte counts and SHA-256 values matched release_manifest.json.
- An HTTPS clone with credential helpers disabled passed all 15 portable-code tests and both historical offline vote checks.
- Before upload, the sanitized main package reproduced all 198 selected table scores and both aggregation results; the corpus verifier rebuilt all 120,238 frozen documents with unchanged encoded text, labels and row order.
- Only machine-local path prefixes in unused structured mapping provenance were redacted. No historical inference was rerun and no original frozen input was overwritten.

See public_access_verification.json for access/hash evidence and sanitization_manifest.json for transformation scope. The recorded checks cover public access, file hashes, corpus reconstruction and saved-score reproduction.
