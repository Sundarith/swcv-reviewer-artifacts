# Baselines

A portable OpenAI-compatible generation client is available:

```bash
PYTHONPATH=src python -m swcv.inference --help
```

See [API usage](../docs/API_USAGE.md) for endpoint/model configuration, exact input format, top-20 evidence handling, exception-only retries and saved response provenance. Naive RAG and closed-book transport are supported. SWCV voting remains in `src/swcv/`.

The client consumes prepared questions and reference records. Endpoint-specific parameters and context capacity should match the selected model.

Recorded additional-method and stronger-baseline regeneration source is now provided in [regeneration](../regeneration/README.md), with original-source hashes, staging instructions and explicit dependency/input boundaries.
