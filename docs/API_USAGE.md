# Run through an OpenAI-compatible API

The client supports the HTTP Chat Completions protocol, not every provider's native API. Set the complete API base URL (usually ending in `/v1`); the client appends `/chat/completions`. OpenAI-compatible hosted providers and compatible self-hosted servers can use the same client. Native APIs using a different protocol need an adapter. Compatibility has been tested with a local mock server; no live commercial-provider validation is claimed.

## Configuration

See [provider configuration](PROVIDERS.md) for the shared client interface and supported connection fields. Endpoints using `max_completion_tokens` can select it with `token_parameter` in the configuration or `--token-parameter max_completion_tokens`.

Read credentials from environment variables; never put an API key in a config, command-line argument or committed file:

```bash
export SWCV_API_BASE_URL='https://YOUR_PROVIDER/v1'
export SWCV_API_MODEL='YOUR_EXACT_MODEL_IDENTIFIER'
# Supply SWCV_API_KEY using your environment or secret manager.
```

A local server that does not require authorization can leave `SWCV_API_KEY` unset. `--api-key-env` selects another environment variable. The manifest records only its name, not its value. Do not put credentials in the endpoint URL.

## Prepared input schema

Supply JSONL with one question per line:

```json
{"question_id":"0","question":"The original benchmark question verbatim","references":[{"doc_id":"reference identifier","title":"reference title","text":"Original reference text"}]}
```

The example illustrates the schema only. Actual top-20 runs require exactly **20 references** per question, already ordered by the intended retriever. Provide them from the correct versioned corpus and exclusion manifest. This client does not build a corpus or retrieve references. The included SWCV retrieval logs contain IDs/scores/labels, not full reference text; full new inference requires separately obtaining the corresponding source records.

The exact CTIConnect Naive RAG prompt in `prompts/naive_rag_cticonnect.json` formats each record as `[doc_id] title`, followed by the first 600 characters of its text, separated by blank lines. This is the explicit evidence format used in the current adaptation; the client does not silently discard references or shorten the complete prompt to fit a context window. Gold fields are ignored and never included in the request.

```bash
PYTHONPATH=src python -m swcv.inference \
  --inputs YOUR_PREPARED_1000_QUESTIONS.jsonl \
  --out outputs/YOUR_UNIQUE_RUN_ID \
  --workers 16 --max-tokens 512 --temperature 0
```

Or install with `pip install -e .` and invoke `swcv-infer`. An existing output directory is rejected, avoiding accidental overwrite or answer retries. A different output directory constitutes a new run.

Closed-book generation is also supported with `--method closed_book --prompt prompts/closed_book_cticonnect.json`. That prompt must include the intended `SYSTEM`; inference sends the original question as the user message. Do not use the Naive RAG system prompt for a closed-book comparison.

## Outputs and behavior

- `manifest.json`: endpoint, requested model, UTC start time, actual settings, code/input/prompt hashes and status.
- `prepared.jsonl`: exact messages and references, without gold.
- `predictions.jsonl`: question ID, original prediction, complete raw response, requested payload, response-model identity when provided, finish reason, usage, fingerprint, errors, latency and attempt records. `id`/`prediction` support a CTIConnect-style scorer; question IDs must match the benchmark adapter.
- `progress.json`: count of returned records written.
- `prediction_freeze.json`: output hash/count after the full ledger is saved; inference never opens gold for scoring.

Defaults are temperature 0, max_tokens 512, n=1 and 16 client workers. Top-p and seed are omitted. No JSON schema, output allowlist, repair or fallback is sent. Four identical attempts with 1/2/4-second waits apply only to request/HTTP failures. Empty, wrong, malformed and output-limit responses returned with HTTP success are terminal and remain in the ledger. Exhausted failures remain empty predictions and must be retained in the scoring denominator. A provider context-limit rejection is logged as an HTTP failure; the client never silently shortens the input or changes the output allowance.

## Provider-specific options

Pass supported provider options through `--extra-body OPTIONS.json` or the `extra_body` object in a configuration file. These options are saved in the run manifest. Check the selected endpoint's supported parameters and context capacity before running.

## Connection examples and experiment mapping

See [examples](../examples/README.md) for GraphGen-style connection/generation configurations, and [API requests](API_REQUESTS.md) for the exact request shape, generation settings and saved outputs.
