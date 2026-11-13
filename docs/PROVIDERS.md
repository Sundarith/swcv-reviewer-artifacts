# Provider configuration

Use one API client for compatible Chat Completions endpoints. Changing the endpoint, model identifier, and credential environment variable leaves prepared questions, retrieved evidence, and evaluation code unchanged.

```bash
swcv-infer --config configs/providers/openrouter.example.json \
  --inputs YOUR_PREPARED_INPUTS.jsonl --out outputs/YOUR_RUN
```

Copy the example and enter the exact endpoint and model identifier. Supply credentials through the configured environment variable. The examples cover compatible hosted endpoints, OpenRouter, and self-hosted endpoints.

## Configuration fields

| Field | Purpose |
|---|---|
| `backend` | `openai_compatible`: HTTP Chat Completions protocol |
| `base_url` | API root; the client appends `/chat/completions` |
| `model` | Exact identifier requested from the provider |
| `api_key_env` | Name of the environment variable holding the credential |
| `max_tokens` | Output-token allowance used by the experiment |
| `token_parameter` | Request field: `max_tokens` or `max_completion_tokens` |
| `temperature` | Explicit sampling temperature |
| `omit_parameters` | Explicit list of unsupported optional fields to omit: `temperature`, `n` |
| `capabilities` | Explicitly enabled interfaces: `tools`, `logprobs` |
| `extra_body` | Supported provider-specific request options |

See [the alternate token-field example](../configs/providers/completion_tokens.example.json). Request settings are recorded in the manifest and the exact transmitted payload is retained with each prediction. A different parameter name does not guarantee identical generation behavior across providers; use the selected model's documented token-budget semantics when defining the experiment.

## Architecture integration

Retrievers supply evidence independently of the API client. Architecture implementations use the same client for intermediate and final generation:

```python
from swcv.providers import create_client

client = create_client(configuration)
result = client.complete(messages, max_tokens=512, temperature=0)
answer = result["prediction"]
```

A method that requires tool calls or token probabilities declares those requirements before execution:

```python
client.require_capabilities("tools")
result = client.complete(messages, tools=search_tools, tool_choice="auto")
```

For probability-based methods, enable `logprobs` and request `logprobs=True, top_logprobs=20`. The result preserves native `tool_calls` and `logprobs` alongside the original response. Configuration declares intended support; verify the selected endpoint with a synthetic interface check before a full run. The client does not emulate missing provider features.

## Outputs

Each response has the same envelope: exact request, raw response, prediction text, requested and returned model identifiers, generation ID, usage, finish reason, tool calls, token probabilities, attempt records, and latency. Retrieval and scoring do not depend on provider selection. Inference writes the complete prediction ledger and its hash before evaluation.

The current backend supports the compatible Chat Completions protocol. Other native protocols require separate transport adapters implementing the same result envelope. Contract tests use a mock HTTP server and require no paid API calls.
