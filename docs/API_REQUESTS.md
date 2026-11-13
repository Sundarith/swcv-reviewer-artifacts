# API requests and configuration

## Shared interface

All endpoints supported by this release use the OpenAI-compatible **Chat Completions protocol**. One client sends messages; configuration selects the endpoint/model and optional provider-specific controls. A provider with a different native protocol requires an adapter. This release does not implement GraphGen's full backend list.

The interface is:

```python
from swcv.api import ChatClient

client = ChatClient(base_url=endpoint, model=exact_model_id, api_key=key_from_environment)
result = client.complete(messages, max_tokens=512, temperature=0)
```

The API key is sent in an Authorization header and is not stored in the run's request payload or manifest. The wire request is:

```http
POST {base_url}/chat/completions
Authorization: Bearer {key supplied through the environment, if required}
Content-Type: application/json
```

```json
{
  "model": "EXACT_MODEL_IDENTIFIER",
  "messages": [
    {"role": "system", "content": "EXACT_SYSTEM_PROMPT"},
    {"role": "user", "content": "EXACT_BENCHMARK_QUESTION_AND_FORMATTED_EVIDENCE"}
  ],
  "max_tokens": 512,
  "temperature": 0,
  "n": 1
}
```

Top-p/seed, output schema, allowlists and tools are omitted. The current Naive RAG adaptation uses twenty retrieved CVE records formatted with the CTIConnect prompt; each reference body supplies its first 600 characters. The original full benchmark question is preserved. For closed-book, only the original question is supplied as the user message, with the separately versioned closed-book system prompt.

## Model configuration and saved outputs

`configs/models/current_local_roster.json` lists the exact model identifiers, revisions, context capacities and serving layouts for the current filtered CTI-Bench1000 NaiveRAG20 experiment. Its common request settings are temperature0, max_tokens512, n=1 and sixteen API workers.

For another endpoint, choose the exact model identifier and supported options in a provider configuration. Requested and returned model identifiers are saved separately, together with raw responses, generation IDs, token usage and finish reasons when available. Each run records its input, prompt, code and configuration hashes.

The portable transport has mock-based protocol tests. The supported protocol is OpenAI-compatible Chat Completions; endpoint-specific parameter support must be checked for the selected model.

## Configuration and provider controls

See `examples/README.md` and `configs/providers/` for examples. The OpenRouter example demonstrates `provider.only`, `allow_fallbacks: false`, and `require_parameters: true`. Provider-specific options in `extra_body` are forwarded unchanged and recorded. The client does not silently translate, omit or replace unsupported options. Provider rejection is an error record, never permission to change settings.

Successful returned answers are preserved. No answer is repaired, retried because it is incorrect, or replaced through JSON fallback. Four identical request attempts with 1/2/4-second backoff cover request/HTTP failures only. Final failures remain empty predictions. This is our declared evaluation policy, independent of GraphGen's policies.

## Organization reference

GraphGen's useful pattern is the separation of API configuration (`model`, `base_url`, key/backend), shared client implementation, generation parameters and runnable examples:

- https://github.com/Sundarith/GraphGen-v.1/blob/main/.env.example
- https://github.com/Sundarith/GraphGen-v.1/blob/main/graphgen/models/llm/api/http_client.py
- https://github.com/Sundarith/GraphGen-v.1/blob/main/graphgen/models/llm/api/openai_client.py

Our implementation is independent. No GraphGen code was copied. We adopt the configuration pattern, preserving our experiment-specific request and raw-output policies.
