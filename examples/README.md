# Configure an API and run Naive RAG

The client keeps connection settings separate from method inputs, following the useful configuration pattern in GraphGen. The examples show how to select an endpoint and model while preserving a recorded experiment configuration.

1. Copy `configs/providers/openai_compatible.example.json`, `openrouter.example.json`, or `self_hosted.example.json` to your own configuration file.
2. Replace model/endpoint/provider placeholders with your actual settings. API keys remain in the named environment variable.
3. Prepare question JSONL with exactly twenty ordered reference records per question, as described in `docs/API_USAGE.md`.
4. Run from the repository root:

```bash
bash examples/run_naive_rag20.sh YOUR_CONFIG.json YOUR_INPUTS.jsonl outputs/YOUR_NEW_RUN
```

Explicit command-line flags override environment endpoint/model overrides; those override the JSON configuration. JSON configuration overrides other built-in defaults. `.env.example` documents variable names but is not automatically loaded.

OpenRouter's example pins a provider and disables routing fallback. It intentionally contains placeholders, since provider/model availability and supported parameters must be checked for the selected endpoint. Additional provider options must be supported by the selected endpoint.

See `configs/models/current_local_roster.json` for the exact model identifiers used in the current local queue. Change `model` to select an LLM; keep each experiment's actual runtime, prompt, decoding and corpus configuration separate. `docs/API_REQUESTS.md` explains the connection to recorded experiments.
