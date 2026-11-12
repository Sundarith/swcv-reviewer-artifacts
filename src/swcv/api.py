"""Minimal OpenAI-compatible Chat Completions transport using the standard library."""
import json
import http.client
import time
from urllib import error, parse, request

class ChatClient:
    def __init__(self, base_url, model, api_key=None, timeout=1800, sleep=time.sleep,
                 token_parameter="max_tokens", omit_parameters=(), capabilities=()):
        parsed = parse.urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an HTTP(S) API base URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Keep credentials and query parameters out of base_url")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.sleep = sleep
        if token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("Unsupported output-token parameter")
        if not set(omit_parameters) <= {"temperature", "n"}:
            raise ValueError("Only temperature and n may be explicitly omitted")
        if not set(capabilities) <= {"tools", "logprobs"}:
            raise ValueError("Unknown API capability")
        self.token_parameter = token_parameter
        self.omit_parameters = frozenset(omit_parameters)
        self.capabilities = frozenset(capabilities)

    def require_capabilities(self, *names):
        missing = set(names) - self.capabilities
        if missing:
            raise ValueError("Provider configuration lacks required capabilities: " + ", ".join(sorted(missing)))

    def _redact(self, text):
        return text.replace(self.api_key, "[REDACTED]") if self.api_key else text

    def complete(self, messages, *, max_tokens=512, temperature=0, extra_body=None,
                 tools=None, tool_choice=None, logprobs=False, top_logprobs=None):
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        payload = {"model": self.model, "messages": messages, self.token_parameter: max_tokens, "temperature": temperature, "n": 1}
        for name in self.omit_parameters:
            payload.pop(name, None)
        if tools is not None:
            self.require_capabilities("tools")
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        elif tool_choice is not None:
            raise ValueError("tool_choice requires tools")
        if logprobs:
            self.require_capabilities("logprobs")
            payload["logprobs"] = True
            if top_logprobs is not None:
                if not isinstance(top_logprobs, int) or isinstance(top_logprobs, bool) or not 0 <= top_logprobs <= 20:
                    raise ValueError("top_logprobs must be an integer between 0 and 20")
                payload["top_logprobs"] = top_logprobs
        elif top_logprobs is not None:
            raise ValueError("top_logprobs requires logprobs=True")
        extra_body = extra_body or {}
        allowed = {"chat_template_kwargs", "reasoning_effort", "reasoning", "provider"}
        if not set(extra_body) <= allowed:
            raise ValueError("Only explicit thinking/template and provider-routing options are supported in extra_body")
        payload.update(extra_body)
        data = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        attempts = []
        started = time.monotonic()
        for attempt in range(1, 5):
            req = request.Request(self.url, data=data, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=self.timeout) as response:
                    status = response.status
                    raw = response.read().decode("utf-8", errors="replace")
            except error.HTTPError as exc:
                failure = {"attempt": attempt, "http_status": exc.code, "error": "HTTP request failed", "response": self._redact(exc.read().decode("utf-8", errors="replace"))}
            except (error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
                failure = {"attempt": attempt, "error": self._redact(str(exc))}
            else:
                attempts.append({"attempt": attempt, "http_status": status})
                result = {"request": payload, "attempts": attempts, "raw_response": self._redact(raw), "prediction": "", "finish_reason": None}
                # A returned 2xx answer is terminal, including malformed/empty/truncated answers.
                try:
                    body = json.loads(raw)
                    choice = body["choices"][0]
                    content = choice["message"].get("content")
                    if content is not None and not isinstance(content, str):
                        raise TypeError("Expected string content")
                    result.update(generation_id=body.get("id"), provider=body.get("provider"), prediction=content or "", finish_reason=choice.get("finish_reason"), response_model=body.get("model"), usage=body.get("usage"), system_fingerprint=body.get("system_fingerprint"), reasoning=choice["message"].get("reasoning_content") or choice["message"].get("reasoning"))
                    result["tool_calls"] = choice["message"].get("tool_calls")
                    result["logprobs"] = choice.get("logprobs")
                except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
                    result["returned_response_error"] = str(exc)
                result["latency_seconds"] = time.monotonic() - started
                return result
            attempts.append(failure)
            if attempt < 4:
                self.sleep([1, 2, 4][attempt - 1])
        return {"request": payload, "attempts": attempts, "prediction": "", "finish_reason": None, "request_error": "Request failed after four identical attempts", "latency_seconds": time.monotonic() - started}
