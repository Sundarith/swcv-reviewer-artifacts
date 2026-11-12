"""Generate raw answers from prepared, gold-free inputs; never score during inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from .providers import create_client


def prepare(inputs, prompt, method="naive_rag", top_k=20):
    prepared = []
    seen = set()
    for item in inputs:
        key = str(item["question_id"])
        if key in seen:
            raise ValueError("Duplicate question_id: " + key)
        seen.add(key)
        if not isinstance(item["question"], str) or not item["question"].strip():
            raise ValueError("question must be nonempty original benchmark text")
        record = {"question_id": key}
        if method == "naive_rag":
            refs = item["references"]
            if len(refs) != top_k:
                raise ValueError(f"{key}: expected exactly {top_k} references; never silently clip")
            candidates = "\n\n".join(f'[{r["doc_id"]}] {r["title"]}\n{r["text"][:600]}' for r in refs)
            user = prompt["PROMPT"].format(question=item["question"], candidates=candidates)
            record["references"] = refs
        else:
            user = item["question"]
        record["messages"] = [{"role": "system", "content": prompt["SYSTEM"]}, {"role": "user", "content": user}]
        prepared.append(record)
    if not prepared:
        raise ValueError("No input questions")
    return prepared


def main():
    # Read non-secret configuration first; explicit flags override these defaults.
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    preliminary, _ = config_parser.parse_known_args()
    configuration = json.loads(preliminary.config.read_text()) if preliminary.config else {}
    allowed_configuration = {"backend", "base_url", "model", "api_key_env", "method", "top_k", "workers", "max_tokens", "temperature", "timeout", "extra_body", "token_parameter", "omit_parameters", "capabilities"}
    if not isinstance(configuration, dict) or not set(configuration) <= allowed_configuration:
        config_parser.error("Config contains unsupported fields; API keys must be supplied through the environment")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, help="Non-secret JSON connection/generation settings")
    ap.add_argument("--backend", default="openai_compatible", choices=["openai_compatible"])
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="New run directory; must not exist")
    ap.add_argument("--base-url", default=os.environ.get("SWCV_API_BASE_URL"))
    ap.add_argument("--model", default=os.environ.get("SWCV_API_MODEL"))
    ap.add_argument("--api-key-env", default="SWCV_API_KEY")
    ap.add_argument("--prompt", type=Path, default=None)
    ap.add_argument("--method", choices=["naive_rag", "closed_book"], default="naive_rag")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--token-parameter", choices=["max_tokens", "max_completion_tokens"], default="max_tokens")
    ap.add_argument("--extra-body", type=Path, help="Explicit provider thinking/template options JSON")
    ap.set_defaults(**{k: v for k, v in configuration.items() if k != "extra_body"})
    if os.environ.get("SWCV_API_BASE_URL"):
        ap.set_defaults(base_url=os.environ["SWCV_API_BASE_URL"])
    if os.environ.get("SWCV_API_MODEL"):
        ap.set_defaults(model=os.environ["SWCV_API_MODEL"])
    args = ap.parse_args()
    if args.backend != "openai_compatible":
        ap.error("This release supports the OpenAI-compatible Chat Completions backend")
    if not args.base_url or not args.model:
        ap.error("Specify --base-url and --model or SWCV_API_BASE_URL/SWCV_API_MODEL")
    if args.workers < 1 or args.max_tokens < 1 or args.top_k < 1:
        ap.error("workers/max-tokens/top-k must be positive")
    client = create_client({**configuration, "backend": args.backend, "base_url": args.base_url,
                            "model": args.model, "api_key_env": args.api_key_env,
                            "timeout": args.timeout, "token_parameter": args.token_parameter})
    if args.prompt is None:
        args.prompt = Path("prompts") / ("naive_rag_cticonnect.json" if args.method == "naive_rag" else "closed_book_cticonnect.json")
    prompt = json.loads(args.prompt.read_text())
    extra = json.loads(args.extra_body.read_text()) if args.extra_body else configuration.get("extra_body", {})
    if not set(extra) <= {"chat_template_kwargs", "reasoning_effort", "reasoning", "provider"}:
        ap.error("extra-body may only contain template, reasoning or provider-routing options")
    # Ignore gold and all unrelated input fields rather than transmitting them.
    items = prepare([json.loads(l) for l in args.inputs.read_text().splitlines() if l.strip()], prompt, args.method, args.top_k)
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "method": args.method, "backend": args.backend, "model_requested": args.model, "base_url": args.base_url, "api_key_env": args.api_key_env, "workers": args.workers, "top_k": args.top_k, "max_tokens": args.max_tokens, "temperature": args.temperature, "extra_body": extra, "thinking_mode": "Provider-controlled; explicit options recorded, not universally enforced", "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest() if args.config else None, "question_count": len(items), "inputs_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(), "prompt_sha256": hashlib.sha256(args.prompt.read_bytes()).hexdigest(), "code_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in ["api.py", "inference.py"]}, "retry_policy": "Four identical attempts on request/HTTP failures only, backoff 1/2/4 seconds", "returned_answers": "No retries, constraints, repair, or fallback", "gold_accessed_for_scoring": False, "context_capacity": "Provider-specific; no client clipping or automatic budget reduction", "status": "running"}
    manifest.update(token_parameter=client.token_parameter, omit_parameters=sorted(client.omit_parameters), capabilities=sorted(client.capabilities))
    manifest["code_sha256"]["providers.py"] = hashlib.sha256((Path(__file__).parent / "providers.py").read_bytes()).hexdigest()
    def save(name, value):
        target = args.out / name
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(value, indent=2) + "\n")
        temp.replace(target)
    save("manifest.json", manifest)
    (args.out / "prepared.jsonl").write_text("".join(json.dumps(q) + "\n" for q in items))
    def answer(item):
        result = client.complete(item["messages"], max_tokens=args.max_tokens, temperature=args.temperature, extra_body=extra)
        return {"question_id": item["question_id"], "id": item["question_id"], **result}
    try:
        with (args.out / "predictions.jsonl").open("x") as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
            for count, result in enumerate(pool.map(answer, items), 1):
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                save("progress.json", {"done": count, "total": len(items)})
        output = args.out / "predictions.jsonl"
        save("prediction_freeze.json", {"sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "count": len(items), "gold_accessed_for_scoring": False})
        manifest["status"] = "completed"
    except BaseException:
        manifest["status"] = "failed_or_interrupted"
        raise
    finally:
        save("manifest.json", manifest)

if __name__ == "__main__":
    main()
