"""Run one frozen CTIConnect condition with the existing method implementation."""
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import requests
from transformers import AutoTokenizer, TokenizersBackend
from common import CODE, CONNECT, ELIGIBLE, FIXED, MASK, N, OUT, ROOT, SUITE, questions, read, rows, save
from api import ChatClient
from host_lifecycle import freeze, sha, start_host, stop_host, ticks
from reader import score

PYTHON = '__LOCAL_ASSET_ROOT__/miniconda3/bin/python'
RETRIEVAL = '__LOCAL_ASSET_ROOT__/miniconda3/envs/cyber-ft/bin/python'
NATIVE = str(ROOT / 'logs/experiments/rcm1000_source_priority_all_models_20261017_v1/runtime/bin/python')
AGENT = str(SUITE / 'client_runtime/bin/python')
BASE_RANKING = OUT / 'inputs/full_naive_prepared.jsonl'
GEMMA_JOB = ROOT / 'analysis/gemma31_ctibench1000_remaining_20261108/job.json'
PROMPT = OUT / 'inputs/naive_rag_cticonnect.json'


def check_capacity():
    assert shutil.disk_usage(OUT).free >= 8 * 1024**3, 'Disk reserve reached'
    state = read(GEMMA_JOB)
    assert state['status'] in {'completed', 'completed_with_blockers'}
    proc = Path(f"/proc/{state['pid']}/stat")
    try:
        fields = proc.read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return
    assert fields[0] == 'Z' or int(fields[19]) != state['start_ticks'], 'Gemma CTI-Bench still owns the GPUs'


def process(args, directory, name, *, cpu=False):
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'PYTHONHASHSEED': '0', 'TOKENIZERS_PARALLELISM': 'false'}
    if cpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
    with (directory / name).open('ab') as log:
        child = subprocess.Popen(args, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT)
    save(directory / 'child.json', {'pid': child.pid, 'start_ticks': ticks(child.pid), 'command': args})
    return child


def execute(args, directory, name, allowed=(0,)):
    result = process(args, directory, name).wait()
    if result not in allowed:
        raise RuntimeError(f'Child exited with code {result}; see {directory / name}')
    return result


@contextmanager
def sparse_service(directory, enabled):
    if not enabled:
        yield
        return
    with socket.socket() as sock:
        assert sock.connect_ex(('127.0.0.1', 8497)) != 0, 'Sparse port is already in use'
    child = process([NATIVE, str(CODE / 'sparse_service.py'), str(directory / 'sparse')],
                    directory, 'sparse.log', cpu=True)
    try:
        for _ in range(900):
            assert child.poll() is None, 'Sparse service exited'
            try:
                if requests.get('http://127.0.0.1:8497/health', timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        else:
            raise RuntimeError('Sparse service readiness deadline')
        yield
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=30)


def corpus():
    eligible = set(read(MASK)['filtered_eligible_rows'])
    docs = {}
    targets = {q['source_cve_id'] for q in questions()}
    with (FIXED / 'documents.jsonl').open() as stream:
        for row, line in enumerate(stream):
            if row in eligible:
                doc = json.loads(line)
                meta = json.loads(doc['raw_data'])
                assert meta['source'] == 'CVE' and meta['identifier'] not in targets
                docs[meta['identifier']] = doc
    assert len(docs) == len(eligible) == ELIGIBLE
    return docs


def prepare_messages(method, source, profile):
    queries = questions()
    entries = {q['question_id']: q for q in rows(source)} if source else {}
    assert not source or len(entries) == N and set(entries) == {q['question_id'] for q in queries}
    docs = corpus() if source else {}
    prompt = read(OUT / 'inputs/closed_book_cticonnect.json' if method == 'closed_book' else PROMPT)
    loader = TokenizersBackend if profile.get('tokenizer_loader') == 'TokenizersBackend' else AutoTokenizer
    tokenizer = loader.from_pretrained(profile['tokenizer_path'], local_files_only=True)
    template = Path(profile['chat_template']).read_text()
    prepared = []
    for q in queries:
        entry = entries.get(q['question_id'], {})
        refs = entry.get('references', [])
        assert all(r['cve_id'] in docs for r in refs)
        failure = entry.get('pipeline_failure')
        if method in {'naive_rag', 'reranked_rag', 'hyde', 'fusion', 'etr'} and not failure:
            assert len(refs) == 20
        if method == 'closed_book':
            user = q['prompt']
        else:
            candidates = []
            for ref in refs:
                doc = docs[ref['cve_id']]
                text = ref.get('text_override', doc['text'])
                candidates.append(f"[{doc['id']}] {doc['title']}\n{text[:600]}")
            user = prompt['PROMPT'].format(question=q['prompt'], candidates='\n\n'.join(candidates))
        messages = entry.get('custom_messages') or [
            {'role': 'system', 'content': prompt['SYSTEM']}, {'role': 'user', 'content': user}]
        count = len(tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
            add_generation_prompt=True, chat_template=template, **profile['template']))
        if not failure and count + 512 > profile['context']:
            raise ValueError(f"{q['question_id']} requires {count+512} tokens, profile limit is {profile['context']}")
        prepared.append({'question_id': q['question_id'], 'prompt': q['prompt'], 'messages': messages,
                         'input_tokens': count, 'pipeline_failure': failure})
    return prepared


def long_context_gate(directory, profile):
    tokenizer = AutoTokenizer.from_pretrained(profile['tokenizer_path'], local_files_only=True)
    template = Path(profile['chat_template']).read_text()

    def request(index):
        messages = [{'role': 'user', 'content': f'Synthetic capacity case {index}. Reply hello. ' + ' hello' * 7400}]
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
            add_generation_prompt=True, chat_template=template, **profile['template'])
        assert 7372 <= len(tokens) and len(tokens) + 512 <= profile['context']
        response = requests.post('http://127.0.0.1:8480/v1/chat/completions', json={
            'model': profile['model_name'], 'messages': messages, 'max_tokens': 512,
            'temperature': 0, 'n': 1}, timeout=300)
        response.raise_for_status()
        result = response.json()
        assert result['model'] == profile['model_name'] and result['choices'][0]['message']['content']
        return {'input_tokens': len(tokens), 'response': result}

    with ThreadPoolExecutor(max_workers=16) as pool:
        save(directory / 'long_context_gate.json', {'responses': list(pool.map(request, range(16)))})


def reader(method, directory, source, profile, item):
    prepared = prepare_messages(method, source, profile)
    target = directory / 'prepared.jsonl'
    with target.open('x') as stream:
        for q in prepared:
            stream.write(json.dumps(q) + '\n')
    save(directory / 'preflight.json', {'passed': True, 'n': N, 'context': profile['context'],
        'max_input_tokens': max(q['input_tokens'] for q in prepared),
        'prepared_sha256': sha(target), 'source_sha256': sha(source) if source else None})
    client = ChatClient('http://127.0.0.1:8480/v1', profile['model_name'])
    if any(not q['pipeline_failure'] for q in prepared):
        check_capacity()
        start_host(item, profile)
        if item['model'] == 'gemma31_qat' and method in {'naive_rag', 'reranked_rag'}:
            long_context_gate(directory, profile)

    def answer(q):
        if q['pipeline_failure']:
            return {'question_id': q['question_id'], 'prediction': '', 'pipeline_failure': q['pipeline_failure']}
        if shutil.disk_usage(directory).free < 5 * 1024**3:
            raise RuntimeError('Disk reserve reached before request')
        return {'question_id': q['question_id'], **client.complete(q['messages'], max_tokens=512, temperature=0)}

    ledger = directory / 'predictions.jsonl'
    with ledger.open('x') as stream, ThreadPoolExecutor(max_workers=16) as pool:
        for count, future in enumerate(as_completed([pool.submit(answer, q) for q in prepared]), 1):
            stream.write(json.dumps(future.result()) + '\n')
            stream.flush()
            save(directory / 'progress.json', {'done': count, 'total': N})
    score(directory, ledger, prepared)
    predictions = rows(ledger)
    report = read(directory / 'report.json')
    report['pipeline_failures'] = sum(bool(q.get('pipeline_failure')) for q in predictions)
    report['request_failures'] = sum(bool(q.get('request_error')) for q in predictions)
    report['truncated'] = sum(q.get('finish_reason') == 'length' for q in predictions)
    save(directory / 'report.json', report)


def agent_helpers():
    source = SUITE / 'code/final_methods_controller.py'
    tree = ast.parse(source.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {'gate', 'tool_profile'}]
    namespace = {'read': read, 'Path': Path, 'requests': requests, 'json': json}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


def model_profile(method, model, directory):
    path = OUT / 'full_profiles' / f'{model}.json'
    profile_path = directory / 'runtime_profile.json'
    shutil.copyfile(path, profile_path)
    item = {'directory': str(directory), 'profile': str(profile_path), 'model': model}
    if method == 'arag':
        agent_helpers()['tool_profile'](item)
    return item, freeze(item)


def run(method, model, directory):
    assert not (directory / 'predictions.jsonl').exists()
    assert not (directory / 'native_run/predictions.jsonl').exists()
    item, profile = model_profile(method, model, directory)
    save(directory / 'protocol.json', {'dataset': 'CTIConnect RCM', 'questions': N,
        'eligible_records': ELIGIBLE, 'method': method, 'model': model,
        'profile_sha256': sha(Path(item['profile'])), 'mask_sha256': sha(MASK),
        'workers': 16, 'temperature': 0, 'reader_max_tokens': 512,
        'generation_retries': 'HTTP/request exceptions only',
        'official_question_ids': True})
    if method in {'closed_book', 'naive_rag', 'reranked_rag', 'local_crag', 'linearrag'}:
        source = {'closed_book': None, 'naive_rag': BASE_RANKING,
            'local_crag': OUT / 'local_crag_prepared.jsonl', 'linearrag': OUT / 'linearrag_prepared.jsonl'}.get(method)
        if method == 'reranked_rag':
            source = OUT / 'inputs/full_reranked_prepared.jsonl'
        reader(method, directory, source, profile, item)
        return
    if method in {'hyde', 'fusion', 'etr'}:
        intermediate = directory / 'intermediate'
        start_host(item, profile)
        execute([PYTHON, str(CODE / 'generate_queries.py'), item['profile'], method, str(intermediate)],
                directory, 'intermediate.log')
        stop_host()
        execute([RETRIEVAL, str(CODE / 'query_retrieval.py'), method, str(intermediate)],
                directory, 'retrieval.log')
        reader(method, directory, intermediate / 'retrieval.jsonl', profile, item)
        return

    with sparse_service(directory, method in {'ircot', 'ragenta'}):
        start_host(item, profile)
        target = directory / 'native_run'
        if method == 'ircot':
            execute([AGENT, str(CODE / 'ircot_run.py'), item['profile'], str(target)], directory, 'native.log')
            return
        save(directory / 'native_gate.json', {'response': agent_helpers()['gate'](method, profile)})
        for wave in range(32):
            result = execute([AGENT, str(CODE / 'agent_run.py'), item['profile'], method, str(target)],
                             directory, 'native.log', allowed=(0, 75))
            if result == 0:
                return
            stop_host()
            execute([RETRIEVAL, str(CODE / 'agent_rank.py'), str(target), str(wave)],
                    directory, 'dynamic_retrieval.log')
            start_host(item, profile)
        raise RuntimeError('Native agent deferred-retrieval wave limit reached')


def main():
    method, model, target = sys.argv[1:]
    directory = Path(target)
    directory.mkdir(parents=True, exist_ok=True)
    check_capacity()
    if method == 'prepare':
        stop_host()
        if model == 'local_crag':
            with sparse_service(directory, True):
                execute([RETRIEVAL, str(CODE / 'prepare_crag.py')], directory, 'preparation.log')
        elif model == 'linearrag':
            execute([NATIVE, str(CODE / 'prepare_linear.py')], directory, 'preparation.log')
        else:
            raise ValueError(model)
    else:
        run(method, model, directory)


if __name__ == '__main__':
    main()
