"""Validate the corrected Phi parser, smoke-test A-RAG, then rerun RCM290."""
import ast
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

import requests

ROOT = Path('__EXPERIMENT_WORKSPACE__')
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'analysis/cticonnect_rcm290_complete_20261111/phi'
CONNECT = ROOT / 'logs/experiments/cticonnect_rcm290_full_suite_20261107_v1'
FULL = CONNECT / 'code/full'
sys.path.insert(0, str(ROOT / 'analysis/ctibench1000_high_failure_recovery_20261110'))
import live_smoke as host
from prepare import digest, read
host.WORK = HERE


def prepare():
    profile = read(BASE / 'host_config.json')
    old_template = profile['chat_template']
    template = HERE / 'chat_template.jinja'
    template.write_bytes(Path(old_template).read_bytes())
    profile['chat_template'] = str(template)
    for server in profile['servers']:
        args = [str(template) if a == old_template else a for a in server['args']]
        assert args[args.index('--tool-call-parser') + 1] == 'phi4_mini_json'
        args[args.index('--tool-call-parser') + 1] = 'phi4_nested_json'
        args += ['--tool-parser-plugin', str(HERE / 'parser_plugin.py')]
        server['args'] = args
    profile['tool_adapter'] = {**profile['tool_adapter'], 'parser': 'phi4_nested_json',
                              'change': 'Decode complete JSON arrays including nested argument arrays'}
    for asset in profile['assets']:
        assert Path(asset['path']).is_file() and digest(Path(asset['path'])) == asset['sha256']
    host.save(HERE / 'host_config.json', profile)
    ids = [r['question_id'] for r in
           map(json.loads, (CONNECT / 'inputs/queries.jsonl').read_text().splitlines())]
    smoke = sorted(ids, key=lambda q: hashlib.sha256(('phi-parser-v1:' + q).encode()).digest())[:4]
    original = (BASE / 'agent_run.py').read_text()
    for name, selected in (('smoke', smoke), ('full', ids)):
        source = original
        if name == 'smoke':
            marker = "completed={str(x['question_id'])"
            source = source.replace(marker, 'queries=[q for q in queries if q["question_id"] in '
                                    + repr(set(selected)) + ']\n' + marker)
        compile(source, str(HERE / (name + '_agent.py')), 'exec')
        (HERE / (name + '_agent.py')).write_text(source)
    shutil.copyfile(BASE / 'template_tools.py', HERE / 'template_tools.py')
    target = HERE / 'native_run'
    target.mkdir()
    cached = BASE / 'native_run/retrieval_cache.json'
    if cached.exists():
        shutil.copyfile(cached, target / 'retrieval_cache.json')
    host.save(HERE / 'protocol.json', {
        'dataset': 'CTIConnect RCM290', 'question_ids': ids, 'smoke_ids': smoke,
        'change': 'Nonstreaming tool-call parser now decodes complete nested JSON arrays',
        'prompt_template': 'Byte-identical to prior Phi-4 A-RAG template',
        'context': profile['context'], 'generation_cap': 512, 'workers': 16, 'temperature': 0,
        'evidence': 'Full original records and complete retained history',
        'original_profile_sha256': digest(BASE / 'host_config.json'),
        'source_ledger_sha256': digest(BASE / 'native_run/predictions.jsonl'),
        'sha256': {p.name: digest(p) for p in
                   (HERE / 'parser_plugin.py', HERE / 'host_config.json', template,
                    HERE / 'smoke_agent.py', HERE / 'full_agent.py', Path(__file__))},
        'gold_opened': False,
    })
    return smoke


def start():
    assert host.free()
    profile = read(HERE / 'host_config.json')
    child = host.launch([host.PY, '-u', str(host.SUITE / 'code/agent_api_host.py'), 'serve'],
                        HERE / 'host.log', {'CTI_HOST_CONFIG': str(HERE / 'host_config.json')})
    host.HOST = child, host.ticks(child.pid)
    for _ in range(450):
        assert child.poll() is None, 'Owned Phi model host failed'
        try:
            response = requests.get('http://127.0.0.1:8480/health', timeout=2)
            if response.ok and response.json()['model'] == profile['model_name']:
                return profile
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError('Model readiness deadline')


def gate(profile):
    tools = [{'type': 'function', 'function': {
        'name': 'chunk_read', 'description': 'Read fictional chunks by ID.',
        'parameters': {'type': 'object', 'properties': {
            'chunk_ids': {'type': 'array', 'items': {'type': 'string'}}}, 'required': ['chunk_ids']},
    }}]
    messages = [{'role': 'user', 'content':
                 'Call chunk_read exactly once with chunk_ids ["fixture-1", "fixture-2"]. '
                 'Then report the two values from its result.'}]
    payload = {'model': profile['model_name'], 'messages': messages, 'tools': tools,
               'tool_choice': 'auto', 'max_tokens': 512, 'temperature': 0, 'n': 1}
    first = host.request(payload)
    host.save(HERE / 'array_gate_initial.json', first)
    choice = first['response']['choices'][0]
    assistant = choice['message']
    assert choice['finish_reason'] == 'tool_calls', first
    calls = assistant['tool_calls']
    assert len(calls) == 1 and calls[0]['function']['name'] == 'chunk_read'
    assert json.loads(calls[0]['function']['arguments']) == {'chunk_ids': ['fixture-1', 'fixture-2']}
    follow = {**payload, 'messages': messages + [assistant, {
        'role': 'tool', 'tool_call_id': calls[0]['id'],
        'content': 'fixture-1 contains bluebird. fixture-2 contains foxglove.'}]}
    second = host.request(follow)
    host.save(HERE / 'array_gate_continuation.json', second)
    text = second['response']['choices'][0]['message'].get('content') or ''
    assert 'bluebird' in text.lower() and 'foxglove' in text.lower(), second
    host.save(HERE / 'array_gate.json', {'passed': True, 'nested_array': True, 'tool_result_round_trip': True})


def run_agent(name, wave):
    script = HERE / (name + '_agent.py')
    for offset in range(32):
        host.state('running', phase=name, wave=wave)
        result = host.execute([host.CLIENT, str(script), str(HERE / 'host_config.json'),
                              'arag', str(HERE / 'native_run')], HERE / 'agent.log', (0, 75))
        if result == 0:
            return wave
        host.stop_host()
        host.state('retrieving', phase=name, wave=wave)
        host.execute([host.RPY, str(FULL / 'agent_rank.py'), str(HERE / 'native_run'), str(wave)], HERE / 'retrieval.log')
        wave += 1
        start()
    raise RuntimeError('Agent deferred retrieval limit')


def score():
    directory = HERE / 'native_run'
    ledger = directory / 'predictions.jsonl'
    rows = [json.loads(line) for line in ledger.open()]
    assert len(rows) == len({r['question_id'] for r in rows}) == 290
    host.save(directory / 'prediction_freeze.json', {'sha256': digest(ledger), 'n': 290, 'gold_opened': False})
    sys.path.insert(0, str(CONNECT / 'code'))
    from reader import score as official_score
    queries = [json.loads(line) for line in (CONNECT / 'inputs/queries.jsonl').open()]
    official_score(directory, ledger, queries)
    report = read(directory / 'report.json')
    report['pipeline_failures'] = sum(bool(r.get('pipeline_failure')) for r in rows)
    report['cases_with_documents_read'] = sum(bool(r.get('trace', {}).get('chunks_read_ids')) for r in rows)
    host.save(HERE / 'results.json', report)


def main():
    with (HERE / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (HERE / 'job.json').exists()
        assert host.free()
        signal.signal(signal.SIGTERM, host.interrupt)
        host.state('validating_assets')
        try:
            assert read(HERE / 'parser_tests.json')['passed']
            smoke = prepare()
            host.state('starting_model')
            profile = start()
            host.state('validating_array_tool_round_trip')
            gate(profile)
            wave = run_agent('smoke', 0)
            rows = [json.loads(line) for line in (HERE / 'native_run/predictions.jsonl').open()]
            assert len(rows) == 4 and {r['question_id'] for r in rows} == set(smoke)
            assert not any(r.get('pipeline_failure') for r in rows), rows
            reads = sum(bool(r.get('trace', {}).get('chunks_read_ids')) for r in rows)
            host.save(HERE / 'smoke_result.json', {'n': 4, 'pipeline_failures': 0, 'cases_with_documents_read': reads})
            assert reads > 0, 'No benchmark document reads in smoke test'
            run_agent('full', wave)
            host.state('scoring')
            score()
            host.state('completed', report=read(HERE / 'results.json'))
        except BaseException:
            host.state('failed', error=traceback.format_exc())
            raise
        finally:
            host.stop_host()


if __name__ == '__main__':
    main()
