"""Recover RCM290 Fusion parser failures and Local CRAG empty evidence."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import importlib.util
import json
from pathlib import Path
import re
import shutil
import signal
import sys
import time
import traceback

import requests
from transformers import AutoTokenizer, TokenizersBackend

ROOT = Path('__EXPERIMENT_WORKSPACE__')
HERE = Path(__file__).resolve().parent
CONNECT = ROOT / 'logs/experiments/cticonnect_rcm290_full_suite_20261107_v1'
FULL = CONNECT / 'code/full'
HELPERS = ROOT / 'analysis/ctibench1000_high_failure_recovery_20261110'
sys.path.insert(0, str(HELPERS))
import live_smoke as host
from prepare import digest, read
host.WORK = HERE
sys.path.insert(0, str(FULL))
from condition import prepare_messages
sys.path.insert(0, str(CONNECT / 'code'))
from reader import score as official_score
PREDECESSOR = ROOT / 'analysis/cticonnect_rcm290_context_recovery_20261111/job.json'
MODELS = ['gemmae4b', 'granite8b', 'llama', 'qwen', 'phi', 'gptoss', 'granite30_fp8', 'muse', 'nemotron', 'qwen38', 'gemma31_qat']


def rows(path):
    return [json.loads(line) for line in path.open()]


def validated(directory):
    p = directory / 'predictions.jsonl'
    assert digest(p) == read(directory / 'prediction_freeze.json')['sha256']
    data = rows(p)
    assert len(data) == len({r['question_id'] for r in data}) == 290
    return {r['question_id']: r for r in data}


def signature(profile):
    value = json.loads(json.dumps(profile))
    template = Path(value.pop('chat_template'))
    value.pop('assets', None)
    value.pop('historical_asset_hashes_matched', None)
    for server in value['servers']:
        args = server['args']
        args[args.index('--port') + 1] = '<port>'
        args[args.index('--chat-template') + 1] = '<template>'
    return value, digest(template)


def score(directory, data):
    ledger = directory / 'predictions.jsonl'
    with ledger.open('x') as stream:
        stream.write(''.join(json.dumps(r) + '\n' for r in data))
    official_score(directory, ledger, rows(CONNECT / 'inputs/queries.jsonl'))
    report = read(directory / 'report.json')
    # The adopted variant normalizes only Unicode hyphens inside CWE identifiers.
    spec = importlib.util.spec_from_file_location('official_metrics', CONNECT / 'inputs/official_metrics.py')
    metrics = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = metrics
    spec.loader.exec_module(metrics)
    gold_path = ROOT / 'logs/experiments/cticonnect_rcm290_nvd_first_all_models_20261018_v1/shared/official_rcm.jsonl'
    gold = {r['id']: r for r in rows(gold_path)}
    pattern = re.compile(r'(?<=CWE)[\u2010-\u2015\u2212\ufe63\uff0d](?=\d)', re.IGNORECASE)
    items = [metrics.score_id_item(pattern.sub('-', r['prediction']), gold[r['question_id']]['ground_truth'], task='rcm', item_id=r['question_id']) for r in data]
    report['adopted_unicode_normalized'] = metrics.aggregate_scores(items)['overall']
    report['pipeline_failures'] = sum(bool(r.get('pipeline_failure')) for r in data)
    host.save(directory / 'report.json', report)
    return report


def crag():
    ids = {r['question_id'] for r in rows(CONNECT / 'local_crag_prepared.jsonl') if r.get('pipeline_failure')}
    assert len(ids) == 61
    for model in MODELS:
        original = CONNECT / 'runs/local_crag' / model
        naive = CONNECT / 'runs/naive_rag' / model
        old = validated(original)
        fallback = validated(naive)
        source_profile = original / 'runtime_profile.json'
        naive_profile = naive / ('runtime_profile.json' if (naive / 'runtime_profile.json').exists() else 'profile.json')
        same = signature(read(source_profile)) == signature(read(naive_profile))
        directory = HERE / ('local_crag_' + model)
        directory.mkdir()
        source = read(naive_profile)
        # Generate the fallback using the exact saved Naive RAG messages.
        prepared = {r['question_id']: r for r in rows(naive / 'prepared.jsonl')}
        reusable = same and all(fallback[qid].get('request', {}).get('messages') == prepared[qid]['messages'] for qid in ids)
        if reusable:
            data = [{**(fallback[qid] if qid in ids else old[qid]),
                     'reused_source': 'naive_rag' if qid in ids else 'local_crag'} for qid in sorted(old)]
            host.save(directory / 'protocol.json', {'variant': 'Local CRAG dense-top20 fallback on zero surviving strips',
                'changed_ids': sorted(ids), 'reused_identical_reader': True,
                'source_sha256': digest(original / 'predictions.jsonl'),
                'fallback_sha256': digest(naive / 'predictions.jsonl'), 'gold_opened': False})
            host.RESULTS['local_crag/' + model] = {'completed': True, **score(directory, data)}
        else:
            host.save(directory / 'pending.json', {'reason': 'Archived reader profile or request differs',
                'changed_ids': sorted(ids), 'source': str(original), 'fallback': str(naive)})
            host.RESULTS['local_crag/' + model] = {'completed': False, 'needs_inference': True}
    host.save(HERE / 'results.json', host.RESULTS)


def prepare_fusion():
    parser = ROOT / 'logs/experiments/rcm335_hyde_fusion_nonreasoning_full335_20261024_v1/native/fusion/code'
    script = (FULL / 'query_retrieval.py').read_text()
    script = "import sys\nsys.path.insert(0," + repr(str(FULL)) + ")\n" + script
    script = script.replace('inputs=[];index={};failed={}',
        'sys.path.insert(0,' + repr(str(parser)) + ')\nfrom contracts import rewrite_text\ninputs=[];index={};failed={}')
    lines = script.splitlines(keepends=True)
    replaced = 0
    for i, line in enumerate(lines):
        if line.startswith(' texts=[text] if method'):
            lines[i] = " texts=[text] if method!='fusion' else (rewrite_text(text,r.get('finish_reason'),queries[i]['prompt']) or [])\n"
            replaced += 1
        elif "if method=='fusion' and len(texts)!=4:" in line:
            lines[i] = " if method=='fusion' and not texts:failed[i]='no_accepted_rewrites';continue\n"
            replaced += 1
    assert replaced == 2
    destination = HERE / 'query_retrieval.py'
    destination.write_text(''.join(lines))
    compile(destination.read_text(), str(destination), 'exec')
    for model in MODELS:
        original = CONNECT / 'runs/rag_fusion' / model
        failures = [r for r in validated(original).values() if r.get('pipeline_failure')]
        if not failures:
            continue
        directory = HERE / ('fusion_' + model)
        directory.mkdir()
        intermediate = directory / 'intermediate'
        intermediate.mkdir()
        for name in ('outputs.jsonl', 'prediction_freeze.json'):
            shutil.copyfile(original / 'intermediate' / name, intermediate / name)
        assert digest(intermediate / 'outputs.jsonl') == read(intermediate / 'prediction_freeze.json')['sha256']
        source = original / 'runtime_profile.json'
        if model == 'gemma31_qat':
            source = CONNECT / 'runs/naive_rag/gemma31_qat/runtime_profile.json'
        profile = read(source)
        template = directory / 'chat_template.jinja'
        old_template = profile['chat_template']
        template.write_bytes(Path(old_template).read_bytes())
        profile['chat_template'] = str(template)
        for server in profile['servers']:
            server['args'] = [str(template) if a == old_template else a for a in server['args']]
        host.save(directory / 'host_config.json', profile)
        host.save(directory / 'protocol.json', {'variant': 'Released Fusion rewrite parser with archived generated queries',
            'source_ledger_sha256': digest(original / 'predictions.jsonl'), 'parser_sha256': digest(parser / 'contracts.py'),
            'retrieval_code_sha256': digest(destination), 'gold_opened': False, 'workers': 16, 'max_tokens': 512})


def start(directory):
    profile_path = directory / 'host_config.json'
    profile = read(profile_path)
    assert host.free()
    for asset in profile['assets']:
        assert Path(asset['path']).is_file() and digest(Path(asset['path'])) == asset['sha256']
    child = host.launch([host.PY, '-u', str(ROOT / 'scripts/local_llm_api_host.py'), 'serve'],
                        directory / 'host.log', {'CTI_HOST_CONFIG': str(profile_path)})
    host.HOST = child, host.ticks(child.pid)
    for _ in range(450):
        assert child.poll() is None, 'Model startup failed'
        try:
            response = requests.get('http://127.0.0.1:8480/health', timeout=2)
            if response.ok and response.json()['model'] == profile['model_name']:
                return profile
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError('Model readiness deadline')


def fusion(model):
    directory = HERE / ('fusion_' + model)
    host.execute([host.RPY, str(HERE / 'query_retrieval.py'), 'fusion', str(directory / 'intermediate')], directory / 'retrieval.log')
    profile = read(directory / 'host_config.json')
    prepared = prepare_messages('fusion', directory / 'intermediate/retrieval.jsonl', profile)
    host.save(directory / 'prepared.json', prepared)
    profile = start(directory)
    original = CONNECT / 'runs/rag_fusion' / model
    old = validated(original)
    def answer(row):
        qid = row['question_id']
        if row.get('pipeline_failure'):
            return {'question_id': qid, 'prediction': '', 'pipeline_failure': row['pipeline_failure']}
        payload = {'model': profile['model_name'], 'messages': row['messages'], 'max_tokens': 512, 'temperature': 0, 'n': 1}
        if old[qid].get('request') == payload:
            return {**old[qid], 'reused_identical_request': True}
        response = host.request(payload)
        choice = response.get('response', {}).get('choices', [{}])[0]
        return {**response, 'question_id': qid, 'prediction': choice.get('message', {}).get('content') or '',
                'finish_reason': choice.get('finish_reason')}
    data = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for i, future in enumerate(as_completed([pool.submit(answer, r) for r in prepared]), 1):
            data.append(future.result())
            host.save(directory / 'progress.json', {'completed': i, 'total': 290})
    return score(directory, data)


def main():
    with (HERE / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        resume='--resume' in sys.argv
        if resume:
            previous=read(HERE/'job.json')
            assert host.ticks(previous['pid'])!=previous['start_ticks']
            host.RESULTS.update(read(HERE/'results.json'))
        else:
            assert not (HERE/'job.json').exists()
        signal.signal(signal.SIGTERM, host.interrupt)
        if not resume:
            host.state('auditing_and_assembling_crag_fallback')
            crag()
            prepare_fusion()
        host.state('waiting_for_context_recovery')
        while True:
            job = read(PREDECESSOR)
            if host.ticks(job['pid']) != job['start_ticks']:
                assert job['status'].startswith('completed'), job
                break
            time.sleep(20)
        assert host.free()
        for model in MODELS:
            if not (HERE / ('fusion_' + model)).exists():
                continue
            key = 'fusion/' + model
            host.state('running', condition=key)
            try:
                host.RESULTS[key] = {'completed': True, **fusion(model)}
            except Exception:
                host.RESULTS[key] = {'completed': False, 'error': traceback.format_exc()}
            finally:
                host.stop_host()
            host.save(HERE / 'results.json', host.RESULTS)
        host.state('completed' if all(r['completed'] for r in host.RESULTS.values()) else 'completed_with_failures')


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        host.state('failed', error=traceback.format_exc())
        host.stop_host()
        raise
