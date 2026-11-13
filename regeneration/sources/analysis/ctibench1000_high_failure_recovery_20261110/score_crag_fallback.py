"""Assemble and score Local CRAG plus dense fallback from identical archived requests."""
from collections import Counter
import csv
import importlib.util
import json
from pathlib import Path
import sys

from prepare import OUT, ROOT, SUITE, digest, read, rows, write_rows


def profile(directory):
    host = read(directory.parent / 'host_config.json')
    template = Path(host.pop('chat_template'))
    host.pop('assets', None)
    for server in host['servers']:
        args = server['args']
        args[args.index('--port') + 1] = '<port>'
        args[args.index('--chat-template') + 1] = '<template>'
    return host, digest(template)


def asset_signature(protocol):
    return sorted(
        (Path(asset['path']).name, asset['bytes'], asset['sha256'])
        for asset in protocol['model_assets']
    )


def indexed(path):
    return {r['question_id']: r for r in rows(path)}


def main():
    freeze = read(OUT / 'prepared/input_freeze.json')
    ids = set(freeze['crag_fallback_ids'])
    variants = OUT / 'crag_dense_fallback_v2'
    variants.mkdir(exist_ok=False)
    pending = {}
    completed = []
    for entry in freeze['conditions']:
        if entry['method'] != 'local_crag':
            continue
        model = entry['model']
        crag = (ROOT / entry['report']).parent
        naive = SUITE / 'stages/naive_rag20/runs' / model / 'dense'
        if model == 'gemma31_qat':
            naive = SUITE / ('stages/naive_rag20_gemma31_fp8kv7168_static_scale'
                            '/runs/gemma31_qat/dense')
        if not (naive / 'prediction_freeze.json').is_file() or profile(crag) != profile(naive):
            pending[model] = 'Reader profile or frozen Naive RAG ledger differs'
            continue
        for directory in (crag, naive):
            path = directory / 'predictions.jsonl'
            assert digest(path) == read(directory / 'prediction_freeze.json')['sha256']
        crag_protocol = read(crag / 'protocol.json')
        naive_protocol = read(naive / 'protocol.json')
        if any(crag_protocol[k] != naive_protocol[k] for k in (
            'generation', 'workers_total', 'request_exception_policy',
        )) or asset_signature(crag_protocol) != asset_signature(naive_protocol):
            pending[model] = 'Frozen generation, worker or model-asset settings differ'
            continue
        previous = indexed(crag / 'predictions.jsonl')
        fallback = indexed(naive / 'predictions.jsonl')
        prepared = indexed(naive / 'prepared.jsonl')
        for qid in ids:
            assert previous[qid].get('pipeline_failure')
            r = fallback[qid]
            assert r['request'] == {
                'model': naive_protocol['model'], 'messages': prepared[qid]['messages'],
                'max_tokens': 512, 'temperature': 0, 'n': 1,
            }
            assert previous[qid]['source_cve_id'] == r['source_cve_id']
        merged = [
            {**(fallback[qid] if qid in ids else previous[qid]),
             'reused_from': 'naive_rag20' if qid in ids else 'local_crag'}
            for qid in sorted(previous, key=int)
        ]
        destination = variants / model
        destination.mkdir()
        ledger_hash = write_rows(destination / 'predictions.jsonl', merged)
        receipt = {
            'sha256': ledger_hash, 'rows': 1000, 'gold_opened_for_scoring': False,
            'crag_ledger_sha256': entry['prediction_sha256'],
            'naive_ledger_sha256': digest(naive / 'predictions.jsonl'),
            'reused_naive_requests': len(ids),
        }
        (destination / 'prediction_freeze.json').write_text(json.dumps(receipt, indent=2) + '\n')
        completed.append((model, destination, crag, merged, prepared))

    benchmark = ROOT / 'data/cti-bench/data/cti-rcm.tsv'
    with benchmark.open() as stream:
        gold = list(csv.DictReader(stream, delimiter='\t'))
    assert len(gold) == 1000
    results = {}
    for model, destination, crag, merged, prepared in completed:
        metrics_path = crag / 'code/official_metrics.py'
        spec = importlib.util.spec_from_file_location('official_metrics', metrics_path)
        metrics = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = metrics
        spec.loader.exec_module(metrics)
        items = []
        for row in merged:
            qid = row['question_id']
            case = gold[int(qid)]
            assert prepared[qid]['prompt'] == case['Prompt']
            target = 'CWE-' + str(int(case['GT'].split('-')[1]))
            items.append(metrics.score_id_item(
                row['content'], {'target_type': 'cwe', 'target_id': target},
                task='rcm', item_id=qid,
            ))
        report = {
            'variant': 'Local CRAG with dense fallback for no surviving strips',
            'official_mean_question_metrics': metrics.aggregate_scores(items)['overall'],
            'empty_ID_sets': sum(not x.pred_ids for x in items),
            'finish_reasons': Counter(row.get('finish_reason') or 'absent' for row in merged),
            'prediction_sha256': digest(destination / 'predictions.jsonl'),
        }
        (destination / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        results[model] = report
    (variants / 'summary.json').write_text(json.dumps({
        'completed': results, 'pending': pending,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
