"""Freeze gold-free recovery inputs from the existing CTI-Bench 1000 ledgers."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SUITE = ROOT / 'logs/experiments/ctibench1000_filtered_architectures_20261107_v1'
AUDIT = OUT.parent / 'ctibench1000_agent_continuation_20261110/coverage.json'
FUSION = ROOT / 'logs/experiments/rcm335_hyde_fusion_nonreasoning_full335_20261024_v1/native/fusion/code'
IDS = {str(i) for i in range(1000)}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rows(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1000 and {str(r['question_id']) for r in records} == IDS, path
    return records


def write_rows(path, records):
    with path.open('x') as stream:
        for row in records:
            stream.write(json.dumps(row) + '\n')
    return digest(path)


def main():
    sys.path.insert(0, str(FUSION))
    from contracts import rewrite_text

    conditions = [
        row for row in read(AUDIT)['conditions']
        if row['method'] != 'ragenta' and row['pipeline_failures'] >= 100
    ]
    assert Counter(row['method'] for row in conditions) == {
        'local_crag': 11, 'fusion': 8, 'arag': 1,
    }
    naive = SUITE / 'stages/naive_rag20/shared_prepared.jsonl'
    crag = SUITE / 'local_crag_prepared.jsonl'
    fallback = {r['question_id']: r for r in rows(naive)}
    expected_reason = 'No evidence strips passed fixed evaluator threshold'
    crag_ids = set()
    repaired = []
    for row in rows(crag):
        if row.get('pipeline_failure'):
            assert row['pipeline_failure'] == expected_reason and not row['references']
            original = fallback[row['question_id']]
            assert original['prompt'] == row['prompt'] and len(original['references']) == 20
            crag_ids.add(row['question_id'])
            row = {**original, 'recovery_evidence': 'original_dense_top20'}
        repaired.append(row)
    assert len(crag_ids) == 373
    receipt = {
        'gold_opened': False,
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in (
            Path(__file__), AUDIT, naive, crag, FUSION / 'contracts.py',
            FUSION / 'reference_functions.py',
        )},
        'crag_fallback_ids': sorted(crag_ids, key=int),
        'conditions': [],
    }
    generated = OUT / 'prepared'
    generated.mkdir(exist_ok=False)
    receipt['crag_fallback_input_sha256'] = write_rows(generated / 'crag_dense_fallback.jsonl', repaired)
    for condition in conditions:
        method, model = condition['method'], condition['model']
        report = ROOT / condition['report']
        ledger = report.with_name('predictions.jsonl')
        source_hash = digest(ledger)
        assert source_hash == read(report.with_name('prediction_freeze.json'))['sha256']
        failures = [r for r in rows(ledger) if r.get('pipeline_failure')]
        assert len(failures) == condition['pipeline_failures']
        item = {**condition, 'prediction_sha256': source_hash}
        if method == 'local_crag':
            assert {r['question_id'] for r in failures} == crag_ids
        elif method == 'fusion':
            intermediate = SUITE / 'stages/fusion' / model / 'intermediate'
            source = intermediate / 'outputs.jsonl'
            source_hash = digest(source)
            assert source_hash == read(intermediate / 'prediction_freeze.json')['sha256']
            failed_ids = {r['question_id'] for r in failures}
            parsed = []
            for row in rows(source):
                qid = str(row['question_id'])
                assert int(qid) == row['row_index']
                queries = None
                if not (row.get('request_error') or row.get('returned_response_error')):
                    queries = rewrite_text(
                        row['prediction'], row['finish_reason'], fallback[qid]['prompt'],
                    )
                parsed.append({'question_id': qid, 'rewrites': queries})
            item.update(
                intermediate_sha256=source_hash,
                rewrite_input_sha256=write_rows(generated / f'fusion_{model}.jsonl', parsed),
                previously_failed_with_rewrites=sum(
                    row['question_id'] in failed_ids and bool(row['rewrites']) for row in parsed
                ),
                without_rewrites=sum(not row['rewrites'] for row in parsed),
            )
        else:
            item['failure_kinds'] = Counter(
                r['pipeline_failure'].split(':', 3)[2].strip()
                if r['pipeline_failure'].count(':') >= 2 else r['pipeline_failure']
                for r in failures
            )
        receipt['conditions'].append(item)
    with (generated / 'input_freeze.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
