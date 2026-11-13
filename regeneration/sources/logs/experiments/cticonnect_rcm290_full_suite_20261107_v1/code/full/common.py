"""Shared paths and validation for the full CTIConnect continuation."""
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parents[2]
ROOT = OUT.parents[2]
SUITE = ROOT / 'logs/experiments/ctibench1000_filtered_architectures_20261107_v1'
FIXED = ROOT / 'logs/experiments/rcm335_nvd_first_fixed_pool_all_models_20261018_v1/shared'
CONNECT = ROOT / 'logs/experiments/cticonnect_rcm290_nvd_first_all_models_20261018_v1/shared'
QUERY_PATH = OUT / 'inputs/queries.jsonl'
MASK = OUT / 'inputs/exclusions.json'
CODE = OUT / 'code/full'
N = 290
ELIGIBLE = 119889
sys.path.insert(0, str(OUT / 'code'))


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def questions():
    result = rows(QUERY_PATH)
    assert len(result) == len({q['question_id'] for q in result}) == N
    assert [q['row_index'] for q in result] == list(range(N))
    assert all(q['question_id'] == q['official_question_id'] for q in result)
    return result


def save(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def score_native(directory, ledger):
    from reader import score
    records = rows(ledger)
    by_id = {r['question_id']: r for r in records}
    assert len(records) == len(by_id) == N
    assert all('prediction' in r for r in records)
    score(directory, ledger, questions())
    report = read(directory / 'report.json')
    report['pipeline_failures'] = sum(bool(r.get('pipeline_failure')) for r in records)
    save(directory / 'report.json', report)
