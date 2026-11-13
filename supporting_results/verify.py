"""Verify frozen bytes, question coverage, Muse metrics and representation controls offline."""
import gzip
import hashlib
import json
from pathlib import Path
from official_metrics import aggregate_scores, score_id_item

ROOT = Path(__file__).resolve().parent

def sha(data):
    return hashlib.sha256(data).hexdigest()

def read(name):
    data = (ROOT / name).read_bytes()
    return gzip.decompress(data) if name.endswith('.gz') else data

def rows(name):
    result = [json.loads(line) for line in read(name).splitlines()]
    ids = [str(r.get('question_id', r.get('row_index'))) for r in result]
    assert len(result) == 1000 and set(ids) == {str(i) for i in range(1000)}, name
    return result

def main():
    manifest = json.loads(read('manifest.json'))
    for entry in manifest['files']:
        name = entry['path']
        assert sha((ROOT / name).read_bytes()) == entry['sha256'], name
        assert sha(read(name)) == entry['uncompressed_sha256'], name
        if entry['frozen_sha256']:
            assert sha(read(name)) == entry['frozen_sha256'], name
    gold_bytes = (ROOT.parent / 'benchmarks/ctibench_rcm1000_gold.jsonl').read_bytes()
    assert sha(gold_bytes) == manifest['gold_sha256']
    gold = {r['question_id']: r['gold'] for r in map(json.loads, gold_bytes.splitlines())}
    protocol = json.loads(read('muse/protocol_and_results.json'))
    prepared = rows('muse/prepared.jsonl.gz')
    suffix = protocol['original_user'].split('{candidates}')[1]
    for row in prepared:
        assert len(row['references']) == 20
        assert row['messages'][0]['content'] == protocol['original_system']
        assert row['messages'][1]['content'].endswith(suffix)
    output = {'muse': {}, 'ablations_correct': {}}
    for run in protocol['run_order']:
        predictions = rows(f'muse/{run}.jsonl.gz')
        assert [r['question_id'] for r in predictions] == [r['question_id'] for r in prepared]
        scores = [score_id_item(r['content'], {'target_type': 'cwe', 'target_id': gold[r['question_id']]}, task='rcm', item_id=r['question_id']) for r in predictions]
        result = aggregate_scores(scores)['overall']
        for metric in ('precision', 'recall', 'f1', 'exact_match'):
            assert abs(result[metric] - protocol['runs'][run][metric]) < 1e-12, (run, metric)
        output['muse'][run] = result
    point4 = rows('ablations/point4/predictions.jsonl.gz')
    point5 = rows('ablations/point5/predictions.jsonl.gz')
    control = rows('ablations/point5/baseline_predictions.jsonl.gz')
    arms = [(point4, 'full_question', 763), (point4, 'description_only', 714),
            (point5, 'full_reference', 759), (point5, 'description_product', 734),
            (control, 'source_supported', 763)]
    for records, arm, expected in arms:
        correct = 0
        for r in records:
            neighbors = r[arm]['neighbors']
            assert len(neighbors) == 20
            votes = {}
            for n in neighbors:
                for label in n['cwe_ids']:
                    votes[label] = votes.get(label, 0) + n['similarity']
            winner = sorted(votes, key=lambda label: (-votes[label], label))[0]
            assert winner == r[arm]['prediction'], (arm, r['row_index'])
            correct += winner == gold[str(r['row_index'])]
        assert correct == expected, (arm, correct, expected)
        output['ablations_correct'][arm] = correct
    for a, b, c in zip(point4, point5, control):
        assert a['row_index'] == b['row_index'] == c['row_index']
        assert a['prompt'] == b['prompt'] == c['prompt']
        assert a['full_question'] == c['source_supported']
    print(json.dumps(output, indent=2))

if __name__ == '__main__':
    main()
