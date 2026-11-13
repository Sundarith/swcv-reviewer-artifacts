"""Verify published hashes and independently reconstruct overlap-condition votes."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import official_metrics as official

ROOT=Path(__file__).resolve().parent
for name,expected in json.loads((ROOT/'checksums.json').read_text()).items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
gold={r['id']:r['ground_truth'] for r in map(json.loads,(ROOT/'gold.jsonl').read_text().splitlines())}
O=ROOT/'overlap'
raw=gzip.decompress((O/'predictions.jsonl.gz').read_bytes())
assert hashlib.sha256(raw).hexdigest()==json.loads((O/'overlap_prediction_freeze.json').read_text())['sha256']
records=[json.loads(line) for line in raw.decode().splitlines()]
assert len(records)==len({r['question_id'] for r in records})==290
assert {r['question_id'] for r in records}==set(gold)
results=json.loads((O/'overlap_results.json').read_text())
for name in results:
    mask=json.loads((O/(name+'_mask.json')).read_text());allowed=set(mask['filtered_eligible_rows'])
    assert len(allowed)==mask['remaining_references']
    items=[]
    for row in records:
        arm=row['conditions'][name];votes=Counter()
        assert len(arm['neighbors'])==20
        assert all(n['row'] in allowed for n in arm['neighbors'])
        for hit in arm['neighbors'][:10]:
            for c in hit['cwe_ids']:votes[c]+=hit['similarity']
        prediction=min(votes,key=lambda c:(-votes[c],c))
        assert prediction==arm['prediction']
        items.append(official.score_id_item(prediction,gold[row['question_id']]))
    assert sum(x.exact_match for x in items)==results[name]['official_correct']
    assert abs(official.aggregate_scores(items)['overall']['f1']-results[name]['official_mean_question_metrics']['f1'])<1e-12
metrics=json.loads((ROOT/'metrics.json').read_text())
for condition in metrics:
    with (ROOT/'class_results'/f'{condition}_confusion.csv').open() as stream:
        import csv
        assert sum(int(r['questions']) for r in csv.DictReader(stream))==290
print('All hashes, 104 confusion summaries, 870 overlap-condition votes and accepted-alternative scores verified.')
