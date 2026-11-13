"""Offline reconstruction of all six frozen k-sensitivity results."""
from pathlib import Path
from collections import Counter
import gzip,hashlib,importlib.util,json,sys
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('official_metrics',P.parent/'official_metrics.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
gold={str(r['id']):r for r in map(json.loads,(P.parent/'gold.jsonl').read_text().splitlines())}
with gzip.open(P/'ranked50.jsonl.gz','rt') as f:ranked=[json.loads(l) for l in f]
assert len(ranked)==len(gold)==290
reports=json.loads((P/'results.json').read_text())
for k in [1,5,10,20,30,50]:
 ledger=P/f'predictions_k{k}.jsonl'
 assert hashlib.sha256(ledger.read_bytes()).hexdigest()==reports[str(k)]['prediction_sha256']
 saved={str(r['question_id']):r['prediction'] for r in map(json.loads,ledger.read_text().splitlines())}
 assert set(saved)==set(gold)
 scores=[]
 for r in ranked:
  weights=Counter()
  for h in r['hits'][:k]:
   for label in h['cwe_ids']:weights[label]+=h['similarity']
  prediction=min(weights,key=lambda c:(-weights[c],c));qid=str(r['question_id'])
  assert prediction==saved[qid]
  scores.append(m.score_id_item(prediction,gold[qid]['ground_truth'],task='rcm',item_id=qid))
 actual=m.aggregate_scores(scores)['overall'];expected=reports[str(k)]['official_mean_question_metrics']
 assert actual==expected,(k,actual,expected)
 print(k,round(actual['f1']*100,2))
print('All six vote ledgers and official scores reproduced.')
