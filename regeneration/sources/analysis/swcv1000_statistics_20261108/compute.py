from pathlib import Path
from collections import Counter
import json,hashlib,csv,math
import numpy as np
R=Path('__EXPERIMENT_WORKSPACE__');O=Path(__file__).resolve().parent;S=R/'analysis/swcv_final_group_sensitivity_20261107';source=S/'predictions.jsonl'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,d):(O/n).write_text(json.dumps(d,indent=2)+'\n')
assert sha(source)==json.loads((S/'prediction_freeze.json').read_text())['sha256']
assert not (O/'predictions.jsonl').exists()
records=[]
for line in source.open():
 r=json.loads(line);x=r['source_supported'];counts=Counter(label for n in x['neighbors'] for label in n['cwe_ids']);weighted=Counter()
 for n in x['neighbors']:
  for label in n['cwe_ids']:weighted[label]+=n['similarity']
 assert x['prediction']==sorted(weighted,key=lambda label:(-weighted[label],label))[0]
 records.append({'row_index':r['row_index'],'prompt':r['prompt'],'swcv':x['prediction'],'majority':sorted(counts,key=lambda label:(-counts[label],label))[0]})
assert len(records)==len({r['row_index'] for r in records})==1000
with (O/'predictions.jsonl').open('x') as f:
 for r in records:f.write(json.dumps(r)+'\n')
save('prediction_freeze.json',{'sha256':sha(O/'predictions.jsonl'),'n':1000,'gold_opened_for_this_analysis':False,'source_sha256':sha(source),'majority':'one vote per attached structured CWE label; lexical CWE tie break; same frozen top20 references as SWCV','selection':'fixed control, no label-driven selection'})
with (R/'data/cti-bench/data/cti-rcm.tsv').open() as f:benchmark=list(csv.DictReader(f,delimiter='\t'))
counts=np.zeros(4,dtype=np.int64)
for r in records:
 b=benchmark[r['row_index']];assert r['prompt']==b['Prompt'];gold='CWE-'+str(int(b['GT'].split('-')[1]));s=r['swcv']==gold;m=r['majority']==gold;counts[0 if s and m else 1 if s else 2 if m else 3]+=1
both,wins,losses,neither=map(int,counts);n=len(records);d=wins+losses;p=min(1.,2*sum(math.comb(d,k) for k in range(min(wins,losses)+1))/2**d) if d else 1.
rng=np.random.default_rng(20261108);draws=rng.multinomial(n,counts/n,size=200000);ci=lambda a:list(map(float,np.percentile(a,[2.5,97.5])))
result={'condition':'all1000 CTI-Bench, source-supported433 exclusions,120238 references,top20','n':n,'swcv_correct':both+wins,'majority_correct':both+losses,'swcv_accuracy_percent':100*(both+wins)/n,'majority_accuracy_percent':100*(both+losses)/n,'discordant':{'swcv_only_correct':wins,'majority_only_correct':losses,'both_correct':both,'neither_correct':neither},'difference_points':100*(wins-losses)/n,'difference_95_percentile_bootstrap_CI_points':ci(100*(draws[:,1]-draws[:,2])/n),'swcv_accuracy_95_percentile_bootstrap_CI_percent':ci(100*(draws[:,0]+draws[:,1])/n),'majority_accuracy_95_percentile_bootstrap_CI_percent':ci(100*(draws[:,0]+draws[:,2])/n),'exact_two_sided_mcnemar_p':p,'bootstrap_resamples':200000,'seed':20261108,'unit':'paired question outcomes; multinomial draws equivalent to resampling paired cases','limitations':'Question-level independent-case assumption; conditional on frozen predictions and retrospective design. Does not estimate run-to-run variation or account for vulnerability-family clustering. One fixed majority-vote comparison; p unadjusted.','prediction_sha256':sha(O/'predictions.jsonl'),'code_sha256':sha(Path(__file__))}
save('results.json',result);print(json.dumps(result,indent=2))
